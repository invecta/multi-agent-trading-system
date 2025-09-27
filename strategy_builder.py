"""
Interactive strategy builder with drag-and-drop components
Allows users to create custom trading strategies with visual components
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta

class StrategyComponent:
    """Base class for strategy components"""
    
    def __init__(self, component_id: str, component_type: str, parameters: Dict = None):
        self.id = component_id
        self.type = component_type
        self.parameters = parameters or {}
        self.connections = []
        
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'parameters': self.parameters,
            'connections': self.connections
        }
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the component logic"""
        raise NotImplementedError

class DataSourceComponent(StrategyComponent):
    """Data source component"""
    
    def __init__(self, component_id: str, symbol: str = 'AAPL', timeframe: str = '1d'):
        super().__init__(component_id, 'data_source', {
            'symbol': symbol,
            'timeframe': timeframe
        })
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        # In a real implementation, this would fetch live data
        # For now, return the input data
        return data

class MovingAverageComponent(StrategyComponent):
    """Moving average component"""
    
    def __init__(self, component_id: str, period: int = 20, ma_type: str = 'SMA'):
        super().__init__(component_id, 'moving_average', {
            'period': period,
            'ma_type': ma_type
        })
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Close' not in data.columns:
            return data
            
        period = self.parameters['period']
        ma_type = self.parameters['ma_type']
        
        if ma_type == 'SMA':
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        elif ma_type == 'EMA':
            data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
            
        return data

class RSIComponent(StrategyComponent):
    """RSI component"""
    
    def __init__(self, component_id: str, period: int = 14):
        super().__init__(component_id, 'rsi', {'period': period})
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Close' not in data.columns:
            return data
            
        period = self.parameters['period']
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data

class SignalComponent(StrategyComponent):
    """Signal generation component"""
    
    def __init__(self, component_id: str, signal_type: str = 'crossover', 
                 buy_condition: str = 'MA_20 > MA_50', sell_condition: str = 'MA_20 < MA_50'):
        super().__init__(component_id, 'signal', {
            'signal_type': signal_type,
            'buy_condition': buy_condition,
            'sell_condition': sell_condition
        })
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        # Generate signals based on conditions
        data['Signal'] = 0
        
        # Simple crossover example
        if 'MA_20' in data.columns and 'MA_50' in data.columns:
            data['Signal'] = np.where(data['MA_20'] > data['MA_50'], 1, 
                                    np.where(data['MA_20'] < data['MA_50'], -1, 0))
        
        return data

class StrategyBuilder:
    """Main strategy builder class"""
    
    def __init__(self):
        self.components = {}
        self.connections = []
        self.component_types = {
            'data_source': {
                'name': 'Data Source',
                'icon': '📊',
                'color': '#1f77b4',
                'inputs': 0,
                'outputs': 1
            },
            'moving_average': {
                'name': 'Moving Average',
                'icon': '📈',
                'color': '#ff7f0e',
                'inputs': 1,
                'outputs': 1
            },
            'rsi': {
                'name': 'RSI',
                'icon': '📉',
                'color': '#2ca02c',
                'inputs': 1,
                'outputs': 1
            },
            'signal': {
                'name': 'Signal Generator',
                'icon': '⚡',
                'color': '#d62728',
                'inputs': 1,
                'outputs': 1
            }
        }
        
    def add_component(self, component: StrategyComponent):
        """Add a component to the strategy"""
        self.components[component.id] = component
        
    def remove_component(self, component_id: str):
        """Remove a component from the strategy"""
        if component_id in self.components:
            del self.components[component_id]
            # Remove connections
            self.connections = [conn for conn in self.connections 
                              if conn['from'] != component_id and conn['to'] != component_id]
            
    def add_connection(self, from_component: str, to_component: str):
        """Add a connection between components"""
        if from_component in self.components and to_component in self.components:
            connection = {'from': from_component, 'to': to_component}
            if connection not in self.connections:
                self.connections.append(connection)
                self.components[to_component].connections.append(from_component)
                
    def remove_connection(self, from_component: str, to_component: str):
        """Remove a connection between components"""
        connection = {'from': from_component, 'to': to_component}
        if connection in self.connections:
            self.connections.remove(connection)
            if from_component in self.components[to_component].connections:
                self.components[to_component].connections.remove(from_component)
                
    def execute_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete strategy"""
        if not self.components:
            return data
            
        # Find data source component
        data_source = None
        for component in self.components.values():
            if component.type == 'data_source':
                data_source = component
                break
                
        if not data_source:
            return data
            
        # Execute components in order
        current_data = data_source.execute(data)
        
        # Simple execution order (in a real implementation, this would be more sophisticated)
        for component in self.components.values():
            if component.type != 'data_source':
                current_data = component.execute(current_data)
                
        return current_data
        
    def get_strategy_json(self) -> str:
        """Export strategy as JSON"""
        strategy_data = {
            'components': {comp_id: comp.to_dict() for comp_id, comp in self.components.items()},
            'connections': self.connections
        }
        return json.dumps(strategy_data, indent=2)
        
    def load_strategy_json(self, strategy_json: str):
        """Load strategy from JSON"""
        try:
            strategy_data = json.loads(strategy_json)
            
            # Clear existing components
            self.components.clear()
            self.connections.clear()
            
            # Load components
            for comp_id, comp_data in strategy_data['components'].items():
                component = self._create_component_from_dict(comp_data)
                if component:
                    self.components[comp_id] = component
                    
            # Load connections
            self.connections = strategy_data['connections']
            
        except Exception as e:
            print(f"Error loading strategy: {e}")
            
    def _create_component_from_dict(self, comp_data: Dict) -> Optional[StrategyComponent]:
        """Create component from dictionary data"""
        comp_type = comp_data['type']
        comp_id = comp_data['id']
        parameters = comp_data.get('parameters', {})
        
        if comp_type == 'data_source':
            return DataSourceComponent(comp_id, 
                                     parameters.get('symbol', 'AAPL'),
                                     parameters.get('timeframe', '1d'))
        elif comp_type == 'moving_average':
            return MovingAverageComponent(comp_id,
                                        parameters.get('period', 20),
                                        parameters.get('ma_type', 'SMA'))
        elif comp_type == 'rsi':
            return RSIComponent(comp_id, parameters.get('period', 14))
        elif comp_type == 'signal':
            return SignalComponent(comp_id,
                                 parameters.get('signal_type', 'crossover'),
                                 parameters.get('buy_condition', 'MA_20 > MA_50'),
                                 parameters.get('sell_condition', 'MA_20 < MA_50'))
        
        return None

def create_strategy_builder_layout():
    """Create the strategy builder layout"""
    
    return html.Div([
        # Header
        html.Div([
            html.H2("Strategy Builder", className="text-center mb-4"),
            html.P("Drag and drop components to build your trading strategy", 
                   className="text-center text-muted")
        ]),
        
        # Main content area
        html.Div([
            # Component palette
            html.Div([
                html.H4("Components", className="mb-3"),
                html.Div(id="component-palette", children=[
                    create_component_card('data_source', 'Data Source', '📊', '#1f77b4'),
                    create_component_card('moving_average', 'Moving Average', '📈', '#ff7f0e'),
                    create_component_card('rsi', 'RSI', '📉', '#2ca02c'),
                    create_component_card('signal', 'Signal Generator', '⚡', '#d62728')
                ])
            ], className="col-md-3", style={'border-right': '1px solid #dee2e6'}),
            
            # Canvas area
            html.Div([
                html.Div(id="strategy-canvas", 
                        style={'min-height': '500px', 'border': '2px dashed #dee2e6', 
                               'border-radius': '8px', 'padding': '20px'}),
                html.Div(id="canvas-drop-zone", 
                        style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
                               'z-index': 1000, 'display': 'none'})
            ], className="col-md-6", style={'position': 'relative'}),
            
            # Properties panel
            html.Div([
                html.H4("Properties", className="mb-3"),
                html.Div(id="properties-panel", children=[
                    html.P("Select a component to edit its properties", 
                           className="text-muted text-center")
                ])
            ], className="col-md-3")
            
        ], className="row"),
        
        # Control buttons
        html.Div([
            dbc.Button("Execute Strategy", id="execute-strategy-btn", 
                      color="primary", className="me-2"),
            dbc.Button("Save Strategy", id="save-strategy-btn", 
                      color="success", className="me-2"),
            dbc.Button("Load Strategy", id="load-strategy-btn", 
                      color="info", className="me-2"),
            dbc.Button("Clear Canvas", id="clear-canvas-btn", 
                      color="warning", className="me-2")
        ], className="text-center mt-4"),
        
        # Results area
        html.Div([
            html.H4("Strategy Results", className="mt-4"),
            html.Div(id="strategy-results")
        ])
        
    ], className="container-fluid")

def create_component_card(comp_type: str, name: str, icon: str, color: str):
    """Create a draggable component card"""
    
    return html.Div([
        html.Div([
            html.Span(icon, style={'font-size': '24px', 'margin-right': '8px'}),
            html.Span(name, style={'font-weight': 'bold'})
        ], style={
            'padding': '10px',
            'border': f'2px solid {color}',
            'border-radius': '8px',
            'background-color': f'{color}20',
            'cursor': 'grab',
            'margin-bottom': '10px',
            'text-align': 'center'
        }, 
        id=f"component-{comp_type}",
        draggable=True,
        **{"data-component-type": comp_type})
    ])

def create_strategy_builder_callbacks(app, strategy_builder: StrategyBuilder):
    """Create callbacks for the strategy builder"""
    
    @app.callback(
        [Output("strategy-canvas", "children"),
         Output("properties-panel", "children"),
         Output("strategy-results", "children")],
        [Input("execute-strategy-btn", "n_clicks"),
         Input("save-strategy-btn", "n_clicks"),
         Input("load-strategy-btn", "n_clicks"),
         Input("clear-canvas-btn", "n_clicks")],
        [State("strategy-canvas", "children"),
         State("properties-panel", "children")]
    )
    def handle_strategy_actions(execute_clicks, save_clicks, load_clicks, clear_clicks,
                               canvas_children, properties_children):
        
        ctx = callback_context
        if not ctx.triggered:
            return canvas_children or [], properties_children or [], []
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "execute-strategy-btn":
            # Execute strategy
            results = execute_strategy(strategy_builder)
            return canvas_children or [], properties_children or [], results
            
        elif button_id == "save-strategy-btn":
            # Save strategy
            strategy_json = strategy_builder.get_strategy_json()
            return canvas_children or [], properties_children or [], [
                html.Div([
                    html.H5("Strategy Saved"),
                    html.Pre(strategy_json, style={'background': '#f8f9fa', 'padding': '10px'})
                ])
            ]
            
        elif button_id == "load-strategy-btn":
            # Load strategy (placeholder)
            return canvas_children or [], properties_children or [], [
                html.Div([
                    html.H5("Load Strategy"),
                    html.P("Strategy loading functionality would be implemented here")
                ])
            ]
            
        elif button_id == "clear-canvas-btn":
            # Clear canvas
            strategy_builder.components.clear()
            strategy_builder.connections.clear()
            return [], [
                html.P("Select a component to edit its properties", 
                       className="text-muted text-center")
            ], []
            
        return canvas_children or [], properties_children or [], []

def execute_strategy(strategy_builder: StrategyBuilder):
    """Execute the strategy and return results"""
    
    if not strategy_builder.components:
        return [html.P("No components in strategy", className="text-muted")]
        
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(100) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Execute strategy
    result_data = strategy_builder.execute_strategy(sample_data)
    
    # Create results visualization
    fig = go.Figure()
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=result_data['Date'],
        y=result_data['Close'],
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add moving averages if they exist
    for col in result_data.columns:
        if col.startswith('MA_') or col.startswith('EMA_'):
            fig.add_trace(go.Scatter(
                x=result_data['Date'],
                y=result_data[col],
                name=col,
                line=dict(width=1)
            ))
    
    # Add signals if they exist
    if 'Signal' in result_data.columns:
        buy_signals = result_data[result_data['Signal'] == 1]
        sell_signals = result_data[result_data['Signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Signal'
            ))
            
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell Signal'
            ))
    
    fig.update_layout(
        title='Strategy Execution Results',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400,
        template='plotly_dark'
    )
    
    return [dcc.Graph(figure=fig)]

# Global strategy builder instance
strategy_builder = StrategyBuilder()
