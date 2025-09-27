#!/usr/bin/env python3
"""
Professional Multi-Agent Trading System Dashboard
Enhanced for Data Analyst Portfolio Showcase
"""
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Initialize Dash app with professional theme
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])
app.title = "Multi-Agent Trading System - Professional Analytics Dashboard"

# Available symbols with sector information
symbols_data = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "base_price": 150},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "base_price": 2800},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "base_price": 300},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "base_price": 200},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "base_price": 400},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer", "base_price": 3000},
    "META": {"name": "Meta Platforms", "sector": "Technology", "base_price": 300},
    "NFLX": {"name": "Netflix Inc.", "sector": "Media", "base_price": 400}
}

# Enhanced data generator with realistic market behavior
def generate_professional_data(symbol, start_date, end_date):
    """Generate realistic market data with sector-specific behavior"""
    print(f"Generating professional market data for {symbol}")
    
    symbol_info = symbols_data[symbol]
    base_price = symbol_info["base_price"]
    sector = symbol_info["sector"]
    
    # Sector-specific volatility
    volatility_multipliers = {
        "Technology": 1.2,
        "Automotive": 1.5,
        "Consumer": 0.8,
        "Media": 1.1
    }
    
    volatility = volatility_multipliers.get(sector, 1.0)
    n_days = 250
    
    # Generate realistic price series with trends
    np.random.seed(hash(symbol) % 2**32)
    
    # Create trend component
    trend = np.linspace(0, 0.1, n_days)  # 10% upward trend over year
    
    # Generate returns with sector-specific volatility
    returns = np.random.normal(0.0005, 0.02 * volatility, n_days) + trend/n_days
    
    prices = [base_price]
    volumes = []
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))
        
        # Volume correlates with price movement
        volume_base = random.randint(1000000, 10000000)
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
        volumes.append(int(volume_base * volume_multiplier))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    })
    
    # Ensure High >= Low >= Close
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Add date index
    df['Date'] = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    df.set_index('Date', inplace=True)
    
    print(f"Generated {len(df)} days of {sector} sector data")
    return df

def calculate_advanced_indicators(df):
    """Calculate professional technical indicators"""
    print("Calculating advanced technical indicators...")
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    print("Advanced indicators calculated")
    return df

def generate_professional_signals(df):
    """Generate sophisticated trading signals"""
    print("Generating professional trading signals...")
    
    df['Signal'] = 0
    df['Signal_Strength'] = 0.0
    
    # Multi-factor signal generation
    for i in range(50, len(df)):  # Start after indicators stabilize
        signal_strength = 0
        
        # SMA Crossover
        if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]:
            signal_strength += 0.3
        elif df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] and df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1]:
            signal_strength -= 0.3
        
        # RSI signals
        if df['RSI'].iloc[i] < 30:
            signal_strength += 0.2
        elif df['RSI'].iloc[i] > 70:
            signal_strength -= 0.2
        
        # MACD signals
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
            signal_strength += 0.2
        elif df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
            signal_strength -= 0.2
        
        # Bollinger Bands
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            signal_strength += 0.1
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            signal_strength -= 0.1
        
        # Volume confirmation
        if df['Volume_Ratio'].iloc[i] > 1.5:  # High volume
            signal_strength *= 1.2
        
        df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
        
        # Generate signal based on strength
        if signal_strength > 0.5:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif signal_strength < -0.5:
            df.iloc[i, df.columns.get_loc('Signal')] = -1
    
    signal_count = len(df[df['Signal'] != 0])
    print(f"Generated {signal_count} professional signals")
    return df

def run_professional_backtest(df, initial_capital=100000):
    """Run sophisticated backtest with detailed analytics"""
    print(f"Running professional backtest with ${initial_capital:,}")
    
    capital = initial_capital
    position = 0
    portfolio_values = [initial_capital]
    trades = []
    daily_returns = []
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        signal_strength = df['Signal_Strength'].iloc[i]
        
        if signal == 1 and position == 0:  # Buy
            # Position sizing based on signal strength
            position_size = capital * 0.95 * min(abs(signal_strength), 1.0) / current_price
            position = position_size
            capital -= position_size * current_price
            trades.append({
                'Date': f"Day {i}",
                'Type': 'BUY',
                'Price': round(current_price, 2),
                'Quantity': round(position_size, 2),
                'Value': round(position_size * current_price, 2),
                'Signal_Strength': round(signal_strength, 2)
            })
        elif signal == -1 and position > 0:  # Sell
            capital += position * current_price
            trades.append({
                'Date': f"Day {i}",
                'Type': 'SELL',
                'Price': round(current_price, 2),
                'Quantity': round(position, 2),
                'Value': round(position * current_price, 2),
                'Signal_Strength': round(signal_strength, 2)
            })
            position = 0
        
        # Calculate portfolio value and returns
        current_value = capital + (position * current_price)
        portfolio_values.append(current_value)
        
        if len(portfolio_values) > 1:
            daily_return = (current_value - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
    
    # Close position
    if position > 0:
        capital += position * df['Close'].iloc[-1]
        trades.append({
            'Date': f"Day {len(df)}",
            'Type': 'SELL',
            'Price': round(df['Close'].iloc[-1], 2),
            'Quantity': round(position, 2),
            'Value': round(position * df['Close'].iloc[-1], 2),
            'Signal_Strength': 0
        })
        portfolio_values[-1] = capital
    
    # Calculate comprehensive metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    days = len(df)
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Advanced risk metrics
    returns_series = pd.Series(daily_returns)
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() != 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns_series[returns_series < 0]
    sortino_ratio = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
    
    # Max drawdown
    portfolio_series = pd.Series(portfolio_values)
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    if trades:
        profitable_trades = 0
        for i in range(1, len(trades), 2):  # Check sell trades
            if i < len(trades) and trades[i]['Type'] == 'SELL':
                buy_price = trades[i-1]['Price']
                sell_price = trades[i]['Price']
                if sell_price > buy_price:
                    profitable_trades += 1
        win_rate = profitable_trades / (len(trades) // 2) if len(trades) > 1 else 0
    else:
        win_rate = 0
    
    print(f"Professional backtest completed: {total_return:.2%} return, {len(trades)} trades")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'final_value': final_value,
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'trade_history': trades
    }

# Professional Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
                "Multi-Agent Trading System"
            ], style={'color': '#2c3e50', 'marginBottom': '5px'}),
            html.H3("Professional Analytics Dashboard", style={'color': '#7f8c8d', 'marginBottom': '0px'}),
            html.P("Advanced AI-Powered Trading Analytics & Risk Management", 
                   style={'color': '#95a5a6', 'fontSize': '14px', 'marginTop': '5px'})
        ], style={'textAlign': 'center', 'marginBottom': '30px'})
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-cogs", style={'marginRight': '8px'}),
                    "Trading Controls"
                ], style={'color': '#2c3e50', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-chart-bar", style={'marginRight': '5px'}),
                        "Symbol:"
                    ], style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': f"{s} - {symbols_data[s]['name']}", 'value': s} for s in symbols_data.keys()],
                        value='AAPL',
                        style={'marginBottom': '20px'}
                    )
                ]),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-dollar-sign", style={'marginRight': '5px'}),
                        "Initial Capital:"
                    ], style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(
                        id='initial-capital',
                        type='number',
                        value=100000,
                        min=1000,
                        max=1000000,
                        step=1000,
                        style={'width': '100%', 'marginBottom': '20px'}
                    )
                ]),
                
                html.Button([
                    html.I(className="fas fa-play", style={'marginRight': '8px'}),
                    "Run Professional Analysis"
                ], id='run-backtest-btn', style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 30px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                })
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
        
        # Performance Summary
        html.Div([
            html.H4([
                html.I(className="fas fa-trophy", style={'marginRight': '8px'}),
                "Performance Analytics"
            ], style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div(id='performance-summary')
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    # Advanced Charts Section
    html.Div([
        html.Div([
            html.H4([
                html.I(className="fas fa-chart-area", style={'marginRight': '8px'}),
                "Portfolio Performance"
            ], style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='portfolio-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4([
                html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px'}),
                "Risk Analysis"
            ], style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='risk-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.H4([
                html.I(className="fas fa-chart-pie", style={'marginRight': '8px'}),
                "Technical Indicators"
            ], style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='technical-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4([
                html.I(className="fas fa-table", style={'marginRight': '8px'}),
                "Trade Analysis"
            ], style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(id='trade-analysis')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'marginBottom': '30px'}),
    
    # Sector Analysis
    html.Div([
        html.H4([
            html.I(className="fas fa-industry", style={'marginRight': '8px'}),
            "Sector Performance Analysis"
        ], style={'color': '#2c3e50', 'marginBottom': '20px'}),
        html.Div([
            html.Label("Select Multiple Symbols for Sector Analysis:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='sector-analysis-symbols',
                options=[{'label': f"{s} - {symbols_data[s]['name']} ({symbols_data[s]['sector']})", 'value': s} for s in symbols_data.keys()],
                value=['AAPL', 'MSFT', 'NVDA'],
                multi=True,
                style={'marginBottom': '20px'}
            ),
            html.Button([
                html.I(className="fas fa-chart-bar", style={'marginRight': '8px'}),
                "Analyze Sectors"
            ], id='sector-analysis-btn', style={
                'backgroundColor': '#27ae60',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'marginBottom': '20px'
            })
        ]),
        dcc.Graph(id='sector-analysis-chart')
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# Global variable to store results
backtest_results = {}

# Callbacks
@app.callback(
    [Output('performance-summary', 'children'),
     Output('portfolio-chart', 'figure'),
     Output('risk-chart', 'figure'),
     Output('technical-chart', 'figure'),
     Output('trade-analysis', 'children')],
    [Input('run-backtest-btn', 'n_clicks')],
    [Input('symbol-dropdown', 'value'),
     Input('initial-capital', 'value')]
)
def run_professional_analysis(n_clicks, symbol, initial_capital):
    """Run professional analysis and update dashboard"""
    if n_clicks is None:
        empty_fig = go.Figure()
        empty_table = html.Div("Click 'Run Professional Analysis' to see results", 
                              style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        return "No analysis run yet", empty_fig, empty_fig, empty_fig, empty_table
    
    try:
        print(f"\n=== Starting Professional Analysis for {symbol} ===")
        
        # Generate data
        df = generate_professional_data(symbol, '2023-01-01', '2024-01-01')
        
        # Calculate indicators
        df = calculate_advanced_indicators(df)
        
        # Generate signals
        df = generate_professional_signals(df)
        
        # Run backtest
        result = run_professional_backtest(df, initial_capital)
        
        # Store result
        backtest_results[symbol] = result
        
        # Create professional performance summary
        summary = html.Div([
            html.Div([
                html.Div([
                    html.H2(f"{result['total_return']:.2%}", style={'color': '#27ae60', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Total Return", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['sharpe_ratio']:.2f}", style={'color': '#3498db', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Sharpe Ratio", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['sortino_ratio']:.2f}", style={'color': '#9b59b6', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Sortino Ratio", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['max_drawdown']:.2%}", style={'color': '#e74c3c', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Max Drawdown", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['win_rate']:.1%}", style={'color': '#f39c12', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Win Rate", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"${result['final_value']:,.0f}", style={'color': '#27ae60', 'margin': '0', 'fontSize': '28px'}),
                    html.P("Final Value", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '16%', 'display': 'inline-block'})
        ])
        
        # Create professional portfolio chart
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            y=result['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3498db', width=3),
            hovertemplate='Day: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add benchmark (buy and hold)
        benchmark_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        benchmark_values = [initial_capital * (1 + benchmark_return * i / len(df)) for i in range(len(result['portfolio_values']))]
        portfolio_fig.add_trace(go.Scatter(
            y=benchmark_values,
            mode='lines',
            name='Buy & Hold Benchmark',
            line=dict(color='#95a5a6', width=2, dash='dash'),
            hovertemplate='Day: %{x}<br>Benchmark: $%{y:,.2f}<extra></extra>'
        ))
        
        portfolio_fig.update_layout(
            title=f"Portfolio Performance vs Benchmark - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=400,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        # Create risk analysis chart
        portfolio_series = pd.Series(result['portfolio_values'])
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='#e74c3c'),
            fillcolor='rgba(231,76,60,0.3)',
            hovertemplate='Day: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        risk_fig.update_layout(
            title=f"Risk Analysis - Drawdown Profile",
            xaxis_title="Trading Days",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        # Create technical indicators chart
        technical_fig = go.Figure()
        
        # Price and moving averages
        technical_fig.add_trace(go.Scatter(
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2c3e50', width=2)
        ))
        technical_fig.add_trace(go.Scatter(
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#3498db', width=1)
        ))
        technical_fig.add_trace(go.Scatter(
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#e74c3c', width=1)
        ))
        
        # Add Bollinger Bands
        technical_fig.add_trace(go.Scatter(
            y=df['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='#95a5a6', width=1, dash='dot'),
            showlegend=False
        ))
        technical_fig.add_trace(go.Scatter(
            y=df['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='#95a5a6', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(149,165,166,0.1)',
            showlegend=False
        ))
        
        # Add signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        if not buy_signals.empty:
            technical_fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='#27ae60', size=8, symbol='triangle-up'),
                hovertemplate='Buy Signal<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        if not sell_signals.empty:
            technical_fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='#e74c3c', size=8, symbol='triangle-down'),
                hovertemplate='Sell Signal<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        technical_fig.update_layout(
            title=f"Technical Analysis - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        # Create trade analysis table
        if result['trade_history']:
            trade_df = pd.DataFrame(result['trade_history'])
            
            # Calculate trade performance
            trade_performance = []
            for i in range(1, len(trade_df), 2):
                if i < len(trade_df) and trade_df.iloc[i]['Type'] == 'SELL':
                    buy_price = trade_df.iloc[i-1]['Price']
                    sell_price = trade_df.iloc[i]['Price']
                    pnl = sell_price - buy_price
                    pnl_pct = (pnl / buy_price) * 100
                    trade_performance.append({
                        'Trade': f"Trade {len(trade_performance) + 1}",
                        'Buy Price': f"${buy_price:.2f}",
                        'Sell Price': f"${sell_price:.2f}",
                        'PnL': f"${pnl:.2f}",
                        'PnL %': f"{pnl_pct:.2f}%",
                        'Signal Strength': f"{trade_df.iloc[i-1]['Signal_Strength']:.2f}"
                    })
            
            if trade_performance:
                perf_df = pd.DataFrame(trade_performance)
                trade_table = html.Table([
                    html.Thead([
                        html.Tr([html.Th(col, style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}) for col in perf_df.columns])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(perf_df.iloc[i][col], style={'padding': '8px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'center'}) for col in perf_df.columns
                        ]) for i in range(len(perf_df))
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px', 'backgroundColor': '#ffffff'})
            else:
                trade_table = html.P("No completed trades", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        else:
            trade_table = html.P("No trades executed", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        
        print(f"=== Professional analysis completed for {symbol} ===")
        return summary, portfolio_fig, risk_fig, technical_fig, trade_table
        
    except Exception as e:
        error_msg = f"Error running analysis: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, go.Figure(), go.Figure(), go.Figure(), error_msg

@app.callback(
    Output('sector-analysis-chart', 'figure'),
    [Input('sector-analysis-btn', 'n_clicks')],
    [Input('sector-analysis-symbols', 'value')]
)
def update_sector_analysis(n_clicks, symbols):
    """Update sector analysis chart"""
    if n_clicks is None or not symbols:
        return go.Figure()
    
    try:
        sector_data = []
        
        for symbol in symbols:
            if symbol in backtest_results:
                result = backtest_results[symbol]
                sector_info = symbols_data[symbol]
                sector_data.append({
                    'Symbol': symbol,
                    'Company': sector_info['name'],
                    'Sector': sector_info['sector'],
                    'Total Return': result['total_return'] * 100,
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown': abs(result['max_drawdown']) * 100,
                    'Win Rate': result['win_rate'] * 100
                })
        
        if sector_data:
            df_sector = pd.DataFrame(sector_data)
            
            # Create subplot with multiple metrics
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Return by Sector', 'Sharpe Ratio Comparison', 
                              'Risk vs Return', 'Win Rate Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Color by sector
            colors = {'Technology': '#3498db', 'Automotive': '#e74c3c', 
                     'Consumer': '#27ae60', 'Media': '#f39c12'}
            
            for sector in df_sector['Sector'].unique():
                sector_df = df_sector[df_sector['Sector'] == sector]
                color = colors.get(sector, '#95a5a6')
                
                fig.add_trace(
                    go.Bar(x=sector_df['Symbol'], y=sector_df['Total Return'], 
                          name=f'{sector} Return', marker_color=color, showlegend=True),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=sector_df['Symbol'], y=sector_df['Sharpe Ratio'], 
                          name=f'{sector} Sharpe', marker_color=color, showlegend=False),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=sector_df['Max Drawdown'], y=sector_df['Total Return'], 
                              mode='markers+text', text=sector_df['Symbol'], 
                              name=f'{sector} Risk-Return', marker_color=color, showlegend=False),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=sector_df['Symbol'], y=sector_df['Win Rate'], 
                          name=f'{sector} Win Rate', marker_color=color, showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Comprehensive Sector Performance Analysis",
                height=600,
                showlegend=True
            )
            
            return fig
        else:
            return go.Figure()
            
    except Exception as e:
        return go.Figure()

# Run the professional dashboard
if __name__ == "__main__":
    print("Starting Professional Multi-Agent Trading System Dashboard...")
    print("Dashboard will be available at: http://localhost:8056")
    print("Enhanced for Data Analyst Portfolio Showcase!")
    print("Features: Advanced Analytics, Sector Analysis, Professional Visualizations")
    
    app.run(debug=True, host='0.0.0.0', port=8056)
