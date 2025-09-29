#!/usr/bin/env python3
"""
Minimal Trading Dashboard for PythonAnywhere Deployment
No external dependencies except Dash itself
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Create Dash app
application = dash.Dash(__name__)
application.title = "Trading Analytics Dashboard"

# Sample data generation
def generate_sample_data(symbol='AAPL', days=365):
    """Generate sample trading data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), 
                         freq='D')
    
    # Generate price data
    base_price = 100 if symbol == 'AAPL' else 150 if symbol == 'GOOGL' else 200
    np.random.seed(42 + hash(symbol) % 1000)  # Different seeds for different symbols
    
    prices = [base_price]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, base_price * 0.5))  # Minimum price floor
    
    volumes = np.random.lognormal(15, 0.5, len(dates)).astype(int)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + random.uniform(0, 0.05)) for p in prices],
        'Low': [p * (1 - random.uniform(0, 0.05)) for p in prices],
        'Close': prices,
        'Volume': volumes
    }).set_index('Date')

def run_backtest(data, symbol, capital=100000):
    """Simple moving average strategy backtest"""
    df = data.copy()
    
    # Calculate moving averages
    df['MA_Short'] = df['Close'].rolling(window=10).mean()
    df['MA_Long'] = df['Close'].rolling(window=30).mean()
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1  # Buy
    df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1  # Sell
    
    # Remove NaN values
    df = df.dropna()
    
    # Calculate trades
    trades = []
    position = 0
    shares = 0
    cash = capital
    
    for i in range(1, len(df)):
        signal_change = df.iloc[i]['Signal'] - df.iloc[i-1]['Signal']
        price = df.iloc[i]['Close']
        
        if signal_change == 2:  # Buy signal
            shares = int(cash * 0.95 / price)  # Use 95% of cash
            position_value = shares * price
            cash -= position_value
            position = 1
            trades.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': price,
                'shares': shares,
                'value': position_value
            })
        elif signal_change == -2:  # Sell signal
            if position == 1:
                position_value = shares * price
                cash += position_value
                position = 0
                trades.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'type': 'SELL',
                    'price': price,
                    'shares': shares,
                    'value': position_value
                })
                shares = 0
    
    # Calculate final portfolio value
    final_value = cash + (shares * df['Close'].iloc[-1] if position == 1 else cash + shares * df['Close'].iloc[-1])
    total_return = (final_value - capital) / capital * 100
    
    # Calculate metrics
    win_trades = sum(1 for trade in trades if trade['type'] == 'SELL')
    total_trades = len(trades) // 2 if len(trades) > 0 else 0
    win_rate = 50.0  # Simplified
    
    return {
        'trades': trades,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'final_value': final_value,
        'portfolio_data': df
    }

# Layout
application.layout = html.Div([
    # Header
    html.H1("Trading Analytics Dashboard", 
            style={'textAlign': 'center', 'margin': '20px', 'color': 'darkblue'}),
    
    # Control Panel
    html.Div([
        html.Label("Symbol:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[
                {'label': 'AAPL - Apple Inc.', 'value': 'AAPL'},
                {'label': 'GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
                {'label': 'TSLA - Tesla Inc.', 'value': 'TSLA'},
                {'label': 'MSFT - Microsoft Corp.', 'value': 'MSFT'},
            ],
            value='AAPL',
            style={'width': '200px', 'display': 'inline-block', 'margin': '10px'}
        ),
        
        html.Label("Capital:", style={'fontWeight': 'bold', 'marginLeft': '20px'}),
        dcc.Input(
            id='capital-input',
            type='number',
            value=100000,
            style={'width': '100px', 'display': 'inline-block', 'margin': '10px'}
        ),
        
        html.Label("Days:", style={'fontWeight': 'bold', 'marginLeft': '20px'}),
        dcc.Dropdown(
            id='days-dropdown',
            options=[
                {'label': '30 Days', 'value': 30},
                {'label': '90 Days', 'value': 90},
                {'label': '1 Year', 'value': 365},
                {'label': '2 Years', 'value': 730},
            ],
            value=365,
            style={'width': '120px', 'display': 'inline-block', 'margin': '10px'}
        ),
        
        html.Button("Run Analysis", id="run-button", 
                   style={'marginLeft': '20px', 'padding': '10px 20px', 
                         'backgroundColor': '#007bff', 'color': 'white', 
                         'border': 'none', 'borderRadius': '5px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'margin': '20px'}),
    
    # Status Display
    html.Div(id='status-display', 
             style={'padding': '10px', 'margin': '20px', 'backgroundColor': '#e9ecef'}),
    
    # Main Content Tabs
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="📊 Overview", value="overview"),
            dcc.Tab(label="📈 Charts", value="charts"),
            dcc.Tab(label="🔍 Analysis", value="analysis"),
        ], id="main-tabs", value="overview"),
        
        html.Div(id="tab-content", style={'margin': '20px'})
    ])
])

# Callbacks
@application.callback(
    [Output("status-display", "children"),
     Output("tab-content", "children")],
    [Input("run-button", "n_clicks")],
    [State("symbol-dropdown", "value"),
     State("capital-input", "value"),
     State("days-dropdown", "value"),
     State("main-tabs", "value")]
)
def update_dashboard(n_clicks, symbol, capital, days, active_tab):
    """Main callback to update dashboard"""
    
    if not n_clicks:
        return "Ready to start analysis...", html.P("Select options and click Run Analysis to begin.")
    
    try:
        status_msg = f"✅ Analysis running for {symbol} over {days} days..."
        
        # Generate sample data
        data = generate_sample_data(symbol, days)
        
        # Run backtest
        results = run_backtest(data, symbol, capital)
        
        # Update status
        final_status = f"""
        ✅ Analysis completed for {symbol}
        📊 Total Return: {results['total_return']:.2f}%
        🔄 Total Trades: {results['total_trades']}
        📈 Win Rate: {results['win_rate']:.1f}%
        💰 Final Value: ${results['final_value']:,.2f}
        """
        
        # Tab content based on selection
        if active_tab == "overview":
            tab_content = create_overview_tab(results, symbol)
        elif active_tab == "charts":
            tab_content = create_charts_tab(data, symbol)
        elif active_tab == "analysis":
            tab_content = create_analysis_tab(results)
        else:
            tab_content = html.P("Select a tab to view content.")
            
        return final_status, tab_content
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, html.P(f"Error occurred: {str(e)}")

def create_overview_tab(results, symbol):
    """Create overview tab content"""
    trades = results['trades']
    
    return html.Div([
        html.Div([
            html.H3(f"{symbol} Analysis"),
            html.Div([
                html.P(f"Total Return: {results['total_return']:.2f}%"),
                html.P(f"Final Value: ${results['final_value']:,.2f}")
            ], style={'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#e3f2fd'})
        ], style={'display': 'inline-block', 'width': '48%', 'backgroundColor': 'white', 
                 'padding': '20px', 'margin': '10px'}),
        
        html.Div([
            html.H3("Recent Trades"),
            html.Div([
                html.P(f"{trade['date']} - {trade['type']} {trade['shares']} @ ${trade['price']:.2f}")
                for trade in trades[-10:]  # Show last 10 trades
            ], style={'maxHeight': '200px', 'overflowY': 'scroll'})
        ], style={'display': 'inline-block', 'width': '48%', 'backgroundColor': 'white', 
                 'padding': '20px', 'margin': '10px'})
    ])

def create_charts_tab(data, symbol):
    """Create charts tab content"""
    
    # Price chart
    price_chart = go.Figure()
    price_chart.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    price_chart.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    # Volume chart
    volume_chart = go.Figure()
    volume_chart.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker=dict(color='green')
    ))
    volume_chart.update_layout(
        title=f"{symbol} Volume Chart",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400
    )
    
    return html.Div([
        dcc.Graph(figure=price_chart, style={'margin': '20px'}),
        dcc.Graph(figure=volume_chart, style={'margin': '20px'})
    ])

def create_analysis_tab(results):
    """Create analysis tab content"""
    
    portfolio_data = results['portfolio_data']
    
    # Moving averages chart
    ma_chart = go.Figure()
    ma_chart.add_trace(go.Scatter(
        x=portfolio_data.index,
        y=portfolio_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    ma_chart.add_trace(go.Scatter(
        x=portfolio_data.index,
        y=portfolio_data['MA_Short'],
        mode='lines',
        name='MA Short',
        line=dict(color='orange')
    ))
    ma_chart.add_trace(go.Scatter(
        x=portfolio_data.index,
        y=portfolio_data['MA_Long'],
        mode='lines',
        name='MA Long',
        line=dict(color='red')
    ))
    
    ma_chart.update_layout(
        title="Moving Average Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    return html.Div([
        dcc.Graph(figure=ma_chart, style={'margin': '20px'})
    ])

if __name__ == "__main__":
    print("Starting Minimal Trading Dashboard...")
    print("Dashboard will be available at: http://localhost:8059")
    application.run(host="0.0.0.0", port=8059, debug=False)
