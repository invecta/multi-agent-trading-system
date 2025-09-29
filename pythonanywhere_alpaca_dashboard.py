#!/usr/bin/env python3
"""
ALPACA ENHANCED TRADING DASHBOARD - PYTHONANYWHERE VERSION
Combines working simple dashboard with Alpaca integration
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import json

print("Starting Alpaca Enhanced Trading Dashboard...")

# Create Dash app
application = dash.Dash(__name__)
application.title = "Alpaca Enhanced Trading Dashboard"

# Alpaca Configuration
try:
    ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "PKOEKMI4RY0LHF565WDO")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb")
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    
    def test_alpaca_connection():
        try:
            headers = {
                'APCA-API-KEY-ID': ALPACA_API_KEY,
                'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
            }
            response = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=headers)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, None
        except Exception as e:
            return False, str(e)
    
    alpaca_connected, account_data = test_alpaca_connection()
    if alpaca_connected:
        print(f"Alpaca Connected: True - Account: {account_data.get('account_number', 'N/A')}")
    else:
        print(f"Alpaca Connection Failed: {account_data}")
        # Set default values when connection fails
        account_data = {
            'buying_power': '0',
            'portfolio_value': '0', 
            'cash': '0'
        }
            
except Exception as e:
    print(f"Alpaca setup error: {e}")
    alpaca_connected = False
    account_data = {
        'buying_power': '0',
        'portfolio_value': '0', 
        'cash': '0'
    }

# Simple symbols for fallback
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA']

def get_alpaca_positions():
    """Get positions from Alpaca API"""
    if not alpaca_connected:
        return []
    
    try:
        headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
        }
        response = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []

def generate_data(symbol, days):
    """Generate simple market data"""
    np.random.seed(hash(symbol) % 1000)
    
    # Start prices for each symbol
    base_prices = {'AAPL': 150, 'GOOGL': 140, 'TSLA': 200, 'MSFT': 300, 'AMZN': 120, 'NVDA': 400}
    initial_price = base_prices.get(symbol, 100)
    
    # Generate price data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = [initial_price]
    
    for i in range(days - 1):
        change = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, initial_price * 0.5))
    
    volumes = np.random.randint(1000000, 5000000, days)
    
    prices_Method = pd.Series(prices)
    sma_20 = prices_Method.rolling(20).mean().ffill()
    sma_50 = prices_Method.rolling(50).mean().ffill()
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes,
        'SMA_20': sma_20,
        'SMA_50': sma_50
    })

def run_backtest(data, capital):
    """Simple moving average crossover strategy"""
    df = data.copy()
    
    trades = []
    position = 0
    shares = 0
    cash = capital
    
    for i in range(50, len(df)):
        current_price = df['Close'].iloc[i]
        
        if (df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and 
            df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1] and position == 0):
            
            shares = int(cash * 0.95 / current_price)
            if shares > 0:
                cash -= shares * current_price
                position = 1
                trades.append({
                    'date': df['Date'].iloc[i].strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': round(current_price, 2),
                    'shares': shares
                })
        
        elif (df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] and 
              df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1] and position == 1):
            
            cash += shares * current_price
            trades.append({
                'date': df['Date'].iloc[i].strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': round(current_price, 2),
                'shares': shares
            })
            position = 0
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position else 0)
    total_return = ((final_value - capital) / capital) * 100
    
    return {
        'trades': trades,
        'total_return': round(total_return, 2),
        'total_trades': len(trades),
        'final_value': round(final_value, 2)
    }

# Enhanced Layout with Alpaca Integration
application.layout = html.Div([
    html.Div([
        html.H1("Alpaca Enhanced Trading Dashboard", 
                style={'textAlign': 'center', 'margin': '20px', 'color': 'white'}),
        
        # Connection Status
        html.Div([
            html.Span("●", 
                     style={'color': 'green' if alpaca_connected else 'red', 
                           'fontSize': '20px', 'marginRight': '10px'}),
            html.Span(f"Alpaca Status: {'Connected' if alpaca_connected else 'Disconnected'}",
                     style={'color': 'white', 'fontSize': '16px'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'})
        
    ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              'padding': '20px', 'borderRadius': '15px', 'marginBottom': '20px'}),
    
    # Main Dashboard Cards
    html.Div([
        # Alpaca Account Info
        html.Div([
            html.H3("Alpaca Account", style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H4(f"${float(account_data.get('buying_power', 0)):,.2f}"),
                    html.P("Buying Power")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${float(account_data.get('portfolio_value', 0)):,.2f}"),
                    html.P("Portfolio Value")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${float(account_data.get('cash', 0)):,.2f}"),
                    html.P("Cash")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'margin': '5px'})
                
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
            
        ], className='dashboard-card'),
        
        # Alpaca Positions
        html.Div([
            html.H3("Live Positions", style={'marginBottom': '15px'}),
            html.Div(id='positions-content')
        ], className='dashboard-card'),
        
        # Backtest Controls
        html.Div([
            html.H3("Strategy Backtest", style={'marginBottom': '15px'}),
            
            html.Div([
                html.Div([
                    html.Label("Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': symbol, 'value': symbol} for symbol in SYMBOLS],
                        value='AAPL',
                        style={'width': '150px', 'margin-bottom': '10px'}
                    )
                ], style={'margin': '10px'}),
                
                html.Div([
                    html.Label("Capital:"),
                    dcc.Input(
                        id='capital-input',
                        type='number',
                        value=100000,
                        style={'width': '150px', 'margin-bottom': '10px'}
                    )
                ], style={'margin': '10px'}),
                
                html.Div([
                    html.Label("Days:"),
                    dcc.Dropdown(
                        id='days-dropdown',
                        options=[
                            {'label': '30 days', 'value': 30},
                            {'label': '90 days', 'value': 90},
                            {'label': '180 days', 'value': 180},
                            {'label': '365 days', 'value': 365}
                        ],
                        value=365,
                        style={'width': '150px', 'margin-bottom': '10px'}
                    )
                ], style={'margin': '10px'}),
                
                html.Button('Run Analysis', id='analyze-btn', n_clicks=0, 
                           style={'background': '#667eea', 'color': 'white', 'border': 'none', 
                                 'padding': '10px 20px', 'border-radius': '5px', 'cursor': 'pointer',
                                 'margin': '10px'})
                
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'})
            
        ], className='dashboard-card'),
        
        # Results Area
        html.Div([
            html.H3("Analysis Results"),
            html.Div(id='results-content')
        ], className='dashboard-card')
        
    ], style={'margin': '0 20px'})
])

@application.callback(
    Output('positions-content', 'children'),
    Input('analyze-btn', 'n_clicks')
)
def update_positions(n_clicks):
    try:
        positions = get_alpaca_positions()
        if not positions:
            return html.P("No positions found or Alpaca disconnected")
        
        position_elements = []
        for pos in positions:
            market_value = float(pos['market_value'])
            unrealized_pl = float(pos['unrealized_pl'])
            unrealized_pl_pct = float(pos['unrealized_plpc']) * 100
            
            position_elements.append(
                html.Div([
                    html.H5(f"{pos['symbol']}"),
                    html.P(f"Quantity: {pos['qty']}"),
                    html.P(f"Market Value: ${market_value:,.2f}"),
                    html.P(f"Unrealized P&L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:.2f}%)",
                          style={'color': 'green' if unrealized_pl >= 0 else 'red'})
                ], style={'background': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)' if unrealized_pl >= 0 else 'linear-gradient(135deg, #dc3545 0%, #fd7e14 100%)',
                     'color': 'white', 'padding': '15px', 'borderRadius': '10px', 'margin': '10px 0'})
            )
        
        return position_elements
    except Exception as e:
        return html.P(f"Error loading positions: {str(e)}")

@application.callback(
    Output('results-content', 'children'),
    [Input('analyze-btn', 'n_clicks')],
    [State('symbol-dropdown', 'value'),
     State('capital-input', 'value'),
     State('days-dropdown', 'value')]
)
def update_analysis(n_clicks, symbol, capital, days):
    if n_clicks == 0:
        return html.P("Select parameters and click 'Run Analysis' to start.")
    
    try:
        data = generate_data(symbol, days)
        results = run_backtest(data, capital)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50', line=dict(color='red')))
        
        for trade in results['trades']:
            trade_date = pd.to_datetime(trade['date'])
            trade_price = trade['price']
            trade_type = trade['type']
            
            fig.add_trace(go.Scatter(
                x=[trade_date], 
                y=[trade_price],
                mode='markers',
                name=trade_type,
                marker=dict(
                    color='green' if trade_type == 'BUY' else 'red',
                    size=10,
                    symbol='triangle-up' if trade_type == 'BUY' else 'triangle-down'
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'{symbol} Backtest Results',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            hovermode='x unified'
        )
        
        summary = html.Div([
            html.H4(f"Strategy Performance - {symbol}"),
            html.Div([
                html.Div([
                    html.H4(f"{results['total_return']}%"),
                    html.P("Total Return")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{results['total_trades']}"),
                    html.P("Total Trades")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${results['final_value']:,}"),
                    html.P("Final Value")
                ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px', 'margin': '5px'})
                
            ], style={'display': 'flex', 'justify-content': 'space-around'}),
            
            dcc.Graph(figure=fig)
            
        ])
        
        return summary
        
    except Exception as e:
        return html.Div([
            html.H4("Error"),
            html.P(f"Something went wrong: {str(e)}")
        ])

# Custom CSS
application.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .dashboard-card {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("Alpaca Enhanced Dashboard Ready!")
    print("Features: Live Alpaca positions, Account info, Strategy backtest")
    print("Strategy: Simple moving average crossover (20/50)")
    application.run(host='127.0.0.1', port=8061, debug=True)
