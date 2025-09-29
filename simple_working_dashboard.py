#!/usr/bin/env python3
"""
SIMPLE WORKING TRADING DASHBOARD
Fresh start - only essential features that work
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Starting Simple Working Dashboard...")

# Create Dash app
app = dash.Dash(__name__)
app.title = "Simple Trading Dashboard"

# Simple symbols list
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA']

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
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, initial_price * 0.5))  # Minimum price floor
    
    # Generate volume
    volumes = np.random.randint(1000000, 5000000, days)
    
    # Calculate moving averages
    prices_series = pd.Series(prices)
    sma_20 = prices_series.rolling(20).mean().fillna(method='bfill')
    sma_50 = prices_series.rolling(50).mean().fillna(method='bfill')
    
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
        
        # Buy signal: SMA_20 crosses above SMA_50
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
        
        # Sell signal: SMA_20 crosses below SMA_50
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
    
    # Final portfolio value
    final_value = cash + (shares * data['Close'].iloc[-1] if position else 0)
    total_return = ((final_value - capital) / capital) * 100
    
    return {
        'trades': trades,
        'total_return': round(total_return, 2),
        'total_trades': len(trades),
        'final_value': round(final_value, 2)
    }

# Layout
app.layout = html.Div([
    html.H1("Simple Trading Dashboard", style={'textAlign': 'center', 'margin': '20px'}),
    
    # Controls
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
                   style={'background': '#007bff', 'color': 'white', 'border': 'none', 
                         'padding': '10px 20px', 'border-radius': '5px', 'cursor': 'pointer',
                         'margin': '10px'})
        
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),
    
    # Results
    html.Div(id='results-content', style={'margin': '20px'})
])

@app.callback(
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
        # Generate data
        data = generate_data(symbol, days)
        
        # Run backtest
        results = run_backtest(data, capital)
        
        # Create price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50', line=dict(color='red')))
        
        # Add trade markers
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
            title=f'{symbol} Stock Price & Strategy Trades',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
        
        # Create results summary
        summary = html.Div([
            html.H3(f"Analysis Results for {symbol}"),
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
            
            html.H4("Recent Trades:"),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date"), html.Th("Type"), 
                        html.Th("Price"), html.Th("Shares")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(trade['date']),
                        html.Td(trade['type'], style={'color': 'green' if trade['type'] == 'BUY' else 'red'}),
                        html.Td(f"${trade['price']}"),
                        html.Td(trade['shares'])
                    ]) for trade in results['trades'][-10:]  # Last 10 trades
                ])
            ], style={'width': '100%', 'border-collapse': 'collapse', 'border': '1px solid black'})
            
        ])
        
        return [summary, dcc.Graph(figure=fig)]
        
    except Exception as e:
        return html.Div([
            html.H3("Error"),
            html.P(f"Something went wrong: {str(e)}")
        ])

if __name__ == '__main__':
    print("Simple Dashboard Ready!")
    print("Features: Symbol selection, Capital input, Time period, Backtest analysis")
    print("Strategy: Simple moving average crossover (20/50)")
    app.run(host='127.0.0.1', port=8060, debug=True)