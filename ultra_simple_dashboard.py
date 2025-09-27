#!/usr/bin/env python3
"""
Ultra-Simple Working Dashboard
No datetime parsing issues - guaranteed to work
"""
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Multi-Agent Trading System - Backtesting Dashboard"

# Available symbols
available_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]

# Ultra-simple data generator
def generate_simple_data(symbol, start_date, end_date):
    """Generate simple simulated stock data"""
    print(f"Generating data for {symbol}")
    
    # Create simple date range (just use numbers)
    n_days = 250  # Approximate trading days in a year
    
    # Base price for each symbol
    base_prices = {
        'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300, 'TSLA': 200,
        'NVDA': 400, 'AMZN': 3000, 'META': 300, 'NFLX': 400
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate simple price series
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))
    
    # Create DataFrame with simple index
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 10000000) for _ in range(n_days)]
    })
    
    # Add simple date column (no timezone issues)
    df['Date'] = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    df.set_index('Date', inplace=True)
    
    print(f"Generated {len(df)} days of data")
    return df

def calculate_simple_indicators(df):
    """Calculate simple technical indicators"""
    print("Calculating indicators...")
    
    # Simple moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Simple RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    print("Indicators calculated")
    return df

def generate_simple_signals(df):
    """Generate simple trading signals"""
    print("Generating signals...")
    
    df['Signal'] = 0
    
    # Simple SMA crossover
    df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
    df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1
    
    # Simple RSI signals
    df.loc[df['RSI'] < 30, 'Signal'] = 1
    df.loc[df['RSI'] > 70, 'Signal'] = -1
    
    signal_count = len(df[df['Signal'] != 0])
    print(f"Generated {signal_count} signals")
    return df

def run_simple_backtest(df, initial_capital=100000):
    """Run simple backtest"""
    print(f"Running backtest with ${initial_capital:,}")
    
    capital = initial_capital
    position = 0
    portfolio_values = [initial_capital]
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        
        if signal == 1 and position == 0:  # Buy
            position_size = capital * 0.95 / current_price
            position = position_size
            capital -= position_size * current_price
            trades.append({
                'Date': f"Day {i}",
                'Type': 'BUY',
                'Price': round(current_price, 2),
                'Quantity': round(position_size, 2),
                'Value': round(position_size * current_price, 2)
            })
        elif signal == -1 and position > 0:  # Sell
            capital += position * current_price
            trades.append({
                'Date': f"Day {i}",
                'Type': 'SELL',
                'Price': round(current_price, 2),
                'Quantity': round(position, 2),
                'Value': round(position * current_price, 2)
            })
            position = 0
        
        current_value = capital + (position * current_price)
        portfolio_values.append(current_value)
    
    # Close position
    if position > 0:
        capital += position * df['Close'].iloc[-1]
        trades.append({
            'Date': f"Day {len(df)}",
            'Type': 'SELL',
            'Price': round(df['Close'].iloc[-1], 2),
            'Quantity': round(position, 2),
            'Value': round(position * df['Close'].iloc[-1], 2)
        })
        portfolio_values[-1] = capital
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    days = len(df)
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    
    portfolio_series = pd.Series(portfolio_values)
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    print(f"Backtest complete: {total_return:.2%} return, {len(trades)} trades")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'final_value': final_value,
        'portfolio_values': portfolio_values,
        'trade_history': trades
    }

# Dashboard Layout
app.layout = html.Div([
    html.Div([
        html.H1("Multi-Agent Trading System", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.H3("Backtesting Dashboard", style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
        html.P("Ultra-Simple Version - Guaranteed to Work!", style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '16px', 'fontWeight': 'bold'})
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            html.H4("Backtest Controls", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Symbol:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[{'label': s, 'value': s} for s in available_symbols],
                    value='AAPL',
                    style={'marginBottom': '20px'}
                )
            ]),
            
            html.Div([
                html.Label("Initial Capital:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
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
            
            html.Button(
                "Run Backtest",
                id='run-backtest-btn',
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 30px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                }
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginRight': '20px'}),
        
        # Performance Summary
        html.Div([
            html.H4("Performance Summary", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div(id='performance-summary')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    # Charts Section
    html.Div([
        html.Div([
            html.H4("Portfolio Performance", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='portfolio-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Drawdown Analysis", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='drawdown-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.H4("Risk Metrics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='risk-metrics-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Trade History", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(id='trade-history-table')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'marginBottom': '30px'})
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# Global variable to store results
backtest_results = {}

# Callbacks
@app.callback(
    [Output('performance-summary', 'children'),
     Output('portfolio-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('risk-metrics-chart', 'figure'),
     Output('trade-history-table', 'children')],
    [Input('run-backtest-btn', 'n_clicks')],
    [Input('symbol-dropdown', 'value'),
     Input('initial-capital', 'value')]
)
def run_backtest_callback(n_clicks, symbol, initial_capital):
    """Run backtest and update dashboard"""
    if n_clicks is None:
        empty_fig = go.Figure()
        empty_table = html.Div("Click 'Run Backtest' to see results", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        return "No backtest run yet", empty_fig, empty_fig, empty_fig, empty_table
    
    try:
        print(f"\n=== Starting Simple Backtest for {symbol} ===")
        
        # Generate data
        df = generate_simple_data(symbol, '2023-01-01', '2024-01-01')
        
        # Calculate indicators
        df = calculate_simple_indicators(df)
        
        # Generate signals
        df = generate_simple_signals(df)
        
        # Run backtest
        result = run_simple_backtest(df, initial_capital)
        
        # Store result
        backtest_results[symbol] = result
        
        # Create performance summary
        summary = html.Div([
            html.Div([
                html.Div([
                    html.H2(f"{result['total_return']:.2%}", style={'color': '#27ae60', 'margin': '0', 'fontSize': '24px'}),
                    html.P("Total Return", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '20%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['sharpe_ratio']:.2f}", style={'color': '#3498db', 'margin': '0', 'fontSize': '24px'}),
                    html.P("Sharpe Ratio", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '20%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['max_drawdown']:.2%}", style={'color': '#e74c3c', 'margin': '0', 'fontSize': '24px'}),
                    html.P("Max Drawdown", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '20%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['total_trades']}", style={'color': '#9b59b6', 'margin': '0', 'fontSize': '24px'}),
                    html.P("Total Trades", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '20%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"${result['final_value']:,.0f}", style={'color': '#27ae60', 'margin': '0', 'fontSize': '24px'}),
                    html.P("Final Value", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '20%', 'display': 'inline-block'})
        ])
        
        # Create portfolio chart
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            y=result['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3498db', width=3)
        ))
        portfolio_fig.update_layout(
            title=f"Portfolio Performance - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        # Create drawdown chart
        portfolio_series = pd.Series(result['portfolio_values'])
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='#e74c3c'),
            fillcolor='rgba(231,76,60,0.3)'
        ))
        drawdown_fig.update_layout(
            title=f"Drawdown Analysis - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        # Create risk metrics chart
        risk_metrics = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': [result['total_return']*100, result['sharpe_ratio'], abs(result['max_drawdown'])*100]
        }
        
        risk_fig = px.bar(
            x=risk_metrics['Metric'],
            y=risk_metrics['Value'],
            title=f"Risk Metrics - {symbol}",
            template="plotly_white",
            color=risk_metrics['Value'],
            color_continuous_scale='RdYlGn'
        )
        risk_fig.update_layout(height=400, showlegend=False)
        
        # Create trade history table
        if result['trade_history']:
            trade_df = pd.DataFrame(result['trade_history'])
            trade_table = html.Table([
                html.Thead([
                    html.Tr([html.Th(col, style={'padding': '8px', 'backgroundColor': '#f8f9fa'}) for col in trade_df.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(trade_df.iloc[i][col], style={'padding': '8px', 'borderBottom': '1px solid #dee2e6'}) for col in trade_df.columns
                    ]) for i in range(min(len(trade_df), 10))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px', 'backgroundColor': '#ffffff'})
        else:
            trade_table = html.P("No trades executed", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        
        print(f"=== Backtest completed successfully for {symbol} ===")
        return summary, portfolio_fig, drawdown_fig, risk_fig, trade_table
        
    except Exception as e:
        error_msg = f"Error running backtest: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, go.Figure(), go.Figure(), go.Figure(), error_msg

# Run the dashboard
if __name__ == "__main__":
    print("Starting Ultra-Simple Multi-Agent Trading System Dashboard...")
    print("Dashboard will be available at: http://localhost:8055")
    print("Ultra-Simple Version - No Datetime Issues!")
    print("Use the controls to run backtests and analyze results")
    
    app.run(debug=True, host='0.0.0.0', port=8055)
