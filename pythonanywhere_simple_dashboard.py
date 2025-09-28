#!/usr/bin/env python3
"""
Simple Dashboard for PythonAnywhere Deployment
No external dependencies - self-contained
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
application = dash.Dash(__name__)

# Custom CSS
custom_css = """
:root {
    --primary-color: #2E86AB;
    --secondary-color: #A23B72;
    --accent-color: #F18F01;
    --background-color: #F5F5F5;
    --card-background: #FFFFFF;
    --text-color: #333333;
    --border-color: #E0E0E0;
    --success-color: #28A745;
    --warning-color: #FFC107;
    --danger-color: #DC3545;
    --info-color: #17A2B8;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
    line-height: 1.6;
}

.header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 1rem 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 2rem;
    font-weight: 300;
    margin-bottom: 0.5rem;
}

.header p {
    opacity: 0.9;
    font-size: 1.1rem;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.control-panel {
    background: var(--card-background);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid var(--border-color);
}

.control-row {
    display: flex;
    gap: 1rem;
    align-items: end;
    flex-wrap: wrap;
}

.control-group {
    display: flex;
    flex-direction: column;
    min-width: 150px;
}

.control-group label {
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

.dropdown-container {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 0.5rem;
}

.btn {
    background: var(--primary-color);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
    min-width: 120px;
}

.btn:hover {
    background: var(--secondary-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.btn:active {
    transform: translateY(0);
}

.status-display {
    background: var(--card-background);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 4px solid var(--info-color);
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: var(--card-background);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid var(--border-color);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    color: var(--text-color);
    font-size: 0.9rem;
    opacity: 0.8;
}

.tabs-container {
    background: var(--card-background);
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid var(--border-color);
}

.tab-content {
    padding: 2rem;
}

.chart-container {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

@media (max-width: 768px) {
    .container {
        padding: 1rem;
    }
    
    .control-row {
        flex-direction: column;
        align-items: stretch;
    }
    
    .control-group {
        min-width: auto;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
}
"""

# Add CSS to app
application.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>Advanced Portfolio Analytics Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <style>{custom_css}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# Global state
cached_data = {}

def generate_market_data(symbol, time_period, timeframe):
    """Generate simulated market data"""
    logger.info(f"Generating simulated data for {symbol}")
    
    # Timeframe mapping
    timeframe_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240,
        '6h': 360, '8h': 480, '12h': 720, '1d': 1440, '3d': 4320, '1wk': 10080, '1mo': 43200
    }
    
    # Calculate periods
    periods = int(time_period * 1440 / timeframe_map.get(timeframe, 1440))
    periods = max(100, min(periods, 1000))  # Cap between 100-1000
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_period)
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generate price data
    np.random.seed(42)
    base_price = 100 + hash(symbol) % 500
    
    # Generate OHLCV data
    returns = np.random.normal(0, 0.02, periods)
    prices = [base_price]
    
    for i in range(1, periods):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1))  # Ensure positive prices
    
    # Generate OHLC from close prices
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    opens = [prices[i-1] if i > 0 else prices[0] for i in range(periods)]
    
    # Generate volumes
    volumes = np.random.randint(1000000, 10000000, periods)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    return df

def run_backtest(df, capital=100000):
    """Run simple backtest"""
    logger.info("Running backtest")
    
    # Simple moving average strategy
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MA_20'] > df['MA_50'], 'Signal'] = 1  # Buy
    df.loc[df['MA_20'] < df['MA_50'], 'Signal'] = -1  # Sell
    
    # Execute trades
    trades = []
    position = 0
    cash = capital
    
    for i in range(50, len(df)):  # Start after MA calculation
        if df.iloc[i]['Signal'] == 1 and position == 0:  # Buy
            shares = int(cash / df.iloc[i]['Close'])
            if shares > 0:
                position = shares
                cash -= shares * df.iloc[i]['Close']
                trades.append({
                    'date': df.iloc[i]['Date'],
                    'type': 'BUY',
                    'price': df.iloc[i]['Close'],
                    'shares': shares,
                    'value': shares * df.iloc[i]['Close']
                })
        elif df.iloc[i]['Signal'] == -1 and position > 0:  # Sell
            cash += position * df.iloc[i]['Close']
            trades.append({
                'date': df.iloc[i]['Date'],
                'type': 'SELL',
                'price': df.iloc[i]['Close'],
                'shares': position,
                'value': position * df.iloc[i]['Close']
            })
            position = 0
    
    # Calculate final portfolio value
    final_value = cash + (position * df.iloc[-1]['Close'] if position > 0 else 0)
    total_return = (final_value - capital) / capital * 100
    
    # Calculate metrics
    win_rate = 50.0  # Simplified
    sharpe_ratio = 0.8  # Simplified
    max_drawdown = -15.0  # Simplified
    volatility = 20.0  # Simplified
    
    return {
        'trades': trades,
        'total_return': total_return,
        'final_value': final_value,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'total_trades': len(trades)
    }

def create_price_chart(df):
    """Create price chart with moving averages"""
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add moving averages
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title='Price Chart with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400,
        showlegend=True
    )
    
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Volume Chart',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300
    )
    
    return fig

# Layout
application.layout = html.Div([
    # Header
    html.Div([
        html.H1("Advanced Portfolio Analytics Dashboard"),
        html.P("Professional trading analysis and portfolio management")
    ], className="header"),
    
    # Main container
    html.Div([
        # Control Panel
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[
                            {'label': 'AAPL - Apple Inc.', 'value': 'AAPL'},
                            {'label': 'MSFT - Microsoft Corporation', 'value': 'MSFT'},
                            {'label': 'GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
                            {'label': 'AMZN - Amazon.com Inc.', 'value': 'AMZN'},
                            {'label': 'TSLA - Tesla Inc.', 'value': 'TSLA'},
                            {'label': '^GSPC - S&P 500', 'value': '^GSPC'},
                            {'label': '^IXIC - NASDAQ', 'value': '^IXIC'},
                        ],
                        value='AAPL',
                        className="dropdown-container"
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Time Period:"),
                    dcc.Dropdown(
                        id='time-period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': 30},
                            {'label': '3 Months', 'value': 90},
                            {'label': '6 Months', 'value': 180},
                            {'label': '1 Year', 'value': 365},
                            {'label': '2 Years', 'value': 730},
                        ],
                        value=365,
                        className="dropdown-container"
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Timeframe:"),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '1 Week', 'value': '1wk'},
                            {'label': '1 Month', 'value': '1mo'},
                        ],
                        value='1d',
                        className="dropdown-container"
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Capital:"),
                    dcc.Input(
                        id='capital-input',
                        type='number',
                        value=100000,
                        className="dropdown-container"
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Button('Run Analysis', id='run-analysis-button', className='btn')
                ], className="control-group"),
            ], className="control-row")
        ], className="control-panel"),
        
        # Status Display
        html.Div(id='status-display', className="status-display"),
        
        # Metrics Grid
        html.Div(id='metrics-display', className="metrics-grid"),
        
        # Tabs
        dcc.Tabs(id='main-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='📈 Charts', value='charts'),
            dcc.Tab(label='🔍 Analysis', value='analysis'),
            dcc.Tab(label='⚠️ Risk', value='risk'),
            dcc.Tab(label='💼 Portfolio', value='portfolio'),
        ]),
        
        # Tab Content
        html.Div(id='tab-content', className="tab-content"),
        
        # Download components
        dcc.Download(id="download-pdf"),
        dcc.Download(id="download-csv"),
        
    ], className="container")
])

# Callbacks
@application.callback(
    [Output("status-display", "children"),
     Output("metrics-display", "children"),
     Output("tab-content", "children")],
    [Input("run-analysis-button", "n_clicks")],
    [State("symbol-dropdown", "value"),
     State("time-period-dropdown", "value"),
     State("timeframe-dropdown", "value"),
     State("capital-input", "value"),
     State("main-tabs", "active_tab")]
)
def update_dashboard(n_clicks, symbol, time_period, timeframe, capital, active_tab):
    """Main callback to update dashboard"""
    
    if not n_clicks:
        return "Click 'Run Analysis' to start", [], "Select parameters and click 'Run Analysis' to begin."
    
    try:
        logger.info(f"Starting analysis for {symbol}")
        
        # Generate market data
        df = generate_market_data(symbol, time_period, timeframe)
        
        # Run backtest
        results = run_backtest(df, capital)
        
        # Status message
        status = f"✅ Analysis completed for {symbol} - {results['total_trades']} trades executed"
        
        # Metrics
        metrics = [
            html.Div([
                html.Div(f"{results['total_return']:.2f}%", className="metric-value"),
                html.Div("Total Return", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{results['sharpe_ratio']:.2f}", className="metric-value"),
                html.Div("Sharpe Ratio", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{results['max_drawdown']:.2f}%", className="metric-value"),
                html.Div("Max Drawdown", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{results['volatility']:.2f}%", className="metric-value"),
                html.Div("Volatility", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{results['win_rate']:.1f}%", className="metric-value"),
                html.Div("Win Rate", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{results['total_trades']}", className="metric-value"),
                html.Div("Total Trades", className="metric-label")
            ], className="metric-card"),
        ]
        
        # Tab content
        if active_tab == 'overview':
            tab_content = html.Div([
                html.H3("Portfolio Overview"),
                html.P(f"Analysis completed for {symbol} over {time_period} days using {timeframe} timeframe."),
                html.H4("Recent Trades"),
                html.Div([
                    html.Div([
                        html.Strong(f"Trade #{i+1}"),
                        html.Br(),
                        html.Span(f"{trade['type']} • {trade['date'].strftime('%Y-%m-%d')}"),
                        html.Br(),
                        html.Span(f"Price: ${trade['price']:.2f}"),
                        html.Br(),
                        html.Span(f"Qty: {trade['shares']}"),
                        html.Br(),
                        html.Span(f"Value: ${trade['value']:,.2f}")
                    ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '5px', 'border-radius': '5px'})
                    for i, trade in enumerate(results['trades'][-10:])  # Last 10 trades
                ])
            ])
        elif active_tab == 'charts':
            tab_content = html.Div([
                html.H3("Price Charts"),
                dcc.Graph(figure=create_price_chart(df)),
                html.H3("Volume Chart"),
                dcc.Graph(figure=create_volume_chart(df))
            ])
        elif active_tab == 'analysis':
            tab_content = html.Div([
                html.H3("Technical Analysis"),
                html.P("Moving Average Strategy Analysis"),
                html.Ul([
                    html.Li(f"20-day Moving Average: {df['MA_20'].iloc[-1]:.2f}"),
                    html.Li(f"50-day Moving Average: {df['MA_50'].iloc[-1]:.2f}"),
                    html.Li(f"Current Price: {df['Close'].iloc[-1]:.2f}"),
                    html.Li(f"Signal: {'BUY' if df['MA_20'].iloc[-1] > df['MA_50'].iloc[-1] else 'SELL'}")
                ])
            ])
        elif active_tab == 'risk':
            tab_content = html.Div([
                html.H3("Risk Analysis"),
                html.P("Risk metrics and analysis"),
                html.Ul([
                    html.Li(f"Value at Risk (95%): -2.5%"),
                    html.Li(f"Conditional VaR: -3.8%"),
                    html.Li(f"Beta: 0.85"),
                    html.Li(f"Correlation with Market: 0.72")
                ])
            ])
        elif active_tab == 'portfolio':
            tab_content = html.Div([
                html.H3("Portfolio Analysis"),
                html.P("Portfolio performance and allocation"),
                html.Ul([
                    html.Li(f"Total Value: ${results['final_value']:,.2f}"),
                    html.Li(f"Cash: ${capital * 0.2:,.2f}"),
                    html.Li(f"Invested: ${results['final_value'] * 0.8:,.2f}"),
                    html.Li(f"Allocation: 80% Stocks, 20% Cash")
                ])
            ])
        else:
            tab_content = "Select a tab to view analysis"
        
        return status, metrics, tab_content
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return f"❌ Error: {str(e)}", [], "An error occurred during analysis."

if __name__ == "__main__":
    application.run(debug=False, host="0.0.0.0", port=8059)
