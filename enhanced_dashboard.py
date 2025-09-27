"""
Enhanced Multi-Agent Trading System Dashboard
Integrates ML forecasting, sentiment analysis, portfolio optimization, and real-time alerts
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Import our new modules
from ml_forecasting import MLPriceForecaster
from sentiment_analysis import SentimentAnalyzer
from portfolio_optimization import PortfolioOptimizer
from real_time_alerts import AlertManager
from economic_calendar import EconomicCalendar
from options_analysis import OptionsAnalyzer
from crypto_integration import CryptoAnalyzer

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Enhanced Multi-Agent Trading System"

# Global variables
backtest_results = {}
ml_forecaster = MLPriceForecaster()
sentiment_analyzer = SentimentAnalyzer()
portfolio_optimizer = PortfolioOptimizer()
alert_manager = AlertManager()
economic_calendar = EconomicCalendar()
options_analyzer = OptionsAnalyzer()
crypto_analyzer = CryptoAnalyzer()

# Layout
app.layout = html.Div([
    html.H1("🚀 Enhanced Multi-Agent Trading System", 
            style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': '30px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Symbol:", style={'fontWeight': 'bold'}),
            dcc.Input(id='symbol-input', value='AAPL', type='text', 
                     style={'width': '100px', 'marginRight': '20px'})
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Sector:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[
                    {'label': 'Technology', 'value': 'Technology'},
                    {'label': 'Healthcare', 'value': 'Healthcare'},
                    {'label': 'Finance', 'value': 'Finance'},
                    {'label': 'Energy', 'value': 'Energy'},
                    {'label': 'Consumer', 'value': 'Consumer'}
                ],
                value='Technology',
                style={'width': '150px', 'marginRight': '20px'}
            )
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Capital ($):", style={'fontWeight': 'bold'}),
            dcc.Input(id='capital-input', value=100000, type='number',
                     style={'width': '120px', 'marginRight': '20px'})
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Time Period:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='time-period-dropdown',
                options=[
                    {'label': '1 Month', 'value': 30},
                    {'label': '3 Months', 'value': 90},
                    {'label': '6 Months', 'value': 180},
                    {'label': '1 Year', 'value': 365},
                    {'label': '2 Years', 'value': 730},
                    {'label': '5 Years', 'value': 1825}
                ],
                value=365,
                style={'width': '150px', 'marginRight': '20px'}
            )
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Button('🚀 Run Enhanced Analysis', id='run-button', n_clicks=0,
                   style={'backgroundColor': '#2E86AB', 'color': 'white', 'border': 'none',
                         'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'})
    ], style={'textAlign': 'center', 'marginBottom': '30px', 'padding': '20px',
              'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Status Display
    html.Div(id='status-display', style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Metrics Display
    html.Div(id='metrics-display', style={'marginBottom': '30px'}),
    
    # Enhanced Tabs
    dcc.Tabs(id='main-tabs', value='overview-tab', children=[
        dcc.Tab(label='📊 Overview', value='overview-tab'),
        dcc.Tab(label='🤖 ML Forecasting', value='ml-tab'),
        dcc.Tab(label='📰 Sentiment Analysis', value='sentiment-tab'),
        dcc.Tab(label='📈 Portfolio Optimization', value='portfolio-tab'),
        dcc.Tab(label='🚨 Real-time Alerts', value='alerts-tab'),
        dcc.Tab(label='📅 Economic Calendar', value='economic-tab'),
        dcc.Tab(label='📊 Options Analysis', value='options-tab'),
        dcc.Tab(label='₿ Crypto Market', value='crypto-tab'),
        dcc.Tab(label='📋 Export Report', value='export-tab')
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='tab-content', style={'padding': '20px'}),
    
    # Export button (always present)
    html.Div([
        html.Button('📄 Export Enhanced PDF Report', id='export-pdf-button', n_clicks=0,
                   style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                         'padding': '15px 30px', 'borderRadius': '5px', 'cursor': 'pointer',
                         'fontSize': '16px', 'marginTop': '20px', 'display': 'none'})
    ], style={'textAlign': 'center', 'padding': '20px'}),
    
    # Download component
    dcc.Download(id="download-pdf")
])

# Callbacks
@app.callback(
    [Output('status-display', 'children'),
     Output('metrics-display', 'children'),
     Output('tab-content', 'children'),
     Output('export-pdf-button', 'style')],
    [Input('run-button', 'n_clicks'),
     Input('main-tabs', 'value')],
    [State('symbol-input', 'value'),
     State('sector-dropdown', 'value'),
     State('capital-input', 'value'),
     State('time-period-dropdown', 'value')]
)
def update_dashboard(n_clicks, active_tab, symbol, sector, capital, time_period):
    # Handle initial state
    if n_clicks == 0 and active_tab == 'overview-tab':
        return "Ready to analyze", "", "Select parameters and click 'Run Enhanced Analysis'", {'display': 'none'}
    
    # If run button was clicked, run the analysis
    if n_clicks and n_clicks > 0:
        try:
            results = run_enhanced_analysis(symbol, sector, capital, time_period)
            backtest_results[symbol] = results
        except Exception as e:
            return f"❌ Error: {str(e)}", "", "Please try again", {'display': 'none'}
    
    # If no analysis has been run yet, show message
    if symbol not in backtest_results:
        return "Ready to analyze", "", "Please run analysis first", {'display': 'none'}
    
    results = backtest_results[symbol]
    
    # Status
    status = f"✅ Enhanced analysis completed for {symbol}"
    
    # Metrics
    metrics = create_metrics_display(results)
    
    # Tab content based on active tab
    if active_tab == 'overview-tab':
        tab_content = create_overview_tab(results)
    elif active_tab == 'ml-tab':
        tab_content = create_ml_tab(results)
    elif active_tab == 'sentiment-tab':
        tab_content = create_sentiment_tab(results)
    elif active_tab == 'portfolio-tab':
        tab_content = create_portfolio_tab(results)
    elif active_tab == 'alerts-tab':
        tab_content = create_alerts_tab(results)
    elif active_tab == 'economic-tab':
        tab_content = create_economic_tab(results)
    elif active_tab == 'options-tab':
        tab_content = create_options_tab(results)
    elif active_tab == 'crypto-tab':
        tab_content = create_crypto_tab(results)
    elif active_tab == 'export-tab':
        tab_content = create_export_tab(results)
    else:
        tab_content = "Tab content not found"
    
    # Show export button when analysis is available
    export_button_style = {
        'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
        'padding': '15px 30px', 'borderRadius': '5px', 'cursor': 'pointer',
        'fontSize': '16px', 'marginTop': '20px', 'display': 'block'
    }
    
    return status, metrics, tab_content, export_button_style

def run_enhanced_analysis(symbol, sector, capital, time_period=365):
    """Run complete enhanced analysis"""
    print(f"Starting enhanced analysis for {symbol} over {time_period} days...")
    
    # Generate market data
    df = generate_enhanced_market_data(symbol, sector, time_period)
    
    # Run backtest
    backtest_results = run_enhanced_backtest(df, capital)
    
    # ML Forecasting
    ml_results = ml_forecaster.run_complete_forecast(df, symbol)
    
    # Sentiment Analysis
    sentiment_results = sentiment_analyzer.run_complete_sentiment_analysis(symbol)
    
    # Portfolio Optimization
    returns = df['Close'].pct_change().dropna()
    # Convert to numpy array to avoid pandas Series issues
    returns_array = returns.values
    portfolio_results = portfolio_optimizer.generate_optimization_report(returns_array, [symbol])
    
    # Set up alerts
    setup_alerts(symbol, backtest_results, ml_results, sentiment_results)
    
    return {
        'symbol': symbol,
        'sector': sector,
        'capital': capital,
        'time_period': time_period,
        'backtest': backtest_results,
        'ml_forecast': ml_results,
        'sentiment': sentiment_results,
        'portfolio': portfolio_results,
        'market_data': df
    }

def generate_enhanced_market_data(symbol, sector, time_period=365):
    """Generate enhanced market data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=time_period, freq='D')
    
    # Base price with trend
    base_price = 150 + np.cumsum(np.random.randn(time_period) * 0.5)
    
    # Add sector-specific volatility
    sector_volatility = {'Technology': 0.02, 'Healthcare': 0.015, 'Finance': 0.025, 
                        'Energy': 0.03, 'Consumer': 0.018}
    volatility = sector_volatility.get(sector, 0.02)
    
    prices = base_price * (1 + np.random.randn(time_period) * volatility)
    volumes = np.random.randint(1000000, 5000000, time_period)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(time_period) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(time_period)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(time_period)) * 0.01),
        'Close': prices,
        'Volume': volumes
    })
    
    return df

def run_enhanced_backtest(df, capital):
    """Run enhanced backtest with multiple strategies"""
    initial_capital = capital
    current_capital = initial_capital
    position = 0
    trades = []
    
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    # Generate signals
    for i in range(50, len(df)):
        current_price = df['Close'].iloc[i]
        sma_20 = df['SMA_20'].iloc[i]
        sma_50 = df['SMA_50'].iloc[i]
        rsi = df['RSI'].iloc[i]
        
        # Multiple strategy signals
        signal_strength = 0
        
        # Moving average crossover
        if sma_20 > sma_50 and df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]:
            signal_strength += 0.3
        
        # RSI signals
        if rsi < 30:
            signal_strength += 0.2
        elif rsi > 70:
            signal_strength -= 0.2
        
        # Volume confirmation
        avg_volume = df['Volume'].iloc[i-20:i].mean()
        if df['Volume'].iloc[i] > avg_volume * 1.5:
            signal_strength += 0.1
        
        # Execute trades
        if signal_strength > 0.3 and position == 0:
            # Buy signal
            shares = int(current_capital * 0.95 / current_price)
            if shares > 0:
                position = shares
                current_capital -= shares * current_price
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'BUY',
                    'Price': current_price,
                    'Shares': shares,
                    'Value': shares * current_price,
                    'Strategy': 'Enhanced Multi-Strategy'
                })
        
        elif signal_strength < -0.3 and position > 0:
            # Sell signal
            current_capital += position * current_price
            trades.append({
                'Date': df['Date'].iloc[i],
                'Type': 'SELL',
                'Price': current_price,
                'Shares': position,
                'Value': position * current_price,
                'Strategy': 'Enhanced Multi-Strategy'
            })
            position = 0
    
    # Final portfolio value
    if position > 0:
        current_capital += position * df['Close'].iloc[-1]
    
    total_return = (current_capital - initial_capital) / initial_capital
    
    return {
        'initial_capital': initial_capital,
        'final_value': current_capital,
        'total_return': total_return,
        'trades': trades,
        'position': position
    }

def setup_alerts(symbol, backtest_results, ml_results, sentiment_results):
    """Set up real-time alerts"""
    # Price alerts
    current_price = backtest_results.get('final_value', 0) / backtest_results.get('initial_capital', 1) * 150
    
    alert_manager.add_alert(
        alert_type='price_breakout',
        symbol=symbol,
        message=f'{symbol} price breakout detected',
        priority='high',
        channels=['console'],
        conditions={'price_above': current_price * 1.1}
    )
    
    # ML prediction alerts
    if ml_results and 'forecast_summary' in ml_results:
        expected_return = ml_results['forecast_summary']['expected_return']
        
        alert_manager.add_alert(
            alert_type='ml_prediction',
            symbol=symbol,
            message=f'{symbol} ML model predicts {expected_return:.2f}% return',
            priority='medium',
            channels=['console'],
            conditions={'portfolio_return_above': expected_return / 100}
        )
    
    # Sentiment alerts
    if sentiment_results and 'combined_analysis' in sentiment_results:
        sentiment_score = sentiment_results['combined_analysis']['combined_score']
        
        alert_manager.add_alert(
            alert_type='sentiment_change',
            symbol=symbol,
            message=f'{symbol} sentiment changed to {sentiment_results["combined_analysis"]["sentiment_category"]}',
            priority='medium',
            channels=['console'],
            conditions={'sentiment_above': sentiment_score}
        )

def create_metrics_display(results):
    """Create enhanced metrics display"""
    backtest = results['backtest']
    ml_forecast = results['ml_forecast']
    sentiment = results['sentiment']
    
    metrics = [
        html.Div([
            html.H3("📊 Performance Metrics", style={'color': '#2E86AB'}),
            html.Div([
                html.Div([
                    html.H4(f"${backtest['final_value']:,.2f}", style={'color': '#28a745', 'margin': '0'}),
                    html.P("Final Portfolio Value", style={'margin': '0', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{backtest['total_return']:.2%}", style={'color': '#28a745' if backtest['total_return'] > 0 else '#dc3545', 'margin': '0'}),
                    html.P("Total Return", style={'margin': '0', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{len(backtest['trades'])}", style={'color': '#2E86AB', 'margin': '0'}),
                    html.P("Total Trades", style={'margin': '0', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{ml_forecast['forecast_summary']['expected_return']:.2f}%" if ml_forecast else "N/A", 
                           style={'color': '#ffc107', 'margin': '0'}),
                    html.P("ML Forecast", style={'margin': '0', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{sentiment['combined_analysis']['sentiment_category']}" if sentiment else "N/A", 
                           style={'color': '#17a2b8', 'margin': '0'}),
                    html.P("Sentiment", style={'margin': '0', 'fontSize': '14px'})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
        ])
    ]
    
    return metrics

def create_overview_tab(results):
    """Create overview tab content"""
    df = results['market_data']
    backtest = results['backtest']
    
    # Price chart with indicators
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Chart with Technical Indicators', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
    
    fig.update_layout(height=600, title=f"{results['symbol']} - Enhanced Analysis Overview")
    
    return html.Div([
        html.H3(f"📊 {results['symbol']} Analysis Overview"),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Key Insights"),
            html.Ul([
                html.Li(f"Total Return: {backtest['total_return']:.2%}"),
                html.Li(f"Number of Trades: {len(backtest['trades'])}"),
                html.Li(f"ML Forecast: {results['ml_forecast']['forecast_summary']['expected_return']:.2f}%" if results['ml_forecast'] else "N/A"),
                html.Li(f"Sentiment: {results['sentiment']['combined_analysis']['sentiment_category']}" if results['sentiment'] else "N/A")
            ])
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_ml_tab(results):
    """Create ML forecasting tab"""
    if not results['ml_forecast']:
        return html.Div("ML forecasting not available")
    
    ml_data = results['ml_forecast']
    
    # Forecast chart
    fig = go.Figure()
    
    # Historical prices
    df = results['market_data']
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical Price', line=dict(color='blue')))
    
    # Forecast prices
    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=len(ml_data['predictions'])+1, freq='D')[1:]
    fig.add_trace(go.Scatter(x=forecast_dates, y=ml_data['predictions'], name='ML Forecast', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title=f"{results['symbol']} - ML Price Forecast", height=400)
    
    return html.Div([
        html.H3(f"🤖 ML Forecasting for {results['symbol']}"),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Forecast Summary"),
            html.P(f"Model Used: {ml_data['model_used']}"),
            html.P(f"Expected Return: {ml_data['forecast_summary']['expected_return']:.2f}%"),
            html.P(f"Forecast Volatility: {ml_data['forecast_summary']['volatility']:.2f}%"),
            html.P(f"Model R² Score: {ml_data['model_performance']['r2']:.4f}")
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_sentiment_tab(results):
    """Create sentiment analysis tab"""
    if not results['sentiment']:
        return html.Div("Sentiment analysis not available")
    
    sentiment_data = results['sentiment']
    
    # Sentiment chart
    fig = go.Figure()
    
    # News sentiment
    fig.add_trace(go.Bar(x=['News', 'Social', 'Combined'], 
                        y=[sentiment_data['news_analysis']['overall_sentiment'],
                           sentiment_data['social_analysis']['overall_sentiment'],
                           sentiment_data['combined_analysis']['combined_score']],
                        name='Sentiment Score',
                        marker_color=['lightblue', 'lightgreen', 'orange']))
    
    fig.update_layout(title=f"{results['symbol']} - Sentiment Analysis", height=400)
    
    return html.Div([
        html.H3(f"📰 Sentiment Analysis for {results['symbol']}"),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Sentiment Summary"),
            html.P(f"Overall Sentiment: {sentiment_data['combined_analysis']['sentiment_category']}"),
            html.P(f"Trading Signal: {sentiment_data['combined_analysis']['trading_signal']}"),
            html.P(f"Confidence: {sentiment_data['combined_analysis']['confidence']:.3f}"),
            html.P(f"News Articles: {sentiment_data['news_analysis']['article_count']}"),
            html.P(f"Social Posts: {sentiment_data['social_analysis']['post_count']}")
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_portfolio_tab(results):
    """Create portfolio optimization tab"""
    if not results['portfolio']:
        return html.Div("Portfolio optimization not available")
    
    portfolio_data = results['portfolio']
    
    # Efficient frontier chart
    fig = go.Figure()
    
    # Random portfolios
    fig.add_trace(go.Scatter(x=portfolio_data['efficient_frontier']['portfolio_volatilities'],
                            y=portfolio_data['efficient_frontier']['portfolio_returns'],
                            mode='markers',
                            name='Random Portfolios',
                            marker=dict(color='lightblue', size=4)))
    
    # Optimal portfolios
    min_vol = portfolio_data['efficient_frontier']['min_volatility_portfolio']
    max_sharpe = portfolio_data['efficient_frontier']['max_sharpe_portfolio']
    
    fig.add_trace(go.Scatter(x=[min_vol['volatility']], y=[min_vol['expected_return']],
                            mode='markers', name='Min Volatility',
                            marker=dict(color='red', size=10)))
    
    fig.add_trace(go.Scatter(x=[max_sharpe['volatility']], y=[max_sharpe['expected_return']],
                            mode='markers', name='Max Sharpe',
                            marker=dict(color='green', size=10)))
    
    fig.update_layout(title=f"{results['symbol']} - Portfolio Optimization", 
                     xaxis_title='Volatility', yaxis_title='Expected Return', height=400)
    
    return html.Div([
        html.H3(f"📈 Portfolio Optimization for {results['symbol']}"),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Optimization Results"),
            html.P(f"Max Sharpe Return: {max_sharpe['expected_return']:.2%}"),
            html.P(f"Max Sharpe Volatility: {max_sharpe['volatility']:.2%}"),
            html.P(f"Min Volatility Return: {min_vol['expected_return']:.2%}"),
            html.P(f"Min Volatility Volatility: {min_vol['volatility']:.2%}")
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_alerts_tab(results):
    """Create real-time alerts tab"""
    # Start alert monitoring
    alert_manager.start_monitoring()
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    
    return html.Div([
        html.H3(f"🚨 Real-time Alerts for {results['symbol']}"),
        html.Div([
            html.H4("Alert System Status"),
            html.P(f"Total Alerts: {summary['total_alerts']}"),
            html.P(f"Active Alerts: {summary['active_alerts']}"),
            html.P(f"Triggered Alerts: {summary['triggered_alerts']}"),
            html.P(f"Monitoring Active: {summary['monitoring_active']}"),
            html.P(f"Recent Alerts (24h): {summary['recent_alerts_24h']}")
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Alert Types"),
            html.Ul([html.Li(f"{alert_type}: {count}") for alert_type, count in summary['type_distribution'].items()])
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_economic_tab(results):
    """Create economic calendar tab"""
    try:
        # Get economic calendar data
        symbols = [results['symbol']]
        calendar_data = economic_calendar.get_economic_calendar_data(symbols, 30)
        
        # Create economic events table
        economic_events = calendar_data['economic_events'][:10]  # Show first 10
        earnings_events = calendar_data['earnings_events'][:10]  # Show first 10
        
        economic_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Date'), html.Th('Time'), html.Th('Event'), 
                    html.Th('Impact'), html.Th('Forecast'), html.Th('Previous')
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(event['date']), html.Td(event['time']), 
                    html.Td(event['event']), html.Td(event['impact'], 
                    style={'color': 'red' if event['impact'] == 'High' else 'orange' if event['impact'] == 'Medium' else 'green'}),
                    html.Td(event['forecast']), html.Td(event['previous'])
                ]) for event in economic_events
            ])
        ], style={'width': '100%', 'border': '1px solid #ddd'})
        
        earnings_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Date'), html.Th('Time'), html.Th('Symbol'), 
                    html.Th('Impact'), html.Th('EPS Forecast'), html.Th('Revenue Forecast')
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(event['date']), html.Td(event['time']), 
                    html.Td(event['symbol']), html.Td(event['impact'],
                    style={'color': 'red' if event['impact'] == 'High' else 'orange' if event['impact'] == 'Medium' else 'green'}),
                    html.Td(f"${event['eps_forecast']}"), html.Td(f"${event['revenue_forecast']}B")
                ]) for event in earnings_events
            ])
        ], style={'width': '100%', 'border': '1px solid #ddd'})
        
        # Market impact analysis
        impact_analysis = calendar_data['market_impact_analysis']
        
        return html.Div([
            html.H3(f"📅 Economic Calendar & Earnings for {results['symbol']}"),
            
            # Market Impact Summary
            html.Div([
                html.H4("Market Impact Analysis"),
                html.P(f"Total Events: {impact_analysis['total_events']}"),
                html.P(f"High Impact: {impact_analysis['high_impact_count']}"),
                html.P(f"Medium Impact: {impact_analysis['medium_impact_count']}"),
                html.P(f"Low Impact: {impact_analysis['low_impact_count']}"),
                html.P(f"Impact Score: {impact_analysis['impact_score']}"),
                html.P(f"Market Sentiment: {impact_analysis['market_sentiment']}"),
                html.H5("Recommendations:"),
                html.Ul([html.Li(rec) for rec in impact_analysis['recommendations']])
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Economic Events
            html.Div([
                html.H4("Upcoming Economic Events"),
                economic_table
            ], style={'marginBottom': '20px'}),
            
            # Earnings Events
            html.Div([
                html.H4("Upcoming Earnings Events"),
                earnings_table
            ])
        ])
    except Exception as e:
        return html.Div([
            html.H3("📅 Economic Calendar"),
            html.P(f"Error loading economic calendar: {str(e)}")
        ])

def create_options_tab(results):
    """Create options analysis tab"""
    try:
        # Get options data
        current_price = results['backtest']['final_value'] / 100000 * 150  # Approximate current price
        options_data = options_analyzer.generate_options_report(results['symbol'], current_price, 30)
        
        # Create options chain table (first 10 rows)
        options_chain = options_data['options_chain'][:10]
        
        options_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Strike'), html.Th('Call Price'), html.Th('Put Price'),
                    html.Th('Call Volume'), html.Th('Put Volume'), html.Th('Call Delta'), html.Th('Put Delta')
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(f"${option['strike']}"), html.Td(f"${option['call_price']}"), 
                    html.Td(f"${option['put_price']}"), html.Td(option['call_volume']), 
                    html.Td(option['put_volume']), html.Td(f"{option['call_delta']:.3f}"), 
                    html.Td(f"{option['put_delta']:.3f}")
                ]) for option in options_chain
            ])
        ], style={'width': '100%', 'border': '1px solid #ddd'})
        
        # Volatility analysis
        vol_analysis = options_data['volatility_analysis']
        pcr_analysis = options_data['put_call_analysis']
        
        return html.Div([
            html.H3(f"📊 Options Analysis for {results['symbol']}"),
            html.P(f"Current Price: ${current_price:.2f} | Days to Expiry: 30"),
            
            # Volatility Analysis
            html.Div([
                html.H4("Volatility Analysis"),
                html.P(f"ITM Average IV: {vol_analysis['itm_avg_iv']:.2f}%"),
                html.P(f"ATM Average IV: {vol_analysis['atm_avg_iv']:.2f}%"),
                html.P(f"OTM Average IV: {vol_analysis['otm_avg_iv']:.2f}%"),
                html.P(f"Volatility Skew: {vol_analysis['volatility_skew']:.2f}%"),
                html.P(f"Skew Direction: {vol_analysis['skew_direction']}")
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Put/Call Analysis
            html.Div([
                html.H4("Put/Call Analysis"),
                html.P(f"Volume Put/Call Ratio: {pcr_analysis['volume_put_call_ratio']:.3f}"),
                html.P(f"OI Put/Call Ratio: {pcr_analysis['oi_put_call_ratio']:.3f}"),
                html.P(f"Volume Sentiment: {pcr_analysis['volume_sentiment']}"),
                html.P(f"OI Sentiment: {pcr_analysis['oi_sentiment']}"),
                html.P(f"Max Pain Strike: ${options_data['max_pain_strike']:.2f}")
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Options Chain
            html.Div([
                html.H4("Options Chain (Sample)"),
                options_table
            ])
        ])
    except Exception as e:
        return html.Div([
            html.H3("📊 Options Analysis"),
            html.P(f"Error loading options data: {str(e)}")
        ])

def create_crypto_tab(results):
    """Create crypto market tab"""
    try:
        # Get crypto data for major cryptocurrencies
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
        crypto_data = {}
        
        for symbol in crypto_symbols:
            crypto_data[symbol] = crypto_analyzer.generate_crypto_report(symbol, 30)
        
        # Create crypto overview table
        crypto_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Symbol'), html.Th('Price'), html.Th('24h Change'), 
                    html.Th('7d Change'), html.Th('Market Cap'), html.Th('Volume')
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(symbol), html.Td(f"${data['metrics']['current_price']:,.2f}"),
                    html.Td(f"{data['metrics']['price_change_24h']:+.2f}%", 
                    style={'color': 'green' if data['metrics']['price_change_24h'] > 0 else 'red'}),
                    html.Td(f"{data['metrics']['price_change_7d']:+.2f}%",
                    style={'color': 'green' if data['metrics']['price_change_7d'] > 0 else 'red'}),
                    html.Td(f"${data['metrics']['current_market_cap']:,.0f}"),
                    html.Td(f"{data['metrics']['avg_volume_7d']:,.0f}")
                ]) for symbol, data in crypto_data.items()
            ])
        ], style={'width': '100%', 'border': '1px solid #ddd'})
        
        # Market overview
        market_overview = crypto_data['BTC']['market_overview']
        fear_greed = crypto_data['BTC']['metrics']['fear_greed_index']
        
        return html.Div([
            html.H3("₿ Cryptocurrency Market Overview"),
            
            # Market Overview
            html.Div([
                html.H4("Market Overview"),
                html.P(f"Total Market Cap: ${market_overview['total_market_cap']:,.0f}"),
                html.P(f"24h Volume: ${market_overview['total_volume_24h']:,.0f}"),
                html.P(f"BTC Dominance: {market_overview['btc_dominance']:.1f}%"),
                html.P(f"ETH Dominance: {market_overview['eth_dominance']:.1f}%"),
                html.P(f"Market Sentiment: {market_overview['market_sentiment']['sentiment']}"),
                html.P(f"Active Cryptocurrencies: {market_overview['active_cryptocurrencies']:,}")
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Fear & Greed Index
            html.Div([
                html.H4("Fear & Greed Index"),
                html.P(f"Index: {fear_greed['index']}/100"),
                html.P(f"Sentiment: {fear_greed['sentiment']}"),
                html.P(f"Description: {fear_greed['description']}")
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Top Gainers and Losers
            html.Div([
                html.H4("Top Gainers"),
                html.Ul([html.Li(f"{gainer['symbol']}: {gainer['change']:+.2f}%") for gainer in market_overview['top_gainers']])
            ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top', 'marginRight': '2%'}),
            
            html.Div([
                html.H4("Top Losers"),
                html.Ul([html.Li(f"{loser['symbol']}: {loser['change']:+.2f}%") for loser in market_overview['top_losers']])
            ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top'}),
            
            # Crypto Table
            html.Div([
                html.H4("Major Cryptocurrencies"),
                crypto_table
            ], style={'marginTop': '20px'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("₿ Crypto Market"),
            html.P(f"Error loading crypto data: {str(e)}")
        ])

def create_export_tab(results):
    """Create export tab"""
    return html.Div([
        html.H3(f"📋 Export Enhanced Report for {results['symbol']}"),
        html.Div([
            html.P("Generate a comprehensive PDF report including all analysis results."),
            html.P("Click the 'Export Enhanced PDF Report' button below to download your report.")
        ], style={'textAlign': 'center', 'padding': '40px'})
    ])

# PDF Export Callback
@app.callback(
    Output("download-pdf", "data"),
    [Input('export-pdf-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('sector-dropdown', 'value'),
     State('capital-input', 'value'),
     State('time-period-dropdown', 'value')]
)
def export_to_pdf(n_clicks, symbol, sector, capital, time_period):
    if n_clicks and n_clicks > 0 and symbol in backtest_results:
        try:
            pdf_content = generate_enhanced_pdf_report(symbol, sector, capital, time_period)
            return dict(
                content=pdf_content,
                filename=f"{symbol}_enhanced_trading_report.pdf",
                base64=True,
                type="application/pdf"
            )
        except Exception as e:
            print(f"PDF export error: {e}")
            return None
    return None

def generate_enhanced_pdf_report(symbol, sector, capital, time_period):
    """Generate enhanced PDF report"""
    if symbol not in backtest_results:
        return None
    
    results = backtest_results[symbol]
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)
    
    story = []
    
    # Title
    story.append(Paragraph("Enhanced Multi-Agent Trading System Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    backtest = results['backtest']
    ml_forecast = results['ml_forecast']
    sentiment = results['sentiment']
    
    summary_text = f"""
    <b>Symbol:</b> {symbol}<br/>
    <b>Sector:</b> {sector}<br/>
    <b>Time Period:</b> {time_period} days<br/>
    <b>Initial Capital:</b> ${capital:,.2f}<br/>
    <b>Final Portfolio Value:</b> ${backtest['final_value']:,.2f}<br/>
    <b>Total Return:</b> {backtest['total_return']:.2%}<br/>
    <b>Number of Trades:</b> {len(backtest['trades'])}<br/>
    <b>ML Forecast Return:</b> {ml_forecast['forecast_summary']['expected_return']:.2f}%<br/>
    <b>Sentiment:</b> {sentiment['combined_analysis']['sentiment_category']}<br/>
    <b>Confidence:</b> {sentiment['combined_analysis']['confidence']:.3f}
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # ML Forecasting Section
    story.append(Paragraph("Machine Learning Forecasting", heading_style))
    ml_text = f"""
    <b>Model Used:</b> {ml_forecast['model_used']}<br/>
    <b>Model R² Score:</b> {ml_forecast['model_performance']['r2']:.4f}<br/>
    <b>Expected Return:</b> {ml_forecast['forecast_summary']['expected_return']:.2f}%<br/>
    <b>Forecast Volatility:</b> {ml_forecast['forecast_summary']['volatility']:.2f}%<br/>
    <b>Max Gain:</b> {ml_forecast['forecast_summary']['max_gain']:.2f}%<br/>
    <b>Max Loss:</b> {ml_forecast['forecast_summary']['max_loss']:.2f}%
    """
    story.append(Paragraph(ml_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Sentiment Analysis Section
    story.append(Paragraph("Sentiment Analysis", heading_style))
    sentiment_text = f"""
    <b>Overall Sentiment:</b> {sentiment['combined_analysis']['sentiment_category']}<br/>
    <b>Trading Signal:</b> {sentiment['combined_analysis']['trading_signal']}<br/>
    <b>News Sentiment:</b> {sentiment['news_analysis']['overall_sentiment']:.3f}<br/>
    <b>Social Sentiment:</b> {sentiment['social_analysis']['overall_sentiment']:.3f}<br/>
    <b>Combined Score:</b> {sentiment['combined_analysis']['combined_score']:.3f}<br/>
    <b>Confidence:</b> {sentiment['combined_analysis']['confidence']:.3f}
    """
    story.append(Paragraph(sentiment_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Portfolio Optimization Section
    portfolio = results['portfolio']
    story.append(Paragraph("Portfolio Optimization", heading_style))
    portfolio_text = f"""
    <b>Max Sharpe Return:</b> {portfolio['efficient_frontier']['max_sharpe_portfolio']['expected_return']:.2%}<br/>
    <b>Max Sharpe Volatility:</b> {portfolio['efficient_frontier']['max_sharpe_portfolio']['volatility']:.2%}<br/>
    <b>Min Volatility Return:</b> {portfolio['efficient_frontier']['min_volatility_portfolio']['expected_return']:.2%}<br/>
    <b>Min Volatility Volatility:</b> {portfolio['efficient_frontier']['min_volatility_portfolio']['volatility']:.2%}
    """
    story.append(Paragraph(portfolio_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"""
    <i>Enhanced report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    Multi-Agent Trading System - Advanced Analytics Dashboard<br/>
    This report contains simulated trading data for educational and portfolio demonstration purposes.</i>
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    pdf_content = buffer.getvalue()
    
    # Encode to base64
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    return pdf_base64

# Helper functions
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

if __name__ == '__main__':
    print("Starting Enhanced Multi-Agent Trading System Dashboard...")
    print("Dashboard will be available at: http://localhost:8058")
    print("Enhanced Features: ML Forecasting, Sentiment Analysis, Portfolio Optimization, Real-time Alerts")
    app.run(debug=True, host='0.0.0.0', port=8058)
