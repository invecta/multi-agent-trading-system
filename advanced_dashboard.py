"""
Advanced Multi-Agent Trading System Dashboard
Enhanced with Professional Analytics, Risk Management, and Advanced Charting
"""

import dash
from dash import dcc, html, Input, Output, callback_context, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import base64
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import plotly.io as pio

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

app.title = "Advanced Multi-Agent Trading System"

# Global storage for backtest results
backtest_results = {}

def generate_enhanced_market_data(symbol, sector='Technology', n_days=250):
    """Generate enhanced market data with realistic patterns"""
    print(f"Generating enhanced market data for {symbol}")
    
    # Sector-specific base prices and volatility
    sector_configs = {
        'Technology': {'base_price': 150, 'volatility': 1.2},
        'Finance': {'base_price': 80, 'volatility': 1.0},
        'Healthcare': {'base_price': 120, 'volatility': 0.8},
        'Automotive': {'base_price': 200, 'volatility': 1.5},
        'Energy': {'base_price': 60, 'volatility': 1.8}
    }
    
    config = sector_configs.get(sector, sector_configs['Technology'])
    base_price = config['base_price']
    volatility = config['volatility']
    
    # Generate realistic price movements with trends and cycles
    np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
    
    # Create multiple trend components
    trend = np.linspace(0, 0.15, n_days)  # 15% upward trend
    cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, n_days))  # Market cycle
    noise = np.random.normal(0, 0.02 * volatility, n_days)
    
    returns = trend/n_days + cycle/n_days + noise
    
    prices = [base_price]
    volumes = [random.randint(2000000, 8000000)]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))
        
        # Volume with realistic patterns
        volume_base = random.randint(1500000, 12000000)
        volume_multiplier = 1 + abs(returns[i]) * 15 + random.random() * 0.5
        volumes.append(int(volume_base * volume_multiplier))
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'Close': prices,
        'Volume': volumes
    })
    
    # Ensure High >= Low >= Close
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Add date index
    start_date = datetime.now() - timedelta(days=n_days)
    df.index = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    print(f"Generated {n_days} days of {sector} sector data")
    return df

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    print("Calculating advanced technical indicators...")
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
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
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Price momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    print("Advanced indicators calculated")
    return df

def generate_enhanced_signals(df):
    """Generate enhanced trading signals with multiple strategies"""
    print("Generating enhanced trading signals...")
    
    df['Signal'] = 0
    df['Signal_Strength'] = 0.0
    df['Strategy'] = ''
    
    for i in range(50, len(df)):  # Start after indicators are calculated
        signal_strength = 0
        strategies = []
        
        # Strategy 1: Moving Average Crossover
        if df['EMA_12'].iloc[i] > df['EMA_26'].iloc[i] and df['EMA_12'].iloc[i-1] <= df['EMA_26'].iloc[i-1]:
            signal_strength += 0.3
            strategies.append('MA_Cross')
        elif df['EMA_12'].iloc[i] < df['EMA_26'].iloc[i] and df['EMA_12'].iloc[i-1] >= df['EMA_26'].iloc[i-1]:
            signal_strength -= 0.3
            strategies.append('MA_Cross')
        
        # Strategy 2: RSI Mean Reversion
        if df['RSI'].iloc[i] < 30:  # Oversold
            signal_strength += 0.4
            strategies.append('RSI_Oversold')
        elif df['RSI'].iloc[i] > 70:  # Overbought
            signal_strength -= 0.4
            strategies.append('RSI_Overbought')
        
        # Strategy 3: Bollinger Bands
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            signal_strength += 0.3
            strategies.append('BB_Oversold')
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            signal_strength -= 0.3
            strategies.append('BB_Overbought')
        
        # Strategy 4: MACD Divergence
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
            signal_strength += 0.2
            strategies.append('MACD_Bull')
        elif df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
            signal_strength -= 0.2
            strategies.append('MACD_Bear')
        
        # Strategy 5: Volume Confirmation
        if df['Volume_Ratio'].iloc[i] > 1.5:  # High volume
            signal_strength *= 1.3
            strategies.append('Volume_Confirm')
        
        # Strategy 6: Momentum
        if df['Momentum'].iloc[i] > 0.05:  # Strong upward momentum
            signal_strength += 0.2
            strategies.append('Momentum_Up')
        elif df['Momentum'].iloc[i] < -0.05:  # Strong downward momentum
            signal_strength -= 0.2
            strategies.append('Momentum_Down')
        
        # Add some random signals for demo (reduced probability)
        if random.random() < 0.02:  # 2% chance
            signal_strength += random.choice([0.5, -0.5])
            strategies.append('Random')
        
        df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
        df.iloc[i, df.columns.get_loc('Strategy')] = ', '.join(strategies)
        
        # Generate signal based on strength
        if signal_strength > 0.4:  # Lowered threshold
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif signal_strength < -0.4:  # Lowered threshold
            df.iloc[i, df.columns.get_loc('Signal')] = -1
    
    signal_count = len(df[df['Signal'] != 0])
    print(f"Generated {signal_count} enhanced signals")
    return df

def run_enhanced_backtest(df, initial_capital=100000):
    """Run enhanced backtest with detailed tracking"""
    print(f"Running enhanced backtest with ${initial_capital:,}")
    
    capital = initial_capital
    position = 0
    trades = []
    portfolio_values = []
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        
        if signal == 1 and position == 0:  # Buy signal
            shares = int(capital * 0.95 / current_price)  # Use 95% of capital
            if shares > 0:
                position = shares
                capital -= shares * current_price
                trades.append({
                    'Date': df.index[i],
                    'Type': 'BUY',
                    'Price': current_price,
                    'Shares': shares,
                    'Value': shares * current_price,
                    'Strategy': df['Strategy'].iloc[i],
                    'Signal_Strength': df['Signal_Strength'].iloc[i]
                })
        
        elif signal == -1 and position > 0:  # Sell signal
            capital += position * current_price
            trades.append({
                'Date': df.index[i],
                'Type': 'SELL',
                'Price': current_price,
                'Shares': position,
                'Value': position * current_price,
                'Strategy': df['Strategy'].iloc[i],
                'Signal_Strength': df['Signal_Strength'].iloc[i]
            })
            position = 0
        
        # Calculate portfolio value
        portfolio_value = capital + (position * current_price)
        portfolio_values.append(portfolio_value)
    
    # Final portfolio value
    final_value = capital + (position * df['Close'].iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate metrics
    portfolio_df = pd.DataFrame({'Portfolio_Value': portfolio_values}, index=df.index)
    portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
    
    # Risk metrics
    volatility = portfolio_df['Returns'].std() * np.sqrt(252)
    sharpe_ratio = (portfolio_df['Returns'].mean() * 252) / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
    max_drawdown = portfolio_df['Drawdown'].min()
    
    # VaR calculation (95% confidence)
    var_95 = np.percentile(portfolio_df['Returns'].dropna(), 5)
    
    # Win rate
    profitable_trades = len([t for t in trades if t['Type'] == 'SELL' and 
                           any(t2['Type'] == 'BUY' and t2['Date'] < t['Date'] and 
                               t2['Price'] < t['Price'] for t2 in trades)])
    total_sell_trades = len([t for t in trades if t['Type'] == 'SELL'])
    win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0
    
    print(f"Enhanced backtest completed: {total_return:.2%} return, {len(trades)} trades")
    
    return {
        'trades': trades,
        'portfolio_values': portfolio_values,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'win_rate': win_rate,
        'final_value': final_value,
        'portfolio_df': portfolio_df
    }

def create_market_heatmap():
    """Create market sector heatmap"""
    sectors = ['Technology', 'Finance', 'Healthcare', 'Automotive', 'Energy']
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'JNJ', 'PFE', 'TSLA', 'F', 'XOM', 'CVX']
    
    # Generate random performance data
    np.random.seed(42)
    performance_data = []
    
    for stock in stocks:
        for sector in sectors:
            # Assign stocks to sectors
            if stock in ['AAPL', 'MSFT', 'GOOGL']:
                stock_sector = 'Technology'
            elif stock in ['JPM', 'BAC']:
                stock_sector = 'Finance'
            elif stock in ['JNJ', 'PFE']:
                stock_sector = 'Healthcare'
            elif stock in ['TSLA', 'F']:
                stock_sector = 'Automotive'
            else:
                stock_sector = 'Energy'
            
            if sector == stock_sector:
                performance = np.random.normal(0.08, 0.15)  # 8% mean return, 15% volatility
                performance_data.append({
                    'Stock': stock,
                    'Sector': sector,
                    'Performance': performance,
                    'Volatility': np.random.uniform(0.1, 0.3)
                })
    
    return pd.DataFrame(performance_data)

# App layout
app.layout = html.Div([
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-line", style={'margin-right': '10px'}),
            "Advanced Multi-Agent Trading System"
        ], style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.Label([
                    html.I(className="fas fa-search", style={'margin-right': '5px'}),
                    "Symbol"
                ]),
                dcc.Input(
                    id='symbol-input',
                    type='text',
                    value='AAPL',
                    style={'width': '100%'}
                )
            ], className='three columns'),
            
            html.Div([
                html.Label([
                    html.I(className="fas fa-industry", style={'margin-right': '5px'}),
                    "Sector"
                ]),
                dcc.Dropdown(
                    id='sector-dropdown',
                    options=[
                        {'label': 'Technology', 'value': 'Technology'},
                        {'label': 'Finance', 'value': 'Finance'},
                        {'label': 'Healthcare', 'value': 'Healthcare'},
                        {'label': 'Automotive', 'value': 'Automotive'},
                        {'label': 'Energy', 'value': 'Energy'}
                    ],
                    value='Technology',
                    style={'width': '100%'}
                )
            ], className='three columns'),
            
            html.Div([
                html.Label([
                    html.I(className="fas fa-dollar-sign", style={'margin-right': '5px'}),
                    "Initial Capital"
                ]),
                dcc.Input(
                    id='capital-input',
                    type='number',
                    value=100000,
                    style={'width': '100%'}
                )
            ], className='three columns'),
            
            html.Div([
                html.Br(),
                html.Button([
                    html.I(className="fas fa-play", style={'margin-right': '5px'}),
                    "Run Analysis"
                ], id='run-button', n_clicks=0, className='button-primary',
                style={'width': '100%', 'height': '40px', 'margin-bottom': '5px'}),
                html.Button([
                    html.I(className="fas fa-file-pdf", style={'margin-right': '5px'}),
                    "Export PDF"
                ], id='export-pdf-button', n_clicks=0, className='button-secondary',
                style={'width': '100%', 'height': '40px'})
            ], className='three columns')
        ], className='row', style={'marginBottom': '30px'}),
        
        # Status and metrics row
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-info-circle", style={'margin-right': '5px'}),
                    "Analysis Status"
                ]),
                html.Div(id='status-display', style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], className='six columns'),
            
            html.Div([
                html.H4([
                    html.I(className="fas fa-trophy", style={'margin-right': '5px'}),
                    "Key Metrics"
                ]),
                html.Div(id='metrics-display', style={'fontSize': '14px', 'color': '#27ae60'})
            ], className='six columns')
        ], className='row', style={'marginBottom': '30px'})
    ], style={'padding': '20px'}),
    
    # Main content tabs
    dcc.Tabs(id='main-tabs', value='charts-tab', children=[
        dcc.Tab(label='📊 Advanced Charts', value='charts-tab'),
        dcc.Tab(label='📈 Portfolio Analysis', value='portfolio-tab'),
        dcc.Tab(label='⚠️ Risk Management', value='risk-tab'),
        dcc.Tab(label='🔥 Market Heatmap', value='heatmap-tab'),
        dcc.Tab(label='📋 Trade History', value='trades-tab')
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='tab-content', style={'padding': '20px'}),
    
    # Hidden div for PDF download
    html.Div(id='pdf-download', style={'display': 'none'}),
    
    # Download component
    dcc.Download(id="download-pdf")
])

# Callbacks
@app.callback(
    [Output('status-display', 'children'),
     Output('metrics-display', 'children'),
     Output('tab-content', 'children')],
    [Input('run-button', 'n_clicks'),
     Input('symbol-input', 'value'),
     Input('sector-dropdown', 'value'),
     Input('capital-input', 'value'),
     Input('main-tabs', 'value')]
)
def update_dashboard(n_clicks, symbol, sector, capital, active_tab):
    try:
        ctx = callback_context
        if not ctx.triggered:
            return "Ready to analyze", "No data yet", "Select a tab to view analysis"
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle initial load or tab changes
        if trigger_id in ['symbol-input', 'sector-dropdown', 'capital-input', 'main-tabs']:
            if symbol and symbol in backtest_results:
                status = f"📊 Viewing {symbol} analysis"
                results = backtest_results[symbol]['results']
                metrics = f"Return: {results['total_return']:.2%} | Trades: {len(results['trades'])} | Sharpe: {results['sharpe_ratio']:.2f}"
                content = generate_tab_content(active_tab, backtest_results[symbol])
                return status, metrics, content
            else:
                return "Ready to analyze", "No data yet", "Select a tab to view analysis"
        
        if trigger_id == 'run-button' and n_clicks > 0:
            # Run analysis
            print(f"=== Starting Enhanced Analysis for {symbol} ===")
            
            # Generate data
            df = generate_enhanced_market_data(symbol, sector)
            df = calculate_advanced_indicators(df)
            df = generate_enhanced_signals(df)
            
            # Run backtest
            results = run_enhanced_backtest(df, capital)
            
            # Store results
            backtest_results[symbol] = {
                'data': df,
                'results': results,
                'symbol': symbol,
                'sector': sector,
                'capital': capital
            }
            
            status = f"✅ Analysis completed for {symbol} ({sector})"
            metrics = f"Return: {results['total_return']:.2%} | Trades: {len(results['trades'])} | Sharpe: {results['sharpe_ratio']:.2f}"
            
        elif symbol and symbol in backtest_results:
            status = f"📊 Viewing {symbol} analysis"
            results = backtest_results[symbol]['results']
            metrics = f"Return: {results['total_return']:.2%} | Trades: {len(results['trades'])} | Sharpe: {results['sharpe_ratio']:.2f}"
        else:
            status = "Ready to analyze"
            metrics = "No data yet"
        
        # Generate tab content
        if symbol and symbol in backtest_results:
            content = generate_tab_content(active_tab, backtest_results[symbol])
        else:
            content = html.Div([
                html.H3("No Analysis Data"),
                html.P("Please run an analysis first by clicking the 'Run Analysis' button.")
            ])
        
        return status, metrics, content
    
    except Exception as e:
        print(f"Callback error: {e}")
        return f"Error: {str(e)}", "Error occurred", html.Div([
            html.H3("Error"),
            html.P(f"An error occurred: {str(e)}")
        ])

@app.callback(
    Output("download-pdf", "data"),
    [Input('export-pdf-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('sector-dropdown', 'value'),
     State('capital-input', 'value')]
)
def export_to_pdf(n_clicks, symbol, sector, capital):
    if n_clicks and n_clicks > 0 and symbol and symbol in backtest_results:
        try:
            print(f"Generating PDF for {symbol}...")
            # Generate PDF
            pdf_content = generate_pdf_report(symbol, sector, capital)
            
            # Return download data
            return dict(
                content=pdf_content,
                filename=f"{symbol}_trading_report.pdf",
                type="application/pdf"
            )
        except Exception as e:
            print(f"PDF export error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

def generate_pdf_report(symbol, sector, capital):
    """Generate comprehensive PDF report"""
    try:
        print(f"PDF generation started for {symbol}")
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get data
        print(f"Accessing backtest results for {symbol}")
        data = backtest_results[symbol]
        df = data['data']
        results = data['results']
        print(f"Data accessed successfully. Trades: {len(results['trades'])}")
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Build PDF content
        story = []
    
        # Title
        story.append(Paragraph("Multi-Agent Trading System Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        <b>Symbol:</b> {symbol}<br/>
        <b>Sector:</b> {sector}<br/>
        <b>Initial Capital:</b> ${capital:,.2f}<br/>
        <b>Final Value:</b> ${results['final_value']:,.2f}<br/>
        <b>Total Return:</b> {results['total_return']:.2%}<br/>
        <b>Total Trades:</b> {len(results['trades'])}<br/>
        <b>Win Rate:</b> {results['win_rate']:.2%}<br/>
        <b>Sharpe Ratio:</b> {results['sharpe_ratio']:.2f}<br/>
        <b>Max Drawdown:</b> {results['max_drawdown']:.2%}<br/>
        <b>Volatility:</b> {results['volatility']:.2%}<br/>
        <b>VaR (95%):</b> {results['var_95']:.2%}
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
        # Performance Metrics Table
        story.append(Paragraph("Performance Metrics", heading_style))
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{results['total_return']:.2%}"],
            ['Annualized Return', f"{results['total_return'] * 252/250:.2%}"],
            ['Volatility', f"{results['volatility']:.2%}"],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
            ['VaR (95%)', f"{results['var_95']:.2%}"],
            ['Win Rate', f"{results['win_rate']:.2%}"],
            ['Total Trades', f"{len(results['trades'])}"],
            ['Final Portfolio Value', f"${results['final_value']:,.2f}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
    
        # Trade History
        if results['trades']:
            story.append(Paragraph("Trade History", heading_style))
            trades_data = [['Date', 'Type', 'Price', 'Shares', 'Value', 'Strategy']]
            
            for trade in results['trades']:
                trades_data.append([
                    trade['Date'].strftime('%Y-%m-%d'),
                    trade['Type'],
                    f"${trade['Price']:.2f}",
                    f"{trade['Shares']:,}",
                    f"${trade['Value']:,.2f}",
                    trade['Strategy']
                ])
            
            trades_table = Table(trades_data, colWidths=[1*inch, 0.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1.5*inch])
            trades_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(trades_table)
            story.append(Spacer(1, 20))
    
        # Risk Analysis
        story.append(Paragraph("Risk Analysis", heading_style))
        risk_text = f"""
        <b>Value at Risk (95%):</b> {results['var_95']:.2%} - This means there's a 5% chance of losing more than this amount in a single day.<br/>
        <b>Maximum Drawdown:</b> {results['max_drawdown']:.2%} - The largest peak-to-trough decline in portfolio value.<br/>
        <b>Volatility:</b> {results['volatility']:.2%} - Annualized standard deviation of returns, measuring price fluctuations.<br/>
        <b>Sharpe Ratio:</b> {results['sharpe_ratio']:.2f} - Risk-adjusted return measure (higher is better).<br/>
        <b>Win Rate:</b> {results['win_rate']:.2%} - Percentage of profitable trades.
        """
        story.append(Paragraph(risk_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Technical Analysis Summary
        story.append(Paragraph("Technical Analysis Summary", heading_style))
        
        # Calculate some technical summary stats
        avg_rsi = df['RSI'].mean()
        avg_volume_ratio = df['Volume_Ratio'].mean()
        signal_count = len(df[df['Signal'] != 0])
        
        tech_text = f"""
        <b>Average RSI:</b> {avg_rsi:.1f} (30-70 range indicates normal market conditions)<br/>
        <b>Average Volume Ratio:</b> {avg_volume_ratio:.2f} (1.0 = average volume)<br/>
        <b>Total Signals Generated:</b> {signal_count}<br/>
        <b>Analysis Period:</b> {len(df)} trading days<br/>
        <b>Price Range:</b> ${df['Close'].min():.2f} - ${df['Close'].max():.2f}
        """
        story.append(Paragraph(tech_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"""
        <i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        Multi-Agent Trading System - Advanced Analytics Dashboard<br/>
        This report contains simulated trading data for educational and portfolio demonstration purposes.</i>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
    
        # Build PDF
        print("Building PDF document...")
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        print(f"PDF generated successfully. Size: {len(pdf_content)} bytes")
        return pdf_content
    
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple error PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = [Paragraph("Error generating PDF report", getSampleStyleSheet()['Heading1'])]
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

def generate_tab_content(active_tab, data):
    """Generate content for each tab"""
    df = data['data']
    results = data['results']
    symbol = data['symbol']
    
    if active_tab == 'charts-tab':
        return generate_charts_tab(df, results, symbol)
    elif active_tab == 'portfolio-tab':
        return generate_portfolio_tab(df, results, symbol)
    elif active_tab == 'risk-tab':
        return generate_risk_tab(df, results, symbol)
    elif active_tab == 'heatmap-tab':
        return generate_heatmap_tab()
    elif active_tab == 'trades-tab':
        return generate_trades_tab(results, symbol)
    else:
        return html.Div("Select a tab")

def generate_charts_tab(df, results, symbol):
    """Generate advanced charts tab"""
    
    # Price chart with indicators
    fig_price = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price & Signals', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price data
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Moving averages
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dot'), showlegend=False),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dot'), 
                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False),
        row=1, col=1
    )
    
    # Buy/Sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    fig_price.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                  mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                  name='Buy Signal'),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                  mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                  name='Sell Signal'),
        row=1, col=1
    )
    
    # RSI
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig_price.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_price.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
        row=3, col=1
    )
    fig_price.add_trace(
        go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color='gray'),
        row=3, col=1
    )
    
    fig_price.update_layout(height=800, title_text=f"Advanced Technical Analysis - {symbol}")
    
    # Volume chart
    fig_volume = go.Figure()
    fig_volume.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue')
    )
    fig_volume.add_trace(
        go.Scatter(x=df.index, y=df['Volume_SMA'], name='Volume SMA', line=dict(color='red'))
    )
    fig_volume.update_layout(title_text=f"Volume Analysis - {symbol}", height=400)
    
    return html.Div([
        html.H3([
            html.I(className="fas fa-chart-line", style={'margin-right': '10px'}),
            f"Advanced Charts - {symbol}"
        ]),
        dcc.Graph(figure=fig_price),
        dcc.Graph(figure=fig_volume)
    ])

def generate_portfolio_tab(df, results, symbol):
    """Generate portfolio analysis tab"""
    
    # Portfolio performance
    fig_portfolio = go.Figure()
    
    # Portfolio value
    fig_portfolio.add_trace(
        go.Scatter(x=df.index, y=results['portfolio_values'], 
                  name='Portfolio Value', line=dict(color='blue', width=3))
    )
    
    # Buy & Hold comparison
    initial_price = df['Close'].iloc[0]
    buy_hold_values = [(df['Close'].iloc[i] / initial_price) * results['final_value'] for i in range(len(df))]
    fig_portfolio.add_trace(
        go.Scatter(x=df.index, y=buy_hold_values, 
                  name='Buy & Hold', line=dict(color='gray', dash='dash'))
    )
    
    fig_portfolio.update_layout(
        title_text=f"Portfolio Performance - {symbol}",
        height=400,
        yaxis_title="Portfolio Value ($)"
    )
    
    # Drawdown chart
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(
        go.Scatter(x=results['portfolio_df'].index, y=results['portfolio_df']['Drawdown'] * 100,
                  fill='tonexty', line=dict(color='red'), name='Drawdown %')
    )
    fig_drawdown.update_layout(
        title_text="Drawdown Analysis",
        height=300,
        yaxis_title="Drawdown (%)"
    )
    
    # Performance metrics
    metrics_data = [
        ['Total Return', f"{results['total_return']:.2%}"],
        ['Volatility', f"{results['volatility']:.2%}"],
        ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"],
        ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
        ['VaR (95%)', f"{results['var_95']:.2%}"],
        ['Win Rate', f"{results['win_rate']:.2%}"],
        ['Total Trades', f"{len(results['trades'])}"]
    ]
    
    fig_metrics = go.Figure(data=[
        go.Bar(x=[m[0] for m in metrics_data], y=[float(m[1].rstrip('%')) for m in metrics_data],
               text=[m[1] for m in metrics_data], textposition='auto')
    ])
    fig_metrics.update_layout(title_text="Performance Metrics", height=400)
    
    return html.Div([
        html.H3([
            html.I(className="fas fa-chart-pie", style={'margin-right': '10px'}),
            f"Portfolio Analysis - {symbol}"
        ]),
        dcc.Graph(figure=fig_portfolio),
        dcc.Graph(figure=fig_drawdown),
        dcc.Graph(figure=fig_metrics)
    ])

def generate_risk_tab(df, results, symbol):
    """Generate risk management tab"""
    
    # Risk-return scatter
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(x=[results['volatility']], y=[results['total_return']],
                  mode='markers', marker=dict(size=20, color='blue'),
                  text=[f"{symbol}"], textposition="top center")
    )
    fig_scatter.update_layout(
        title_text="Risk-Return Analysis",
        xaxis_title="Volatility",
        yaxis_title="Return",
        height=400
    )
    
    # VaR visualization
    returns = results['portfolio_df']['Returns'].dropna()
    fig_var = go.Figure()
    fig_var.add_trace(
        go.Histogram(x=returns, nbinsx=50, name='Returns Distribution')
    )
    fig_var.add_vline(x=results['var_95'], line_dash="dash", line_color="red",
                      annotation_text=f"VaR 95%: {results['var_95']:.2%}")
    fig_var.update_layout(
        title_text="Value at Risk Analysis",
        xaxis_title="Daily Returns",
        yaxis_title="Frequency",
        height=400
    )
    
    # Correlation matrix (simulated)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    np.random.seed(42)
    corr_matrix = np.random.rand(len(symbols), len(symbols))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=symbols,
        y=symbols,
        colorscale='RdBu',
        zmid=0
    ))
    fig_corr.update_layout(title_text="Correlation Matrix", height=400)
    
    return html.Div([
        html.H3([
            html.I(className="fas fa-shield-alt", style={'margin-right': '10px'}),
            f"Risk Management - {symbol}"
        ]),
        dcc.Graph(figure=fig_scatter),
        dcc.Graph(figure=fig_var),
        dcc.Graph(figure=fig_corr)
    ])

def generate_heatmap_tab():
    """Generate market heatmap tab"""
    
    heatmap_data = create_market_heatmap()
    
    fig_heatmap = px.treemap(
        heatmap_data,
        path=['Sector', 'Stock'],
        values='Performance',
        color='Performance',
        color_continuous_scale='RdYlGn',
        title="Market Performance Heatmap"
    )
    
    fig_heatmap.update_layout(height=600)
    
    # Sector performance bar chart
    sector_perf = heatmap_data.groupby('Sector')['Performance'].mean().reset_index()
    fig_sector = px.bar(
        sector_perf,
        x='Sector',
        y='Performance',
        title="Sector Performance Comparison",
        color='Performance',
        color_continuous_scale='RdYlGn'
    )
    fig_sector.update_layout(height=400)
    
    return html.Div([
        html.H3([
            html.I(className="fas fa-fire", style={'margin-right': '10px'}),
            "Market Heatmap & Sector Analysis"
        ]),
        dcc.Graph(figure=fig_heatmap),
        dcc.Graph(figure=fig_sector)
    ])

def generate_trades_tab(results, symbol):
    """Generate trade history tab"""
    
    trades_df = pd.DataFrame(results['trades'])
    
    if len(trades_df) == 0:
        return html.Div([
            html.H3([
                html.I(className="fas fa-list", style={'margin-right': '10px'}),
                f"Trade History - {symbol}"
            ]),
            html.P("No trades executed.")
        ])
    
    # Trade timeline
    fig_timeline = go.Figure()
    
    buy_trades = trades_df[trades_df['Type'] == 'BUY']
    sell_trades = trades_df[trades_df['Type'] == 'SELL']
    
    fig_timeline.add_trace(
        go.Scatter(x=buy_trades['Date'], y=buy_trades['Price'],
                  mode='markers', marker=dict(color='green', size=15, symbol='triangle-up'),
                  name='Buy Trades')
    )
    fig_timeline.add_trace(
        go.Scatter(x=sell_trades['Date'], y=sell_trades['Price'],
                  mode='markers', marker=dict(color='red', size=15, symbol='triangle-down'),
                  name='Sell Trades')
    )
    
    fig_timeline.update_layout(
        title_text=f"Trade Timeline - {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400
    )
    
    # Trade performance
    trade_performance = []
    for i, trade in enumerate(trades_df.itertuples()):
        if trade.Type == 'SELL':
            # Find corresponding buy trade
            buy_trade = trades_df[(trades_df['Type'] == 'BUY') & 
                                 (trades_df['Date'] < trade.Date)].iloc[-1]
            pnl = (trade.Price - buy_trade.Price) * trade.Shares
            trade_performance.append({
                'Date': trade.Date,
                'PnL': pnl,
                'Return': (trade.Price - buy_trade.Price) / buy_trade.Price,
                'Strategy': trade.Strategy
            })
    
    if trade_performance:
        perf_df = pd.DataFrame(trade_performance)
        
        fig_performance = go.Figure()
        fig_performance.add_trace(
            go.Bar(x=perf_df['Date'], y=perf_df['PnL'],
                  marker_color=['green' if p > 0 else 'red' for p in perf_df['PnL']],
                  name='Trade P&L')
        )
        fig_performance.update_layout(
            title_text="Trade Performance",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            height=400
        )
    else:
        fig_performance = go.Figure()
        fig_performance.update_layout(
            title_text="Trade Performance",
            annotations=[dict(text="No completed trades", x=0.5, y=0.5, showarrow=False)]
        )
    
    return html.Div([
        html.H3([
            html.I(className="fas fa-list", style={'margin-right': '10px'}),
            f"Trade History - {symbol}"
        ]),
        dcc.Graph(figure=fig_timeline),
        dcc.Graph(figure=fig_performance),
        html.H4("Trade Details"),
        html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in trades_df.columns])
            ]),
            html.Tbody([
                html.Tr([html.Td(trades_df.iloc[i][col]) for col in trades_df.columns])
                for i in range(len(trades_df))
            ])
        ], style={'width': '100%', 'fontSize': '12px'})
    ])

if __name__ == '__main__':
    print("Starting Advanced Multi-Agent Trading System Dashboard...")
    print("Dashboard will be available at: http://localhost:8057")
    print("Enhanced Features: Advanced Charting, Risk Management, Market Heatmap")
    app.run(debug=True, host='0.0.0.0', port=8057)
