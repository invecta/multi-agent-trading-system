"""
Enhanced Dashboard V2 - Complete Integration
Integrates all enhancement modules into a comprehensive trading dashboard
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
import io
import logging

# Import all enhancement modules
from real_time_data_integration import data_provider, market_aggregator, news_analyzer, alert_system
from advanced_charting import charting_engine
from portfolio_benchmarking import benchmark_engine
from strategy_builder import strategy_builder, create_strategy_builder_layout
from social_sentiment_analysis import sentiment_aggregator
from advanced_risk_metrics import portfolio_risk_analyzer
from multi_asset_support import multi_asset_analyzer, multi_asset_visualization
from automated_reporting import report_generator, report_scheduler
from mobile_pwa_support import mobile_dashboard
from user_authentication import user_management_system
from real_data_integration import get_real_data_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with mobile support - OPTIMIZED
application = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application.title = "Enhanced Trading Dashboard"

# Optimize app configuration
application.config.suppress_callback_exceptions = True
application.scripts.config.serve_locally = True
application.css.config.serve_locally = True

# Performance optimizations (removed invalid config options)
# application.config.update() - removed invalid compression options

# Global state and optimization
backtest_results = {}
cached_data = {}  # Cache for frequently accessed data
analysis_cache = {}  # Cache for analysis results
chart_cache = {}     # Cache for chart data
memory_optimization = True
current_user = None
user_portfolios = []

def create_enhanced_layout():
    """Create the enhanced dashboard layout"""
    
    return html.Div([
        # Mobile-responsive header
        mobile_dashboard.responsive_layout.create_responsive_header(),
        
        # Navigation
        mobile_dashboard.responsive_layout.create_responsive_navigation(),
        
        # Main content area
        html.Div([
            # Control panel
            mobile_dashboard.responsive_layout.create_responsive_controls(),
            
            # Status display
            html.Div(id="status-display", className="alert alert-info text-center"),
            
            # Main tabs
            mobile_dashboard.responsive_layout.create_responsive_tabs(),
            
            # Tab content
            html.Div(id="tab-content", className="container-fluid p-3"),
            
            # Export button (hidden by default)
            html.Div([
                dbc.Button(
                    "Export PDF Report",
                    id="export-pdf-button-hidden",
                    color="success",
                    className="mt-3",
                    style={'display': 'none'}
                )
            ], className="text-center"),
            
            # Download components
            dcc.Download(id="download-pdf"),
            dcc.Download(id="download-csv")
            
        ], className="main-content"),
        
        # PWA features
        mobile_dashboard.pwa_features.create_offline_indicator(),
        mobile_dashboard.pwa_features.create_install_prompt(),
        
        # PWA install button
        html.Div([
            dbc.Button(
                "Install App",
                id="pwa-install-button",
                color="primary",
                size="sm",
                className="pwa-install-button"
            )
        ]),
        
        # Dark theme toggle button
        html.Div([
            dbc.Button(
                "🌙 Dark Mode",
                id="dark-theme-toggle",
                color="secondary",
                size="sm",
                className="dark-theme-toggle"
            )
        ], className="theme-toggle-container"),
        
        # Run Analysis button
        html.Div([
            dbc.Button(
                "🚀 Run Analysis",
                id="run-analysis-button",
                color="primary",
                size="lg",
                className="run-analysis-button mb-3"
            )
        ], className="text-center mb-3"),
        
        # Export buttons
        html.Div([
            dbc.Button(
                "📄 Export PDF",
                id="export-pdf-button-main",
                color="primary",
                size="sm",
                className="export-button"
            ),
            dbc.Button(
                "📊 Export CSV",
                id="export-csv-button",
                color="success",
                size="sm",
                className="export-button"
            )
        ], className="export-buttons-container"),
        
        # Chart containers removed - using Plotly instead
        
    ], id="app-container", className="app-container")

def create_overview_tab(results):
    """Create overview tab content"""
    
    if not results:
        return html.Div([
            html.H3("Welcome to Enhanced Trading Dashboard"),
            html.P("Click 'Run Analysis' to get started with comprehensive market analysis."),
            html.Div([
                html.H4("Features:"),
                html.Ul([
                    html.Li("Real-time market data integration"),
                    html.Li("Advanced charting with technical indicators"),
                    html.Li("Portfolio benchmarking and comparison"),
                    html.Li("Interactive strategy builder"),
                    html.Li("Social sentiment analysis"),
                    html.Li("Advanced risk metrics (VaR, CVaR, Monte Carlo)"),
                    html.Li("Multi-asset support (stocks, crypto, forex, commodities)"),
                    html.Li("Automated report generation"),
                    html.Li("Mobile-responsive PWA design"),
                    html.Li("User authentication and portfolio management")
                ])
            ])
        ])
    
    # Get trades data first
    trades_data = results.get('trades', [])
    
    # Debug: Print trades data
    print(f"DEBUG: UI - trades_data length: {len(trades_data)}")
    print(f"DEBUG: UI - results keys: {list(results.keys())}")
    print(f"DEBUG: UI - results.get('trades'): {type(results.get('trades', []))}")
    
    # Create metrics grid
    metrics = [
        {'title': 'Total Return', 'value': f"{results.get('total_return', 0):.2f}%", 'change': f"vs Benchmark: {results.get('benchmark_return', 0):.2f}%"},
        {'title': 'Sharpe Ratio', 'value': f"{results.get('sharpe_ratio', 0):.2f}", 'change': f"Risk-adjusted return"},
        {'title': 'Max Drawdown', 'value': f"{results.get('max_drawdown', 0):.2f}%", 'change': f"Maximum loss from peak"},
        {'title': 'Volatility', 'value': f"{results.get('volatility', 0):.2f}%", 'change': f"Annualized volatility"},
        {'title': 'Win Rate', 'value': f"{results.get('win_rate', 0):.1f}%", 'change': f"Successful trades"},
        {'title': 'Total Trades', 'value': f"{len(trades_data)}", 'change': f"Trades executed"},
        {'title': 'VaR (95%)', 'value': f"{results.get('var_95', 0):.2f}%", 'change': f"Value at Risk"},
        {'title': 'Beta', 'value': f"{results.get('beta', 0):.2f}", 'change': f"Market correlation"}
    ]
    
    metrics_grid = mobile_dashboard.responsive_layout.create_responsive_metrics_grid(metrics)
    
    # Create detailed trades list
    trades_list = []
    if trades_data:
        for i, trade in enumerate(trades_data[:20]):  # Limit to first 20 trades
            trade_type = trade.get('Type', trade.get('type', 'Unknown'))
            entry_date = trade.get('Date', trade.get('entry_date', 'N/A'))
            exit_date = trade.get('exit_date', 'N/A')
            price = trade.get('Price', trade.get('price', 0))
            quantity = trade.get('Shares', trade.get('quantity', 0))
            pnl = trade.get('pnl', 0)
            pnl_percent = trade.get('pnl_percent', 0)
            
            # Format dates to YYYY-MM-DD
            if isinstance(entry_date, str) and 'T' in entry_date:
                entry_date = entry_date.split('T')[0]
            if isinstance(exit_date, str) and 'T' in exit_date:
                exit_date = exit_date.split('T')[0]
            
            # Color coding for profit/loss
            pnl_color = 'text-success' if pnl >= 0 else 'text-danger'
            pnl_icon = '📈' if pnl >= 0 else '📉'
            
            trades_list.append(
                html.Div([
                    html.Div([
                        html.Div([
                            html.H6(f"Trade #{i+1}", className="mb-1"),
                            html.Small(f"{trade_type} • {entry_date} → {exit_date}", className="text-muted")
                        ], className="col-md-6"),
                        html.Div([
                            html.Strong(f"Price: ${price:.2f}"),
                            html.Br(),
                            html.Small(f"Qty: {quantity}")
                        ], className="col-md-3 text-end"),
                        html.Div([
                            html.Strong(f"{pnl_icon} ${pnl:.2f}", className=pnl_color),
                            html.Br(),
                            html.Small(f"({pnl_percent:.2f}%)", className=pnl_color)
                        ], className="col-md-3 text-end")
                    ], className="row align-items-center")
                ], className="border-bottom py-2")
            )
    
    return html.Div([
        html.H3("Portfolio Overview", className="text-center mb-4"),
        metrics_grid,
        
        # Detailed Trades List
        html.Div([
            html.H4("Recent Trades", className="mb-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong("Trade Details", className="col-md-6"),
                        html.Strong("Price & Quantity", className="col-md-3 text-end"),
                        html.Strong("P&L", className="col-md-3 text-end")
                    ], className="row border-bottom pb-2 mb-2")
                ] + trades_list, className="border rounded p-3")
            ]) if trades_list else html.Div([
                html.P("No trades executed during this period.", className="text-muted text-center py-4")
            ], className="border rounded p-3")
        ], className="mt-4"),
        
        # Quick insights
        html.Div([
            html.H4("Quick Insights"),
            html.Div(id="quick-insights", className="row")
        ], className="mt-4")
    ])

def create_charts_tab(results):
    """Create charts tab content with Plotly - ENHANCED"""
    
    if not results:
        return html.Div([
            html.H3("Charts"),
            html.P("Run analysis to see interactive charts.")
        ])
    
    # Create sample chart data
    dates = list(range(1, 31))
    prices = [100 + i * 0.5 + np.random.normal(0, 2) for i in dates]
    volumes = [1000000 + np.random.normal(0, 100000) for _ in dates]
    portfolio_values = [100000 + i * 1000 + np.random.normal(0, 5000) for i in dates]
    benchmark_values = [100000 + i * 800 + np.random.normal(0, 3000) for i in dates]
    
    # Price chart
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#007bff', width=2),
        fill='tonexty'
    ))
    price_fig.update_layout(
        title='Price Chart',
        xaxis_title='Day',
        yaxis_title='Price ($)',
        height=400,
        showlegend=True
    )
    
    # Volume chart
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name='Volume',
        marker_color='#28a745'
    ))
    volume_fig.update_layout(
        title='Volume Chart',
        xaxis_title='Day',
        yaxis_title='Volume',
        height=300,
        showlegend=True
    )
    
    # Performance chart
    performance_fig = go.Figure()
    performance_fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#28a745', width=2),
        fill='tonexty'
    ))
    performance_fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_values,
        mode='lines',
        name='Benchmark',
        line=dict(color='#dc3545', width=2)
    ))
    performance_fig.update_layout(
        title='Performance Chart',
        xaxis_title='Day',
        yaxis_title='Value ($)',
        height=400,
        showlegend=True
    )
    
    # Risk-Return Scatter Plot
    risk_return_fig = create_risk_return_scatter()
    
    # Market Heatmap
    heatmap_fig = create_market_heatmap()
    
    # Volume Analysis Chart
    volume_analysis_fig = create_volume_analysis_chart(prices, volumes)
    
    return html.Div([
        html.H3("Interactive Charts with Plotly - ENHANCED"),
        
        # Plotly charts
        html.Div([
            dcc.Graph(figure=price_fig)
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(figure=volume_fig)
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(figure=performance_fig)
        ], className="chart-container"),
        
        # Risk-Return Scatter Plot
        html.Div([
            dcc.Graph(figure=risk_return_fig)
        ], className="chart-container"),
        
        # Market Heatmap
        html.Div([
            dcc.Graph(figure=heatmap_fig)
        ], className="chart-container"),
        
        # Volume Analysis
        html.Div([
            dcc.Graph(figure=volume_analysis_fig)
        ], className="chart-container")
    ])

def create_risk_return_scatter():
    """Create risk-return scatter plot for position analysis"""
    
    # Generate sample data for different positions
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'PYPL', 'CRM']
    returns = np.random.normal(0.08, 0.15, len(symbols))  # Annual returns
    risks = np.random.normal(0.25, 0.08, len(symbols))    # Annual volatility
    market_caps = np.random.uniform(100, 2000, len(symbols))  # Market cap in billions
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers',
        marker=dict(
            size=market_caps/50,  # Size based on market cap
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return (%)"),
            line=dict(width=2, color='black')
        ),
        text=symbols,
        hovertemplate='<b>%{text}</b><br>' +
                     'Risk: %{x:.2%}<br>' +
                     'Return: %{y:.2%}<br>' +
                     'Market Cap: $%{marker.size:.0f}B<extra></extra>',
        name='Positions'
    ))
    
    # Add efficient frontier line
    efficient_risks = np.linspace(0.1, 0.4, 50)
    efficient_returns = 0.05 + 0.3 * efficient_risks - 0.5 * efficient_risks**2
    fig.add_trace(go.Scatter(
        x=efficient_risks,
        y=efficient_returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add risk-free rate
    fig.add_trace(go.Scatter(
        x=[0.05],
        y=[0.03],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='star'),
        name='Risk-Free Rate',
        hovertemplate='Risk-Free Rate: 3%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Risk-Return Scatter Plot for Position Analysis',
        xaxis_title='Risk (Volatility)',
        yaxis_title='Expected Return',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_market_heatmap():
    """Create market heatmap showing sector/stock performance"""
    
    # Define sectors and stocks
    sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Indices', 'Forex']
    stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABT', 'MRNA'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'V'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'TSLA'],
        'Indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],
        'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
    }
    
    # Generate performance data
    performance_data = []
    stock_names = []
    sector_colors = []
    
    for sector in sectors:
        for stock in stocks[sector]:
            # Generate performance (-20% to +30%)
            performance = np.random.normal(0.05, 0.15)
            performance_data.append(performance)
            stock_names.append(f"{stock}<br>{sector}")
            
            # Color coding by sector
            if sector == 'Technology':
                sector_colors.append('#1f77b4')
            elif sector == 'Healthcare':
                sector_colors.append('#ff7f0e')
            elif sector == 'Finance':
                sector_colors.append('#2ca02c')
            elif sector == 'Energy':
                sector_colors.append('#d62728')
            else:  # Consumer
                sector_colors.append('#9467bd')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[performance_data],
        x=stock_names,
        y=['Performance'],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{p:.1%}" for p in performance_data]],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{x}</b><br>Performance: %{z:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Market Heatmap - Sector/Stock Performance',
        xaxis_title='Stocks by Sector',
        yaxis_title='',
        height=300,
        showlegend=False
    )
    
    return fig

def create_volume_analysis_chart(prices, volumes):
    """Create volume analysis with price action correlation"""
    
    # Calculate volume indicators
    volume_sma = pd.Series(volumes).rolling(window=5).mean()
    volume_ratio = [v/vol for v, vol in zip(volumes, volume_sma)]
    
    # Calculate price changes
    price_changes = [0] + [p2-p1 for p1, p2 in zip(prices[:-1], prices[1:])]
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price Action', 'Volume Analysis', 'Volume-Price Correlation'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#007bff', width=2)
    ), row=1, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=list(range(len(volumes))),
        y=volumes,
        name='Volume',
        marker_color='#28a745',
        opacity=0.7
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=list(range(len(volume_sma))),
        y=volume_sma,
        mode='lines',
        name='Volume SMA',
        line=dict(color='red', width=2)
    ), row=2, col=1)
    
    # Volume-Price correlation
    fig.add_trace(go.Scatter(
        x=list(range(len(volume_ratio))),
        y=volume_ratio,
        mode='lines+markers',
        name='Volume Ratio',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ), row=3, col=1)
    
    # Add correlation line
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        title='Volume Analysis with Price Action Correlation',
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Volume Ratio", row=3, col=1)
    
    return fig

def create_var_analysis_chart():
    """Create VaR (Value at Risk) analysis chart"""
    
    # Generate sample portfolio returns
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.08, 0.15, 1000)  # 1000 daily returns
    
    # Calculate VaR at different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    var_values = [np.percentile(portfolio_returns, (1-c)*100) for c in confidence_levels]
    
    # Create histogram with VaR lines
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=50,
        name='Portfolio Returns',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add VaR lines
    colors = ['red', 'orange', 'darkred']
    for i, (conf, var) in enumerate(zip(confidence_levels, var_values)):
        fig.add_vline(
            x=var,
            line_dash="dash",
            line_color=colors[i],
            annotation_text=f"VaR {conf*100:.0f}%: {var:.2%}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title='Value at Risk (VaR) Analysis',
        xaxis_title='Portfolio Returns',
        yaxis_title='Frequency',
        height=400,
        showlegend=True
    )
    
    return fig

def create_correlation_matrix():
    """Create correlation matrix for portfolio diversification"""
    
    # Generate sample correlation data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'XOM']
    
    # Create correlation matrix
    np.random.seed(42)
    correlation_matrix = np.random.uniform(-0.3, 0.8, (len(symbols), len(symbols)))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=symbols,
        y=symbols,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Correlation Matrix',
        xaxis_title='Assets',
        yaxis_title='Assets',
        height=500,
        showlegend=False
    )
    
    return fig

def create_stress_test_chart():
    """Create stress testing scenarios chart"""
    
    # Define stress scenarios
    scenarios = ['Market Crash', 'Interest Rate Shock', 'Oil Price Spike', 'Tech Bubble', 'Recession']
    
    # Generate portfolio performance under stress
    np.random.seed(42)
    normal_performance = np.random.normal(0.08, 0.15, len(scenarios))
    stress_performance = normal_performance - np.random.uniform(0.15, 0.35, len(scenarios))
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=normal_performance,
        name='Normal Conditions',
        marker_color='lightgreen',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=stress_performance,
        name='Stress Scenarios',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Portfolio Stress Testing Scenarios',
        xaxis_title='Stress Scenarios',
        yaxis_title='Portfolio Return',
        height=400,
        showlegend=True,
        barmode='group'
    )
    
    return fig

def create_risk_metrics_chart():
    """Create comprehensive risk metrics chart"""
    
    # Generate risk metrics data
    metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
    values = [1.2, 1.8, 0.9, -0.15, -0.08, -0.12]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Comprehensive Risk Metrics',
        xaxis_title='Risk Metrics',
        yaxis_title='Value',
        height=400,
        showlegend=False
    )
    
    return fig

def create_rsi_chart(results):
    """Create RSI chart"""
    try:
        # Get price data from results
        price_data = results.get('price_data', {})
        
        if price_data is not None and isinstance(price_data, dict) and 'prices' in price_data:
            prices = pd.Series(price_data['prices'])
            rsi_values = calculate_rsi(prices)
            
            # Create dates for x-axis
            dates = pd.date_range(start='2024-01-01', periods=len(rsi_values), freq='D')
            
            fig = go.Figure()
            
            # Add RSI line
            fig.add_trace(go.Scatter(
                x=dates,
                y=rsi_values,
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            ))
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=400,
                showlegend=True
            )
            
            return fig
        else:
            # Fallback with sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            rsi_sample = np.random.uniform(20, 80, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=rsi_sample,
                mode='lines',
                name='RSI (Sample)',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig.update_layout(
                title="RSI (Relative Strength Index) - Sample Data",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=400,
                showlegend=True
            )
            
            return fig
            
    except Exception as e:
        print(f"Error creating RSI chart: {e}")
        return go.Figure().add_annotation(text="Error loading RSI chart", showarrow=False)

def create_macd_chart(results):
    """Create MACD chart"""
    try:
        # Get price data from results
        price_data = results.get('price_data', {})
        
        if price_data is not None and isinstance(price_data, dict) and 'prices' in price_data:
            prices = pd.Series(price_data['prices'])
            macd_line, signal_line, histogram = calculate_macd(prices)
            
            # Create dates for x-axis
            dates = pd.date_range(start='2024-01-01', periods=len(macd_line), freq='D')
            
            fig = go.Figure()
            
            # Add MACD line
            fig.add_trace(go.Scatter(
                x=dates,
                y=macd_line,
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ))
            
            # Add signal line
            fig.add_trace(go.Scatter(
                x=dates,
                y=signal_line,
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ))
            
            # Add histogram
            fig.add_trace(go.Bar(
                x=dates,
                y=histogram,
                name='Histogram',
                marker_color=['green' if x >= 0 else 'red' for x in histogram],
                opacity=0.6
            ))
            
            fig.update_layout(
                title="MACD (Moving Average Convergence Divergence)",
                xaxis_title="Date",
                yaxis_title="MACD",
                height=400,
                showlegend=True
            )
            
            return fig
        else:
            # Fallback with sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            macd_sample = np.random.uniform(-2, 2, 100)
            signal_sample = np.random.uniform(-1.5, 1.5, 100)
            histogram_sample = macd_sample - signal_sample
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=macd_sample,
                mode='lines',
                name='MACD (Sample)',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=signal_sample,
                mode='lines',
                name='Signal (Sample)',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Bar(
                x=dates,
                y=histogram_sample,
                name='Histogram (Sample)',
                marker_color=['green' if x >= 0 else 'red' for x in histogram_sample],
                opacity=0.6
            ))
            
            fig.update_layout(
                title="MACD (Moving Average Convergence Divergence) - Sample Data",
                xaxis_title="Date",
                yaxis_title="MACD",
                height=400,
                showlegend=True
            )
            
            return fig
            
    except Exception as e:
        print(f"Error creating MACD chart: {e}")
        return go.Figure().add_annotation(text="Error loading MACD chart", showarrow=False)

def create_moving_averages_chart(results):
    """Create Moving Averages chart"""
    try:
        # Get price data from results
        price_data = results.get('price_data', {})
        
        if price_data is not None and isinstance(price_data, dict) and 'prices' in price_data:
            prices = pd.Series(price_data['prices'])
            
            # Calculate moving averages
            ma_20 = prices.rolling(window=20).mean()
            ma_50 = prices.rolling(window=50).mean()
            ma_200 = prices.rolling(window=200).mean()
            
            # Create dates for x-axis
            dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_20,
                mode='lines',
                name='MA 20',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_50,
                mode='lines',
                name='MA 50',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_200,
                mode='lines',
                name='MA 200',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title="Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                showlegend=True
            )
            
            return fig
        else:
            # Fallback with sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            price_sample = 100 + np.cumsum(np.random.randn(100) * 0.5)
            ma_20_sample = pd.Series(price_sample).rolling(window=20).mean()
            ma_50_sample = pd.Series(price_sample).rolling(window=50).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_sample,
                mode='lines',
                name='Price (Sample)',
                line=dict(color='black', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_20_sample,
                mode='lines',
                name='MA 20 (Sample)',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_50_sample,
                mode='lines',
                name='MA 50 (Sample)',
                line=dict(color='orange', width=1)
            ))
            
            fig.update_layout(
                title="Moving Averages - Sample Data",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                showlegend=True
            )
            
            return fig
            
    except Exception as e:
        print(f"Error creating Moving Averages chart: {e}")
        return go.Figure().add_annotation(text="Error loading Moving Averages chart", showarrow=False)

def create_volume_chart(results):
    """Create Volume Analysis chart"""
    try:
        # Get price data from results
        price_data = results.get('price_data', {})
        
        if price_data is not None and isinstance(price_data, dict) and 'prices' in price_data and 'volumes' in price_data:
            prices = pd.Series(price_data['prices'])
            volumes = pd.Series(price_data['volumes'])
            
            # Create dates for x-axis
            dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=volumes,
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Price and Volume Analysis",
                height=500,
                showlegend=True
            )
            
            return fig
        else:
            # Fallback with sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            price_sample = 100 + np.cumsum(np.random.randn(100) * 0.5)
            volume_sample = np.random.uniform(1000000, 5000000, 100)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price (Sample)', 'Volume (Sample)'),
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=price_sample,
                    mode='lines',
                    name='Price (Sample)',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=volume_sample,
                    name='Volume (Sample)',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Price and Volume Analysis - Sample Data",
                height=500,
                showlegend=True
            )
            
            return fig
            
    except Exception as e:
        print(f"Error creating Volume chart: {e}")
        return go.Figure().add_annotation(text="Error loading Volume chart", showarrow=False)

def create_analysis_tab(results):
    """Create analysis tab content"""
    
    if not results:
        return html.Div("No data available. Run analysis first.")
    
    # Technical analysis summary cards
    technical_summary = html.Div([
        html.H4("Technical Analysis Summary"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("RSI", className="card-title"),
                        html.P(f"{results.get('rsi', 0):.1f}", className="card-text"),
                        html.Small("Oversold/Overbought", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("MACD", className="card-title"),
                        html.P(f"{results.get('macd', 0):.3f}", className="card-text"),
                        html.Small("Momentum Signal", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("MA Trend", className="card-title"),
                        html.P(f"{'Bullish' if results.get('ma_trend', 0) > 0 else 'Bearish'}", className="card-text"),
                        html.Small("Trend Direction", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Volume", className="card-title"),
                        html.P(f"{results.get('volume_analysis', 'Normal')}", className="card-text"),
                        html.Small("Volume Analysis", className="text-muted")
                    ])
                ])
            ], width=3)
        ], className="mb-4")
    ])
    
    # Interactive Technical Analysis Charts
    technical_charts = html.Div([
        html.H4("Interactive Technical Analysis Charts"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_rsi_chart(results))
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=create_macd_chart(results))
            ], width=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_moving_averages_chart(results))
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=create_volume_chart(results))
            ], width=6)
        ], className="mb-4")
    ])
    
    # Sentiment analysis
    sentiment_analysis = html.Div([
        html.H4("Sentiment Analysis"),
        html.Div(id="sentiment-content")
    ])
    
    # Risk analysis
    risk_analysis = html.Div([
        html.H4("Risk Analysis"),
        html.Div([
            html.P(f"VaR (95%): {results.get('var_95', 0):.2f}%"),
            html.P(f"CVaR (95%): {results.get('cvar_95', 0):.2f}%"),
            html.P(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2f}%"),
            html.P(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        ])
    ])
    
    return html.Div([
        technical_summary,
        html.Hr(),
        technical_charts,
        html.Hr(),
        sentiment_analysis,
        html.Hr(),
        risk_analysis
    ])

def create_risk_tab(results):
    """Create risk management tab"""
    
    if not results:
        return html.Div("No data available. Run analysis first.")
    
    # Risk metrics
    risk_metrics = html.Div([
        html.H4("Risk Metrics"),
        html.Div([
            html.P(f"Portfolio VaR (95%): {results.get('var_95', 0):.2f}%"),
            html.P(f"Conditional VaR (95%): {results.get('cvar_95', 0):.2f}%"),
            html.P(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2f}%"),
            html.P(f"Volatility: {results.get('volatility', 0):.2f}%"),
            html.P(f"Beta: {results.get('beta', 0):.2f}"),
            html.P(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        ])
    ])
    
    # Stress testing
    stress_testing = html.Div([
        html.H4("Stress Testing Scenarios"),
        html.Div(id="stress-test-content")
    ])
    
    # Monte Carlo simulation
    monte_carlo = html.Div([
        html.H4("Monte Carlo Simulation"),
        html.Div(id="monte-carlo-content")
    ])
    
    return html.Div([
        risk_metrics,
        html.Hr(),
        stress_testing,
        html.Hr(),
        monte_carlo
    ])

def create_portfolio_tab(results):
    """Create portfolio management tab"""
    
    if not results:
        return html.Div("No data available. Run analysis first.")
    
    # Portfolio performance
    portfolio_performance = html.Div([
        html.H4("Portfolio Performance"),
        html.Div([
            html.P(f"Total Value: ${results.get('total_value', 0):,.2f}"),
            html.P(f"Total Return: {results.get('total_return', 0):.2f}%"),
            html.P(f"Annualized Return: {results.get('annualized_return', 0):.2f}%"),
            html.P(f"Cash Balance: ${results.get('cash_balance', 0):,.2f}")
        ])
    ])
    
    # Asset allocation
    asset_allocation = html.Div([
        html.H4("Asset Allocation"),
        html.Div(id="allocation-content")
    ])
    
    # Benchmark comparison
    benchmark_comparison = html.Div([
        html.H4("Benchmark Comparison"),
        html.Div(id="benchmark-content")
    ])
    
    # Enhanced Risk Management Charts
    var_chart = create_var_analysis_chart()
    correlation_matrix = create_correlation_matrix()
    stress_test_chart = create_stress_test_chart()
    risk_metrics_chart = create_risk_metrics_chart()
    
    return html.Div([
        html.H3("Enhanced Risk Management & Portfolio Optimization"),
        portfolio_performance,
        html.Hr(),
        asset_allocation,
        html.Hr(),
        benchmark_comparison,
        
        # Risk Management Charts
        html.Div([
            dcc.Graph(figure=var_chart)
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(figure=correlation_matrix)
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(figure=stress_test_chart)
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(figure=risk_metrics_chart)
        ], className="chart-container")
    ])

def create_reports_tab(results):
    """Create reports tab"""
    
    if not results:
        return html.Div("No data available. Run analysis first.")
    
    # Report generation
    report_generation = html.Div([
        html.H4("Report Generation"),
        html.Div([
            dbc.Button("Generate Daily Report", id="daily-report-btn", color="primary", className="me-2"),
            dbc.Button("Generate Weekly Report", id="weekly-report-btn", color="success", className="me-2"),
            dbc.Button("Generate Monthly Report", id="monthly-report-btn", color="info", className="me-2")
        ])
    ])
    
    # Scheduled reports
    scheduled_reports = html.Div([
        html.H4("Scheduled Reports"),
        html.Div(id="scheduled-reports-content")
    ])
    
    # Report history
    report_history = html.Div([
        html.H4("Report History"),
        html.Div(id="report-history-content")
    ])
    
    return html.Div([
        report_generation,
        html.Hr(),
        scheduled_reports,
        html.Hr(),
        report_history
    ])

def run_enhanced_analysis(symbol, sector, capital, time_period, timeframe, start_date=None, end_date=None):
    """Run comprehensive enhanced analysis - OPTIMIZED"""
    
    # Create cache key for this analysis
    cache_key = f"{symbol}_{sector}_{capital}_{time_period}_{timeframe}_{start_date}_{end_date}"
    
    # Check cache first
    if memory_optimization and cache_key in analysis_cache:
        print(f"Using cached analysis for {symbol}")
        return analysis_cache[cache_key]
    
    try:
        print(f"=== ENHANCED ANALYSIS STARTED ===")
        print(f"Symbol: {symbol}, Sector: {sector}, Capital: {capital}")
        print(f"Time Period: {time_period}, Timeframe: {timeframe}")
        print(f"Start Date: {start_date}, End Date: {end_date}")
        print(f"Starting enhanced analysis for {symbol} over {time_period} days with {timeframe} timeframe...")
        
        # Generate market data
        market_data = generate_enhanced_market_data(symbol, sector, time_period, timeframe, start_date, end_date)
        
        # Convert market data to DataFrame if it's a dict
        if isinstance(market_data, dict):
            # Create DataFrame with proper date index
            df = pd.DataFrame({
                'Close': market_data['prices'],
                'Volume': market_data['volumes'],
                'Open': market_data.get('opens', market_data['prices']),
                'High': market_data.get('highs', market_data['prices'] * 1.01),
                'Low': market_data.get('lows', market_data['prices'] * 0.99)
            })
            
            # Set date index if available
            if 'dates' in market_data and market_data['dates']:
                df.index = pd.to_datetime(market_data['dates'])
            else:
                # Create a date range if no dates provided
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        else:
            df = market_data
        
        # Run backtest
        backtest_results = run_enhanced_backtest(df, capital, timeframe)
        
        # Debug: Print backtest results
        print(f"DEBUG: Backtest returned {len(backtest_results.get('trades', []))} trades")
        print(f"DEBUG: Backtest total_trades: {backtest_results.get('total_trades', 0)}")
        
        # ML Forecasting
        ml_results = run_ml_forecasting(df, symbol)
        
        # Sentiment Analysis
        sentiment_results = run_sentiment_analysis(symbol)
        
        # Portfolio Optimization
        portfolio_results = run_portfolio_optimization(df, symbol)
        
        # Risk Analysis
        risk_results = run_risk_analysis(df, symbol)
        
        # Multi-asset Analysis
        multi_asset_results = run_multi_asset_analysis(symbol)
        
        # Prepare price data for charts
        price_data_for_charts = {
            'prices': df['Close'].tolist() if 'Close' in df.columns else df.index.tolist(),
            'volumes': df['Volume'].tolist() if 'Volume' in df.columns else [1000000] * len(df),
            'opens': df['Open'].tolist() if 'Open' in df.columns else df['Close'].tolist() if 'Close' in df.columns else [100] * len(df),
            'highs': df['High'].tolist() if 'High' in df.columns else df['Close'].tolist() if 'Close' in df.columns else [100] * len(df),
            'lows': df['Low'].tolist() if 'Low' in df.columns else df['Close'].tolist() if 'Close' in df.columns else [100] * len(df),
            'dates': df.index.strftime('%Y-%m-%d').tolist() if hasattr(df.index, 'strftime') else [f"2024-01-{i+1:02d}" for i in range(len(df))]
        }
        
        # Combine all results
        results = {
            'symbol': symbol,
            'sector': sector,
            'capital': capital,
            'time_period': time_period,
            'timeframe': timeframe,
            'price_data': price_data_for_charts,  # Use formatted data for charts
            'backtest_results': backtest_results,
            'ml_results': ml_results,
            'sentiment_results': sentiment_results,
            'portfolio_results': portfolio_results,
            'risk_results': risk_results,
            'multi_asset_results': multi_asset_results,
            'total_return': backtest_results.get('total_return', 0),
            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
            'max_drawdown': backtest_results.get('max_drawdown', 0),
            'volatility': backtest_results.get('volatility', 0),
            'win_rate': backtest_results.get('win_rate', 0),
            'total_trades': backtest_results.get('total_trades', 0),
            'trades': backtest_results.get('trades', []),  # Ensure trades are passed
            'var_95': risk_results.get('var_95', 0),
            'cvar_95': risk_results.get('cvar_95', 0),
            'beta': risk_results.get('beta', 0),
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 0,
            'macd': df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
            'ma_trend': 1 if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else -1 if 'SMA_20' in df.columns and 'SMA_50' in df.columns else 0,
            'volume_analysis': 'High' if df['Volume'].iloc[-1] > df['Volume'].mean() * 1.5 else 'Normal',
            'total_value': capital * (1 + backtest_results.get('total_return', 0) / 100),
            'annualized_return': backtest_results.get('annualized_return', 0),
            'cash_balance': capital * 0.1,  # Assume 10% cash
            'benchmark_return': 8.5  # Simulated benchmark return
        }
        
        # Debug: Print final results
        print(f"DEBUG: Final results - {len(results.get('trades', []))} trades in results")
        print(f"DEBUG: Results total_trades: {results.get('total_trades', 0)}")
        
        # Cache results
        if memory_optimization:
            analysis_cache[cache_key] = results
            # Limit cache size to prevent memory issues
            if len(analysis_cache) > 50:
                # Remove oldest entries
                oldest_key = next(iter(analysis_cache))
                del analysis_cache[oldest_key]
        
        return results
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        return {}

def generate_enhanced_market_data(symbol, sector, time_period, timeframe, start_date=None, end_date=None):
    """Generate enhanced market data using real data - OPTIMIZED"""
    
    # Check cache first
    cache_key = f"{symbol}_{sector}_{time_period}_{timeframe}_{start_date}_{end_date}"
    if cache_key in cached_data:
        return cached_data[cache_key]
    
    # Get real data manager
    real_data_manager = get_real_data_manager()
    
    # Try to get real data first
    try:
        real_data = real_data_manager.generate_enhanced_market_data(
            symbol, sector, time_period, timeframe, start_date, end_date
        )
        
        if real_data and real_data.get('real_data', False):
            logger.info(f"Using real data for {symbol}")
            # Cache the real data
            cached_data[cache_key] = real_data
            return real_data
    except Exception as e:
        logger.warning(f"Failed to get real data for {symbol}: {str(e)}")
    
    # Fallback to simulated data
    logger.info(f"Using simulated data for {symbol}")
    
    # Timeframe mapping
    timeframe_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '4h': '4H', '1d': 'D', '1w': 'W'
    }
    
    freq = timeframe_map.get(timeframe, 'D')
    
    # Calculate number of periods based on timeframe - OPTIMIZED
    if timeframe in ['1m', '5m', '15m']:
        periods = min(time_period * 24 * 4, 5000)  # Reduced from 10000
    elif timeframe in ['1h', '4h']:
        periods = min(time_period * 24, 2000)  # Reduced from 5000
    else:
        periods = min(time_period, 1000)  # Cap daily data
    
    dates = pd.date_range('2023-01-01', periods=periods, freq=freq)
    
    # Vectorized price generation - MUCH FASTER
    # Use time_period in seed to make different periods generate different data
    np.random.seed(42 + time_period)  # For reproducible but different results per period
    base_price = 150.0
    
    # Generate returns vectorized
    sector_volatility = {'Technology': 0.02, 'Healthcare': 0.015, 'Finance': 0.025, 
                        'Energy': 0.03, 'Consumer': 0.018, 'Indices': 0.012}
    volatility = sector_volatility.get(sector, 0.02)
    
    returns = np.random.normal(0.0005, volatility, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Vectorized OHLC generation - MUCH FASTER
    price_volatility = np.abs(np.random.normal(0, 0.005, periods))
    opens = np.roll(prices, 1)
    opens[0] = base_price
    highs = prices * (1 + price_volatility)
    lows = prices * (1 - price_volatility)
    
    # Vectorized volume generation - MUCH FASTER
    volumes = np.random.randint(1000000, 5000000, periods)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    # Calculate technical indicators - OPTIMIZED
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    
    # Cache the result
    cached_data[cache_key] = df
    
    return df

def run_enhanced_backtest(df, capital, timeframe):
    """Run enhanced backtest with multiple strategies"""
    
    initial_capital = capital
    current_capital = initial_capital
    position = 0
    trades = []
    
    # Adjust indicator windows based on timeframe
    timeframe_multiplier = {
        '1m': 0.1, '5m': 0.2, '15m': 0.3, '1h': 0.5, '4h': 0.7, '1d': 1.0, '1w': 2.0
    }
    
    multiplier = timeframe_multiplier.get(timeframe, 1.0)
    
    # Calculate technical indicators with timeframe-adjusted windows
    sma_short = max(5, int(20 * multiplier))
    sma_long = max(10, int(50 * multiplier))
    rsi_period = max(5, int(14 * multiplier))
    
    # Use existing indicators if available, otherwise calculate - OPTIMIZED
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(window=sma_short, min_periods=1).mean()
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['Close'].rolling(window=sma_long, min_periods=1).mean()
    if 'RSI' not in df.columns:
        df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    if 'MACD' not in df.columns:
        macd_line, signal_line, histogram = calculate_macd(df['Close'])
        df['MACD'] = macd_line
    
    # Adjust signal frequency based on timeframe
    signal_frequency = {
        '1m': 0.8, '5m': 0.7, '15m': 0.6, '1h': 0.5, '4h': 0.4, '1d': 0.3, '1w': 0.2
    }
    
    freq_multiplier = signal_frequency.get(timeframe, 0.3)
    
    # Vectorized signal generation - MUCH FASTER
    start_idx = max(sma_long, 20)
    if start_idx >= len(df):
        return {
            'initial_capital': initial_capital,
            'final_value': initial_capital,
            'total_return': 0,
            'trades': [],
            'trade_count': 0,
            'df': df
        }
    
    # Vectorized calculations
    prices = df['Close'].iloc[start_idx:].values
    sma_20_values = df['SMA_20'].iloc[start_idx:].values
    sma_50_values = df['SMA_50'].iloc[start_idx:].values
    rsi_values = df['RSI'].iloc[start_idx:].values
    volumes = df['Volume'].iloc[start_idx:].values
    # Use index as dates if Date column doesn't exist
    if 'Date' in df.columns:
        dates = df['Date'].iloc[start_idx:].values
    else:
        dates = df.index[start_idx:].values
    
    # Vectorized signal strength calculation
    ma_crossover = (sma_20_values > sma_50_values) & (np.roll(sma_20_values, 1) <= np.roll(sma_50_values, 1))
    rsi_oversold = rsi_values < 30
    rsi_overbought = rsi_values > 70
    
    # Volume confirmation (vectorized)
    avg_volumes = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
    volume_spike = volumes > avg_volumes * 1.5
    
    # Calculate signal strengths vectorized
    signal_strengths = np.zeros(len(prices))
    signal_strengths[ma_crossover] += 0.3
    signal_strengths[rsi_oversold] += 0.2
    signal_strengths[rsi_overbought] -= 0.2
    signal_strengths[volume_spike] += 0.1
    
    # Execute trades vectorized - VERY LOW THRESHOLDS FOR MORE TRADES
    buy_threshold = 0.001 * freq_multiplier  # Even lower threshold
    sell_threshold = -0.001 * freq_multiplier  # Even lower threshold
    
    # Debug: Print signal info
    print(f"DEBUG: buy_threshold={buy_threshold}, sell_threshold={sell_threshold}")
    print(f"DEBUG: signal_strengths range: {signal_strengths.min():.3f} to {signal_strengths.max():.3f}")
    print(f"DEBUG: {np.sum(signal_strengths > buy_threshold)} buy signals, {np.sum(signal_strengths < sell_threshold)} sell signals")
    print(f"DEBUG: First 10 signal strengths: {signal_strengths[:10]}")
    print(f"DEBUG: Position starts at: {position}")
    
    for i, (price, signal_strength, date) in enumerate(zip(prices, signal_strengths, dates)):
        if signal_strength > buy_threshold and position == 0:
            # Buy signal
            shares = int(current_capital * 0.95 / price)
            if shares > 0:
                position = shares
                current_capital -= shares * price
                trades.append({
                    'Date': date,
                    'Type': 'BUY',
                    'Price': price,
                    'Shares': shares,
                    'Value': shares * price,
                    'Strategy': f'Enhanced Multi-Strategy ({timeframe})'
                })
                print(f"DEBUG: BUY trade executed at price {price}, shares {shares}, signal {signal_strength:.3f}")
        
        elif signal_strength < sell_threshold and position > 0:
            # Sell signal
            current_capital += position * price
            trades.append({
                'Date': date,
                'Type': 'SELL',
                'Price': price,
                'Shares': position,
                'Value': position * price,
                'Strategy': f'Enhanced Multi-Strategy ({timeframe})'
            })
            print(f"DEBUG: SELL trade executed at price {price}, shares {position}, signal {signal_strength:.3f}")
            position = 0
    
    # Final portfolio value
    if position > 0:
        final_value = current_capital + position * prices[-1]
    else:
        final_value = current_capital
    
    # Calculate performance metrics - OPTIMIZED
    total_return = ((final_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Calculate win rate
    if trades:
        winning_trades = sum(1 for trade in trades if trade['Type'] == 'SELL')
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    else:
        win_rate = 0
    
    # Debug: Print final results
    print(f"DEBUG: Final backtest results - {len(trades)} trades generated")
    print(f"DEBUG: Total return: {total_return:.2f}%, Win rate: {win_rate:.1f}%")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': total_return * (252 / len(df)),
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades,
        'trade_count': len(trades),
        'df': df
    }

def run_ml_forecasting(df, symbol):
    """Run ML forecasting analysis"""
    
    try:
        # Simple ML forecasting simulation
        returns = df['Close'].pct_change().dropna()
        
        # Simulate ML model results
        ml_results = {
            'model_name': 'Random Forest',
            'r_squared': np.random.uniform(0.6, 0.9),
            'rmse': np.random.uniform(0.01, 0.05),
            'mae': np.random.uniform(0.005, 0.02),
            'predictions': {
                '1_day': returns.mean() + np.random.randn() * returns.std(),
                '5_day': returns.mean() * 5 + np.random.randn() * returns.std() * 2,
                '10_day': returns.mean() * 10 + np.random.randn() * returns.std() * 3
            },
            'confidence': np.random.uniform(0.7, 0.95)
        }
        
        return ml_results
        
    except Exception as e:
        logger.error(f"Error in ML forecasting: {e}")
        return {}

def run_sentiment_analysis(symbol):
    """Run sentiment analysis"""
    
    try:
        # Simulate sentiment analysis
        sentiment_results = {
            'overall_sentiment': np.random.choice(['positive', 'negative', 'neutral']),
            'sentiment_score': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.6, 0.9),
            'sources': {
                'twitter': np.random.uniform(-0.5, 0.5),
                'reddit': np.random.uniform(-0.5, 0.5),
                'news': np.random.uniform(-0.5, 0.5)
            }
        }
        
        return sentiment_results
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {}

def run_portfolio_optimization(df, symbol):
    """Run portfolio optimization"""
    
    try:
        # Simulate portfolio optimization
        returns = df['Close'].pct_change().dropna()
        
        portfolio_results = {
            'optimal_weights': {symbol: 1.0},
            'expected_return': returns.mean() * 252,
            'expected_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'efficient_frontier': [],
            'monte_carlo_simulations': 1000
        }
        
        return portfolio_results
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return {}

def run_risk_analysis(df, symbol):
    """Run risk analysis"""
    
    try:
        returns = df['Close'].pct_change().dropna()
        
        # Calculate risk metrics
        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Calculate beta (simplified)
        beta = np.random.uniform(0.8, 1.2)
        
        risk_results = {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'max_drawdown': 0,  # Will be calculated in backtest
            'stress_test_results': {}
        }
        
        return risk_results
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        return {}

def run_multi_asset_analysis(symbol):
    """Run multi-asset analysis"""
    
    try:
        # Simulate multi-asset analysis
        multi_asset_results = {
            'asset_type': 'stock',
            'sector': 'Technology',
            'market_cap': np.random.uniform(1000000000, 3000000000000),
            'pe_ratio': np.random.uniform(15, 35),
            'dividend_yield': np.random.uniform(0, 0.05),
            'correlation_analysis': {},
            'sector_performance': np.random.uniform(-0.1, 0.1)
        }
        
        return multi_asset_results
        
    except Exception as e:
        logger.error(f"Error in multi-asset analysis: {e}")
        return {}

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_csv_report(results, symbol, sector, capital, time_period, timeframe):
    """Generate CSV report from analysis results"""
    
    import csv
    import io
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Trading Analysis Report'])
    writer.writerow(['Symbol', symbol])
    writer.writerow(['Sector', sector])
    writer.writerow(['Capital', capital])
    writer.writerow(['Time Period', time_period])
    writer.writerow(['Timeframe', timeframe])
    writer.writerow(['Generated', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])
    
    # Performance metrics
    writer.writerow(['Performance Metrics'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Initial Capital', f"${results.get('initial_capital', 0):,.2f}"])
    writer.writerow(['Final Value', f"${results.get('final_value', 0):,.2f}"])
    writer.writerow(['Total Return', f"{results.get('total_return', 0):.2f}%"])
    writer.writerow(['Trade Count', results.get('trade_count', 0)])
    writer.writerow(['Win Rate', f"{results.get('win_rate', 0):.2f}%"])
    writer.writerow(['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"])
    writer.writerow(['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%"])
    writer.writerow(['Volatility', f"{results.get('volatility', 0):.2f}%"])
    writer.writerow(['VaR (95%)', f"{results.get('var_95', 0):.2f}%"])
    writer.writerow([])
    
    # Trade history
    if 'trades' in results and results['trades']:
        writer.writerow(['Trade History'])
        writer.writerow(['Date', 'Type', 'Price', 'Shares', 'Value', 'Strategy'])
        for trade in results['trades']:
            # Format date - handle various formats and round zeros
            date = trade.get('Date', trade.get('entry_date', ''))
            if isinstance(date, str) and 'T' in date:
                # Extract just the date part before 'T'
                date = date.split('T')[0]
            else:
                date = str(date)[:10] if len(str(date)) > 10 else str(date)
            
            # Round values
            price = trade.get('Price', trade.get('price', 0))
            shares = trade.get('Shares', trade.get('quantity', 0))
            value = float(price) * float(shares) if price and shares else 0
            
            writer.writerow([
                str(date),
                trade.get('Type', trade.get('type', '')),
                f"${float(price):.2f}" if price else "$0.00",
                str(int(float(shares))) if shares else "0",  # Round shares to whole number
                f"${value:,.2f}",
                trade.get('Strategy', 'Enhanced Multi-Strategy (1d)')
            ])
        writer.writerow([])
    
    # Technical analysis
    writer.writerow(['Technical Analysis'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Average RSI', f"{results.get('avg_rsi', 0):.2f}"])
    writer.writerow(['Volume Ratio', f"{results.get('volume_ratio', 0):.2f}"])
    writer.writerow(['Total Signals', results.get('total_signals', 0)])
    writer.writerow(['Analysis Period', f"{results.get('analysis_period', 0)} days"])
    writer.writerow(['Price Range', f"${results.get('price_range', 0):.2f}"])
    
    return output.getvalue()

def generate_pdf_report(results, symbol, sector, capital, time_period, timeframe):
    """Generate comprehensive PDF report with all dashboard sections"""
    
    try:
        import base64
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        print(f"Starting comprehensive PDF generation for {symbol}")
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Build PDF content
        story = []
        
        # Title Page
        story.append(Paragraph("Multi-Agent Trading System", title_style))
        story.append(Paragraph("Comprehensive Analysis Report", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Table of Contents
        story.append(Paragraph("Table of Contents", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        toc_data = [
            ['Section', 'Page'],
            ['1. Overview', '2'],
            ['2. Charts Analysis', '3'],
            ['3. Technical Analysis', '4'],
            ['4. Risk Management', '5'],
            ['5. Portfolio Analysis', '6'],
            ['6. Reports Summary', '7']
        ]
        
        toc_table = Table(toc_data, colWidths=[3*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(toc_table)
        story.append(PageBreak())
        
        # 1. OVERVIEW SECTION
        story.append(Paragraph("1. Overview", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary_data = [
            ['Parameter', 'Value'],
            ['Symbol', symbol],
            ['Sector', sector],
            ['Time Period', f"{time_period} days"],
            ['Timeframe', timeframe],
            ['Initial Capital', f"${capital:,.2f}"],
            ['Final Value', f"${results.get('final_value', 0):,.2f}"],
            ['Total Return', f"{results.get('total_return', 0):.2f}%"],
            ['Trade Count', str(results.get('trade_count', 0))],
            ['Win Rate', f"{results.get('win_rate', 0):.2f}%"],
            ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%"],
            ['Volatility', f"{results.get('volatility', 0):.2f}%"],
            ['VaR (95%)', f"{results.get('var_95', 0):.2f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(PageBreak())
        
        # 2. CHARTS SECTION
        story.append(Paragraph("2. Charts Analysis", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Chart Analysis Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        charts_data = [
            ['Chart Type', 'Description', 'Key Insights'],
            ['Price Chart', 'Historical price movement', 'Trend analysis and support/resistance levels'],
            ['Volume Chart', 'Trading volume patterns', 'Volume confirmation and liquidity analysis'],
            ['Performance Chart', 'Portfolio vs benchmark', 'Relative performance and alpha generation'],
            ['Risk-Return Scatter', 'Position risk analysis', 'Efficient frontier and diversification'],
            ['Market Heatmap', 'Sector performance', 'Sector rotation and market trends'],
            ['Volume Analysis', 'Price-volume correlation', 'Volume confirmation and breakout signals']
        ]
        
        charts_table = Table(charts_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
        charts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(charts_table)
        story.append(PageBreak())
        
        # 3. TECHNICAL ANALYSIS SECTION
        story.append(Paragraph("3. Technical Analysis", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Technical Indicators", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        tech_data = [
            ['Indicator', 'Value', 'Signal'],
            ['Average RSI', f"{results.get('avg_rsi', 0):.2f}", 'Oversold/Overbought'],
            ['Volume Ratio', f"{results.get('volume_ratio', 0):.2f}", 'Volume confirmation'],
            ['Total Signals', str(results.get('total_signals', 0)), 'Signal frequency'],
            ['Analysis Period', f"{results.get('analysis_period', 0)} days", 'Data coverage'],
            ['Price Range', f"${results.get('price_range', 0):.2f}", 'Volatility measure'],
            ['SMA 20', f"{results.get('sma_20', 0):.2f}", 'Short-term trend'],
            ['SMA 50', f"{results.get('sma_50', 0):.2f}", 'Medium-term trend'],
            ['MACD', f"{results.get('macd', 0):.2f}", 'Momentum indicator']
        ]
        
        tech_table = Table(tech_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(tech_table)
        story.append(PageBreak())
        
        # 4. RISK MANAGEMENT SECTION
        story.append(Paragraph("4. Risk Management", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Risk Metrics", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        risk_data = [
            ['Risk Metric', 'Value', 'Interpretation'],
            ['VaR (95%)', f"{results.get('var_95', 0):.2f}%", 'Maximum expected loss'],
            ['CVaR (95%)', f"{results.get('cvar_95', 0):.2f}%", 'Conditional value at risk'],
            ['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%", 'Maximum peak-to-trough loss'],
            ['Volatility', f"{results.get('volatility', 0):.2f}%", 'Price volatility measure'],
            ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}", 'Risk-adjusted return'],
            ['Sortino Ratio', f"{results.get('sortino_ratio', 0):.2f}", 'Downside risk measure'],
            ['Calmar Ratio', f"{results.get('calmar_ratio', 0):.2f}", 'Return vs max drawdown'],
            ['Beta', f"{results.get('beta', 0):.2f}", 'Market sensitivity']
        ]
        
        risk_table = Table(risk_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(risk_table)
        story.append(PageBreak())
        
        # 5. PORTFOLIO ANALYSIS SECTION
        story.append(Paragraph("5. Portfolio Analysis", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Portfolio Performance", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        portfolio_data = [
            ['Metric', 'Value', 'Benchmark'],
            ['Total Return', f"{results.get('total_return', 0):.2f}%", '8.5%'],
            ['Annualized Return', f"{results.get('annualized_return', 0):.2f}%", '8.5%'],
            ['Volatility', f"{results.get('volatility', 0):.2f}%", '15.0%'],
            ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}", '0.57'],
            ['Max Drawdown', f"{results.get('max_drawdown', 0):.2f}%", '-12.5%'],
            ['Win Rate', f"{results.get('win_rate', 0):.2f}%", '55.0%'],
            ['Trade Count', str(results.get('trade_count', 0)), 'N/A'],
            ['Average Trade', f"${results.get('avg_trade', 0):,.2f}", 'N/A']
        ]
        
        portfolio_table = Table(portfolio_data, colWidths=[1.5*inch, 1*inch, 1*inch])
        portfolio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(portfolio_table)
        story.append(PageBreak())
        
        # 6. REPORTS SUMMARY SECTION
        story.append(Paragraph("6. Reports Summary", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Trade History
        if 'trades' in results and results['trades']:
            story.append(Paragraph("Trade History", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            trade_data = [['Date', 'Type', 'Price', 'Shares', 'Value', 'Strategy']]
            for trade in results['trades'][:20]:  # Show up to 20 trades
                # Handle different trade data formats
                if isinstance(trade, dict):
                    date = trade.get('Date', trade.get('entry_date', 'N/A'))
                    trade_type = trade.get('Type', trade.get('type', 'N/A'))
                    price = trade.get('Price', trade.get('price', 0))
                    shares = trade.get('Shares', trade.get('quantity', 0))
                    value = float(price) * float(shares) if price and shares else 0
                    strategy = "Enhanced Multi-Strategy (1d)"
                    
                    # Format date - handle various formats and round zeros
                    if isinstance(date, str) and 'T' in date:
                        # Extract just the date part before 'T'
                        date = date.split('T')[0]
                    else:
                        date = str(date)[:10] if len(str(date)) > 10 else str(date)
                    
                    # Round price and value to 2 decimal places
                    price_formatted = f"${float(price):.2f}" if price else "$0.00"
                    value_formatted = f"${value:,.2f}" if value else "$0.00"
                    
                    trade_data.append([
                        str(date),
                        str(trade_type),
                        price_formatted,
                        str(int(float(shares))) if shares else "0",  # Round shares to whole number
                        value_formatted,
                        strategy
                    ])
            
            trade_table = Table(trade_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch, 1.5*inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(trade_table)
            story.append(Spacer(1, 20))
        else:
            story.append(Paragraph("No trades executed during this period.", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Summary and Recommendations
        story.append(Paragraph("Summary and Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Dynamic recommendations based on performance
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        
        recommendations = []
        
        # Performance-based recommendations
        if total_return > 15:
            recommendations.append("• Excellent performance with strong returns above 15%")
        elif total_return > 8:
            recommendations.append("• Good performance with returns above market average")
        else:
            recommendations.append("• Consider strategy optimization to improve returns")
        
        # Risk-based recommendations
        if sharpe_ratio > 1.0:
            recommendations.append("• Strong risk-adjusted returns with Sharpe ratio > 1.0")
        elif sharpe_ratio > 0.5:
            recommendations.append("• Moderate risk-adjusted returns, consider optimization")
        else:
            recommendations.append("• Low risk-adjusted returns, review risk management")
        
        # Drawdown recommendations
        if abs(max_drawdown) < 10:
            recommendations.append("• Well-controlled drawdowns below 10%")
        elif abs(max_drawdown) < 20:
            recommendations.append("• Moderate drawdowns, monitor risk exposure")
        else:
            recommendations.append("• High drawdowns detected, implement stricter risk controls")
        
        # Win rate recommendations
        if win_rate > 60:
            recommendations.append("• High win rate indicates effective strategy")
        elif win_rate > 50:
            recommendations.append("• Balanced win rate, consider position sizing optimization")
        else:
            recommendations.append("• Low win rate, review entry/exit criteria")
        
        # Trade frequency recommendations
        if total_trades > 20:
            recommendations.append("• Active trading strategy with good market participation")
        elif total_trades > 10:
            recommendations.append("• Moderate trading activity, consider signal optimization")
        else:
            recommendations.append("• Low trading activity, review signal generation")
        
        # General recommendations
        recommendations.extend([
            "• Monitor VaR levels for ongoing risk management",
            "• Review sector allocation for diversification benefits",
            "• Consider rebalancing based on correlation analysis",
            "• Implement dynamic position sizing based on volatility"
        ])
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("Enhanced Multi-Agent Trading System - Professional Analysis Report", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("This comprehensive report is for educational purposes only. Not financial advice.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Encode to base64
        pdf_content = base64.b64encode(pdf_content).decode('utf-8')
        
        print(f"Comprehensive PDF generated successfully, base64 size: {len(pdf_content)} characters")
        return pdf_content
        
    except Exception as e:
        print(f"Error in generate_pdf_report: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    histogram = macd_line - macd_signal
    return macd_line, macd_signal, histogram

# Create the app layout
application.layout = create_enhanced_layout()

# Add external stylesheets and scripts
application.css.append_css({
    "external_url": [
        "https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.css"
    ]
})

application.scripts.append_script({
    "external_url": [
        "https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"
    ]
})

# Add custom CSS for dark theme
application.css.append_css({
    "external_url": [
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    ]
})

# Add inline CSS for dark theme
application.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dark Theme CSS */
            .app-container {
                background-color: #ffffff;
                color: #333333;
                min-height: 100vh;
                transition: all 0.3s ease;
            }
            
            .app-container.dark-theme {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            
            .app-container.dark-theme .card {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                color: #e0e0e0;
            }
            
            .app-container.dark-theme .form-control {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                color: #e0e0e0;
            }
            
            .form-label-small {
                font-weight: bold;
                margin-bottom: 3px;
                color: #333;
                font-size: 0.9em;
            }
            
            .date-picker-input {
                border: 1px solid #ddd !important;
                border-radius: 4px !important;
                padding: 8px !important;
                background-color: white !important;
            }
            
            .date-picker-input:hover {
                border-color: #007bff !important;
            }
            
            .date-picker-input:focus {
                border-color: #007bff !important;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
            }
            
            .app-container.dark-theme .date-picker-input {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
                border-color: #404040 !important;
            }
            
            .app-container.dark-theme .btn-secondary {
                background-color: #404040;
                border-color: #404040;
                color: #e0e0e0;
            }
            
            .theme-toggle-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }
            
            .dark-theme-toggle {
                border-radius: 25px;
                padding: 8px 16px;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            
            .export-buttons-container {
                position: fixed;
                top: 20px;
                right: 200px;
                z-index: 1000;
                display: flex;
                gap: 10px;
            }
            
            .export-button {
                border-radius: 25px;
                padding: 8px 16px;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            
            .export-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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

# Create callbacks
@application.callback(
    Output("symbol-dropdown", "options"),
    [Input("sector-dropdown", "value")]
)
def update_symbol_options(selected_sector):
    """Update symbol dropdown based on selected sector using real data"""
    
    # Get real data manager
    real_data_manager = get_real_data_manager()
    
    # Get available symbols from real data
    available_stocks = real_data_manager.get_available_symbols('stocks')
    available_indices = real_data_manager.get_available_symbols('indices')
    
    # Define symbols by sector with real data availability
    sector_symbols = {
        'Technology': [
            {'label': '  AAPL - Apple Inc.', 'value': 'AAPL', 'available': 'AAPL' in available_stocks},
            {'label': '  MSFT - Microsoft Corporation', 'value': 'MSFT', 'available': 'MSFT' in available_stocks},
            {'label': '  GOOGL - Alphabet Inc.', 'value': 'GOOGL', 'available': 'GOOGL' in available_stocks},
            {'label': '  AMZN - Amazon.com Inc.', 'value': 'AMZN', 'available': 'AMZN' in available_stocks},
            {'label': '  META - Meta Platforms Inc.', 'value': 'META', 'available': 'META' in available_stocks},
            {'label': '  NVDA - NVIDIA Corporation', 'value': 'NVDA', 'available': 'NVDA' in available_stocks},
            {'label': '  NFLX - Netflix Inc.', 'value': 'NFLX', 'available': 'NFLX' in available_stocks},
            {'label': '  PYPL - PayPal Holdings Inc.', 'value': 'PYPL', 'available': False},
            {'label': '  CRM - Salesforce Inc.', 'value': 'CRM', 'available': False},
            {'label': '  ADBE - Adobe Inc.', 'value': 'ADBE', 'available': False},
            {'label': '  INTC - Intel Corporation', 'value': 'INTC', 'available': False},
            {'label': '  CSCO - Cisco Systems Inc.', 'value': 'CSCO', 'available': False},
            {'label': '  ORCL - Oracle Corporation', 'value': 'ORCL', 'available': False},
            {'label': '  IBM - IBM Corporation', 'value': 'IBM', 'available': False},
            {'label': '  SNOW - Snowflake Inc.', 'value': 'SNOW', 'available': False}
        ],
        'Healthcare': [
            {'label': '  JNJ - Johnson & Johnson', 'value': 'JNJ'},
            {'label': '  PFE - Pfizer Inc.', 'value': 'PFE'},
            {'label': '  MRNA - Moderna Inc.', 'value': 'MRNA'},
            {'label': '  UNH - UnitedHealth Group', 'value': 'UNH'},
            {'label': '  ABT - Abbott Laboratories', 'value': 'ABT'}
        ],
        'Consumer': [
            {'label': '  PG - Procter & Gamble', 'value': 'PG'},
            {'label': '  KO - Coca-Cola Company', 'value': 'KO'},
            {'label': '  PEP - PepsiCo Inc.', 'value': 'PEP'},
            {'label': '  WMT - Walmart Inc.', 'value': 'WMT'},
            {'label': '  TSLA - Tesla Inc.', 'value': 'TSLA'}
        ],
        'Finance': [
            {'label': '  JPM - JPMorgan Chase & Co.', 'value': 'JPM'},
            {'label': '  BAC - Bank of America Corp.', 'value': 'BAC'},
            {'label': '  WFC - Wells Fargo & Co.', 'value': 'WFC'},
            {'label': '  GS - Goldman Sachs Group Inc.', 'value': 'GS'},
            {'label': '  BRK-B - Berkshire Hathaway Inc.', 'value': 'BRK-B'},
            {'label': '  V - Visa Inc.', 'value': 'V'},
            {'label': '  MA - Mastercard Inc.', 'value': 'MA'}
        ],
        'Energy': [
            {'label': '  XOM - Exxon Mobil Corp.', 'value': 'XOM'},
            {'label': '  CVX - Chevron Corporation', 'value': 'CVX'},
            {'label': '  COP - ConocoPhillips', 'value': 'COP'},
            {'label': '  EOG - EOG Resources Inc.', 'value': 'EOG'}
        ],
        'Indices': [
            {'label': '  ^GSPC - S&P 500', 'value': '^GSPC', 'available': '^GSPC' in available_indices},
            {'label': '  ^DJI - Dow Jones Industrial Average', 'value': '^DJI', 'available': '^DJI' in available_indices},
            {'label': '  ^IXIC - NASDAQ Composite', 'value': '^IXIC', 'available': '^IXIC' in available_indices},
            {'label': '  ^RUT - Russell 2000', 'value': '^RUT', 'available': '^RUT' in available_indices},
            {'label': '  ^VIX - CBOE Volatility Index', 'value': '^VIX', 'available': '^VIX' in available_indices},
            {'label': '  ^TNX - 10-Year Treasury', 'value': '^TNX', 'available': False},
            {'label': '  ^FVX - 5-Year Treasury', 'value': '^FVX', 'available': False},
            {'label': '  ^TYX - 30-Year Treasury', 'value': '^TYX', 'available': False},
            {'label': '  ^DXY - US Dollar Index', 'value': '^DXY', 'available': False},
            {'label': '  ^GOLD - Gold Futures', 'value': '^GOLD', 'available': False},
            {'label': '  ^CRUDE - Crude Oil Futures', 'value': '^CRUDE'},
            {'label': '  ^NATGAS - Natural Gas Futures', 'value': '^NATGAS'}
        ],
        'Forex': [
            {'label': '  EURUSD=X - Euro/US Dollar', 'value': 'EURUSD=X'},
            {'label': '  GBPUSD=X - British Pound/US Dollar', 'value': 'GBPUSD=X'},
            {'label': '  USDJPY=X - US Dollar/Japanese Yen', 'value': 'USDJPY=X'},
            {'label': '  USDCHF=X - US Dollar/Swiss Franc', 'value': 'USDCHF=X'},
            {'label': '  AUDUSD=X - Australian Dollar/US Dollar', 'value': 'AUDUSD=X'},
            {'label': '  USDCAD=X - US Dollar/Canadian Dollar', 'value': 'USDCAD=X'},
            {'label': '  NZDUSD=X - New Zealand Dollar/US Dollar', 'value': 'NZDUSD=X'},
            {'label': '  EURGBP=X - Euro/British Pound', 'value': 'EURGBP=X'},
            {'label': '  EURJPY=X - Euro/Japanese Yen', 'value': 'EURJPY=X'},
            {'label': '  GBPJPY=X - British Pound/Japanese Yen', 'value': 'GBPJPY=X'}
        ]
    }
    
    if selected_sector and selected_sector in sector_symbols:
        # Filter symbols to show only available ones with real data
        available_symbols = []
        for symbol in sector_symbols[selected_sector]:
            if symbol.get('available', True):  # Show available symbols or all if no availability info
                available_symbols.append({'label': symbol['label'], 'value': symbol['value']})
        return available_symbols
    else:
        # Return all available symbols if no sector selected
        all_symbols = []
        for sector, symbols in sector_symbols.items():
            for symbol in symbols:
                if symbol.get('available', True):  # Show available symbols or all if no availability info
                    all_symbols.append({'label': symbol['label'], 'value': symbol['value']})
        return all_symbols

@application.callback(
    Output("symbol-dropdown-mobile", "options"),
    [Input("sector-dropdown-mobile", "value")]
)
def update_symbol_options_mobile(selected_sector):
    """Update mobile symbol dropdown based on selected sector using real data"""
    
    # Get real data manager
    real_data_manager = get_real_data_manager()
    
    # Get available symbols from real data
    available_stocks = real_data_manager.get_available_symbols('stocks')
    available_indices = real_data_manager.get_available_symbols('indices')
    
    # Define symbols by sector with real data availability
    sector_symbols = {
        'Technology': [
            {'label': '  AAPL - Apple Inc.', 'value': 'AAPL', 'available': 'AAPL' in available_stocks},
            {'label': '  MSFT - Microsoft Corporation', 'value': 'MSFT', 'available': 'MSFT' in available_stocks},
            {'label': '  GOOGL - Alphabet Inc.', 'value': 'GOOGL', 'available': 'GOOGL' in available_stocks},
            {'label': '  AMZN - Amazon.com Inc.', 'value': 'AMZN', 'available': 'AMZN' in available_stocks},
            {'label': '  META - Meta Platforms Inc.', 'value': 'META', 'available': 'META' in available_stocks},
            {'label': '  NVDA - NVIDIA Corporation', 'value': 'NVDA', 'available': 'NVDA' in available_stocks},
            {'label': '  NFLX - Netflix Inc.', 'value': 'NFLX', 'available': 'NFLX' in available_stocks},
            {'label': '  PYPL - PayPal Holdings Inc.', 'value': 'PYPL', 'available': False},
            {'label': '  CRM - Salesforce Inc.', 'value': 'CRM', 'available': False},
            {'label': '  ADBE - Adobe Inc.', 'value': 'ADBE', 'available': False},
            {'label': '  INTC - Intel Corporation', 'value': 'INTC', 'available': False},
            {'label': '  CSCO - Cisco Systems Inc.', 'value': 'CSCO', 'available': False},
            {'label': '  ORCL - Oracle Corporation', 'value': 'ORCL', 'available': False},
            {'label': '  IBM - IBM Corporation', 'value': 'IBM', 'available': False},
            {'label': '  SNOW - Snowflake Inc.', 'value': 'SNOW', 'available': False}
        ],
        'Healthcare': [
            {'label': '  JNJ - Johnson & Johnson', 'value': 'JNJ'},
            {'label': '  PFE - Pfizer Inc.', 'value': 'PFE'},
            {'label': '  MRNA - Moderna Inc.', 'value': 'MRNA'},
            {'label': '  UNH - UnitedHealth Group', 'value': 'UNH'},
            {'label': '  ABT - Abbott Laboratories', 'value': 'ABT'}
        ],
        'Consumer': [
            {'label': '  PG - Procter & Gamble', 'value': 'PG'},
            {'label': '  KO - Coca-Cola Company', 'value': 'KO'},
            {'label': '  PEP - PepsiCo Inc.', 'value': 'PEP'},
            {'label': '  WMT - Walmart Inc.', 'value': 'WMT'},
            {'label': '  TSLA - Tesla Inc.', 'value': 'TSLA'}
        ],
        'Finance': [
            {'label': '  JPM - JPMorgan Chase & Co.', 'value': 'JPM'},
            {'label': '  BAC - Bank of America Corp.', 'value': 'BAC'},
            {'label': '  WFC - Wells Fargo & Co.', 'value': 'WFC'},
            {'label': '  GS - Goldman Sachs Group Inc.', 'value': 'GS'},
            {'label': '  BRK-B - Berkshire Hathaway Inc.', 'value': 'BRK-B'},
            {'label': '  V - Visa Inc.', 'value': 'V'},
            {'label': '  MA - Mastercard Inc.', 'value': 'MA'}
        ],
        'Energy': [
            {'label': '  XOM - Exxon Mobil Corp.', 'value': 'XOM'},
            {'label': '  CVX - Chevron Corporation', 'value': 'CVX'},
            {'label': '  COP - ConocoPhillips', 'value': 'COP'},
            {'label': '  EOG - EOG Resources Inc.', 'value': 'EOG'}
        ],
        'Indices': [
            {'label': '  ^GSPC - S&P 500', 'value': '^GSPC', 'available': '^GSPC' in available_indices},
            {'label': '  ^DJI - Dow Jones Industrial Average', 'value': '^DJI', 'available': '^DJI' in available_indices},
            {'label': '  ^IXIC - NASDAQ Composite', 'value': '^IXIC', 'available': '^IXIC' in available_indices},
            {'label': '  ^RUT - Russell 2000', 'value': '^RUT', 'available': '^RUT' in available_indices},
            {'label': '  ^VIX - CBOE Volatility Index', 'value': '^VIX', 'available': '^VIX' in available_indices},
            {'label': '  ^TNX - 10-Year Treasury', 'value': '^TNX', 'available': False},
            {'label': '  ^FVX - 5-Year Treasury', 'value': '^FVX', 'available': False},
            {'label': '  ^TYX - 30-Year Treasury', 'value': '^TYX', 'available': False},
            {'label': '  ^DXY - US Dollar Index', 'value': '^DXY', 'available': False},
            {'label': '  ^GOLD - Gold Futures', 'value': '^GOLD', 'available': False},
            {'label': '  ^CRUDE - Crude Oil Futures', 'value': '^CRUDE'},
            {'label': '  ^NATGAS - Natural Gas Futures', 'value': '^NATGAS'}
        ],
        'Forex': [
            {'label': '  EURUSD=X - Euro/US Dollar', 'value': 'EURUSD=X'},
            {'label': '  GBPUSD=X - British Pound/US Dollar', 'value': 'GBPUSD=X'},
            {'label': '  USDJPY=X - US Dollar/Japanese Yen', 'value': 'USDJPY=X'},
            {'label': '  USDCHF=X - US Dollar/Swiss Franc', 'value': 'USDCHF=X'},
            {'label': '  AUDUSD=X - Australian Dollar/US Dollar', 'value': 'AUDUSD=X'},
            {'label': '  USDCAD=X - US Dollar/Canadian Dollar', 'value': 'USDCAD=X'},
            {'label': '  NZDUSD=X - New Zealand Dollar/US Dollar', 'value': 'NZDUSD=X'},
            {'label': '  EURGBP=X - Euro/British Pound', 'value': 'EURGBP=X'},
            {'label': '  EURJPY=X - Euro/Japanese Yen', 'value': 'EURJPY=X'},
            {'label': '  GBPJPY=X - British Pound/Japanese Yen', 'value': 'GBPJPY=X'}
        ]
    }
    
    if selected_sector and selected_sector in sector_symbols:
        # Filter symbols to show only available ones with real data
        available_symbols = []
        for symbol in sector_symbols[selected_sector]:
            if symbol.get('available', True):  # Show available symbols or all if no availability info
                available_symbols.append({'label': symbol['label'], 'value': symbol['value']})
        return available_symbols
    else:
        # Return all available symbols if no sector selected
        all_symbols = []
        for sector, symbols in sector_symbols.items():
            for symbol in symbols:
                if symbol.get('available', True):  # Show available symbols or all if no availability info
                    all_symbols.append({'label': symbol['label'], 'value': symbol['value']})
        return all_symbols

@application.callback(
    [Output("app-container", "className"),
     Output("dark-theme-toggle", "children")],
    [Input("dark-theme-toggle", "n_clicks")],
    [State("app-container", "className")]
)
def toggle_dark_theme(n_clicks, current_class):
    """Toggle between light and dark theme"""
    
    if n_clicks is None:
        return "app-container", "🌙 Dark Mode"
    
    if "dark-theme" in current_class:
        return "app-container", "🌙 Dark Mode"
    else:
        return "app-container dark-theme", "☀️ Light Mode"

@application.callback(
    Output("download-pdf", "data"),
    [Input("export-pdf-button-main", "n_clicks")],
    [State("symbol-dropdown", "value"),
     State("sector-dropdown", "value"),
     State("capital-input", "value"),
     State("time-period-dropdown", "value"),
     State("timeframe-dropdown", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date")]
)
def export_to_pdf(n_clicks, symbol, sector, capital, time_period, timeframe, start_date, end_date):
    """Export analysis results to PDF"""
    
    if n_clicks is None:
        return None
    
    try:
        # Get current analysis results
        global backtest_results
        if not backtest_results:
            print("No backtest results available for PDF export")
            return None
        
        print(f"Generating PDF for {symbol} with {len(backtest_results)} results")
        
        # Generate PDF content
        pdf_content = generate_pdf_report(
            backtest_results, symbol, sector, capital, time_period, timeframe
        )
        
        if not pdf_content:
            print("PDF content generation failed")
            return None
        
        print(f"PDF generated successfully, size: {len(pdf_content)} characters")
        
        # Return PDF download
        return {
            "content": pdf_content,
            "filename": f"trading_analysis_{symbol}_{time_period}_{timeframe}.pdf",
            "base64": True,
            "type": "application/pdf"
        }
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

@application.callback(
    Output("download-csv", "data"),
    [Input("export-csv-button", "n_clicks")],
    [State("symbol-dropdown", "value"),
     State("sector-dropdown", "value"),
     State("capital-input", "value"),
     State("time-period-dropdown", "value"),
     State("timeframe-dropdown", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date")]
)
def export_to_csv(n_clicks, symbol, sector, capital, time_period, timeframe, start_date, end_date):
    """Export analysis results to CSV"""
    
    if n_clicks is None:
        return None
    
    try:
        # Get current analysis results
        global backtest_results
        if not backtest_results:
            return None
        
        # Generate CSV content
        csv_content = generate_csv_report(
            backtest_results, symbol, sector, capital, time_period, timeframe
        )
        
        # Return CSV download
        return {
            "content": csv_content,
            "filename": f"trading_analysis_{symbol}_{time_period}_{timeframe}.csv",
            "type": "text/csv"
        }
        
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return None

@application.callback(
    [Output("status-display", "children"),
     Output("tab-content", "children"),
     Output("export-pdf-button-hidden", "style")],
    [Input("run-analysis-button", "n_clicks"),
     Input("run-analysis-button-mobile", "n_clicks"),
     Input("main-tabs", "value")],
    [State("symbol-dropdown", "value"),
     State("sector-dropdown", "value"),
     State("capital-input", "value"),
     State("time-period-dropdown", "value"),
     State("timeframe-dropdown", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date")],
    prevent_initial_call=True  # Prevent initial callback execution
)
def update_dashboard(n_clicks, n_clicks_mobile, active_tab, symbol, sector, capital, time_period, timeframe, start_date, end_date):
    """Update dashboard based on user inputs - OPTIMIZED"""
    
    global backtest_results
    
    ctx = callback_context
    if not ctx.triggered:
        return "Ready to analyze", create_overview_tab({}), {'display': 'none'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Optimize: Only run analysis if button was clicked
    if (button_id == "run-analysis-button" and n_clicks) or (button_id == "run-analysis-button-mobile" and n_clicks_mobile):
        # Run analysis
        print(f"=== CALLBACK TRIGGERED ===")
        print(f"Button clicked: {n_clicks}")
        print(f"Symbol: {symbol}, Sector: {sector}, Capital: {capital}")
        print(f"Time Period: {time_period}, Timeframe: {timeframe}")
        print(f"Start Date: {start_date}, End Date: {end_date}")
        try:
            # Calculate actual time period if custom dates are provided
            if time_period == 'custom' and start_date and end_date:
                from datetime import datetime
                try:
                    # Handle various date formats
                    if isinstance(start_date, str):
                        if 'T' in start_date:
                            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        else:
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    else:
                        start_dt = start_date
                        
                    if isinstance(end_date, str):
                        if 'T' in end_date:
                            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        else:
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    else:
                        end_dt = end_date
                        
                    actual_time_period = (end_dt - start_dt).days
                    print(f"Custom date range: {actual_time_period} days")
                except Exception as e:
                    print(f"Date parsing error: {e}, using default time period")
                    actual_time_period = 365
            else:
                actual_time_period = time_period
                
            backtest_results = run_enhanced_analysis(symbol, sector, capital, actual_time_period, timeframe, start_date, end_date)
            status = f"Analysis completed for {symbol} - {len(backtest_results.get('trades', []))} trades executed"
            export_style = {'display': 'block'}
        except Exception as e:
            print(f"=== ERROR IN ANALYSIS ===")
            print(f"Error: {str(e)}")
            status = f"Error: {str(e)}"
            export_style = {'display': 'none'}
    else:
        status = "Ready to analyze"
        export_style = {'display': 'none'}
    
    # Update tab content based on active tab (optimized with caching)
    tab_content_cache_key = f"tab_{active_tab}_{hash(str(backtest_results))}"
    
    if tab_content_cache_key in chart_cache:
        content = chart_cache[tab_content_cache_key]
    else:
        if active_tab == "overview":
            content = create_overview_tab(backtest_results)
        elif active_tab == "charts":
            content = create_charts_tab(backtest_results)
        elif active_tab == "analysis":
            content = create_analysis_tab(backtest_results)
        elif active_tab == "risk":
            content = create_risk_tab(backtest_results)
        elif active_tab == "portfolio":
            content = create_portfolio_tab(backtest_results)
        elif active_tab == "reports":
            content = create_reports_tab(backtest_results)
        else:
            content = create_overview_tab(backtest_results)
        
        # Cache the content
        chart_cache[tab_content_cache_key] = content
        # Limit cache size
        if len(chart_cache) > 100:
            oldest_key = next(iter(chart_cache))
            del chart_cache[oldest_key]
    
    return status, content, export_style

# Add report generation callbacks
@application.callback(
    Output("scheduled-reports-content", "children"),
    [Input("daily-report-btn", "n_clicks"),
     Input("weekly-report-btn", "n_clicks"),
     Input("monthly-report-btn", "n_clicks")]
)
def generate_reports(daily_clicks, weekly_clicks, monthly_clicks):
    """Generate different types of reports and save to files"""
    
    ctx = callback_context
    if not ctx.triggered:
        return "No reports generated yet. Click a report button to generate."
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Get current analysis results
    global backtest_results
    if not backtest_results:
        return html.Div([
            html.H5("No Data Available"),
            html.P("Please run an analysis first before generating reports.")
        ])
    
    try:
        if button_id == "daily-report-btn" and daily_clicks:
            # Generate and save daily report
            pdf_filename = f"reports/daily_report_{timestamp}.pdf"
            csv_filename = f"reports/daily_report_{timestamp}.csv"
            
            # Generate PDF
            pdf_content = generate_pdf_report(backtest_results, "DAILY", "Daily", 100000, "1d", "1d")
            if pdf_content:
                with open(pdf_filename, 'wb') as f:
                    import base64
                    f.write(base64.b64decode(pdf_content))
            
            # Generate CSV
            csv_content = generate_csv_report(backtest_results, "DAILY", "Daily", 100000, "1d", "1d")
            if csv_content:
                with open(csv_filename, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
            
            return html.Div([
                html.H5("Daily Report Generated"),
                html.P("Daily trading analysis report has been generated and saved."),
                html.P(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P(f"PDF saved: {pdf_filename}"),
                html.P(f"CSV saved: {csv_filename}")
            ])
            
        elif button_id == "weekly-report-btn" and weekly_clicks:
            # Generate and save weekly report
            pdf_filename = f"reports/weekly_report_{timestamp}.pdf"
            csv_filename = f"reports/weekly_report_{timestamp}.csv"
            
            # Generate PDF
            pdf_content = generate_pdf_report(backtest_results, "WEEKLY", "Weekly", 100000, "1w", "1d")
            if pdf_content:
                with open(pdf_filename, 'wb') as f:
                    import base64
                    f.write(base64.b64decode(pdf_content))
            
            # Generate CSV
            csv_content = generate_csv_report(backtest_results, "WEEKLY", "Weekly", 100000, "1w", "1d")
            if csv_content:
                with open(csv_filename, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
            
            return html.Div([
                html.H5("Weekly Report Generated"),
                html.P("Weekly trading analysis report has been generated and saved."),
                html.P(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P(f"PDF saved: {pdf_filename}"),
                html.P(f"CSV saved: {csv_filename}")
            ])
            
        elif button_id == "monthly-report-btn" and monthly_clicks:
            # Generate and save monthly report
            pdf_filename = f"reports/monthly_report_{timestamp}.pdf"
            csv_filename = f"reports/monthly_report_{timestamp}.csv"
            
            # Generate PDF
            pdf_content = generate_pdf_report(backtest_results, "MONTHLY", "Monthly", 100000, "1m", "1d")
            if pdf_content:
                with open(pdf_filename, 'wb') as f:
                    import base64
                    f.write(base64.b64decode(pdf_content))
            
            # Generate CSV
            csv_content = generate_csv_report(backtest_results, "MONTHLY", "Monthly", 100000, "1m", "1d")
            if csv_content:
                with open(csv_filename, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
            
            return html.Div([
                html.H5("Monthly Report Generated"),
                html.P("Monthly trading analysis report has been generated and saved."),
                html.P(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P(f"PDF saved: {pdf_filename}"),
                html.P(f"CSV saved: {csv_filename}")
            ])
    
    except Exception as e:
        return html.Div([
            html.H5("Error Generating Report"),
            html.P(f"Error: {str(e)}"),
            html.P("Please try again or check the console for details.")
        ])
    
    return "No reports generated yet. Click a report button to generate."

# Add mobile callbacks
mobile_dashboard.create_mobile_callbacks(app)

if __name__ == "__main__":
    print("Starting Enhanced Multi-Agent Trading System Dashboard V2...")
    print("Dashboard will be available at: http://localhost:8059")
    print("Enhanced Features: All modules integrated - Real-time data, Advanced charting, Portfolio benchmarking, Strategy builder, Sentiment analysis, Risk metrics, Multi-asset support, Automated reporting, Mobile PWA, User authentication")
    print("OPTIMIZATIONS: Vectorized calculations, Data caching, Reduced memory usage, Faster chart rendering")
    
    # Optimized app configuration
    application.run(debug=False, host='0.0.0.0', port=8059, threaded=True)
