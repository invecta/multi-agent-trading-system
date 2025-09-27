#!/usr/bin/env python3
"""
Simple Backtesting Dashboard
Interactive dashboard for Multi-Agent Trading System backtesting
"""
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime, timedelta
from multi_agent_backtest import MultiAgentBacktester

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Multi-Agent Trading System - Backtesting Dashboard"

# Available symbols
available_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]

# Dashboard Layout
app.layout = html.Div([
    html.Div([
        html.H1("🚀 Multi-Agent Trading System", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.H3("Backtesting Dashboard", style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'marginBottom': '30px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.H4("📊 Backtest Controls", style={'color': '#2c3e50'}),
            
            html.Div([
                html.Label("Symbol:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[{'label': s, 'value': s} for s in available_symbols],
                    value='AAPL',
                    style={'marginBottom': '20px'}
                )
            ]),
            
            html.Div([
                html.Label("Start Date:", style={'fontWeight': 'bold'}),
                dcc.DatePickerSingle(
                    id='start-date',
                    date=datetime(2023, 1, 1),
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': '20px'}
                )
            ]),
            
            html.Div([
                html.Label("End Date:", style={'fontWeight': 'bold'}),
                dcc.DatePickerSingle(
                    id='end-date',
                    date=datetime(2024, 1, 1),
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': '20px'}
                )
            ]),
            
            html.Div([
                html.Label("Initial Capital:", style={'fontWeight': 'bold'}),
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
                "🔄 Run Backtest",
                id='run-backtest-btn',
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'fontSize': '16px'
                }
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        # Performance Summary
        html.Div([
            html.H4("📈 Performance Summary", style={'color': '#2c3e50'}),
            html.Div(id='performance-summary')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    # Charts Section
    html.Div([
        html.Div([
            html.H4("📊 Portfolio Performance", style={'color': '#2c3e50'}),
            dcc.Graph(id='portfolio-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H4("📉 Drawdown Analysis", style={'color': '#2c3e50'}),
            dcc.Graph(id='drawdown-chart')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.H4("🎯 Risk Metrics", style={'color': '#2c3e50'}),
            dcc.Graph(id='risk-metrics-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H4("📋 Trade History", style={'color': '#2c3e50'}),
            html.Div(id='trade-history-table')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px'}),
    
    # Comparison Section
    html.Div([
        html.H4("⚖️ Multi-Symbol Comparison", style={'color': '#2c3e50'}),
        html.Div([
            html.Label("Select Symbols to Compare:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='comparison-symbols',
                options=[{'label': s, 'value': s} for s in available_symbols],
                value=['AAPL', 'GOOGL', 'MSFT'],
                multi=True,
                style={'marginBottom': '20px'}
            ),
            html.Button(
                "🔄 Compare",
                id='compare-btn',
                style={
                    'backgroundColor': '#27ae60',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginBottom': '20px'
                }
            )
        ]),
        dcc.Graph(id='comparison-chart')
    ])
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

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
     Input('start-date', 'date'),
     Input('end-date', 'date'),
     Input('initial-capital', 'value')]
)
def run_backtest(n_clicks, symbol, start_date, end_date, initial_capital):
    """Run backtest and update dashboard"""
    if n_clicks is None:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_table = html.Div("Click 'Run Backtest' to see results", style={'textAlign': 'center', 'color': '#7f8c8d'})
        return "No backtest run yet", empty_fig, empty_fig, empty_fig, empty_table
    
    try:
        # Run backtest
        backtester = MultiAgentBacktester()
        
        # Convert dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        # Run async backtest
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            backtester.run_backtest(symbol, start_dt, end_dt, initial_capital)
        )
        loop.close()
        
        if result is None:
            error_msg = "Error: No data available"
            return error_msg, go.Figure(), go.Figure(), go.Figure(), error_msg
        
        # Store result
        backtest_results[symbol] = result
        
        # Create performance summary
        summary = html.Div([
            html.Div([
                html.Div([
                    html.H2(f"{result['total_return']:.2%}", style={'color': '#27ae60', 'margin': '0'}),
                    html.P("Total Return", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['sharpe_ratio']:.2f}", style={'color': '#3498db', 'margin': '0'}),
                    html.P("Sharpe Ratio", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['max_drawdown']:.2%}", style={'color': '#e74c3c', 'margin': '0'}),
                    html.P("Max Drawdown", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['total_trades']}", style={'color': '#9b59b6', 'margin': '0'}),
                    html.P("Total Trades", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"${result['final_value']:,.0f}", style={'color': '#27ae60', 'margin': '0'}),
                    html.P("Final Value", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.H2(f"{result['annualized_return']:.2%}", style={'color': '#f39c12', 'margin': '0'}),
                    html.P("Annualized Return", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'width': '16%', 'display': 'inline-block'})
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
            height=400
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
            height=400
        )
        
        # Create risk metrics chart
        risk_metrics = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Annualized Return'],
            'Value': [result['total_return']*100, result['sharpe_ratio'], 
                     abs(result['max_drawdown'])*100, result['annualized_return']*100]
        }
        
        risk_fig = px.bar(
            x=risk_metrics['Metric'],
            y=risk_metrics['Value'],
            title=f"Risk Metrics - {symbol}",
            template="plotly_white",
            color=risk_metrics['Value'],
            color_continuous_scale='RdYlGn'
        )
        risk_fig.update_layout(height=400)
        
        # Create trade history table
        if result['trade_history']:
            trade_df = pd.DataFrame(result['trade_history'])
            trade_table = html.Table([
                html.Thead([
                    html.Tr([html.Th(col) for col in trade_df.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(trade_df.iloc[i][col]) for col in trade_df.columns
                    ]) for i in range(min(len(trade_df), 10))  # Show first 10 trades
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'})
        else:
            trade_table = html.P("No trades executed", style={'textAlign': 'center', 'color': '#7f8c8d'})
        
        return summary, portfolio_fig, drawdown_fig, risk_fig, trade_table
        
    except Exception as e:
        error_msg = f"Error running backtest: {str(e)}"
        return error_msg, go.Figure(), go.Figure(), go.Figure(), error_msg

@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('compare-btn', 'n_clicks')],
    [Input('comparison-symbols', 'value')]
)
def update_comparison(n_clicks, symbols):
    """Update comparison chart"""
    if n_clicks is None or not symbols:
        return go.Figure()
    
    try:
        comparison_data = []
        
        for symbol in symbols:
            if symbol in backtest_results:
                result = backtest_results[symbol]
                comparison_data.append({
                    'Symbol': symbol,
                    'Total Return': result['total_return'] * 100,
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown': abs(result['max_drawdown']) * 100,
                    'Total Trades': result['total_trades']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Total Return (%)',
                x=df['Symbol'],
                y=df['Total Return'],
                yaxis='y',
                marker_color='#3498db'
            ))
            fig.add_trace(go.Bar(
                name='Sharpe Ratio',
                x=df['Symbol'],
                y=df['Sharpe Ratio'],
                yaxis='y2',
                marker_color='#27ae60'
            ))
            
            fig.update_layout(
                title="Multi-Symbol Performance Comparison",
                xaxis_title="Symbol",
                yaxis=dict(title="Total Return (%)", side="left"),
                yaxis2=dict(title="Sharpe Ratio", side="right", overlaying="y"),
                template="plotly_white",
                height=400
            )
            
            return fig
        else:
            return go.Figure()
            
    except Exception as e:
        return go.Figure()

# Run the dashboard
if __name__ == "__main__":
    print("Starting Multi-Agent Trading System Backtesting Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    print("Use the controls to run backtests and analyze results")
    print("Make sure to run backtests first before comparing symbols")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
