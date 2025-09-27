#!/usr/bin/env python3
"""
Multi-Agent Trading System Backtesting Dashboard
Interactive dashboard for analyzing backtesting results
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
import dash_bootstrap_components as dbc

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi-Agent Trading System - Backtesting Dashboard"

# Global variables for storing backtest results
backtest_results = {}
available_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]

# Dashboard Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Multi-Agent Trading System", className="text-center mb-4"),
            html.H3("Backtesting Dashboard", className="text-center mb-4 text-muted")
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Backtest Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Symbol:"),
                            dcc.Dropdown(
                                id='symbol-dropdown',
                                options=[{'label': s, 'value': s} for s in available_symbols],
                                value='AAPL',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Start Date:"),
                            dcc.DatePickerSingle(
                                id='start-date',
                                date=datetime(2023, 1, 1),
                                display_format='YYYY-MM-DD'
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("End Date:"),
                            dcc.DatePickerSingle(
                                id='end-date',
                                date=datetime(2024, 1, 1),
                                display_format='YYYY-MM-DD'
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Initial Capital:"),
                            dcc.Input(
                                id='initial-capital',
                                type='number',
                                value=100000,
                                min=1000,
                                max=1000000,
                                step=1000
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "🔄 Run Backtest",
                                id='run-backtest-btn',
                                color='primary',
                                size='lg',
                                className="w-100"
                            )
                        ], width=12)
                    ])
                ])
            ])
        ], width=4),
        
        # Performance Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Performance Summary"),
                dbc.CardBody([
                    html.Div(id='performance-summary')
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    # Charts Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Portfolio Performance"),
                dbc.CardBody([
                    dcc.Graph(id='portfolio-chart')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📉 Drawdown Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='drawdown-chart')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Trading Signals"),
                dbc.CardBody([
                    dcc.Graph(id='signals-chart')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Risk Metrics"),
                dbc.CardBody([
                    dcc.Graph(id='risk-metrics-chart')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Trade History Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Trade History"),
                dbc.CardBody([
                    html.Div(id='trade-history-table')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Comparison Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚖️ Multi-Symbol Comparison"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Symbols to Compare:"),
                            dcc.Dropdown(
                                id='comparison-symbols',
                                options=[{'label': s, 'value': s} for s in available_symbols],
                                value=['AAPL', 'GOOGL', 'MSFT'],
                                multi=True
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                "🔄 Compare",
                                id='compare-btn',
                                color='success',
                                className="w-100 mt-4"
                            )
                        ], width=4)
                    ]),
                    dcc.Graph(id='comparison-chart')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('performance-summary', 'children'),
     Output('portfolio-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('signals-chart', 'figure'),
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
        empty_table = html.Div("Click 'Run Backtest' to see results")
        return "No backtest run yet", empty_fig, empty_fig, empty_fig, empty_fig, empty_table
    
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
            return "Error: No data available", go.Figure(), go.Figure(), go.Figure(), go.Figure(), "Error"
        
        # Store result
        backtest_results[symbol] = result
        
        # Create performance summary
        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['total_return']:.2%}", className="text-success"),
                        html.P("Total Return", className="mb-0")
                    ])
                ], color="light")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['sharpe_ratio']:.2f}", className="text-info"),
                        html.P("Sharpe Ratio", className="mb-0")
                    ])
                ], color="light")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['max_drawdown']:.2%}", className="text-danger"),
                        html.P("Max Drawdown", className="mb-0")
                    ])
                ], color="light")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['total_trades']}", className="text-primary"),
                        html.P("Total Trades", className="mb-0")
                    ])
                ], color="light")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${result['final_value']:,.0f}", className="text-success"),
                        html.P("Final Value", className="mb-0")
                    ])
                ], color="light")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['annualized_return']:.2%}", className="text-warning"),
                        html.P("Annualized Return", className="mb-0")
                    ])
                ], color="light")
            ], width=2)
        ])
        
        # Create portfolio chart
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            y=result['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        portfolio_fig.update_layout(
            title=f"Portfolio Performance - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white"
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
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        drawdown_fig.update_layout(
            title=f"Drawdown Analysis - {symbol}",
            xaxis_title="Trading Days",
            yaxis_title="Drawdown (%)",
            template="plotly_white"
        )
        
        # Create signals chart (placeholder - would need price data)
        signals_fig = go.Figure()
        signals_fig.add_trace(go.Scatter(
            y=[1, 2, 3, 4, 5],
            mode='lines+markers',
            name='Price',
            line=dict(color='blue')
        ))
        signals_fig.update_layout(
            title=f"Trading Signals - {symbol}",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template="plotly_white"
        )
        
        # Create risk metrics chart
        risk_metrics = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Value': [result['total_return'], result['sharpe_ratio'], 
                     abs(result['max_drawdown']), 0.6]  # Placeholder win rate
        }
        
        risk_fig = px.bar(
            x=risk_metrics['Metric'],
            y=risk_metrics['Value'],
            title=f"Risk Metrics - {symbol}",
            template="plotly_white"
        )
        
        # Create trade history table
        if result['trade_history']:
            trade_df = pd.DataFrame(result['trade_history'])
            trade_table = dbc.Table.from_dataframe(
                trade_df.head(10),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm"
            )
        else:
            trade_table = html.P("No trades executed")
        
        return summary, portfolio_fig, drawdown_fig, signals_fig, risk_fig, trade_table
        
    except Exception as e:
        error_msg = f"Error running backtest: {str(e)}"
        return error_msg, go.Figure(), go.Figure(), go.Figure(), go.Figure(), error_msg

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
                    'Total Return': result['total_return'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown': abs(result['max_drawdown']),
                    'Total Trades': result['total_trades']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Total Return',
                x=df['Symbol'],
                y=df['Total Return'],
                yaxis='y',
                offsetgroup=1
            ))
            fig.add_trace(go.Bar(
                name='Sharpe Ratio',
                x=df['Symbol'],
                y=df['Sharpe Ratio'],
                yaxis='y2',
                offsetgroup=2
            ))
            
            fig.update_layout(
                title="Multi-Symbol Performance Comparison",
                xaxis_title="Symbol",
                yaxis=dict(title="Total Return (%)", side="left"),
                yaxis2=dict(title="Sharpe Ratio", side="right", overlaying="y"),
                template="plotly_white"
            )
            
            return fig
        else:
            return go.Figure()
            
    except Exception as e:
        return go.Figure()

# Run the dashboard
if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Trading System Backtesting Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🔄 Use the controls to run backtests and analyze results")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
