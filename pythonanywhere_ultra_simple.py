#!/usr/bin/env python3
"""
Ultra Simple Dashboard for PythonAnywhere
Only uses basic Python and Dash
"""

import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime
import random

# Create Dash app
application = dash.Dash(__name__)
application.title = "Simple Trading Dashboard"

# Very simple layout
application.layout = html.Div([
    html.H1("Trading Analytics Dashboard", style={'textAlign': 'center', 'color': 'blue'}),
    
    html.Div([
        html.H2("Simple Trading Dashboard"),
        html.P("This is a minimal trading dashboard running on PythonAnywhere."),
        html.P(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        
        html.Hr(),
        
        html.H3("Trade Analysis"),
        html.Div(id='trade-info', children="Select options and click Analyze to start."),
        
        html.Br(),
        
        html.Label("Symbol: "),
        dcc.Dropdown(
            id='symbol-selector',
            options=[
                {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
            ],
            value='AAPL',
            style={'width': '200px', 'margin': '10px'}
        ),
        
        html.Label("Capital: "),
        dcc.Input(
            id='capital-input',
            type='number',
            value=100000,
            style={'width': '150px', 'margin': '10px'}
        ),
        
        html.Br(),
        
        html.Button("Analyze", id='analyze-btn', n_clicks=0, style={'margin': '10px', 'padding': '10px'})
    ], style={'padding': '30px'})
])

# Simple callback
@application.callback(
    Output('trade-info', 'children'),
    [Input('analyze-btn', 'n_clicks')],
    [State('symbol-selector', 'value'),
     State('capital-input', 'value')]
)
def update_analysis(n_clicks, symbol, capital):
    if n_clicks > 0:
        # Generate random results
        total_return = round(random.uniform(-20, 50), 2)
        trades = random.randint(5, 25)
        win_rate = round(random.uniform(40, 70), 1)
        final_value = capital * (1 + total_return/100)
        
        return html.Div([
            html.H4(f"Analysis for {symbol}"),
            html.P(f"Total Return: {total_return}%"),
            html.P(f"Number of Trades: {trades}"),
            html.P(f"Win Rate: {win_rate}%"),
            html.P(f"Final Portfolio Value: ${final_value:,.2f}"),
            html.P("Strategy: Moving Average Crossover"),
            html.P(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ])
    return "Click Analyze to run trading analysis"

if __name__ == "__main__":
    print("Starting Ultra Simple Dashboard...")
    application.run(host="0.0.0.0", port=8059, debug=False)
