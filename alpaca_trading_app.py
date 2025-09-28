#!/usr/bin/env python3
"""
Alpaca Trading Dashboard - Cloud Deployment Ready
Simple Flask app with Alpaca integration for cloud deployment
"""

import os
import json
from flask import Flask, render_template_string, request, jsonify
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)

# Alpaca API Configuration
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', 'PKOEKMI4RY0LHF565WDO')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb')
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca clients
try:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    alpaca_connected = True
except Exception as e:
    print(f"Alpaca connection failed: {e}")
    alpaca_connected = False

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Alpaca Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .chart { margin: 20px 0; }
        .info { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Alpaca Trading Dashboard</h1>
            <p>Real-time trading with Alpaca API</p>
        </div>
        
        <div class="status {{ 'connected' if alpaca_connected else 'disconnected' }}">
            <strong>Alpaca Status:</strong> {{ 'Connected' if alpaca_connected else 'Disconnected' }}
        </div>
        
        <form method="POST" action="/">
            <div class="form-group">
                <label for="symbol">Stock Symbol:</label>
                <input type="text" id="symbol" name="symbol" value="{{ symbol }}" placeholder="e.g., AAPL, TSLA, MSFT">
            </div>
            
            <div class="form-group">
                <label for="days">Days of Data:</label>
                <select id="days" name="days">
                    <option value="7" {{ 'selected' if days == '7' else '' }}>7 days</option>
                    <option value="30" {{ 'selected' if days == '30' else '' }}>30 days</option>
                    <option value="90" {{ 'selected' if days == '90' else '' }}>90 days</option>
                </select>
            </div>
            
            <button type="submit">Get Market Data</button>
        </form>
        
        {% if chart_data %}
        <div class="chart">
            <h3>Price Chart for {{ symbol }}</h3>
            <div id="chart"></div>
        </div>
        
        <div class="info">
            <h4>Market Data Summary</h4>
            <p><strong>Symbol:</strong> {{ symbol }}</p>
            <p><strong>Current Price:</strong> ${{ "%.2f"|format(current_price) }}</p>
            <p><strong>Data Points:</strong> {{ data_points }}</p>
            <p><strong>Date Range:</strong> {{ start_date }} to {{ end_date }}</p>
        </div>
        {% endif %}
        
        {% if account_info %}
        <div class="info">
            <h4>Account Information</h4>
            <p><strong>Account Status:</strong> {{ account_info.get('status', 'N/A') }}</p>
            <p><strong>Buying Power:</strong> ${{ "%.2f"|format(account_info.get('buying_power', 0)) }}</p>
            <p><strong>Portfolio Value:</strong> ${{ "%.2f"|format(account_info.get('portfolio_value', 0)) }}</p>
        </div>
        {% endif %}
    </div>
    
    {% if chart_data %}
    <script>
        var chartData = {{ chart_data|safe }};
        Plotly.newPlot('chart', chartData.data, chartData.layout);
    </script>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    symbol = request.args.get('symbol', 'AAPL')
    days = request.args.get('days', '30')
    
    chart_data = None
    current_price = 0
    data_points = 0
    start_date = ""
    end_date = ""
    account_info = None
    
    if alpaca_connected:
        try:
            # Get account information
            account = trading_client.get_account()
            account_info = {
                'status': account.status,
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
            
            # Get market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days))
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = data_client.get_stock_bars(request_params)
            
            if symbol in bars.data:
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in bars.data[symbol]])
                
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    data_points = len(df)
                    
                    # Create candlestick chart
                    fig = go.Figure(data=go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=symbol
                    ))
                    
                    fig.update_layout(
                        title=f'{symbol} Price Chart',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500
                    )
                    
                    chart_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
        except Exception as e:
            print(f"Error fetching data: {e}")
    
    return render_template_string(HTML_TEMPLATE,
                                alpaca_connected=alpaca_connected,
                                symbol=symbol,
                                days=days,
                                chart_data=chart_data,
                                current_price=current_price,
                                data_points=data_points,
                                start_date=start_date.strftime('%Y-%m-%d') if start_date else '',
                                end_date=end_date.strftime('%Y-%m-%d') if end_date else '',
                                account_info=account_info)

@app.route('/api/account')
def api_account():
    if not alpaca_connected:
        return jsonify({'error': 'Alpaca not connected'}), 500
    
    try:
        account = trading_client.get_account()
        return jsonify({
            'status': account.status,
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'equity': float(account.equity)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def api_positions():
    if not alpaca_connected:
        return jsonify({'error': 'Alpaca not connected'}), 500
    
    try:
        positions = trading_client.get_all_positions()
        return jsonify([{
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'market_value': float(pos.market_value),
            'unrealized_pl': float(pos.unrealized_pl),
            'unrealized_plpc': float(pos.unrealized_plpc)
        } for pos in positions])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Alpaca Trading Dashboard on port {port}")
    print(f"Alpaca Connected: {alpaca_connected}")
    app.run(host='0.0.0.0', port=port, debug=False)
