#!/usr/bin/env python3
"""
Simple Backtesting Script for Multi-Agent Trading System
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class SimpleBacktester:
    def __init__(self, api_key="SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP"):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    async def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data from Polygon.io"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data:
                        df_data = []
                        for result in data['results']:
                            df_data.append({
                                'Date': pd.to_datetime(result['t'], unit='ms'),
                                'Open': result['o'],
                                'High': result['h'],
                                'Low': result['l'],
                                'Close': result['c'],
                                'Volume': result['v']
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        return df
        return None
    
    def calculate_signals(self, df):
        """Calculate trading signals"""
        # Simple moving average crossover
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1  # Buy
        df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1  # Sell
        
        return df
    
    def backtest(self, df, initial_capital=100000):
        """Run backtest"""
        cash = initial_capital
        shares = 0
        portfolio_values = [initial_capital]
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            if signal == 1 and shares == 0:  # Buy
                shares = int(cash * 0.1 / current_price)  # 10% position
                cash -= shares * current_price
            
            elif signal == -1 and shares > 0:  # Sell
                cash += shares * current_price
                shares = 0
            
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
        
        final_value = cash + shares * df['Close'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'portfolio_values': portfolio_values
        }

async def main():
    """Run backtest example"""
    backtester = SimpleBacktester()
    
    print("Running backtest for AAPL...")
    
    # Get data
    df = await backtester.get_historical_data("AAPL", "2023-01-01", "2024-01-01")
    
    if df is not None:
        # Calculate signals
        df = backtester.calculate_signals(df)
        
        # Run backtest
        result = backtester.backtest(df)
        
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Data Points: {len(df)}")
        
        # Show some trades
        trades = df[df['Signal'] != 0]
        print(f"\nNumber of signals: {len(trades)}")
        print("Sample signals:")
        print(trades[['Close', 'Signal']].head(10))
    
    else:
        print("Failed to get data")

if __name__ == "__main__":
    asyncio.run(main())
