#!/usr/bin/env python3
"""
Multi-Agent Trading System Backtesting Guide
Complete guide for backtesting your trading strategies
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

class MultiAgentBacktester:
    """Advanced backtesting for Multi-Agent Trading System"""
    
    def __init__(self, polygon_api_key="SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP"):
        self.api_key = polygon_api_key
        self.base_url = "https://api.polygon.io"
    
    async def get_historical_data(self, symbol, start_date, end_date, timespan="day"):
        """Get historical data from Polygon.io"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data and len(data['results']) > 0:
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
                        df.sort_index(inplace=True)
                        return df
        return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def generate_trading_signals(self, df):
        """Generate trading signals using multiple strategies"""
        df['Signal'] = 0
        df['Strategy'] = ''
        
        # Strategy 1: Moving Average Crossover
        ma_bullish = df['EMA_12'] > df['EMA_26']
        ma_bearish = df['EMA_12'] < df['EMA_26']
        
        # Strategy 2: RSI Mean Reversion
        rsi_oversold = df['RSI'] < 30
        rsi_overbought = df['RSI'] > 70
        
        # Strategy 3: Bollinger Bands
        bb_oversold = df['Close'] < df['BB_Lower']
        bb_overbought = df['Close'] > df['BB_Upper']
        
        # Strategy 4: MACD
        macd_bullish = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        macd_bearish = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        # Combine signals
        df.loc[ma_bullish, 'Signal'] = 1
        df.loc[ma_bullish, 'Strategy'] = 'MA_CROSSOVER'
        
        df.loc[rsi_oversold, 'Signal'] = 1
        df.loc[rsi_oversold, 'Strategy'] = 'RSI_MEAN_REVERSION'
        
        df.loc[bb_oversold, 'Signal'] = 1
        df.loc[bb_oversold, 'Strategy'] = 'BOLLINGER_BANDS'
        
        df.loc[macd_bullish, 'Signal'] = 1
        df.loc[macd_bullish, 'Strategy'] = 'MACD'
        
        # Sell signals
        df.loc[ma_bearish, 'Signal'] = -1
        df.loc[rsi_overbought, 'Signal'] = -1
        df.loc[bb_overbought, 'Signal'] = -1
        df.loc[macd_bearish, 'Signal'] = -1
        
        return df
    
    def simulate_trading(self, df, initial_capital=100000, position_size=0.1):
        """Simulate trading with risk management"""
        cash = initial_capital
        shares = 0
        portfolio_values = [initial_capital]
        trade_history = []
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            strategy = df['Strategy'].iloc[i]
            
            # Calculate current portfolio value
            current_portfolio_value = cash + (shares * current_price)
            
            # Trading logic
            if signal == 1 and shares == 0:  # Buy signal
                trade_amount = initial_capital * position_size
                if cash >= trade_amount:
                    shares_to_buy = int(trade_amount / current_price)
                    cost = shares_to_buy * current_price
                    cash -= cost
                    shares += shares_to_buy
                    
                    trade_history.append({
                        'date': df.index[i],
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'strategy': strategy,
                        'portfolio_value': current_portfolio_value
                    })
            
            elif signal == -1 and shares > 0:  # Sell signal
                proceeds = shares * current_price
                cash += proceeds
                
                trade_history.append({
                    'date': df.index[i],
                    'action': 'SELL',
                    'shares': shares,
                    'price': current_price,
                    'strategy': strategy,
                    'portfolio_value': current_portfolio_value
                })
                
                shares = 0
            
            # Update portfolio value
            current_portfolio_value = cash + (shares * current_price)
            portfolio_values.append(current_portfolio_value)
        
        # Calculate final metrics
        final_value = cash + (shares * df['Close'].iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate additional metrics
        portfolio_series = pd.Series(portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / len(df)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'trade_history': trade_history,
            'total_trades': len([t for t in trade_history if t['action'] == 'SELL'])
        }
    
    def plot_results(self, df, result, symbol):
        """Plot backtesting results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtesting Results for {symbol}', fontsize=16)
        
        # Portfolio value
        axes[0, 0].plot(result['portfolio_values'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Price and signals
        axes[0, 1].plot(df.index, df['Close'], label='Price', alpha=0.7)
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        axes[0, 1].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy')
        axes[0, 1].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell')
        axes[0, 1].set_title('Price and Trading Signals')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Technical indicators
        axes[1, 0].plot(df.index, df['Close'], label='Price', alpha=0.7)
        axes[1, 0].plot(df.index, df['EMA_12'], label='EMA 12', alpha=0.7)
        axes[1, 0].plot(df.index, df['EMA_26'], label='EMA 26', alpha=0.7)
        axes[1, 0].set_title('Moving Averages')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Performance metrics
        metrics_text = f"""
        Total Return: {result['total_return']:.2%}
        Annualized Return: {result['annualized_return']:.2%}
        Sharpe Ratio: {result['sharpe_ratio']:.2f}
        Max Drawdown: {result['max_drawdown']:.2%}
        Total Trades: {result['total_trades']}
        Final Value: ${result['final_value']:,.2f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    async def run_backtest(self, symbol, start_date, end_date, initial_capital=100000):
        """Run complete backtest"""
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        # Get historical data
        df = await self.get_historical_data(symbol, start_date, end_date)
        if df is None:
            print(f"Failed to get data for {symbol}")
            return None
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate trading signals
        df = self.generate_trading_signals(df)
        
        # Simulate trading
        result = self.simulate_trading(df, initial_capital)
        
        # Print results
        print(f"\nBacktest Results for {symbol}:")
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Annualized Return: {result['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        
        # Plot results
        self.plot_results(df, result, symbol)
        
        return result

async def main():
    """Run backtesting examples"""
    backtester = MultiAgentBacktester()
    
    # Example 1: Single symbol backtest
    print("=" * 60)
    print("MULTI-AGENT TRADING SYSTEM BACKTESTING")
    print("=" * 60)
    
    # Test with AAPL
    result = await backtester.run_backtest(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_capital=100000
    )
    
    if result:
        print(f"\n✅ Backtest completed successfully!")
        print(f"📊 Total Return: {result['total_return']:.2%}")
        print(f"📈 Final Portfolio Value: ${result['final_value']:,.2f}")
        print(f"🎯 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {result['max_drawdown']:.2%}")
    
    print("\n" + "=" * 60)
    print("BACKTESTING COMPLETE!")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Analyze the results and charts")
    print("2. Adjust strategy parameters")
    print("3. Test with different symbols")
    print("4. Run paper trading with live data")
    print("5. Deploy to production when satisfied")

if __name__ == "__main__":
    asyncio.run(main())
