#!/usr/bin/env python3
"""
Custom Strategy Backtest
"""
import asyncio
import pandas as pd
import numpy as np
from multi_agent_backtest import MultiAgentBacktester

class CustomStrategyBacktester(MultiAgentBacktester):
    def generate_trading_signals(self, df):
        """Custom strategy: RSI + Volume confirmation"""
        df['Signal'] = 0
        df['Strategy'] = ''
        
        # RSI signals
        rsi_oversold = df['RSI'] < 25  # More aggressive oversold
        rsi_overbought = df['RSI'] > 75  # More aggressive overbought
        
        # Volume confirmation
        high_volume = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
        
        # Combined signals
        df.loc[rsi_oversold & high_volume, 'Signal'] = 1
        df.loc[rsi_oversold & high_volume, 'Strategy'] = 'RSI_VOLUME_BUY'
        
        df.loc[rsi_overbought & high_volume, 'Signal'] = -1
        df.loc[rsi_overbought & high_volume, 'Strategy'] = 'RSI_VOLUME_SELL'
        
        return df

async def test_custom_strategy():
    backtester = CustomStrategyBacktester()
    
    print("Testing Custom RSI + Volume Strategy:")
    print("=" * 50)
    
    result = await backtester.run_backtest(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_capital=100000
    )
    
    if result:
        print(f"Custom Strategy Results:")
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")

if __name__ == "__main__":
    asyncio.run(test_custom_strategy())
