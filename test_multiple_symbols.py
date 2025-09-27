#!/usr/bin/env python3
"""
Test Multiple Symbols Backtest
"""
import asyncio
from multi_agent_backtest import MultiAgentBacktester

async def test_multiple_symbols():
    backtester = MultiAgentBacktester()
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    print("Testing Multiple Symbols:")
    print("=" * 50)
    
    for symbol in symbols:
        try:
            result = await backtester.run_backtest(
                symbol=symbol,
                start_date="2023-01-01",
                end_date="2024-01-01",
                initial_capital=100000
            )
            
            if result:
                print(f"{symbol}: {result['total_return']:.2%} return, {result['sharpe_ratio']:.2f} Sharpe")
        except Exception as e:
            print(f"{symbol}: Error - {e}")

if __name__ == "__main__":
    asyncio.run(test_multiple_symbols())
