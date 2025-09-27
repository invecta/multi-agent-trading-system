#!/usr/bin/env python3
"""
Test Different Time Periods
"""
import asyncio
from multi_agent_backtest import MultiAgentBacktester

async def test_time_periods():
    backtester = MultiAgentBacktester()
    
    periods = [
        ("2022-01-01", "2023-01-01", "2022"),
        ("2023-01-01", "2024-01-01", "2023"),
        ("2024-01-01", "2025-01-01", "2024")
    ]
    
    print("Testing Different Time Periods for AAPL:")
    print("=" * 50)
    
    for start, end, year in periods:
        try:
            result = await backtester.run_backtest(
                symbol="AAPL",
                start_date=start,
                end_date=end,
                initial_capital=100000
            )
            
            if result:
                print(f"{year}: {result['total_return']:.2%} return, {result['sharpe_ratio']:.2f} Sharpe, {result['total_trades']} trades")
        except Exception as e:
            print(f"{year}: Error - {e}")

if __name__ == "__main__":
    asyncio.run(test_time_periods())
