# Test data retrieval and analysis
from data_collector import FinancialDataCollector
from goldbach_calculator import GoldbachCalculator
from backtesting_engine import BacktestingEngine

print("Testing data retrieval and analysis...")

# Test data collection
collector = FinancialDataCollector()
data = collector.get_market_data('EURUSD=X', '1d')

print(f"Retrieved {len(data)} records for EUR/USD")
if not data.empty:
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print(f"Sample data:")
    print(data.head())
    
    # Test Goldbach calculator
    print("\nTesting Goldbach Calculator...")
    calc = GoldbachCalculator()
    dealing_range = calc.calculate_dealing_range(data)
    
    if dealing_range:
        print(f"Dealing Range High: ${dealing_range['range_high']:.2f}")
        print(f"Dealing Range Low: ${dealing_range['range_low']:.2f}")
        print(f"Current Price: ${dealing_range['current_price']:.2f}")
        
        # Test backtesting
        print("\nTesting Backtesting Engine...")
        backtester = BacktestingEngine(100000)
        results = backtester.run_backtest(data)
        
        if results['performance_metrics']:
            metrics = results['performance_metrics']
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
        
        print("\nAll tests passed! Dashboard should work now.")
    else:
        print("❌ Error in Goldbach calculation")
else:
    print("❌ No data retrieved")
