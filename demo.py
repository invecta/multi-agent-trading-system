"""
Example Usage Script for Goldbach Trading Strategy Analyzer
Demonstrates basic functionality and workflow
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample market data for demonstration"""
    logger.info("Creating sample market data...")
    
    # Generate realistic price data
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 252)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 252))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 252))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    logger.info(f"Created sample data with {len(sample_data)} periods")
    return sample_data

def demonstrate_goldbach_calculator():
    """Demonstrate Goldbach calculator functionality"""
    logger.info("Demonstrating Goldbach Calculator...")
    
    from goldbach_calculator import GoldbachCalculator
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize calculator
    calculator = GoldbachCalculator()
    
    # Calculate dealing range
    dealing_range = calculator.calculate_dealing_range(data)
    
    print("\n" + "="*50)
    print("GOLDBACH CALCULATOR DEMONSTRATION")
    print("="*50)
    
    print(f"\nDealing Range Analysis:")
    print(f"Range High: ${dealing_range['range_high']:.2f}")
    print(f"Range Low: ${dealing_range['range_low']:.2f}")
    print(f"Range Size: ${dealing_range['range_size']:.2f}")
    print(f"Current Price: ${dealing_range['current_price']:.2f}")
    
    print(f"\nGoldbach Levels:")
    for level, price in dealing_range['goldbach_levels'].items():
        if 'gb_level' in level:
            print(f"  {level}: ${price:.2f}")
    
    print(f"\nPO3 Analysis:")
    po3 = dealing_range['po3_levels']
    print(f"  PO3 High: ${po3['po3_high']:.2f}")
    print(f"  PO3 Low: ${po3['po3_low']:.2f}")
    print(f"  PO3 Range: ${po3['po3_range']:.2f}")
    
    # Analyze market structure
    analysis = calculator.analyze_market_structure(data)
    
    print(f"\nMarket Statistics:")
    stats = analysis['statistics']
    print(f"  Total Periods: {stats['total_periods']}")
    print(f"  Volatility: {stats['volatility']:.4f}")
    print(f"  Trend Direction: {stats['trend_direction']}")
    
    return analysis

def demonstrate_backtesting():
    """Demonstrate backtesting engine functionality"""
    logger.info("Demonstrating Backtesting Engine...")
    
    from backtesting_engine import BacktestingEngine
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize backtester
    backtester = BacktestingEngine(initial_capital=100000)
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    print("\n" + "="*50)
    print("BACKTESTING ENGINE DEMONSTRATION")
    print("="*50)
    
    metrics = results['performance_metrics']
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Show trade details
    trades = results['trades']
    completed_trades = [t for t in trades if 'exit_time' in t]
    
    if completed_trades:
        print(f"\nRecent Trades:")
        for i, trade in enumerate(completed_trades[-5:]):  # Show last 5 trades
            print(f"  Trade {i+1}: Entry ${trade['entry_price']:.2f} -> Exit ${trade['exit_price']:.2f}")
            print(f"    P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2%}) - {trade['exit_reason']}")
    
    return results

def demonstrate_optimization():
    """Demonstrate parameter optimization"""
    logger.info("Demonstrating Parameter Optimization...")
    
    from backtesting_engine import BacktestingEngine
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize backtester
    backtester = BacktestingEngine(initial_capital=100000)
    
    # Define parameter ranges
    param_ranges = {
        'po3_multipliers': [0.5, 1.0, 1.5],
        'lookback_periods': [6, 9, 12],
        'max_position_sizes': [0.01, 0.02]
    }
    
    # Run optimization
    optimization_results = backtester.optimize_strategy(data, param_ranges)
    
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")
    print("="*50)
    
    print(f"\nOptimization Results:")
    print(f"  Best Parameters: {optimization_results['best_params']}")
    print(f"  Best Performance: {optimization_results['best_performance']:.4f}")
    
    print(f"\nTop 5 Parameter Combinations:")
    sorted_results = sorted(optimization_results['optimization_results'], 
                           key=lambda x: x['sharpe_ratio'], reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"  {i+1}. Sharpe: {result['sharpe_ratio']:.3f}, "
              f"Return: {result['total_return']:.2%}, "
              f"Drawdown: {result['max_drawdown']:.2%}")
        print(f"     Params: PO3={result['po3_multiplier']}, "
              f"Lookback={result['lookback_period']}, "
              f"Position={result['max_position_size']}")
    
    return optimization_results

def demonstrate_data_collection():
    """Demonstrate data collection functionality"""
    logger.info("Demonstrating Data Collection...")
    
    from data_collector import FinancialDataCollector
    
    # Initialize collector
    collector = FinancialDataCollector("demo_data.db")
    
    print("\n" + "="*50)
    print("DATA COLLECTION DEMONSTRATION")
    print("="*50)
    
    # Collect sample data for EUR/USD
    print("\nCollecting EUR/USD data...")
    eurusd_data = collector.collect_market_data(
        symbols=['EURUSD=X'],
        timeframe='1d',
        period='1y',
        asset_class='forex'
    )
    
    if not eurusd_data.empty:
        print(f"  Collected {len(eurusd_data)} records")
        print(f"  Date range: {eurusd_data.index[0]} to {eurusd_data.index[-1]}")
        print(f"  Price range: ${eurusd_data['low'].min():.2f} - ${eurusd_data['high'].max():.2f}")
    
    # Get data summary
    summary = collector.get_data_summary()
    print(f"\nDatabase Summary:")
    print(f"  Total Records: {summary['total_records']}")
    print(f"  Asset Classes: {summary['asset_class_counts']}")
    print(f"  Timeframes: {summary['timeframe_counts']}")
    
    return collector

def main():
    """Run all demonstrations"""
    print("Goldbach Trading Strategy Analyzer - Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate each component
        print("\n1. Data Collection Demo")
        collector = demonstrate_data_collection()
        
        print("\n2. Goldbach Calculator Demo")
        analysis = demonstrate_goldbach_calculator()
        
        print("\n3. Backtesting Engine Demo")
        backtest_results = demonstrate_backtesting()
        
        print("\n4. Parameter Optimization Demo")
        optimization_results = demonstrate_optimization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nNext Steps:")
        print("1. Run 'python main.py --mode dashboard' to start the interactive dashboard")
        print("2. Run 'python main.py --mode collect' to collect real market data")
        print("3. Run 'python main.py --mode analyze --symbol EURUSD=X' for analysis")
        print("4. Check the README.md for detailed usage instructions")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check that all dependencies are installed correctly.")

if __name__ == "__main__":
    main()
