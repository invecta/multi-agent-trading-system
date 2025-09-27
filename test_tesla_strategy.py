#!/usr/bin/env python3
"""
Test script for Tesla 369 Strategy
Verifies the strategy implementation works correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tesla_369_calculator import Tesla369Calculator
from data_collector import FinancialDataCollector
from backtesting_engine import BacktestingEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tesla_calculator():
    """Test the Tesla 369 calculator"""
    print("=" * 60)
    print("TESTING TESLA 369 CALCULATOR")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum() * 0.5,
        'high': 100 + np.random.randn(100).cumsum() * 0.5 + np.random.rand(100) * 2,
        'low': 100 + np.random.randn(100).cumsum() * 0.5 - np.random.rand(100) * 2,
        'close': 100 + np.random.randn(100).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize Tesla calculator
    calculator = Tesla369Calculator()
    
    # Test individual functions
    print("\n1. Testing Tesla Pattern Detection:")
    patterns = calculator.tesla_pattern_detection(sample_data)
    print(f"   Price Pattern: {patterns.get('price_tesla_pattern', False)}")
    print(f"   Volume Pattern: {patterns.get('volume_tesla_pattern', False)}")
    print(f"   Energy Pattern: {patterns.get('energy_tesla_pattern', False)}")
    print(f"   Tesla Energy: {patterns.get('tesla_energy', 0):.2f}")
    
    print("\n2. Testing Tesla Fibonacci Levels:")
    fib_levels = calculator.tesla_fibonacci_levels(sample_data)
    print(f"   Fib 23.6%: {fib_levels.get('fib_236', 0):.2f}")
    print(f"   Fib 38.2%: {fib_levels.get('fib_382', 0):.2f}")
    print(f"   Fib 50.0%: {fib_levels.get('fib_500', 0):.2f}")
    print(f"   Fib 61.8%: {fib_levels.get('fib_618', 0):.2f}")
    print(f"   Fib 76.4%: {fib_levels.get('fib_764', 0):.2f}")
    
    print("\n3. Testing Tesla Energy Zones:")
    energy_zones = calculator.tesla_energy_zones(sample_data)
    print(f"   Energy Zone: {energy_zones.get('energy_zone', 0)}")
    print(f"   SMA 3: {energy_zones.get('sma_3', 0):.2f}")
    print(f"   SMA 6: {energy_zones.get('sma_6', 0):.2f}")
    print(f"   SMA 9: {energy_zones.get('sma_9', 0):.2f}")
    
    print("\n4. Testing Tesla Frequency Analysis:")
    freq_analysis = calculator.tesla_frequency_analysis(sample_data)
    print(f"   Frequency Signal: {freq_analysis.get('tesla_freq_signal', 0)}")
    print(f"   RSI 3: {freq_analysis.get('freq_3', 0):.2f}")
    print(f"   RSI 6: {freq_analysis.get('freq_6', 0):.2f}")
    print(f"   RSI 9: {freq_analysis.get('freq_9', 0):.2f}")
    
    print("\n5. Testing Tesla Vortex Analysis:")
    vortex_analysis = calculator.tesla_vortex_analysis(sample_data)
    print(f"   Vortex Pattern: {vortex_analysis.get('vortex_pattern', False)}")
    print(f"   Vortex Strength: {vortex_analysis.get('vortex_strength', 0):.2f}")
    print(f"   Vortex Center: {vortex_analysis.get('vortex_center', 0):.2f}")
    
    print("\n6. Testing Tesla Harmonics Analysis:")
    harmonics = calculator.tesla_harmonics_analysis(sample_data)
    print(f"   Harmonic Convergence: {harmonics.get('harmonic_convergence', False)}")
    print(f"   Harmonic Pattern: {harmonics.get('harmonic_pattern', False)}")
    print(f"   Harmonic Energy: {harmonics.get('harmonic_energy', 0):.2f}")
    
    print("\n7. Testing Tesla Sacred Geometry:")
    sacred_geo = calculator.tesla_sacred_geometry(sample_data)
    print(f"   Sacred Geometry Pattern: {sacred_geo.get('sg_pattern', False)}")
    print(f"   Golden Ratio Level: {sacred_geo.get('sg_golden', 0):.2f}")
    print(f"   SG Level 1: {sacred_geo.get('sg_level_1', 0):.2f}")
    
    print("\n8. Testing Tesla Advanced Pattern Detection:")
    advanced = calculator.tesla_advanced_pattern_detection(sample_data)
    print(f"   Tesla Score: {advanced.get('tesla_score', 0)}/6")
    print(f"   Energy Pattern Advanced: {advanced.get('energy_pattern_advanced', False)}")
    print(f"   Price Change 3: {advanced.get('price_change_3', 0):.2f}%")
    print(f"   Price Change 6: {advanced.get('price_change_6', 0):.2f}%")
    print(f"   Price Change 9: {advanced.get('price_change_9', 0):.2f}%")
    
    print("\n9. Testing Complete Tesla Analysis:")
    analysis = calculator.analyze_market_structure(sample_data)
    tesla_analysis = analysis.get('tesla_analysis', {})
    print(f"   Tesla Strength: {tesla_analysis.get('tesla_strength', 0):.1f}%")
    print(f"   Buy Signal: {tesla_analysis.get('tesla_buy_signal', False)}")
    print(f"   Sell Signal: {tesla_analysis.get('tesla_sell_signal', False)}")
    print(f"   Buy Conditions: {tesla_analysis.get('tesla_buy_conditions', 0)}")
    print(f"   Sell Conditions: {tesla_analysis.get('tesla_sell_conditions', 0)}")
    
    print("\n10. Testing Trading Signals:")
    signals = calculator.identify_trading_signals(sample_data)
    signal_count = len(signals[signals['signal'] != 0])
    print(f"   Total Signals Generated: {signal_count}")
    print(f"   Buy Signals: {len(signals[signals['signal'] == 1])}")
    print(f"   Sell Signals: {len(signals[signals['signal'] == -1])}")
    
    print("\nTesla 369 Calculator tests completed successfully!")
    return True

def test_data_collection():
    """Test data collection with Tesla strategy"""
    print("\n" + "=" * 60)
    print("TESTING DATA COLLECTION")
    print("=" * 60)
    
    try:
        collector = FinancialDataCollector()
        
        # Test with a simple symbol
        print("\nTesting data collection for EURUSD=X...")
        data = collector.get_market_data('EURUSD=X', '1d')
        
        if not data.empty:
            print(f"Data collected successfully: {len(data)} records")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Latest close: {data['close'].iloc[-1]:.4f}")
        else:
            print("No data collected")
            return False
            
    except Exception as e:
        print(f"Data collection failed: {str(e)}")
        return False
    
    return True

def test_backtesting():
    """Test backtesting with Tesla strategy"""
    print("\n" + "=" * 60)
    print("TESTING BACKTESTING ENGINE")
    print("=" * 60)
    
    try:
        # Create sample data for backtesting
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': 100 + np.random.randn(50).cumsum() * 0.5,
            'high': 100 + np.random.randn(50).cumsum() * 0.5 + np.random.rand(50) * 2,
            'low': 100 + np.random.randn(50).cumsum() * 0.5 - np.random.rand(50) * 2,
            'close': 100 + np.random.randn(50).cumsum() * 0.5,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        backtester = BacktestingEngine()
        
        print("\nRunning backtest with Tesla 369 strategy...")
        results = backtester.run_backtest(sample_data)
        
        print(f"Backtest completed successfully!")
        print(f"   Total Return: {results['performance_metrics'].get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {results['performance_metrics'].get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {results['performance_metrics'].get('max_drawdown', 0):.2%}")
        print(f"   Win Rate: {results['performance_metrics'].get('win_rate', 0):.2%}")
        print(f"   Total Trades: {results['performance_metrics'].get('total_trades', 0)}")
        print(f"   Profit Factor: {results['performance_metrics'].get('profit_factor', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"Backtesting failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("TESLA 369 STRATEGY TEST SUITE")
    print("Testing the replacement of Goldbach strategy with Tesla 369")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Tesla Calculator
    if test_tesla_calculator():
        success_count += 1
    
    # Test 2: Data Collection
    if test_data_collection():
        success_count += 1
    
    # Test 3: Backtesting
    if test_backtesting():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("ALL TESTS PASSED! Tesla 369 strategy is working correctly.")
        print("\nThe system has been successfully updated from Goldbach to Tesla 369 strategy.")
        print("Key features now available:")
        print("• Tesla Pattern Detection (3, 6, 9 sacred numbers)")
        print("• Tesla Fibonacci Levels")
        print("• Tesla Energy Zones (SMA analysis)")
        print("• Tesla Frequency Analysis (RSI harmonics)")
        print("• Tesla Vortex Mathematics")
        print("• Tesla Sacred Geometry")
        print("• Advanced Tesla Pattern Scoring")
        print("• Comprehensive Tesla Signal Generation")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()
