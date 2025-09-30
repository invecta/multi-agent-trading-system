"""
Goldbach Trading Strategy Calculator
Implements the mathematical concepts from the Goldbach Fundamentals PDF
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)

class GoldbachCalculator:
    """Calculates Goldbach levels and Power of Three dealing ranges"""
    
    def __init__(self):
        self.config = Config()
        self.po3_base = self.config.GOLDBACH_PARAMETERS['power_of_three_base']
        self.goldbach_levels = self.config.GOLDBACH_PARAMETERS['goldbach_levels']
        self.lookback_period = self.config.GOLDBACH_PARAMETERS['lookback_period']
    
    def calculate_power_of_three(self, price: float, multiplier: float = 1.0) -> Dict[str, float]:
        """
        Calculate Power of Three (PO3) levels
        
        Args:
            price: Current price
            multiplier: Multiplier for PO3 calculation
        
        Returns:
            Dictionary with PO3 levels
        """
        po3_value = self.po3_base * multiplier
        
        # Calculate PO3 levels
        po3_levels = {
            'po3_high': price + po3_value,
            'po3_low': price - po3_value,
            'po3_range': po3_value * 2,
            'po3_mid': price,
            'po3_multiplier': multiplier
        }
        
        return po3_levels
    
    def calculate_goldbach_levels(self, high_price: float, low_price: float) -> Dict[str, float]:
        """
        Calculate Goldbach levels within a dealing range
        
        Args:
            high_price: High price of the range
            low_price: Low price of the range
        
        Returns:
            Dictionary with Goldbach levels
        """
        range_size = high_price - low_price
        goldbach_step = range_size / self.goldbach_levels
        
        goldbach_levels = {}
        
        # Calculate 6 Goldbach levels
        for i in range(1, self.goldbach_levels + 1):
            level_price = low_price + (goldbach_step * i)
            goldbach_levels[f'gb_level_{i}'] = level_price
        
        # Add range information
        goldbach_levels.update({
            'range_high': high_price,
            'range_low': low_price,
            'range_size': range_size,
            'goldbach_step': goldbach_step,
            'range_mid': (high_price + low_price) / 2
        })
        
        return goldbach_levels
    
    def calculate_dealing_range(self, data: pd.DataFrame, lookback_periods: int = None) -> Dict[str, float]:
        """
        Calculate optimal dealing range based on historical data
        
        Args:
            data: Price data DataFrame with OHLC columns
            lookback_periods: Number of periods to look back (default: 9)
        
        Returns:
            Dictionary with dealing range information
        """
        if lookback_periods is None:
            lookback_periods = self.lookback_period
        
        # Use recent data for dealing range calculation
        recent_data = data.tail(lookback_periods)
        
        if len(recent_data) < lookback_periods:
            logger.warning(f"Insufficient data for {lookback_periods} periods")
            return {}
        
        # Calculate range extremes
        range_high = recent_data['high'].max()
        range_low = recent_data['low'].min()
        
        # Calculate PO3 dealing range
        current_price = recent_data['close'].iloc[-1]
        
        # Ensure we have valid numeric values
        if pd.isna(range_high) or pd.isna(range_low) or pd.isna(current_price):
            logger.warning("Invalid price data detected")
            return {}
        po3_levels = self.calculate_power_of_three(current_price)
        
        # Determine if PO3 is optimal
        range_size = range_high - range_low
        po3_range = po3_levels['po3_range']
        
        # Calculate optimal PO3 multiplier
        optimal_multiplier = range_size / (2 * self.po3_base)
        
        optimal_po3 = self.calculate_power_of_three(current_price, optimal_multiplier)
        
        dealing_range = {
            'range_high': range_high,
            'range_low': range_low,
            'range_size': range_size,
            'current_price': current_price,
            'po3_levels': po3_levels,
            'optimal_po3': optimal_po3,
            'optimal_multiplier': optimal_multiplier,
            'lookback_periods': lookback_periods,
            'goldbach_levels': self.calculate_goldbach_levels(range_high, range_low)
        }
        
        return dealing_range
    
    def identify_trading_signals(self, data: pd.DataFrame, dealing_range: Dict) -> pd.DataFrame:
        """
        Identify trading signals based on Goldbach levels
        
        Args:
            data: Price data DataFrame
            dealing_range: Dealing range information
        
        Returns:
            DataFrame with trading signals
        """
        signals = data.copy()
        signals['signal'] = 0
        signals['signal_strength'] = 0.0
        signals['entry_price'] = np.nan
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        range_high = dealing_range['range_high']
        range_low = dealing_range['range_low']
        goldbach_levels = dealing_range['goldbach_levels']
        
        for i in range(len(signals)):
            current_price = signals['close'].iloc[i]
            current_high = signals['high'].iloc[i]
            current_low = signals['low'].iloc[i]
            
            # Check for signals at Goldbach levels
            signal_strength = 0
            
            # Buy signals (price approaching Goldbach levels from below)
            for level_name, level_price in goldbach_levels.items():
                if 'gb_level' in level_name:
                    # Check if price is near Goldbach level
                    if abs(current_low - level_price) / level_price < 0.001:  # 0.1% tolerance
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                        signals.iloc[i, signals.columns.get_loc('entry_price')] = level_price
                        signals.iloc[i, signals.columns.get_loc('stop_loss')] = level_price * 0.98  # 2% stop loss
                        signals.iloc[i, signals.columns.get_loc('take_profit')] = level_price * 1.04  # 2:1 risk-reward
                        signal_strength = 1.0
                        break
            
            # Sell signals (price approaching Goldbach levels from above)
            for level_name, level_price in goldbach_levels.items():
                if 'gb_level' in level_name:
                    if abs(current_high - level_price) / level_price < 0.001:  # 0.1% tolerance
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        signals.iloc[i, signals.columns.get_loc('entry_price')] = level_price
                        signals.iloc[i, signals.columns.get_loc('stop_loss')] = level_price * 1.02  # 2% stop loss
                        signals.iloc[i, signals.columns.get_loc('take_profit')] = level_price * 0.96  # 2:1 risk-reward
                        signal_strength = 1.0
                        break
            
            signals.iloc[i, signals.columns.get_loc('signal_strength')] = signal_strength
        
        return signals
    
    def calculate_lookback_partitions(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Calculate lookback partitions based on the number 9
        
        Args:
            data: Price data DataFrame
        
        Returns:
            Dictionary with partition information
        """
        total_periods = len(data)
        partitions = {}
        
        # Calculate partitions based on multiples of 9
        base_partition = 9
        max_partitions = total_periods // base_partition
        
        for i in range(1, max_partitions + 1):
            partition_size = base_partition * i
            if partition_size <= total_periods:
                partitions[f'partition_{i}x9'] = {
                    'size': partition_size,
                    'start_index': max(0, total_periods - partition_size),
                    'end_index': total_periods - 1
                }
        
        return partitions
    
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze market structure using Goldbach concepts
        
        Args:
            data: Price data DataFrame
        
        Returns:
            Dictionary with market structure analysis
        """
        analysis = {}
        
        # Calculate dealing range
        dealing_range = self.calculate_dealing_range(data)
        analysis['dealing_range'] = dealing_range
        
        # Calculate lookback partitions
        partitions = self.calculate_lookback_partitions(data)
        analysis['partitions'] = partitions
        
        # Identify trading signals
        signals = self.identify_trading_signals(data, dealing_range)
        analysis['signals'] = signals
        
        # Calculate market statistics
        analysis['statistics'] = {
            'total_periods': len(data),
            'price_range': data['high'].max() - data['low'].min(),
            'average_volume': data['volume'].mean() if 'volume' in data.columns else 0,
            'volatility': data['close'].pct_change().std(),
            'trend_direction': 'up' if data['close'].iloc[-1] > data['close'].iloc[0] else 'down'
        }
        
        return analysis
    
    def optimize_parameters(self, data: pd.DataFrame, parameter_ranges: Dict) -> Dict[str, float]:
        """
        Optimize Goldbach parameters for best performance
        
        Args:
            data: Historical price data
            parameter_ranges: Dictionary with parameter ranges to test
        
        Returns:
            Dictionary with optimal parameters
        """
        best_params = {}
        best_performance = -np.inf
        
        # Test different parameter combinations
        for po3_mult in parameter_ranges.get('po3_multipliers', [0.5, 1.0, 1.5, 2.0]):
            for lookback in parameter_ranges.get('lookback_periods', [6, 9, 12, 18]):
                # Temporarily update parameters
                original_lookback = self.lookback_period
                self.lookback_period = lookback
                
                # Calculate dealing range with current parameters
                dealing_range = self.calculate_dealing_range(data, lookback)
                
                if dealing_range:
                    # Calculate performance metric (simplified)
                    range_size = dealing_range['range_size']
                    current_price = dealing_range['current_price']
                    
                    # Performance based on range efficiency
                    performance = range_size / current_price if current_price > 0 else 0
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'po3_multiplier': po3_mult,
                            'lookback_period': lookback,
                            'performance': performance
                        }
                
                # Restore original parameter
                self.lookback_period = original_lookback
        
        return best_params

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum() * 0.5,
        'high': 100 + np.random.randn(100).cumsum() * 0.5 + np.random.rand(100) * 2,
        'low': 100 + np.random.randn(100).cumsum() * 0.5 - np.random.rand(100) * 2,
        'close': 100 + np.random.randn(100).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize calculator
    calculator = GoldbachCalculator()
    
    # Calculate dealing range
    dealing_range = calculator.calculate_dealing_range(sample_data)
    print("Dealing Range Analysis:")
    print(f"Range High: {dealing_range['range_high']:.2f}")
    print(f"Range Low: {dealing_range['range_low']:.2f}")
    print(f"Range Size: {dealing_range['range_size']:.2f}")
    
    # Show Goldbach levels
    print("\nGoldbach Levels:")
    for level, price in dealing_range['goldbach_levels'].items():
        if 'gb_level' in level:
            print(f"{level}: {price:.2f}")
    
    # Analyze market structure
    analysis = calculator.analyze_market_structure(sample_data)
    print(f"\nMarket Statistics:")
    print(f"Total Periods: {analysis['statistics']['total_periods']}")
    print(f"Volatility: {analysis['statistics']['volatility']:.4f}")
    print(f"Trend Direction: {analysis['statistics']['trend_direction']}")
