"""
Tesla 369 Trading Strategy Calculator
Implements Nikola Tesla's 369 Theory for financial market analysis
Based on the advanced Pine Script indicator
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)

class Tesla369Calculator:
    """Calculates Tesla 369 patterns and trading signals"""
    
    def __init__(self):
        self.config = Config()
        
        # Tesla's Sacred Numbers
        self.TESLA_THREE = 3
        self.TESLA_SIX = 6
        self.TESLA_NINE = 9
        self.TESLA_TWELVE = 12  # 1+2 = 3
        self.TESLA_EIGHTEEN = 18  # 1+8 = 9
        self.TESLA_TWENTY_SEVEN = 27  # 2+7 = 9
        self.TESLA_THIRTY_SIX = 36  # 3+6 = 9
        
        # Tesla Vortex Mathematics Constants
        self.VORTEX_CENTER = 9
        self.VORTEX_RADIUS_1 = 3
        self.VORTEX_RADIUS_2 = 6
        self.VORTEX_RADIUS_3 = 9
        
        # Signal sensitivity
        self.sensitivity_threshold = 0.6  # Medium sensitivity
    
    def is_tesla_number(self, value: float) -> bool:
        """Check if a number is divisible by Tesla's sacred numbers"""
        abs_value = abs(value)
        return (abs_value % self.TESLA_THREE == 0 or 
                abs_value % self.TESLA_SIX == 0 or 
                abs_value % self.TESLA_NINE == 0)
    
    def tesla_pattern_detection(self, data: pd.DataFrame) -> Dict:
        """Detect basic Tesla patterns in price, volume, and energy"""
        if len(data) < self.TESLA_NINE:
            return {}
        
        # Price patterns based on 369 theory (with safe division)
        def safe_percentage_change(current, previous):
            if pd.isna(current) or pd.isna(previous) or previous == 0:
                return 0
            return (current - previous) / previous * 100
        
        price_change = safe_percentage_change(data['close'].iloc[-1], data['close'].iloc[-self.TESLA_NINE-1])
        volume_change = safe_percentage_change(data['volume'].iloc[-1], data['volume'].iloc[-self.TESLA_SIX-1])
        
        # Check for Tesla number patterns (handle NaN values)
        try:
            price_tesla_pattern = self.is_tesla_number(round(price_change)) if not pd.isna(price_change) else False
            volume_tesla_pattern = self.is_tesla_number(round(volume_change)) if not pd.isna(volume_change) else False
        except (ValueError, OverflowError):
            price_tesla_pattern = False
            volume_tesla_pattern = False
        
        # Tesla Energy Calculation (handle NaN values and zero volume)
        try:
            high_low_diff = data['high'].iloc[-1] - data['low'].iloc[-1]
            volume = data['volume'].iloc[-1]
            
            if pd.isna(high_low_diff) or pd.isna(volume) or volume == 0:
                tesla_energy = 0
                energy_tesla_pattern = False
            else:
                tesla_energy = high_low_diff * volume * self.TESLA_THREE / self.TESLA_SIX
                energy_tesla_pattern = self.is_tesla_number(round(tesla_energy / 1000)) if not pd.isna(tesla_energy) else False
        except (ValueError, OverflowError):
            tesla_energy = 0
            energy_tesla_pattern = False
        
        return {
            'price_tesla_pattern': price_tesla_pattern,
            'volume_tesla_pattern': volume_tesla_pattern,
            'energy_tesla_pattern': energy_tesla_pattern,
            'tesla_energy': tesla_energy,
            'price_change': price_change,
            'volume_change': volume_change
        }
    
    def tesla_fibonacci_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate Tesla-modified Fibonacci levels"""
        if len(data) < self.TESLA_NINE:
            return {}
        
        recent_high = data['high'].tail(self.TESLA_NINE).max()
        recent_low = data['low'].tail(self.TESLA_NINE).min()
        price_range = recent_high - recent_low
        
        # Tesla-modified Fibonacci levels
        fib_levels = {
            'fib_236': recent_low + price_range * 0.236,  # 2+3+6 = 11, reduced: 1+1 = 2
            'fib_382': recent_low + price_range * 0.382,  # 3+8+2 = 13, reduced: 1+3 = 4
            'fib_500': recent_low + price_range * 0.500,  # 5+0+0 = 5
            'fib_618': recent_low + price_range * 0.618,  # 6+1+8 = 15, reduced: 1+5 = 6 (Tesla!)
            'fib_764': recent_low + price_range * 0.764,  # 7+6+4 = 17, reduced: 1+7 = 8
            'recent_high': recent_high,
            'recent_low': recent_low,
            'price_range': price_range
        }
        
        return fib_levels
    
    def tesla_energy_zones(self, data: pd.DataFrame) -> Dict:
        """Calculate Tesla energy zones using SMA analysis"""
        if len(data) < self.TESLA_NINE:
            return {}
        
        sma_3 = data['close'].tail(self.TESLA_THREE).mean()
        sma_6 = data['close'].tail(self.TESLA_SIX).mean()
        sma_9 = data['close'].tail(self.TESLA_NINE).mean()
        
        current_price = data['close'].iloc[-1]
        
        # Energy zone classification
        if current_price > sma_3 and sma_3 > sma_6 and sma_6 > sma_9:
            energy_zone = 1  # Strong positive energy
        elif current_price < sma_3 and sma_3 < sma_6 and sma_6 < sma_9:
            energy_zone = -1  # Strong negative energy
        else:
            energy_zone = 0  # Neutral energy
        
        return {
            'energy_zone': energy_zone,
            'sma_3': sma_3,
            'sma_6': sma_6,
            'sma_9': sma_9,
            'current_price': current_price
        }
    
    def tesla_frequency_analysis(self, data: pd.DataFrame) -> Dict:
        """Tesla frequency analysis using RSI"""
        if len(data) < self.TESLA_NINE:
            return {}
        
        # Calculate RSI for different Tesla periods
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        
        freq_3 = calculate_rsi(data['close'], self.TESLA_THREE)
        freq_6 = calculate_rsi(data['close'], self.TESLA_SIX)
        freq_9 = calculate_rsi(data['close'], self.TESLA_NINE)
        
        # Tesla frequency signal
        tesla_freq_signal = 0
        if freq_3 > 60 and freq_6 > 60 and freq_9 > 60:
            tesla_freq_signal = 1  # High frequency (overbought)
        elif freq_3 < 40 and freq_6 < 40 and freq_9 < 40:
            tesla_freq_signal = -1  # Low frequency (oversold)
        
        return {
            'tesla_freq_signal': tesla_freq_signal,
            'freq_3': freq_3,
            'freq_6': freq_6,
            'freq_9': freq_9
        }
    
    def tesla_vortex_analysis(self, data: pd.DataFrame) -> Dict:
        """Tesla Vortex Mathematics analysis"""
        if len(data) < self.VORTEX_RADIUS_3:
            return {}
        
        # Calculate vortex center and radii
        vortex_center = data['close'].tail(self.VORTEX_CENTER).mean()
        vortex_r1 = data['close'].tail(self.VORTEX_RADIUS_1).mean()
        vortex_r2 = data['close'].tail(self.VORTEX_RADIUS_2).mean()
        vortex_r3 = data['close'].tail(self.VORTEX_RADIUS_3).mean()
        
        # Vortex energy calculation
        vortex_energy = (vortex_r1 - vortex_r2) * (vortex_r2 - vortex_r3) * self.TESLA_THREE
        vortex_pattern = self.is_tesla_number(round(vortex_energy / 100))
        
        # Vortex signal strength
        vortex_strength = abs(vortex_energy) / data['close'].iloc[-1] * 100
        
        return {
            'vortex_pattern': vortex_pattern,
            'vortex_strength': vortex_strength,
            'vortex_center': vortex_center,
            'vortex_r1': vortex_r1,
            'vortex_r2': vortex_r2,
            'vortex_r3': vortex_r3,
            'vortex_energy': vortex_energy
        }
    
    def tesla_harmonics_analysis(self, data: pd.DataFrame) -> Dict:
        """Tesla Frequency Harmonics Analysis"""
        if len(data) < self.TESLA_EIGHTEEN:
            return {}
        
        # Calculate harmonic frequencies using Tesla's sacred numbers
        harmonic_3 = data['close'].tail(self.TESLA_THREE).mean()
        harmonic_6 = data['close'].tail(self.TESLA_SIX).mean()
        harmonic_9 = data['close'].tail(self.TESLA_NINE).mean()
        harmonic_12 = data['close'].tail(self.TESLA_TWELVE).mean()
        harmonic_18 = data['close'].tail(self.TESLA_EIGHTEEN).mean()
        
        current_price = data['close'].iloc[-1]
        
        # Harmonic convergence detection
        harmonic_convergence = (abs(harmonic_3 - harmonic_6) < (current_price * 0.01) and 
                               abs(harmonic_6 - harmonic_9) < (current_price * 0.01) and
                               abs(harmonic_9 - harmonic_12) < (current_price * 0.01))
        
        # Harmonic energy field
        harmonic_energy = (harmonic_3 + harmonic_6 + harmonic_9 + harmonic_12 + harmonic_18) / 5
        harmonic_pattern = self.is_tesla_number(round(harmonic_energy / 10))
        
        return {
            'harmonic_convergence': harmonic_convergence,
            'harmonic_pattern': harmonic_pattern,
            'harmonic_energy': harmonic_energy,
            'harmonic_3': harmonic_3,
            'harmonic_6': harmonic_6,
            'harmonic_9': harmonic_9,
            'harmonic_12': harmonic_12,
            'harmonic_18': harmonic_18
        }
    
    def tesla_sacred_geometry(self, data: pd.DataFrame) -> Dict:
        """Tesla Sacred Geometry Patterns"""
        if len(data) < self.TESLA_TWENTY_SEVEN:
            return {}
        
        # Golden ratio and Tesla's modified ratios
        golden_ratio = 1.618
        tesla_ratio_1 = 3.0  # Tesla's first sacred ratio
        tesla_ratio_2 = 6.0  # Tesla's second sacred ratio
        tesla_ratio_3 = 9.0  # Tesla's third sacred ratio
        
        # Calculate sacred geometry levels
        recent_high = data['high'].tail(self.TESLA_TWENTY_SEVEN).max()
        recent_low = data['low'].tail(self.TESLA_TWENTY_SEVEN).min()
        price_range = recent_high - recent_low
        
        # Sacred geometry levels
        sg_level_1 = recent_low + price_range * (1 / tesla_ratio_1)  # 33.33%
        sg_level_2 = recent_low + price_range * (1 / tesla_ratio_2)  # 16.67%
        sg_level_3 = recent_low + price_range * (1 / tesla_ratio_3)  # 11.11%
        sg_golden = recent_low + price_range * (1 / golden_ratio)    # 61.8%
        
        current_price = data['close'].iloc[-1]
        
        # Sacred geometry pattern detection
        sg_pattern = abs(current_price - sg_golden) < (price_range * 0.02)
        
        return {
            'sg_pattern': sg_pattern,
            'sg_level_1': sg_level_1,
            'sg_level_2': sg_level_2,
            'sg_level_3': sg_level_3,
            'sg_golden': sg_golden,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'price_range': price_range
        }
    
    def tesla_advanced_pattern_detection(self, data: pd.DataFrame) -> Dict:
        """Advanced Tesla Pattern Detection with Multiple Timeframes"""
        if len(data) < self.TESLA_EIGHTEEN:
            return {}
        
        # Multi-timeframe Tesla pattern analysis (with safe division)
        def safe_percentage_change(current, previous):
            if pd.isna(current) or pd.isna(previous) or previous == 0:
                return 0
            return (current - previous) / previous * 100
        
        price_change_3 = safe_percentage_change(data['close'].iloc[-1], data['close'].iloc[-self.TESLA_THREE-1])
        price_change_6 = safe_percentage_change(data['close'].iloc[-1], data['close'].iloc[-self.TESLA_SIX-1])
        price_change_9 = safe_percentage_change(data['close'].iloc[-1], data['close'].iloc[-self.TESLA_NINE-1])
        price_change_18 = safe_percentage_change(data['close'].iloc[-1], data['close'].iloc[-self.TESLA_EIGHTEEN-1])
        
        # Volume pattern analysis (with safe division)
        volume_change_3 = safe_percentage_change(data['volume'].iloc[-1], data['volume'].iloc[-self.TESLA_THREE-1])
        volume_change_6 = safe_percentage_change(data['volume'].iloc[-1], data['volume'].iloc[-self.TESLA_SIX-1])
        volume_change_9 = safe_percentage_change(data['volume'].iloc[-1], data['volume'].iloc[-self.TESLA_NINE-1])
        
        # Tesla pattern scoring
        tesla_score = 0
        if self.is_tesla_number(round(price_change_3)):
            tesla_score += 1
        if self.is_tesla_number(round(price_change_6)):
            tesla_score += 1
        if self.is_tesla_number(round(price_change_9)):
            tesla_score += 1
        if self.is_tesla_number(round(volume_change_3)):
            tesla_score += 1
        if self.is_tesla_number(round(volume_change_6)):
            tesla_score += 1
        if self.is_tesla_number(round(volume_change_9)):
            tesla_score += 1
        
        # Advanced Tesla energy calculation (handle NaN values and zero volume)
        try:
            high_low_diff = data['high'].iloc[-1] - data['low'].iloc[-1]
            volume = data['volume'].iloc[-1]
            
            if pd.isna(high_low_diff) or pd.isna(volume) or volume == 0:
                tesla_energy_advanced = 0
                energy_pattern_advanced = False
            else:
                tesla_energy_advanced = high_low_diff * volume * self.TESLA_THREE / self.TESLA_SIX * self.TESLA_NINE
                energy_pattern_advanced = self.is_tesla_number(round(tesla_energy_advanced / 10000)) if not pd.isna(tesla_energy_advanced) else False
        except (ValueError, OverflowError):
            tesla_energy_advanced = 0
            energy_pattern_advanced = False
        
        return {
            'tesla_score': tesla_score,
            'energy_pattern_advanced': energy_pattern_advanced,
            'tesla_energy_advanced': tesla_energy_advanced,
            'price_change_3': price_change_3,
            'price_change_6': price_change_6,
            'price_change_9': price_change_9,
            'price_change_18': price_change_18,
            'volume_change_3': volume_change_3,
            'volume_change_6': volume_change_6,
            'volume_change_9': volume_change_9
        }
    
    def generate_tesla_signals(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive Tesla 369 trading signals"""
        if len(data) < self.TESLA_TWENTY_SEVEN:
            return {}
        
        # Run all Tesla analyses
        basic_patterns = self.tesla_pattern_detection(data)
        fib_levels = self.tesla_fibonacci_levels(data)
        energy_zones = self.tesla_energy_zones(data)
        frequency_analysis = self.tesla_frequency_analysis(data)
        vortex_analysis = self.tesla_vortex_analysis(data)
        harmonics_analysis = self.tesla_harmonics_analysis(data)
        sacred_geometry = self.tesla_sacred_geometry(data)
        advanced_patterns = self.tesla_advanced_pattern_detection(data)
        
        # Calculate Tesla buy/sell conditions
        tesla_buy_conditions = 0
        tesla_sell_conditions = 0
        
        # Basic pattern conditions
        if basic_patterns.get('price_tesla_pattern', False):
            tesla_buy_conditions += 1
        if basic_patterns.get('volume_tesla_pattern', False):
            tesla_buy_conditions += 1
        if basic_patterns.get('energy_tesla_pattern', False):
            tesla_buy_conditions += 1
        if energy_zones.get('energy_zone', 0) > 0:
            tesla_buy_conditions += 1
        if frequency_analysis.get('tesla_freq_signal', 0) < 0:
            tesla_buy_conditions += 1
        
        # Advanced pattern conditions
        if vortex_analysis.get('vortex_pattern', False):
            tesla_buy_conditions += 1
        if harmonics_analysis.get('harmonic_convergence', False):
            tesla_buy_conditions += 1
        if harmonics_analysis.get('harmonic_pattern', False):
            tesla_buy_conditions += 1
        if sacred_geometry.get('sg_pattern', False):
            tesla_buy_conditions += 1
        if advanced_patterns.get('tesla_score', 0) >= 4:
            tesla_buy_conditions += 1
        
        # Sell conditions (opposite logic)
        if basic_patterns.get('price_tesla_pattern', False):
            tesla_sell_conditions += 1
        if basic_patterns.get('volume_tesla_pattern', False):
            tesla_sell_conditions += 1
        if basic_patterns.get('energy_tesla_pattern', False):
            tesla_sell_conditions += 1
        if energy_zones.get('energy_zone', 0) < 0:
            tesla_sell_conditions += 1
        if frequency_analysis.get('tesla_freq_signal', 0) > 0:
            tesla_sell_conditions += 1
        if vortex_analysis.get('vortex_pattern', False):
            tesla_sell_conditions += 1
        if harmonics_analysis.get('harmonic_convergence', False):
            tesla_sell_conditions += 1
        if harmonics_analysis.get('harmonic_pattern', False):
            tesla_sell_conditions += 1
        if sacred_geometry.get('sg_pattern', False):
            tesla_sell_conditions += 1
        if advanced_patterns.get('tesla_score', 0) >= 4:
            tesla_sell_conditions += 1
        
        # Signal generation based on sensitivity (lowered threshold for more signals)
        tesla_buy_signal = tesla_buy_conditions >= (5 * self.sensitivity_threshold)  # Lowered from 10 to 5
        tesla_sell_signal = tesla_sell_conditions >= (5 * self.sensitivity_threshold)  # Lowered from 10 to 5
        
        # Calculate Tesla Strength Score (0-100)
        tesla_strength = 0
        if basic_patterns.get('price_tesla_pattern', False):
            tesla_strength += 10
        if basic_patterns.get('volume_tesla_pattern', False):
            tesla_strength += 10
        if basic_patterns.get('energy_tesla_pattern', False):
            tesla_strength += 10
        if energy_zones.get('energy_zone', 0) != 0:
            tesla_strength += 10
        if vortex_analysis.get('vortex_pattern', False):
            tesla_strength += 10
        if harmonics_analysis.get('harmonic_convergence', False):
            tesla_strength += 10
        if harmonics_analysis.get('harmonic_pattern', False):
            tesla_strength += 10
        if sacred_geometry.get('sg_pattern', False):
            tesla_strength += 10
        if advanced_patterns.get('tesla_score', 0) >= 4:
            tesla_strength += 10
        if advanced_patterns.get('tesla_score', 0) >= 6:
            tesla_strength += 10
        
        # Combine all results
        tesla_analysis = {
            'tesla_buy_signal': tesla_buy_signal,
            'tesla_sell_signal': tesla_sell_signal,
            'tesla_strength': tesla_strength,
            'tesla_buy_conditions': tesla_buy_conditions,
            'tesla_sell_conditions': tesla_sell_conditions,
            'basic_patterns': basic_patterns,
            'fibonacci_levels': fib_levels,
            'energy_zones': energy_zones,
            'frequency_analysis': frequency_analysis,
            'vortex_analysis': vortex_analysis,
            'harmonics_analysis': harmonics_analysis,
            'sacred_geometry': sacred_geometry,
            'advanced_patterns': advanced_patterns
        }
        
        return tesla_analysis
    
    def identify_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify trading signals based on Tesla 369 analysis"""
        signals = data.copy()
        signals['signal'] = 0
        signals['signal_strength'] = 0.0
        signals['entry_price'] = np.nan
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        signals['tesla_strength'] = 0.0
        
        # Analyze each period
        for i in range(len(signals)):
            if i < self.TESLA_TWENTY_SEVEN:
                continue
                
            current_data = signals.iloc[:i+1]
            tesla_analysis = self.generate_tesla_signals(current_data)
            
            if tesla_analysis:
                tesla_strength = tesla_analysis['tesla_strength']
                signals.iloc[i, signals.columns.get_loc('tesla_strength')] = tesla_strength
                
                # Generate signals based on Tesla strength (lowered threshold for more trades)
                if tesla_analysis['tesla_buy_signal'] and tesla_strength >= 40:  # Lowered from 60 to 40
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('signal_strength')] = tesla_strength / 100.0  # Convert to decimal
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = signals['close'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('stop_loss')] = signals['close'].iloc[i] * 0.98  # 2% stop loss
                    signals.iloc[i, signals.columns.get_loc('take_profit')] = signals['close'].iloc[i] * 1.04  # 2:1 risk-reward
                
                elif tesla_analysis['tesla_sell_signal'] and tesla_strength >= 40:  # Lowered from 60 to 40
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    signals.iloc[i, signals.columns.get_loc('signal_strength')] = tesla_strength / 100.0  # Convert to decimal
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = signals['close'].iloc[i]
                    signals.iloc[i, signals.columns.get_loc('stop_loss')] = signals['close'].iloc[i] * 1.02  # 2% stop loss
                    signals.iloc[i, signals.columns.get_loc('take_profit')] = signals['close'].iloc[i] * 0.96  # 2:1 risk-reward
        
        return signals
    
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure using Tesla 369 concepts"""
        analysis = {}
        
        # Generate Tesla analysis
        tesla_analysis = self.generate_tesla_signals(data)
        analysis['tesla_analysis'] = tesla_analysis
        
        # Identify trading signals
        signals = self.identify_trading_signals(data)
        analysis['signals'] = signals
        
        # Calculate market statistics
        analysis['statistics'] = {
            'total_periods': len(data),
            'price_range': data['high'].max() - data['low'].min(),
            'average_volume': data['volume'].mean() if 'volume' in data.columns else 0,
            'volatility': data['close'].pct_change().std(),
            'trend_direction': 'up' if data['close'].iloc[-1] > data['close'].iloc[0] else 'down',
            'tesla_strength': tesla_analysis.get('tesla_strength', 0) if tesla_analysis else 0
        }
        
        return analysis

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
    
    # Initialize Tesla calculator
    calculator = Tesla369Calculator()
    
    # Analyze market structure
    analysis = calculator.analyze_market_structure(sample_data)
    
    print("Tesla 369 Analysis:")
    if analysis['tesla_analysis']:
        tesla = analysis['tesla_analysis']
        print(f"Tesla Strength: {tesla['tesla_strength']:.1f}%")
        print(f"Buy Signal: {tesla['tesla_buy_signal']}")
        print(f"Sell Signal: {tesla['tesla_sell_signal']}")
        print(f"Tesla Score: {tesla['advanced_patterns']['tesla_score']}/6")
    
    print(f"\nMarket Statistics:")
    stats = analysis['statistics']
    print(f"Total Periods: {stats['total_periods']}")
    print(f"Volatility: {stats['volatility']:.4f}")
    print(f"Trend Direction: {stats['trend_direction']}")
