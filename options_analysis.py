"""
Options Data and Greeks Analysis
Provides options chain data, Greeks calculations, and volatility analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OptionsAnalyzer:
    """Options data and Greeks analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.days_per_year = 365
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option pricing"""
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option pricing"""
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict:
        """Calculate option Greeks"""
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_part1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        theta = (theta_part1 + theta_part2) / 365  # Daily theta
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def generate_options_chain(self, symbol: str, current_price: float, days_to_expiry: int = 30) -> pd.DataFrame:
        """Generate simulated options chain"""
        strikes = []
        option_data = []
        
        # Generate strikes around current price
        strike_range = 0.2  # 20% range
        min_strike = current_price * (1 - strike_range)
        max_strike = current_price * (1 + strike_range)
        
        # Generate strikes every 5% of current price
        strike_step = current_price * 0.05
        current_strike = min_strike
        while current_strike <= max_strike:
            strikes.append(round(current_strike, 2))
            current_strike += strike_step
        
        # Add some specific strikes around current price
        strikes.extend([
            round(current_price * 0.95, 2),
            round(current_price * 0.98, 2),
            round(current_price, 2),
            round(current_price * 1.02, 2),
            round(current_price * 1.05, 2)
        ])
        strikes = sorted(list(set(strikes)))
        
        # Generate option data for each strike
        for strike in strikes:
            T = days_to_expiry / self.days_per_year
            sigma = 0.25 + np.random.uniform(-0.1, 0.1)  # 25% base volatility
            
            # Call option
            call_price = self.black_scholes_call(current_price, strike, T, self.risk_free_rate, sigma)
            call_greeks = self.calculate_greeks(current_price, strike, T, self.risk_free_rate, sigma, 'call')
            
            # Put option
            put_price = self.black_scholes_put(current_price, strike, T, self.risk_free_rate, sigma)
            put_greeks = self.calculate_greeks(current_price, strike, T, self.risk_free_rate, sigma, 'put')
            
            # Calculate volume and open interest (simulated)
            call_volume = int(np.random.uniform(100, 5000))
            put_volume = int(np.random.uniform(100, 5000))
            call_oi = int(np.random.uniform(1000, 10000))
            put_oi = int(np.random.uniform(1000, 10000))
            
            option_data.append({
                'strike': strike,
                'call_price': round(call_price, 2),
                'put_price': round(put_price, 2),
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'call_delta': round(call_greeks['delta'], 4),
                'put_delta': round(put_greeks['delta'], 4),
                'call_gamma': round(call_greeks['gamma'], 4),
                'put_gamma': round(put_greeks['gamma'], 4),
                'call_theta': round(call_greeks['theta'], 4),
                'put_theta': round(put_greeks['theta'], 4),
                'call_vega': round(call_greeks['vega'], 4),
                'put_vega': round(put_greeks['vega'], 4),
                'call_rho': round(call_greeks['rho'], 4),
                'put_rho': round(put_greeks['rho'], 4),
                'implied_volatility': round(sigma * 100, 2)
            })
        
        return pd.DataFrame(option_data)
    
    def analyze_volatility_surface(self, options_chain: pd.DataFrame) -> Dict:
        """Analyze volatility surface and skew"""
        # Calculate moneyness (S/K ratio)
        current_price = options_chain['strike'].median()  # Approximate current price
        options_chain['moneyness'] = current_price / options_chain['strike']
        
        # Volatility skew analysis
        itm_options = options_chain[options_chain['moneyness'] > 1.05]  # In-the-money
        atm_options = options_chain[(options_chain['moneyness'] >= 0.95) & (options_chain['moneyness'] <= 1.05)]  # At-the-money
        otm_options = options_chain[options_chain['moneyness'] < 0.95]  # Out-of-the-money
        
        skew_analysis = {
            'itm_avg_iv': itm_options['implied_volatility'].mean() if len(itm_options) > 0 else 0,
            'atm_avg_iv': atm_options['implied_volatility'].mean() if len(atm_options) > 0 else 0,
            'otm_avg_iv': otm_options['implied_volatility'].mean() if len(otm_options) > 0 else 0,
            'volatility_skew': 0,
            'skew_direction': 'Neutral'
        }
        
        # Calculate skew
        if skew_analysis['otm_avg_iv'] > 0 and skew_analysis['itm_avg_iv'] > 0:
            skew_analysis['volatility_skew'] = skew_analysis['otm_avg_iv'] - skew_analysis['itm_avg_iv']
            if skew_analysis['volatility_skew'] > 2:
                skew_analysis['skew_direction'] = 'Positive Skew (OTM > ITM)'
            elif skew_analysis['volatility_skew'] < -2:
                skew_analysis['skew_direction'] = 'Negative Skew (ITM > OTM)'
            else:
                skew_analysis['skew_direction'] = 'Neutral Skew'
        
        return skew_analysis
    
    def calculate_put_call_ratio(self, options_chain: pd.DataFrame) -> Dict:
        """Calculate put/call ratio and sentiment analysis"""
        total_call_volume = options_chain['call_volume'].sum()
        total_put_volume = options_chain['put_volume'].sum()
        total_call_oi = options_chain['call_oi'].sum()
        total_put_oi = options_chain['put_oi'].sum()
        
        volume_pcr = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        oi_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Sentiment analysis
        if volume_pcr > 1.2:
            volume_sentiment = 'Bearish (High Put Volume)'
        elif volume_pcr < 0.8:
            volume_sentiment = 'Bullish (High Call Volume)'
        else:
            volume_sentiment = 'Neutral'
        
        if oi_pcr > 1.2:
            oi_sentiment = 'Bearish (High Put OI)'
        elif oi_pcr < 0.8:
            oi_sentiment = 'Bullish (High Call OI)'
        else:
            oi_sentiment = 'Neutral'
        
        return {
            'volume_put_call_ratio': round(volume_pcr, 3),
            'oi_put_call_ratio': round(oi_pcr, 3),
            'volume_sentiment': volume_sentiment,
            'oi_sentiment': oi_sentiment,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi
        }
    
    def find_high_volume_options(self, options_chain: pd.DataFrame, top_n: int = 10) -> Dict:
        """Find highest volume options"""
        # Sort by total volume (calls + puts)
        options_chain['total_volume'] = options_chain['call_volume'] + options_chain['put_volume']
        high_volume = options_chain.nlargest(top_n, 'total_volume')
        
        return {
            'high_volume_calls': high_volume.nlargest(5, 'call_volume')[['strike', 'call_price', 'call_volume', 'call_delta']].to_dict('records'),
            'high_volume_puts': high_volume.nlargest(5, 'put_volume')[['strike', 'put_price', 'put_volume', 'put_delta']].to_dict('records')
        }
    
    def generate_options_report(self, symbol: str, current_price: float, days_to_expiry: int = 30) -> Dict:
        """Generate comprehensive options analysis report"""
        # Generate options chain
        options_chain = self.generate_options_chain(symbol, current_price, days_to_expiry)
        
        # Analyze volatility surface
        volatility_analysis = self.analyze_volatility_surface(options_chain)
        
        # Calculate put/call ratios
        pcr_analysis = self.calculate_put_call_ratio(options_chain)
        
        # Find high volume options
        high_volume_analysis = self.find_high_volume_options(options_chain)
        
        # Calculate max pain (simplified)
        max_pain_strike = options_chain.loc[options_chain['total_volume'].idxmax(), 'strike']
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'days_to_expiry': days_to_expiry,
            'options_chain': options_chain.to_dict('records'),
            'volatility_analysis': volatility_analysis,
            'put_call_analysis': pcr_analysis,
            'high_volume_options': high_volume_analysis,
            'max_pain_strike': max_pain_strike,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def demo_options_analysis():
    """Demo function for options analysis"""
    analyzer = OptionsAnalyzer()
    
    # Test with AAPL
    symbol = 'AAPL'
    current_price = 150.0
    days_to_expiry = 30
    
    report = analyzer.generate_options_report(symbol, current_price, days_to_expiry)
    
    print("=== OPTIONS ANALYSIS REPORT ===")
    print(f"Symbol: {report['symbol']}")
    print(f"Current Price: ${report['current_price']}")
    print(f"Days to Expiry: {report['days_to_expiry']}")
    print(f"Analysis Date: {report['analysis_date']}")
    
    print("\n=== VOLATILITY ANALYSIS ===")
    vol_analysis = report['volatility_analysis']
    print(f"ITM Average IV: {vol_analysis['itm_avg_iv']:.2f}%")
    print(f"ATM Average IV: {vol_analysis['atm_avg_iv']:.2f}%")
    print(f"OTM Average IV: {vol_analysis['otm_avg_iv']:.2f}%")
    print(f"Volatility Skew: {vol_analysis['volatility_skew']:.2f}%")
    print(f"Skew Direction: {vol_analysis['skew_direction']}")
    
    print("\n=== PUT/CALL ANALYSIS ===")
    pcr_analysis = report['put_call_analysis']
    print(f"Volume Put/Call Ratio: {pcr_analysis['volume_put_call_ratio']}")
    print(f"OI Put/Call Ratio: {pcr_analysis['oi_put_call_ratio']}")
    print(f"Volume Sentiment: {pcr_analysis['volume_sentiment']}")
    print(f"OI Sentiment: {pcr_analysis['oi_sentiment']}")
    
    print(f"\nMax Pain Strike: ${report['max_pain_strike']}")
    
    print("\n=== HIGH VOLUME OPTIONS ===")
    high_vol = report['high_volume_options']
    print("Top Call Options:")
    for call in high_vol['high_volume_calls'][:3]:
        print(f"  ${call['strike']} - Price: ${call['call_price']}, Volume: {call['call_volume']}, Delta: {call['call_delta']}")
    
    print("Top Put Options:")
    for put in high_vol['high_volume_puts'][:3]:
        print(f"  ${put['strike']} - Price: ${put['put_price']}, Volume: {put['put_volume']}, Delta: {put['put_delta']}")

if __name__ == "__main__":
    demo_options_analysis()
