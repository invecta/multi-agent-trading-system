"""
Crypto Market Integration
Provides cryptocurrency data, DeFi metrics, and blockchain analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class CryptoAnalyzer:
    """Cryptocurrency market analysis and DeFi metrics"""
    
    def __init__(self):
        self.major_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC']
        self.defi_tokens = ['UNI', 'AAVE', 'COMP', 'MKR', 'SUSHI', 'CRV', 'YFI', 'SNX', '1INCH', 'BAL']
        self.stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD']
    
    def generate_crypto_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate simulated cryptocurrency price data"""
        base_price = self.get_base_price(symbol)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price data with higher volatility than stocks
        returns = np.random.normal(0.002, 0.05, days)  # 0.2% daily return, 5% volatility
        prices = [base_price]
        
        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(price)
        
        # Generate volume data (crypto has higher volume)
        volumes = [np.random.uniform(1000000, 10000000) for _ in range(days)]
        
        # Generate market cap data
        market_caps = [price * self.get_circulating_supply(symbol) for price in prices]
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + np.random.uniform(0, 0.1)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.1)) for p in prices],
            'Close': prices,
            'Volume': volumes,
            'Market_Cap': market_caps
        })
        
        return df
    
    def get_base_price(self, symbol: str) -> float:
        """Get base price for cryptocurrency"""
        base_prices = {
            'BTC': 45000, 'ETH': 3000, 'BNB': 300, 'ADA': 0.5, 'SOL': 100,
            'XRP': 0.6, 'DOT': 25, 'DOGE': 0.08, 'AVAX': 35, 'MATIC': 0.8,
            'UNI': 15, 'AAVE': 200, 'COMP': 80, 'MKR': 2500, 'SUSHI': 8,
            'CRV': 3, 'YFI': 8000, 'SNX': 12, '1INCH': 2, 'BAL': 15,
            'USDT': 1.0, 'USDC': 1.0, 'DAI': 1.0, 'BUSD': 1.0, 'TUSD': 1.0
        }
        return base_prices.get(symbol, 100.0)
    
    def get_circulating_supply(self, symbol: str) -> float:
        """Get circulating supply for cryptocurrency"""
        supplies = {
            'BTC': 19000000, 'ETH': 120000000, 'BNB': 160000000, 'ADA': 33000000000,
            'SOL': 400000000, 'XRP': 47000000000, 'DOT': 1000000000, 'DOGE': 130000000000,
            'AVAX': 300000000, 'MATIC': 8000000000, 'UNI': 600000000, 'AAVE': 13000000,
            'COMP': 7000000, 'MKR': 1000000, 'SUSHI': 200000000, 'CRV': 1000000000,
            'YFI': 36000, 'SNX': 100000000, '1INCH': 1000000000, 'BAL': 35000000,
            'USDT': 80000000000, 'USDC': 50000000000, 'DAI': 5000000000, 'BUSD': 20000000000, 'TUSD': 1000000000
        }
        return supplies.get(symbol, 1000000000)
    
    def calculate_crypto_metrics(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate cryptocurrency-specific metrics"""
        # Basic price metrics
        current_price = df['Close'].iloc[-1]
        price_change_24h = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
        price_change_7d = (current_price - df['Close'].iloc[-8]) / df['Close'].iloc[-8] * 100
        price_change_30d = (current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        
        # Volatility metrics
        daily_returns = df['Close'].pct_change().dropna()
        volatility_7d = daily_returns.tail(7).std() * np.sqrt(365) * 100
        volatility_30d = daily_returns.std() * np.sqrt(365) * 100
        
        # Volume metrics
        avg_volume_7d = df['Volume'].tail(7).mean()
        volume_change_24h = (df['Volume'].iloc[-1] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2] * 100
        
        # Market cap metrics
        current_market_cap = df['Market_Cap'].iloc[-1]
        market_cap_change_24h = (current_market_cap - df['Market_Cap'].iloc[-2]) / df['Market_Cap'].iloc[-2] * 100
        
        # Fear & Greed Index (simulated)
        fear_greed_index = self.calculate_fear_greed_index(df)
        
        # Technical indicators
        rsi = self.calculate_rsi(df['Close'], 14)
        macd = self.calculate_macd(df['Close'])
        
        return {
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'price_change_7d': price_change_7d,
            'price_change_30d': price_change_30d,
            'volatility_7d': volatility_7d,
            'volatility_30d': volatility_30d,
            'avg_volume_7d': avg_volume_7d,
            'volume_change_24h': volume_change_24h,
            'current_market_cap': current_market_cap,
            'market_cap_change_24h': market_cap_change_24h,
            'fear_greed_index': fear_greed_index,
            'rsi': rsi,
            'macd': macd
        }
    
    def calculate_fear_greed_index(self, df: pd.DataFrame) -> Dict:
        """Calculate Fear & Greed Index (simulated)"""
        # Simulate fear & greed based on price action and volume
        price_change_7d = (df['Close'].iloc[-1] - df['Close'].iloc[-8]) / df['Close'].iloc[-8] * 100
        volume_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
        
        # Calculate index (0-100)
        if price_change_7d > 10 and volume_ratio > 1.5:
            index = 85  # Extreme Greed
            sentiment = "Extreme Greed"
        elif price_change_7d > 5 and volume_ratio > 1.2:
            index = 70  # Greed
            sentiment = "Greed"
        elif price_change_7d > 0:
            index = 55  # Neutral to Greed
            sentiment = "Neutral to Greed"
        elif price_change_7d > -5:
            index = 45  # Neutral to Fear
            sentiment = "Neutral to Fear"
        elif price_change_7d > -10:
            index = 30  # Fear
            sentiment = "Fear"
        else:
            index = 15  # Extreme Fear
            sentiment = "Extreme Fear"
        
        return {
            'index': index,
            'sentiment': sentiment,
            'description': f"Market sentiment: {sentiment} (Index: {index}/100)"
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def analyze_defi_metrics(self, symbol: str) -> Dict:
        """Analyze DeFi-specific metrics"""
        if symbol not in self.defi_tokens:
            return {'error': 'Not a DeFi token'}
        
        # Simulate DeFi metrics
        tvl = np.random.uniform(100000000, 5000000000)  # Total Value Locked
        daily_volume = np.random.uniform(10000000, 500000000)  # Daily volume
        active_users = np.random.randint(10000, 1000000)  # Active users
        transaction_count = np.random.randint(100000, 10000000)  # Daily transactions
        
        # Calculate DeFi-specific ratios
        volume_tvl_ratio = daily_volume / tvl
        users_tvl_ratio = active_users / (tvl / 1000000)  # Users per $1M TVL
        
        # Protocol-specific metrics
        if symbol == 'UNI':
            protocol_type = 'DEX'
            description = 'Uniswap - Decentralized Exchange'
        elif symbol == 'AAVE':
            protocol_type = 'Lending'
            description = 'Aave - Decentralized Lending Protocol'
        elif symbol == 'COMP':
            protocol_type = 'Lending'
            description = 'Compound - Decentralized Lending Protocol'
        else:
            protocol_type = 'DeFi'
            description = f'{symbol} - DeFi Protocol'
        
        return {
            'protocol_type': protocol_type,
            'description': description,
            'tvl': tvl,
            'daily_volume': daily_volume,
            'active_users': active_users,
            'transaction_count': transaction_count,
            'volume_tvl_ratio': volume_tvl_ratio,
            'users_tvl_ratio': users_tvl_ratio,
            'tvl_change_24h': np.random.uniform(-10, 10),
            'volume_change_24h': np.random.uniform(-20, 20)
        }
    
    def analyze_stablecoin_metrics(self, symbol: str) -> Dict:
        """Analyze stablecoin-specific metrics"""
        if symbol not in self.stablecoins:
            return {'error': 'Not a stablecoin'}
        
        # Simulate stablecoin metrics
        total_supply = np.random.uniform(10000000000, 100000000000)  # Total supply
        circulating_supply = total_supply * np.random.uniform(0.8, 1.0)  # Circulating supply
        daily_volume = np.random.uniform(1000000000, 10000000000)  # Daily volume
        market_cap = circulating_supply  # Market cap (should be close to supply for stablecoins)
        
        # Peg stability (how close to $1)
        peg_deviation = np.random.uniform(-0.02, 0.02)  # -2% to +2% deviation
        current_price = 1.0 + peg_deviation
        
        # Collateralization ratio (for algorithmic stablecoins)
        if symbol == 'DAI':
            collateralization_ratio = np.random.uniform(150, 200)  # 150-200%
        else:
            collateralization_ratio = 100  # Fully collateralized
        
        return {
            'total_supply': total_supply,
            'circulating_supply': circulating_supply,
            'daily_volume': daily_volume,
            'market_cap': market_cap,
            'current_price': current_price,
            'peg_deviation': peg_deviation,
            'peg_stability': 'Stable' if abs(peg_deviation) < 0.01 else 'Unstable',
            'collateralization_ratio': collateralization_ratio,
            'volume_change_24h': np.random.uniform(-30, 30),
            'supply_change_24h': np.random.uniform(-5, 5)
        }
    
    def generate_crypto_market_overview(self) -> Dict:
        """Generate overall crypto market overview"""
        # Simulate market-wide metrics
        total_market_cap = np.random.uniform(1500000000000, 2500000000000)  # $1.5T - $2.5T
        total_volume_24h = np.random.uniform(50000000000, 150000000000)  # $50B - $150B
        btc_dominance = np.random.uniform(40, 50)  # 40-50%
        eth_dominance = np.random.uniform(15, 25)  # 15-25%
        
        # Market sentiment
        market_sentiment = self.calculate_market_sentiment()
        
        # Top gainers and losers (simulated)
        top_gainers = [
            {'symbol': 'SOL', 'change': np.random.uniform(10, 25)},
            {'symbol': 'ADA', 'change': np.random.uniform(8, 20)},
            {'symbol': 'DOT', 'change': np.random.uniform(5, 15)}
        ]
        
        top_losers = [
            {'symbol': 'DOGE', 'change': np.random.uniform(-15, -5)},
            {'symbol': 'XRP', 'change': np.random.uniform(-12, -3)},
            {'symbol': 'MATIC', 'change': np.random.uniform(-10, -2)}
        ]
        
        return {
            'total_market_cap': total_market_cap,
            'total_volume_24h': total_volume_24h,
            'btc_dominance': btc_dominance,
            'eth_dominance': eth_dominance,
            'market_sentiment': market_sentiment,
            'top_gainers': top_gainers,
            'top_losers': top_losers,
            'active_cryptocurrencies': 8000,
            'active_markets': 50000,
            'market_cap_change_24h': np.random.uniform(-5, 5)
        }
    
    def calculate_market_sentiment(self) -> Dict:
        """Calculate overall market sentiment"""
        # Simulate market sentiment based on various factors
        sentiment_score = np.random.uniform(0, 100)
        
        if sentiment_score > 75:
            sentiment = 'Extremely Bullish'
            description = 'Strong buying pressure across all major cryptocurrencies'
        elif sentiment_score > 60:
            sentiment = 'Bullish'
            description = 'Positive momentum with increased institutional adoption'
        elif sentiment_score > 40:
            sentiment = 'Neutral'
            description = 'Mixed signals with sideways price action'
        elif sentiment_score > 25:
            sentiment = 'Bearish'
            description = 'Selling pressure and risk-off sentiment'
        else:
            sentiment = 'Extremely Bearish'
            description = 'Strong selling pressure and fear in the market'
        
        return {
            'score': sentiment_score,
            'sentiment': sentiment,
            'description': description
        }
    
    def generate_crypto_report(self, symbol: str, days: int = 30) -> Dict:
        """Generate comprehensive cryptocurrency analysis report"""
        # Generate price data
        df = self.generate_crypto_data(symbol, days)
        
        # Calculate metrics
        metrics = self.calculate_crypto_metrics(df, symbol)
        
        # Get specific analysis based on token type
        if symbol in self.defi_tokens:
            specific_analysis = self.analyze_defi_metrics(symbol)
        elif symbol in self.stablecoins:
            specific_analysis = self.analyze_stablecoin_metrics(symbol)
        else:
            specific_analysis = {'type': 'Cryptocurrency', 'description': f'{symbol} - Digital Asset'}
        
        # Generate market overview
        market_overview = self.generate_crypto_market_overview()
        
        return {
            'symbol': symbol,
            'price_data': df.to_dict('records'),
            'metrics': metrics,
            'specific_analysis': specific_analysis,
            'market_overview': market_overview,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def demo_crypto_analysis():
    """Demo function for crypto analysis"""
    analyzer = CryptoAnalyzer()
    
    # Test with Bitcoin
    symbol = 'BTC'
    report = analyzer.generate_crypto_report(symbol, 30)
    
    print("=== CRYPTOCURRENCY ANALYSIS REPORT ===")
    print(f"Symbol: {report['symbol']}")
    print(f"Analysis Date: {report['analysis_date']}")
    
    print("\n=== PRICE METRICS ===")
    metrics = report['metrics']
    print(f"Current Price: ${metrics['current_price']:,.2f}")
    print(f"24h Change: {metrics['price_change_24h']:+.2f}%")
    print(f"7d Change: {metrics['price_change_7d']:+.2f}%")
    print(f"30d Change: {metrics['price_change_30d']:+.2f}%")
    print(f"7d Volatility: {metrics['volatility_7d']:.2f}%")
    print(f"30d Volatility: {metrics['volatility_30d']:.2f}%")
    
    print("\n=== MARKET METRICS ===")
    print(f"Market Cap: ${metrics['current_market_cap']:,.0f}")
    print(f"24h Volume: {metrics['avg_volume_7d']:,.0f}")
    print(f"RSI: {metrics['rsi']:.2f}")
    
    print("\n=== FEAR & GREED INDEX ===")
    fgi = metrics['fear_greed_index']
    print(f"Index: {fgi['index']}/100")
    print(f"Sentiment: {fgi['sentiment']}")
    
    print("\n=== MARKET OVERVIEW ===")
    overview = report['market_overview']
    print(f"Total Market Cap: ${overview['total_market_cap']:,.0f}")
    print(f"24h Volume: ${overview['total_volume_24h']:,.0f}")
    print(f"BTC Dominance: {overview['btc_dominance']:.1f}%")
    print(f"ETH Dominance: {overview['eth_dominance']:.1f}%")
    print(f"Market Sentiment: {overview['market_sentiment']['sentiment']}")

if __name__ == "__main__":
    demo_crypto_analysis()
