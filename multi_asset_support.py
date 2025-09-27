"""
Multi-asset support module
Provides support for stocks, crypto, forex, commodities, and other asset classes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import json
import time
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetType(Enum):
    """Asset type enumeration"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    ETF = "etf"
    INDEX = "index"

@dataclass
class Asset:
    """Asset data structure"""
    symbol: str
    name: str
    asset_type: AssetType
    exchange: str = ""
    currency: str = "USD"
    sector: str = ""
    market_cap: float = 0.0
    description: str = ""

class MultiAssetDataProvider:
    """Multi-asset data provider"""
    
    def __init__(self):
        self.asset_registry = self._initialize_asset_registry()
        self.data_sources = {
            'yfinance': self._fetch_yfinance_data,
            'crypto': self._fetch_crypto_data,
            'forex': self._fetch_forex_data,
            'commodity': self._fetch_commodity_data
        }
        
    def _initialize_asset_registry(self) -> Dict[str, Asset]:
        """Initialize asset registry with popular assets"""
        
        registry = {}
        
        # Stocks
        stocks = [
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
            ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology'),
            ('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'Consumer Discretionary'),
            ('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Discretionary'),
            ('META', 'Meta Platforms Inc.', 'NASDAQ', 'Communication'),
            ('NVDA', 'NVIDIA Corporation', 'NASDAQ', 'Technology'),
            ('JPM', 'JPMorgan Chase & Co.', 'NYSE', 'Financial'),
            ('JNJ', 'Johnson & Johnson', 'NYSE', 'Healthcare'),
            ('V', 'Visa Inc.', 'NYSE', 'Financial')
        ]
        
        for symbol, name, exchange, sector in stocks:
            registry[symbol] = Asset(symbol, name, AssetType.STOCK, exchange, 'USD', sector)
            
        # Cryptocurrencies
        cryptos = [
            ('BTC-USD', 'Bitcoin', 'USD'),
            ('ETH-USD', 'Ethereum', 'USD'),
            ('BNB-USD', 'Binance Coin', 'USD'),
            ('ADA-USD', 'Cardano', 'USD'),
            ('SOL-USD', 'Solana', 'USD'),
            ('XRP-USD', 'Ripple', 'USD'),
            ('DOT-USD', 'Polkadot', 'USD'),
            ('DOGE-USD', 'Dogecoin', 'USD')
        ]
        
        for symbol, name, currency in cryptos:
            registry[symbol] = Asset(symbol, name, AssetType.CRYPTO, 'Crypto', currency, 'Cryptocurrency')
            
        # Forex pairs
        forex_pairs = [
            ('EURUSD=X', 'Euro/US Dollar', 'USD'),
            ('GBPUSD=X', 'British Pound/US Dollar', 'USD'),
            ('USDJPY=X', 'US Dollar/Japanese Yen', 'USD'),
            ('USDCHF=X', 'US Dollar/Swiss Franc', 'USD'),
            ('AUDUSD=X', 'Australian Dollar/US Dollar', 'USD'),
            ('USDCAD=X', 'US Dollar/Canadian Dollar', 'USD')
        ]
        
        for symbol, name, currency in forex_pairs:
            registry[symbol] = Asset(symbol, name, AssetType.FOREX, 'Forex', currency, 'Currency')
            
        # Commodities
        commodities = [
            ('GC=F', 'Gold Futures', 'USD'),
            ('SI=F', 'Silver Futures', 'USD'),
            ('CL=F', 'Crude Oil Futures', 'USD'),
            ('NG=F', 'Natural Gas Futures', 'USD'),
            ('ZC=F', 'Corn Futures', 'USD'),
            ('ZS=F', 'Soybean Futures', 'USD')
        ]
        
        for symbol, name, currency in commodities:
            registry[symbol] = Asset(symbol, name, AssetType.COMMODITY, 'Commodity', currency, 'Commodity')
            
        # ETFs
        etfs = [
            ('SPY', 'SPDR S&P 500 ETF', 'NYSE', 'Broad Market'),
            ('QQQ', 'Invesco QQQ Trust', 'NASDAQ', 'Technology'),
            ('IWM', 'iShares Russell 2000 ETF', 'NYSE', 'Small Cap'),
            ('GLD', 'SPDR Gold Trust', 'NYSE', 'Precious Metals'),
            ('TLT', 'iShares 20+ Year Treasury Bond ETF', 'NYSE', 'Bonds')
        ]
        
        for symbol, name, exchange, sector in etfs:
            registry[symbol] = Asset(symbol, name, AssetType.ETF, exchange, 'USD', sector)
            
        return registry
        
    def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """Get asset information"""
        return self.asset_registry.get(symbol)
        
    def get_assets_by_type(self, asset_type: AssetType) -> List[Asset]:
        """Get all assets of a specific type"""
        return [asset for asset in self.asset_registry.values() if asset.asset_type == asset_type]
        
    def search_assets(self, query: str) -> List[Asset]:
        """Search assets by name or symbol"""
        query = query.lower()
        results = []
        
        for asset in self.asset_registry.values():
            if (query in asset.symbol.lower() or 
                query in asset.name.lower() or 
                query in asset.sector.lower()):
                results.append(asset)
                
        return results
        
    def _fetch_yfinance_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _fetch_crypto_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        try:
            # Use yfinance for crypto data
            return self._fetch_yfinance_data(symbol, period)
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _fetch_forex_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch forex data"""
        try:
            # Use yfinance for forex data
            return self._fetch_yfinance_data(symbol, period)
        except Exception as e:
            logger.error(f"Error fetching forex data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _fetch_commodity_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch commodity data"""
        try:
            # Use yfinance for commodity data
            return self._fetch_yfinance_data(symbol, period)
        except Exception as e:
            logger.error(f"Error fetching commodity data for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_asset_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data for any asset type"""
        asset_info = self.get_asset_info(symbol)
        
        if not asset_info:
            logger.warning(f"Asset {symbol} not found in registry")
            return pd.DataFrame()
            
        # Determine data source based on asset type
        if asset_info.asset_type == AssetType.CRYPTO:
            return self._fetch_crypto_data(symbol, period)
        elif asset_info.asset_type == AssetType.FOREX:
            return self._fetch_forex_data(symbol, period)
        elif asset_info.asset_type == AssetType.COMMODITY:
            return self._fetch_commodity_data(symbol, period)
        else:
            return self._fetch_yfinance_data(symbol, period)

class MultiAssetAnalyzer:
    """Multi-asset analysis engine"""
    
    def __init__(self):
        self.data_provider = MultiAssetDataProvider()
        
    def analyze_asset(self, symbol: str, period: str = "1y") -> Dict:
        """Comprehensive asset analysis"""
        
        asset_info = self.data_provider.get_asset_info(symbol)
        if not asset_info:
            return {'error': f'Asset {symbol} not found'}
            
        # Fetch data
        data = self.data_provider.fetch_asset_data(symbol, period)
        if data.empty:
            return {'error': f'No data available for {symbol}'}
            
        # Calculate metrics
        returns = data['Close'].pct_change().dropna()
        
        analysis = {
            'asset_info': {
                'symbol': asset_info.symbol,
                'name': asset_info.name,
                'type': asset_info.asset_type.value,
                'exchange': asset_info.exchange,
                'currency': asset_info.currency,
                'sector': asset_info.sector
            },
            'price_metrics': {
                'current_price': float(data['Close'].iloc[-1]),
                'price_change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
                'price_change_percent': float((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100),
                'high_52w': float(data['High'].max()),
                'low_52w': float(data['Low'].min()),
                'volume': int(data['Volume'].iloc[-1])
            },
            'performance_metrics': {
                'total_return': float((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1),
                'annualized_return': float(((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252 / len(data))) - 1),
                'volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(returns))
            },
            'technical_indicators': self._calculate_technical_indicators(data),
            'data_points': len(data)
        }
        
        return analysis
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = float(data['Close'].rolling(window=20).mean().iloc[-1])
        indicators['sma_50'] = float(data['Close'].rolling(window=50).mean().iloc[-1])
        indicators['ema_12'] = float(data['Close'].ewm(span=12).mean().iloc[-1])
        indicators['ema_26'] = float(data['Close'].ewm(span=26).mean().iloc[-1])
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = float(rsi.iloc[-1])
        
        # MACD
        macd = indicators['ema_12'] - indicators['ema_26']
        macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
        indicators['macd'] = macd
        indicators['macd_signal'] = float(macd_signal)
        indicators['macd_histogram'] = macd - float(macd_signal)
        
        # Bollinger Bands
        bb_middle = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        indicators['bb_upper'] = float((bb_middle + (bb_std * 2)).iloc[-1])
        indicators['bb_middle'] = float(bb_middle.iloc[-1])
        indicators['bb_lower'] = float((bb_middle - (bb_std * 2)).iloc[-1])
        
        return indicators
        
    def compare_assets(self, symbols: List[str], period: str = "1y") -> Dict:
        """Compare multiple assets"""
        
        comparisons = {}
        
        for symbol in symbols:
            analysis = self.analyze_asset(symbol, period)
            if 'error' not in analysis:
                comparisons[symbol] = analysis
                
        if not comparisons:
            return {'error': 'No valid assets to compare'}
            
        # Create comparison metrics
        comparison_data = {
            'assets': comparisons,
            'summary': self._create_comparison_summary(comparisons)
        }
        
        return comparison_data
        
    def _create_comparison_summary(self, comparisons: Dict) -> Dict:
        """Create comparison summary"""
        
        symbols = list(comparisons.keys())
        
        # Performance comparison
        returns = [comp['performance_metrics']['total_return'] for comp in comparisons.values()]
        volatilities = [comp['performance_metrics']['volatility'] for comp in comparisons.values()]
        sharpe_ratios = [comp['performance_metrics']['sharpe_ratio'] for comp in comparisons.values()]
        
        best_performer = symbols[np.argmax(returns)]
        lowest_volatility = symbols[np.argmin(volatilities)]
        best_sharpe = symbols[np.argmax(sharpe_ratios)]
        
        return {
            'best_performer': {
                'symbol': best_performer,
                'return': max(returns)
            },
            'lowest_volatility': {
                'symbol': lowest_volatility,
                'volatility': min(volatilities)
            },
            'best_sharpe': {
                'symbol': best_sharpe,
                'sharpe_ratio': max(sharpe_ratios)
            },
            'average_return': np.mean(returns),
            'average_volatility': np.mean(volatilities),
            'average_sharpe': np.mean(sharpe_ratios)
        }

class MultiAssetVisualization:
    """Multi-asset visualization tools"""
    
    def __init__(self):
        self.analyzer = MultiAssetAnalyzer()
        
    def create_asset_comparison_chart(self, symbols: List[str], period: str = "1y") -> go.Figure:
        """Create asset comparison chart"""
        
        fig = go.Figure()
        
        for symbol in symbols:
            data = self.analyzer.data_provider.fetch_asset_data(symbol, period)
            if not data.empty:
                # Normalize to starting value of 100
                normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    name=symbol,
                    line=dict(width=2)
                ))
                
        fig.update_layout(
            title='Asset Performance Comparison (Normalized)',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base = 100)',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_risk_return_scatter(self, symbols: List[str], period: str = "1y") -> go.Figure:
        """Create risk-return scatter plot"""
        
        data_points = []
        
        for symbol in symbols:
            analysis = self.analyzer.analyze_asset(symbol, period)
            if 'error' not in analysis:
                data_points.append({
                    'symbol': symbol,
                    'return': analysis['performance_metrics']['annualized_return'],
                    'volatility': analysis['performance_metrics']['volatility'],
                    'sharpe': analysis['performance_metrics']['sharpe_ratio'],
                    'type': analysis['asset_info']['type']
                })
                
        if not data_points:
            return go.Figure()
            
        df = pd.DataFrame(data_points)
        
        # Create scatter plot with color coding by asset type
        fig = px.scatter(df, x='volatility', y='return', 
                        color='type', size='sharpe',
                        hover_data=['symbol', 'sharpe'],
                        title='Risk-Return Analysis by Asset Type')
        
        fig.update_layout(
            xaxis_title='Volatility (Risk)',
            yaxis_title='Annualized Return',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_asset_allocation_pie(self, symbols: List[str], weights: List[float] = None) -> go.Figure:
        """Create asset allocation pie chart"""
        
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
            
        # Get asset information
        asset_info = []
        for symbol in symbols:
            info = self.analyzer.data_provider.get_asset_info(symbol)
            if info:
                asset_info.append({
                    'symbol': symbol,
                    'name': info.name,
                    'type': info.asset_type.value,
                    'weight': weights[symbols.index(symbol)]
                })
                
        if not asset_info:
            return go.Figure()
            
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[f"{ai['symbol']} ({ai['type']})" for ai in asset_info],
            values=[ai['weight'] for ai in asset_info],
            hole=0.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Portfolio Asset Allocation',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_correlation_heatmap(self, symbols: List[str], period: str = "1y") -> go.Figure:
        """Create correlation heatmap"""
        
        # Fetch data for all symbols
        data_dict = {}
        for symbol in symbols:
            data = self.analyzer.data_provider.fetch_asset_data(symbol, period)
            if not data.empty:
                data_dict[symbol] = data['Close'].pct_change().dropna()
                
        if len(data_dict) < 2:
            return go.Figure()
            
        # Create correlation matrix
        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=500,
            template='plotly_dark'
        )
        
        return fig

class MultiAssetPortfolio:
    """Multi-asset portfolio management"""
    
    def __init__(self):
        self.analyzer = MultiAssetAnalyzer()
        self.visualization = MultiAssetVisualization()
        
    def create_portfolio(self, assets: Dict[str, float]) -> Dict:
        """Create a multi-asset portfolio"""
        
        portfolio = {
            'assets': assets,
            'total_weight': sum(assets.values()),
            'asset_count': len(assets)
        }
        
        # Validate weights
        if abs(portfolio['total_weight'] - 1.0) > 0.01:
            portfolio['warning'] = f"Portfolio weights sum to {portfolio['total_weight']:.3f}, not 1.0"
            
        # Analyze each asset
        portfolio['asset_analysis'] = {}
        for symbol, weight in assets.items():
            analysis = self.analyzer.analyze_asset(symbol)
            if 'error' not in analysis:
                portfolio['asset_analysis'][symbol] = analysis
                
        # Calculate portfolio metrics
        portfolio['portfolio_metrics'] = self._calculate_portfolio_metrics(portfolio)
        
        return portfolio
        
    def _calculate_portfolio_metrics(self, portfolio: Dict) -> Dict:
        """Calculate portfolio-level metrics"""
        
        assets = portfolio['asset_analysis']
        weights = portfolio['assets']
        
        if not assets:
            return {'error': 'No valid assets in portfolio'}
            
        # Calculate weighted returns and volatility
        total_return = 0
        total_volatility = 0
        
        for symbol, analysis in assets.items():
            weight = weights.get(symbol, 0)
            total_return += weight * analysis['performance_metrics']['annualized_return']
            total_volatility += weight * analysis['performance_metrics']['volatility']
            
        # Simple portfolio metrics (in practice, you'd calculate correlation matrix)
        portfolio_metrics = {
            'expected_return': total_return,
            'expected_volatility': total_volatility,
            'sharpe_ratio': total_return / total_volatility if total_volatility > 0 else 0,
            'diversification_ratio': len(assets) / len(weights) if len(weights) > 0 else 0
        }
        
        return portfolio_metrics
        
    def optimize_portfolio(self, symbols: List[str], target_return: float = None) -> Dict:
        """Simple portfolio optimization"""
        
        # Analyze all assets
        asset_analyses = {}
        for symbol in symbols:
            analysis = self.analyzer.analyze_asset(symbol)
            if 'error' not in analysis:
                asset_analyses[symbol] = analysis
                
        if len(asset_analyses) < 2:
            return {'error': 'Need at least 2 assets for optimization'}
            
        # Simple equal-weight optimization
        equal_weight = 1.0 / len(asset_analyses)
        optimized_weights = {symbol: equal_weight for symbol in asset_analyses.keys()}
        
        # Create optimized portfolio
        optimized_portfolio = self.create_portfolio(optimized_weights)
        
        return {
            'optimized_weights': optimized_weights,
            'portfolio': optimized_portfolio,
            'optimization_method': 'equal_weight'
        }

# Global instances
multi_asset_provider = MultiAssetDataProvider()
multi_asset_analyzer = MultiAssetAnalyzer()
multi_asset_visualization = MultiAssetVisualization()
multi_asset_portfolio = MultiAssetPortfolio()
