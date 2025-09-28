"""
Real Data Integration Module
Integrates downloaded pickle data with the dashboard
"""

import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RealDataManager:
    """Manages real market data from pickle files"""
    
    def __init__(self, data_dir: str = "downloaded_data"):
        """Initialize the real data manager"""
        
        self.data_dir = data_dir
        self.complete_data = None
        self.stocks_data = None
        self.indices_data = None
        
        # Load data on initialization
        self.load_all_data()
    
    def load_all_data(self):
        """Load all pickle data files"""
        
        try:
            # Load complete market data
            complete_file = os.path.join(self.data_dir, "complete_market_data.pkl")
            if os.path.exists(complete_file):
                with open(complete_file, 'rb') as f:
                    self.complete_data = pickle.load(f)
                logger.info("Loaded complete market data")
            
            # Load stocks data
            stocks_file = os.path.join(self.data_dir, "stocks_data.pkl")
            if os.path.exists(stocks_file):
                with open(stocks_file, 'rb') as f:
                    self.stocks_data = pickle.load(f)
                logger.info("Loaded stocks data")
            
            # Load indices data
            indices_file = os.path.join(self.data_dir, "indices_data.pkl")
            if os.path.exists(indices_file):
                with open(indices_file, 'rb') as f:
                    self.indices_data = pickle.load(f)
                logger.info("Loaded indices data")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
    
    def get_symbol_data(self, symbol: str, timeframe: str = '1d', 
                       category: str = 'stocks') -> Optional[pd.DataFrame]:
        """Get data for a specific symbol and timeframe"""
        
        if not self.complete_data:
            return None
        
        try:
            if category in self.complete_data and symbol in self.complete_data[category]:
                if timeframe in self.complete_data[category][symbol]:
                    return self.complete_data[category][symbol][timeframe].copy()
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
        
        return None
    
    def get_available_symbols(self, category: str = 'stocks') -> List[str]:
        """Get list of available symbols for a category"""
        
        if not self.complete_data or category not in self.complete_data:
            return []
        
        return list(self.complete_data[category].keys())
    
    def get_available_timeframes(self, symbol: str, category: str = 'stocks') -> List[str]:
        """Get list of available timeframes for a symbol"""
        
        if not self.complete_data or category not in self.complete_data:
            return []
        
        if symbol not in self.complete_data[category]:
            return []
        
        return list(self.complete_data[category][symbol].keys())
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        
        if not self.complete_data:
            return {}
        
        summary = {}
        for category, category_data in self.complete_data.items():
            summary[category] = {}
            for symbol, symbol_data in category_data.items():
                summary[category][symbol] = {}
                for timeframe, df in symbol_data.items():
                    summary[category][symbol][timeframe] = {
                        'records': len(df),
                        'start_date': df.index.min().strftime('%Y-%m-%d'),
                        'end_date': df.index.max().strftime('%Y-%m-%d'),
                        'latest_price': df['Close'].iloc[-1] if not df.empty else None
                    }
        
        return summary
    
    def generate_enhanced_market_data(self, symbol: str, sector: str, 
                                    time_period: int, timeframe: str, 
                                    custom_start_date: str = None, custom_end_date: str = None) -> Dict:
        """Generate enhanced market data using real data as base"""
        
        # Get real data
        real_data = self.get_symbol_data(symbol, timeframe, 'stocks')
        
        if real_data is None or real_data.empty:
            logger.warning(f"No real data found for {symbol}, using fallback")
            return self._generate_fallback_data(symbol, time_period, timeframe)
        
        # Filter data based on time_period or custom dates
        if custom_start_date and custom_end_date:
            # Use custom date range
            start_date = pd.to_datetime(custom_start_date)
            end_date = pd.to_datetime(custom_end_date)
            logger.info(f"Using custom date range: {start_date} to {end_date}")
        else:
            # Use time_period
            end_date = real_data.index.max()
            start_date = end_date - timedelta(days=time_period)
            
        real_data_filtered = real_data.loc[start_date:end_date]
        
        # Debug: Check if filtering is working
        logger.info(f"DEBUG: Original data range: {real_data.index.min()} to {real_data.index.max()}")
        logger.info(f"DEBUG: Filtering for {time_period} days from {start_date} to {end_date}")
        logger.info(f"DEBUG: Filtered data range: {real_data_filtered.index.min()} to {real_data_filtered.index.max()}")
        
        if real_data_filtered.empty:
            logger.warning(f"Filtered real data for {symbol} is empty. Falling back to simulation.")
            return self._generate_fallback_data(symbol, time_period, timeframe)
        
        # Use filtered real data as base
        logger.info(f"Using real data for {symbol}: {len(real_data_filtered)} records (filtered from {len(real_data)} total)")
        
        # Calculate technical indicators on filtered real data
        prices = real_data_filtered['Close'].values
        volumes = real_data_filtered['Volume'].values
        
        # Ensure we have enough data
        if len(prices) < 50:
            logger.warning(f"Insufficient data for {symbol}, using fallback")
            return self._generate_fallback_data(symbol, time_period, timeframe)
        
        # Calculate indicators
        indicators = self._calculate_indicators(prices, volumes)
        
        # Generate signals based on real data
        signals = self._generate_signals(prices, indicators)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'dates': real_data_filtered.index.tolist(),
            'indicators': indicators,
            'signals': signals,
            'real_data': True,
            'symbol': symbol,
            'timeframe': timeframe
        }
    
    def _calculate_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Calculate technical indicators"""
        
        # Simple Moving Averages
        sma_20 = pd.Series(prices).rolling(window=20).mean().values
        sma_50 = pd.Series(prices).rolling(window=50).mean().values
        
        # RSI
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = pd.Series(prices).ewm(span=12).mean()
        ema_26 = pd.Series(prices).ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20_series = pd.Series(prices).rolling(window=20).mean()
        std_20 = pd.Series(prices).rolling(window=20).std()
        bb_upper = sma_20_series + (std_20 * 2)
        bb_lower = sma_20_series - (std_20 * 2)
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi.values,
            'macd': macd.values,
            'macd_signal': signal_line.values,
            'bb_upper': bb_upper.values,
            'bb_lower': bb_lower.values,
            'volume_sma': pd.Series(volumes).rolling(window=20).mean().values
        }
    
    def _generate_signals(self, prices: np.ndarray, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        
        signals = {
            'buy': np.zeros(len(prices)),
            'sell': np.zeros(len(prices)),
            'strength': np.zeros(len(prices))
        }
        
        # RSI signals
        rsi = indicators['rsi']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        # Buy signals: RSI < 30 and price above SMA
        buy_condition = (rsi < 30) & (prices > sma_20)
        signals['buy'][buy_condition] = 1
        
        # Sell signals: RSI > 70 and price below SMA
        sell_condition = (rsi > 70) & (prices < sma_20)
        signals['sell'][sell_condition] = 1
        
        # Signal strength based on multiple factors
        strength = np.zeros(len(prices))
        
        # RSI strength
        rsi_strength = np.where(rsi < 30, 0.3, np.where(rsi > 70, -0.3, 0))
        
        # Moving average strength
        ma_strength = np.where(prices > sma_20, 0.2, -0.2)
        
        # MACD strength
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_strength = np.where(macd > macd_signal, 0.1, -0.1)
        
        # Combine strengths
        strength = rsi_strength + ma_strength + macd_strength
        signals['strength'] = strength
        
        return signals
    
    def _generate_fallback_data(self, symbol: str, time_period: int, 
                               timeframe: str) -> Dict:
        """Generate fallback data when real data is not available"""
        
        logger.info(f"Generating fallback data for {symbol}")
        
        # Generate synthetic data
        n_days = time_period
        if timeframe == '1h':
            n_points = n_days * 24
        elif timeframe == '5m':
            n_points = n_days * 24 * 12
        else:
            n_points = n_days
        
        # Generate price data
        np.random.seed(42 + time_period)  # For reproducibility but different per period
        returns = np.random.normal(0.0005, 0.02, n_points)
        prices = 100 * np.cumprod(1 + returns)
        
        # Generate volume data
        volumes = np.random.lognormal(10, 0.5, n_points)
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=n_days)
        if timeframe == '1h':
            dates = pd.date_range(start=start_date, periods=n_points, freq='H')
        elif timeframe == '5m':
            dates = pd.date_range(start=start_date, periods=n_points, freq='5T')
        else:
            dates = pd.date_range(start=start_date, periods=n_points, freq='D')
        
        # Calculate indicators
        indicators = self._calculate_indicators(prices, volumes)
        
        # Generate signals
        signals = self._generate_signals(prices, indicators)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'dates': dates.tolist(),
            'indicators': indicators,
            'signals': signals,
            'real_data': False,
            'symbol': symbol,
            'timeframe': timeframe
        }
    
    def get_portfolio_data(self, symbols: List[str], timeframe: str = '1d') -> Dict:
        """Get portfolio data for multiple symbols"""
        
        portfolio_data = {}
        
        for symbol in symbols:
            data = self.get_symbol_data(symbol, timeframe, 'stocks')
            if data is not None and not data.empty:
                portfolio_data[symbol] = {
                    'prices': data['Close'].values,
                    'volumes': data['Volume'].values,
                    'dates': data.index.tolist(),
                    'returns': data['Close'].pct_change().dropna().values
                }
        
        return portfolio_data
    
    def get_market_overview(self) -> Dict:
        """Get market overview data"""
        
        if not self.complete_data:
            return {}
        
        overview = {}
        
        # Get latest prices for major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
        for index in indices:
            data = self.get_symbol_data(index, '1d', 'indices')
            if data is not None and not data.empty:
                latest_price = data['Close'].iloc[-1]
                change = data['Close'].pct_change().iloc[-1] * 100
                overview[index] = {
                    'price': latest_price,
                    'change': change,
                    'change_pct': change
                }
        
        # Get sector performance (using stocks as proxy)
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Consumer': ['AMZN', 'META', 'NFLX'],
            'Automotive': ['TSLA']
        }
        
        sector_performance = {}
        for sector, symbols in sectors.items():
            sector_returns = []
            for symbol in symbols:
                data = self.get_symbol_data(symbol, '1d', 'stocks')
                if data is not None and not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    sector_returns.extend(returns.values)
            
            if sector_returns:
                sector_performance[sector] = {
                    'avg_return': np.mean(sector_returns) * 100,
                    'volatility': np.std(sector_returns) * 100
                }
        
        overview['sectors'] = sector_performance
        return overview

# Global instance
real_data_manager = RealDataManager()

def get_real_data_manager() -> RealDataManager:
    """Get the global real data manager instance"""
    return real_data_manager
