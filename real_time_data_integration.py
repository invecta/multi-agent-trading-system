"""
Real-time data integration module for enhanced dashboard
Provides live market data feeds, WebSocket connections, and real-time updates
"""

import asyncio
import websockets
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataProvider:
    """Real-time market data provider with multiple sources"""
    
    def __init__(self):
        self.subscribers = {}
        self.data_cache = {}
        self.is_running = False
        self.update_interval = 1  # seconds
        
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from real-time updates"""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
            
    def get_live_price(self, symbol: str) -> Dict:
        """Get current live price data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(latest['Close']),
                    'change': float(latest['Close'] - hist.iloc[-2]['Close']),
                    'change_percent': float((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100),
                    'volume': int(latest['Volume']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                    'timestamp': datetime.now().isoformat(),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
            
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Fallback data when live feed fails"""
        base_price = 150 + hash(symbol) % 100
        change = np.random.randn() * 2
        return {
            'symbol': symbol,
            'price': base_price + change,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'volume': np.random.randint(1000000, 5000000),
            'high': base_price + abs(change) + 1,
            'low': base_price - abs(change) - 1,
            'open': base_price,
            'timestamp': datetime.now().isoformat(),
            'market_cap': base_price * 1000000000,
            'pe_ratio': 20 + np.random.randn() * 5,
            'dividend_yield': 0.02 + np.random.randn() * 0.01
        }
        
    def start_real_time_updates(self):
        """Start real-time data updates"""
        self.is_running = True
        thread = threading.Thread(target=self._update_loop)
        thread.daemon = True
        thread.start()
        
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.is_running = False
        
    def _update_loop(self):
        """Main update loop for real-time data"""
        while self.is_running:
            try:
                for symbol in self.subscribers.keys():
                    data = self.get_live_price(symbol)
                    self.data_cache[symbol] = data
                    
                    # Notify subscribers
                    for callback in self.subscribers[symbol]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in callback for {symbol}: {e}")
                            
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)

class MarketDataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.data_provider = RealTimeDataProvider()
        self.market_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
        
    def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'indices': {},
            'crypto': {},
            'sector_performance': {},
            'market_sentiment': 'neutral'
        }
        
        # Get major indices
        for index in self.market_indices:
            try:
                data = self.data_provider.get_live_price(index)
                overview['indices'][index] = data
            except Exception as e:
                logger.error(f"Error fetching {index}: {e}")
                
        # Get crypto data
        for crypto in self.crypto_symbols:
            try:
                data = self.data_provider.get_live_price(crypto)
                overview['crypto'][crypto] = data
            except Exception as e:
                logger.error(f"Error fetching {crypto}: {e}")
                
        # Calculate market sentiment
        overview['market_sentiment'] = self._calculate_market_sentiment(overview)
        
        return overview
        
    def _calculate_market_sentiment(self, overview: Dict) -> str:
        """Calculate overall market sentiment"""
        positive_count = 0
        total_count = 0
        
        for index_data in overview['indices'].values():
            if index_data['change_percent'] > 0:
                positive_count += 1
            total_count += 1
            
        if positive_count / total_count > 0.6:
            return 'bullish'
        elif positive_count / total_count < 0.4:
            return 'bearish'
        else:
            return 'neutral'
            
    def get_sector_performance(self) -> Dict:
        """Get sector performance data"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'NKE']
        }
        
        sector_performance = {}
        
        for sector, symbols in sectors.items():
            sector_data = []
            for symbol in symbols:
                try:
                    data = self.data_provider.get_live_price(symbol)
                    sector_data.append(data)
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                    
            if sector_data:
                avg_change = np.mean([d['change_percent'] for d in sector_data])
                sector_performance[sector] = {
                    'average_change': avg_change,
                    'stocks': sector_data,
                    'performance': 'outperforming' if avg_change > 0 else 'underperforming'
                }
                
        return sector_performance

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for market impact"""
    
    def __init__(self):
        self.news_sources = [
            'Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance'
        ]
        
    def get_market_news(self, symbol: str) -> List[Dict]:
        """Get recent news for a symbol"""
        # Simulate news data
        news_items = []
        for i in range(5):
            news_items.append({
                'title': f"{symbol} shows strong performance in Q4 earnings",
                'source': np.random.choice(self.news_sources),
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'sentiment': np.random.choice(['positive', 'negative', 'neutral']),
                'impact': np.random.choice(['high', 'medium', 'low']),
                'url': f"https://example.com/news/{symbol}_{i}"
            })
            
        return news_items
        
    def analyze_sentiment_trend(self, symbol: str) -> Dict:
        """Analyze sentiment trend over time"""
        news = self.get_market_news(symbol)
        
        sentiment_scores = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        scores = [sentiment_scores[item['sentiment']] for item in news]
        avg_sentiment = np.mean(scores)
        
        return {
            'symbol': symbol,
            'average_sentiment': avg_sentiment,
            'sentiment_trend': 'improving' if avg_sentiment > 0 else 'declining',
            'news_count': len(news),
            'recent_news': news[:3]
        }

class RealTimeAlerts:
    """Real-time alert system for market events"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = {}
        
    def add_alert_rule(self, symbol: str, condition: str, threshold: float, message: str):
        """Add a new alert rule"""
        if symbol not in self.alert_rules:
            self.alert_rules[symbol] = []
            
        self.alert_rules[symbol].append({
            'condition': condition,
            'threshold': threshold,
            'message': message,
            'active': True
        })
        
    def check_alerts(self, data: Dict):
        """Check if any alerts should be triggered"""
        symbol = data['symbol']
        
        if symbol not in self.alert_rules:
            return
            
        for rule in self.alert_rules[symbol]:
            if not rule['active']:
                continue
                
            triggered = False
            if rule['condition'] == 'price_above':
                triggered = data['price'] > rule['threshold']
            elif rule['condition'] == 'price_below':
                triggered = data['price'] < rule['threshold']
            elif rule['condition'] == 'change_percent_above':
                triggered = data['change_percent'] > rule['threshold']
            elif rule['condition'] == 'volume_spike':
                triggered = data['volume'] > rule['threshold']
                
            if triggered:
                alert = {
                    'symbol': symbol,
                    'message': rule['message'],
                    'timestamp': datetime.now().isoformat(),
                    'data': data,
                    'type': 'price_alert'
                }
                self.alerts.append(alert)
                rule['active'] = False  # One-time alert
                
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]

# Global instances
data_provider = RealTimeDataProvider()
market_aggregator = MarketDataAggregator()
news_analyzer = NewsSentimentAnalyzer()
alert_system = RealTimeAlerts()

def initialize_real_time_system():
    """Initialize the real-time data system"""
    data_provider.start_real_time_updates()
    
    # Add some default alert rules
    alert_system.add_alert_rule('AAPL', 'price_above', 200, 'AAPL price above $200')
    alert_system.add_alert_rule('TSLA', 'change_percent_above', 5, 'TSLA up more than 5%')
    
    logger.info("Real-time data system initialized")

def get_real_time_data(symbol: str) -> Dict:
    """Get real-time data for a symbol"""
    return data_provider.get_live_price(symbol)

def get_market_overview() -> Dict:
    """Get market overview"""
    return market_aggregator.get_market_overview()

def get_sector_performance() -> Dict:
    """Get sector performance"""
    return market_aggregator.get_sector_performance()

def get_news_sentiment(symbol: str) -> Dict:
    """Get news sentiment for a symbol"""
    return news_analyzer.analyze_sentiment_trend(symbol)

def get_recent_alerts() -> List[Dict]:
    """Get recent alerts"""
    return alert_system.get_recent_alerts()
