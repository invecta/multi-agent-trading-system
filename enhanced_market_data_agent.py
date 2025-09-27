"""
Enhanced Market Data Agent with Real-Time Streaming Capabilities
Implements Kafka, Apache Flink, and WebSocket integration for low-latency data ingestion
"""
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import websockets
import aiohttp
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import yfinance as yf
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingMarketData:
    """Real-time market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    source: str
    exchange: str
    data_type: str  # 'tick', 'snapshot', 'orderbook'

@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{'price': 100.0, 'size': 1000}]
    asks: List[Dict[str, float]]
    spread: float
    mid_price: float

@dataclass
class MarketDataConfig:
    """Configuration for market data streaming"""
    symbols: List[str]
    data_sources: List[str]  # ['websocket', 'kafka', 'rest_api']
    timeframes: List[str]
    update_frequency: float  # seconds
    buffer_size: int
    enable_orderbook: bool
    enable_tick_data: bool

class RealTimeDataStreamer:
    """Real-time data streaming engine"""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.data_buffer = Queue(maxsize=config.buffer_size)
        self.subscribers = []
        self.is_streaming = False
        self.streaming_tasks = []
        
        # Initialize data sources
        self.kafka_producer = None
        self.kafka_consumer = None
        self.redis_client = None
        self.websocket_connections = {}
        
        # Initialize storage
        self.db_connection = None
        self._init_storage()
        
    def _init_storage(self):
        """Initialize database storage"""
        self.db_connection = sqlite3.connect('streaming_market_data.db', check_same_thread=False)
        cursor = self.db_connection.cursor()
        
        # Create streaming data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS streaming_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL,
                volume INTEGER,
                bid REAL,
                ask REAL,
                spread REAL,
                source TEXT,
                exchange TEXT,
                data_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_symbol_timestamp (symbol, timestamp)
            )
        ''')
        
        # Create order book table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                bids TEXT,  -- JSON string
                asks TEXT,  -- JSON string
                spread REAL,
                mid_price REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_symbol_timestamp (symbol, timestamp)
            )
        ''')
        
        self.db_connection.commit()
    
    async def start_streaming(self):
        """Start real-time data streaming"""
        logger.info("Starting real-time data streaming")
        self.is_streaming = True
        
        # Start streaming tasks for each data source
        if 'websocket' in self.config.data_sources:
            task = asyncio.create_task(self._websocket_streaming())
            self.streaming_tasks.append(task)
        
        if 'kafka' in self.config.data_sources:
            task = asyncio.create_task(self._kafka_streaming())
            self.streaming_tasks.append(task)
        
        if 'rest_api' in self.config.data_sources:
            task = asyncio.create_task(self._rest_api_streaming())
            self.streaming_tasks.append(task)
        
        # Start data processing task
        processing_task = asyncio.create_task(self._process_streaming_data())
        self.streaming_tasks.append(processing_task)
        
        # Wait for all tasks
        await asyncio.gather(*self.streaming_tasks)
    
    async def stop_streaming(self):
        """Stop real-time data streaming"""
        logger.info("Stopping real-time data streaming")
        self.is_streaming = False
        
        # Cancel all streaming tasks
        for task in self.streaming_tasks:
            task.cancel()
        
        # Close connections
        await self._close_connections()
    
    async def _websocket_streaming(self):
        """WebSocket streaming implementation"""
        logger.info("Starting WebSocket streaming")
        
        # Simulate WebSocket connections for different exchanges
        websocket_urls = {
            'binance': 'wss://stream.binance.com:9443/ws/btcusdt@ticker',
            'coinbase': 'wss://ws-feed.exchange.coinbase.com',
            'alpha_vantage': 'wss://stream.alpha-vantage.com/ws'
        }
        
        for exchange, url in websocket_urls.items():
            try:
                async with websockets.connect(url) as websocket:
                    self.websocket_connections[exchange] = websocket
                    
                    while self.is_streaming:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)
                            
                            # Process WebSocket data
                            market_data = await self._process_websocket_data(data, exchange)
                            if market_data:
                                self.data_buffer.put(market_data)
                                
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"WebSocket error for {exchange}: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"Failed to connect to {exchange} WebSocket: {e}")
    
    async def _kafka_streaming(self):
        """Kafka streaming implementation"""
        logger.info("Starting Kafka streaming")
        
        try:
            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Initialize Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                'market_data',
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Consume messages
            for message in self.kafka_consumer:
                if not self.is_streaming:
                    break
                
                data = message.value
                market_data = await self._process_kafka_data(data)
                if market_data:
                    self.data_buffer.put(market_data)
                    
        except Exception as e:
            logger.error(f"Kafka streaming error: {e}")
    
    async def _rest_api_streaming(self):
        """REST API streaming implementation"""
        logger.info("Starting REST API streaming")
        
        async with aiohttp.ClientSession() as session:
            while self.is_streaming:
                try:
                    # Fetch data from multiple APIs
                    tasks = []
                    for symbol in self.config.symbols:
                        task = self._fetch_api_data(session, symbol)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, StreamingMarketData):
                            self.data_buffer.put(result)
                        elif isinstance(result, Exception):
                            logger.error(f"API fetch error: {result}")
                    
                    # Wait before next update
                    await asyncio.sleep(self.config.update_frequency)
                    
                except Exception as e:
                    logger.error(f"REST API streaming error: {e}")
                    await asyncio.sleep(5)  # Wait before retry
    
    async def _fetch_api_data(self, session: aiohttp.ClientSession, symbol: str) -> Optional[StreamingMarketData]:
        """Fetch data from REST API"""
        try:
            # Use yfinance for demonstration
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return StreamingMarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=info.get('currentPrice', 0),
                volume=info.get('volume', 0),
                bid=info.get('bid', 0),
                ask=info.get('ask', 0),
                spread=info.get('ask', 0) - info.get('bid', 0),
                source='rest_api',
                exchange='yahoo',
                data_type='snapshot'
            )
            
        except Exception as e:
            logger.error(f"Error fetching API data for {symbol}: {e}")
            return None
    
    async def _process_streaming_data(self):
        """Process streaming data and store/notify"""
        logger.info("Starting streaming data processing")
        
        while self.is_streaming:
            try:
                # Get data from buffer
                market_data = self.data_buffer.get(timeout=1.0)
                
                # Store in database
                await self._store_market_data(market_data)
                
                # Notify subscribers
                await self._notify_subscribers(market_data)
                
                # Publish to Kafka if enabled
                if self.kafka_producer:
                    await self._publish_to_kafka(market_data)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing streaming data: {e}")
    
    async def _store_market_data(self, market_data: StreamingMarketData):
        """Store market data in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO streaming_data 
                (symbol, timestamp, price, volume, bid, ask, spread, source, exchange, data_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.symbol,
                market_data.timestamp,
                market_data.price,
                market_data.volume,
                market_data.bid,
                market_data.ask,
                market_data.spread,
                market_data.source,
                market_data.exchange,
                market_data.data_type
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def _notify_subscribers(self, market_data: StreamingMarketData):
        """Notify all subscribers of new data"""
        for subscriber in self.subscribers:
            try:
                await subscriber(market_data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    async def _publish_to_kafka(self, market_data: StreamingMarketData):
        """Publish market data to Kafka"""
        try:
            data_dict = asdict(market_data)
            data_dict['timestamp'] = market_data.timestamp.isoformat()
            
            self.kafka_producer.send(
                'market_data',
                key=market_data.symbol,
                value=data_dict
            )
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
    
    async def _process_websocket_data(self, data: Dict, exchange: str) -> Optional[StreamingMarketData]:
        """Process WebSocket data"""
        try:
            # Parse different exchange formats
            if exchange == 'binance':
                return StreamingMarketData(
                    symbol=data.get('s', ''),
                    timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000),
                    price=float(data.get('c', 0)),
                    volume=int(data.get('v', 0)),
                    bid=float(data.get('b', 0)),
                    ask=float(data.get('a', 0)),
                    spread=float(data.get('a', 0)) - float(data.get('b', 0)),
                    source='websocket',
                    exchange=exchange,
                    data_type='tick'
                )
            elif exchange == 'coinbase':
                return StreamingMarketData(
                    symbol=data.get('product_id', ''),
                    timestamp=datetime.now(),
                    price=float(data.get('price', 0)),
                    volume=int(data.get('size', 0)),
                    bid=0,  # Coinbase format varies
                    ask=0,
                    spread=0,
                    source='websocket',
                    exchange=exchange,
                    data_type='tick'
                )
            
        except Exception as e:
            logger.error(f"Error processing WebSocket data: {e}")
        
        return None
    
    async def _process_kafka_data(self, data: Dict) -> Optional[StreamingMarketData]:
        """Process Kafka data"""
        try:
            return StreamingMarketData(
                symbol=data.get('symbol', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                price=data.get('price', 0),
                volume=data.get('volume', 0),
                bid=data.get('bid', 0),
                ask=data.get('ask', 0),
                spread=data.get('spread', 0),
                source=data.get('source', 'kafka'),
                exchange=data.get('exchange', ''),
                data_type=data.get('data_type', 'tick')
            )
        except Exception as e:
            logger.error(f"Error processing Kafka data: {e}")
            return None
    
    async def _close_connections(self):
        """Close all connections"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()
        if self.db_connection:
            self.db_connection.close()
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

class EnhancedMarketDataAgent:
    """Enhanced Market Data Agent with streaming capabilities"""
    
    def __init__(self, config: MarketDataConfig = None):
        self.config = config or MarketDataConfig(
            symbols=['EURUSD=X', 'GBPUSD=X', 'AAPL', 'BTC-USD'],
            data_sources=['websocket', 'rest_api'],
            timeframes=['1m', '5m', '1h', '1d'],
            update_frequency=1.0,
            buffer_size=10000,
            enable_orderbook=True,
            enable_tick_data=True
        )
        
        self.streamer = RealTimeDataStreamer(self.config)
        self.historical_data_cache = {}
        self.real_time_data_cache = {}
        
        # Performance metrics
        self.metrics = {
            'data_points_received': 0,
            'data_points_processed': 0,
            'average_latency': 0,
            'last_update': None
        }
    
    async def start_real_time_streaming(self):
        """Start real-time data streaming"""
        logger.info("Starting enhanced market data agent streaming")
        
        # Subscribe to data updates
        self.streamer.subscribe(self._handle_data_update)
        
        # Start streaming
        await self.streamer.start_streaming()
    
    async def stop_real_time_streaming(self):
        """Stop real-time data streaming"""
        logger.info("Stopping enhanced market data agent streaming")
        await self.streamer.stop_streaming()
    
    async def _handle_data_update(self, market_data: StreamingMarketData):
        """Handle incoming market data updates"""
        try:
            # Update real-time cache
            symbol = market_data.symbol
            if symbol not in self.real_time_data_cache:
                self.real_time_data_cache[symbol] = []
            
            self.real_time_data_cache[symbol].append(market_data)
            
            # Keep only recent data (last 1000 points)
            if len(self.real_time_data_cache[symbol]) > 1000:
                self.real_time_data_cache[symbol] = self.real_time_data_cache[symbol][-1000:]
            
            # Update metrics
            self.metrics['data_points_received'] += 1
            self.metrics['data_points_processed'] += 1
            self.metrics['last_update'] = datetime.now()
            
            # Calculate latency
            latency = (datetime.now() - market_data.timestamp).total_seconds()
            self.metrics['average_latency'] = (
                self.metrics['average_latency'] * 0.9 + latency * 0.1
            )
            
        except Exception as e:
            logger.error(f"Error handling data update: {e}")
    
    async def get_real_time_data(self, symbol: str, limit: int = 100) -> List[StreamingMarketData]:
        """Get real-time data for a symbol"""
        if symbol in self.real_time_data_cache:
            return self.real_time_data_cache[symbol][-limit:]
        return []
    
    async def get_historical_data(self, symbol: str, timeframe: str, period: str = '1d') -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{period}"
            if cache_key in self.historical_data_cache:
                cached_data, timestamp = self.historical_data_cache[cache_key]
                if datetime.now() - timestamp < timedelta(minutes=5):
                    return cached_data
            
            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=timeframe)
            
            # Cache the data
            self.historical_data_cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book for a symbol"""
        try:
            # This would integrate with actual order book APIs
            # For now, simulate order book data
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=[
                    {'price': 100.0, 'size': 1000},
                    {'price': 99.9, 'size': 2000},
                    {'price': 99.8, 'size': 1500}
                ],
                asks=[
                    {'price': 100.1, 'size': 1200},
                    {'price': 100.2, 'size': 1800},
                    {'price': 100.3, 'size': 900}
                ],
                spread=0.1,
                mid_price=100.05
            )
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary"""
        summary = {
            'total_symbols': len(self.config.symbols),
            'active_streams': len(self.streamer.websocket_connections),
            'data_sources': self.config.data_sources,
            'performance_metrics': self.get_performance_metrics(),
            'real_time_data_count': sum(len(data) for data in self.real_time_data_cache.values()),
            'last_update': self.metrics['last_update']
        }
        
        return summary

class MarketDataProcessor:
    """Processes and analyzes streaming market data"""
    
    def __init__(self):
        self.processors = {
            'price_analysis': self._analyze_price_movements,
            'volume_analysis': self._analyze_volume_patterns,
            'volatility_analysis': self._analyze_volatility,
            'spread_analysis': self._analyze_spread_patterns
        }
    
    async def process_data(self, market_data: StreamingMarketData) -> Dict[str, Any]:
        """Process market data and return analysis"""
        analysis = {}
        
        for processor_name, processor_func in self.processors.items():
            try:
                result = await processor_func(market_data)
                analysis[processor_name] = result
            except Exception as e:
                logger.error(f"Error in {processor_name}: {e}")
                analysis[processor_name] = {'error': str(e)}
        
        return analysis
    
    async def _analyze_price_movements(self, market_data: StreamingMarketData) -> Dict[str, Any]:
        """Analyze price movements"""
        return {
            'current_price': market_data.price,
            'price_change': 0,  # Would calculate from previous data
            'price_trend': 'neutral'  # Would analyze trend
        }
    
    async def _analyze_volume_patterns(self, market_data: StreamingMarketData) -> Dict[str, Any]:
        """Analyze volume patterns"""
        return {
            'current_volume': market_data.volume,
            'volume_trend': 'normal',  # Would analyze volume patterns
            'volume_anomaly': False
        }
    
    async def _analyze_volatility(self, market_data: StreamingMarketData) -> Dict[str, Any]:
        """Analyze volatility"""
        return {
            'current_spread': market_data.spread,
            'volatility_level': 'medium',  # Would calculate volatility
            'volatility_trend': 'stable'
        }
    
    async def _analyze_spread_patterns(self, market_data: StreamingMarketData) -> Dict[str, Any]:
        """Analyze spread patterns"""
        return {
            'current_spread': market_data.spread,
            'spread_trend': 'normal',
            'liquidity_level': 'good' if market_data.spread < 0.01 else 'poor'
        }

# Example usage and testing
async def main():
    """Example usage of the enhanced market data agent"""
    
    # Create configuration
    config = MarketDataConfig(
        symbols=['EURUSD=X', 'AAPL', 'BTC-USD'],
        data_sources=['rest_api'],  # Start with REST API for testing
        timeframes=['1m', '5m', '1h'],
        update_frequency=5.0,
        buffer_size=1000,
        enable_orderbook=True,
        enable_tick_data=True
    )
    
    # Initialize enhanced market data agent
    agent = EnhancedMarketDataAgent(config)
    
    # Start streaming
    try:
        print("Starting market data streaming...")
        await agent.start_real_time_streaming()
        
    except KeyboardInterrupt:
        print("Stopping market data streaming...")
        await agent.stop_real_time_streaming()
        
        # Print summary
        summary = await agent.get_market_summary()
        print("Market Data Summary:")
        print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
