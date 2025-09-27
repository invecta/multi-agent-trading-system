"""
Financial Data Collector for Trading Strategy Analysis
Collects market data from multiple sources for Goldbach strategy analysis
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import Dict, List, Optional
import time
from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collects and stores financial market data for analysis"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.config = Config()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timeframe TEXT,
                asset_class TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        # Create economic indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                value REAL,
                timestamp DATETIME NOT NULL,
                country TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON market_data(timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_asset_class ON market_data(asset_class)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def collect_market_data(self, symbols: List[str], timeframe: str = '1d', 
                          period: str = '2y', asset_class: str = 'unknown') -> pd.DataFrame:
        """
        Collect market data for given symbols
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            asset_class: Asset class category (forex, indices, commodities, crypto)
        
        Returns:
            DataFrame with market data
        """
        all_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting data for {symbol} ({timeframe}, {period})")
                
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=timeframe)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Add metadata
                data['symbol'] = symbol
                data['timeframe'] = timeframe
                data['asset_class'] = asset_class
                data.reset_index(inplace=True)
                
                # Rename columns to match database schema
                data.rename(columns={'Date': 'timestamp'}, inplace=True)
                
                # Ensure column names match our database schema (lowercase)
                column_mapping = {
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                data.rename(columns=column_mapping, inplace=True)
                
                # Remove any extra columns that aren't in our schema
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'asset_class']
                extra_columns = [col for col in data.columns if col not in required_columns]
                if extra_columns:
                    data = data.drop(columns=extra_columns)
                
                all_data.append(data)
                
                # Store in database
                self._store_market_data(data)
                
                # Rate limiting to avoid API limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully collected data for {len(combined_data)} records")
            return combined_data
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def _store_market_data(self, data: pd.DataFrame):
        """Store market data in SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            data.to_sql('market_data', conn, if_exists='append', index=False)
            conn.commit()
            logger.debug(f"Stored {len(data)} records in database")
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
        finally:
            conn.close()
    
    def get_market_data(self, symbol: str, timeframe: str = '1d', 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve market data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with market data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume, symbol, timeframe, asset_class
            FROM market_data 
            WHERE symbol = ? AND timeframe = ?
        '''
        params = [symbol, timeframe]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp'
        
        try:
            data = pd.read_sql_query(query, conn, params=params)
            if not data.empty:
                data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
                data.set_index('timestamp', inplace=True)
            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def collect_all_default_data(self):
        """Collect data for all default assets across all timeframes"""
        logger.info("Starting comprehensive data collection")
        
        for asset_class, symbols in self.config.DEFAULT_ASSETS.items():
            logger.info(f"Collecting {asset_class} data")
            
            for timeframe in self.config.DEFAULT_TIMEFRAMES:
                try:
                    self.collect_market_data(
                        symbols=symbols,
                        timeframe=timeframe,
                        period='2y',
                        asset_class=asset_class
                    )
                except Exception as e:
                    logger.error(f"Error collecting {asset_class} data for {timeframe}: {str(e)}")
        
        logger.info("Data collection completed")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all symbols in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT symbol FROM market_data ORDER BY symbol')
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return symbols
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts by asset class
        cursor.execute('''
            SELECT asset_class, COUNT(*) as count 
            FROM market_data 
            GROUP BY asset_class
        ''')
        asset_counts = dict(cursor.fetchall())
        
        # Get counts by timeframe
        cursor.execute('''
            SELECT timeframe, COUNT(*) as count 
            FROM market_data 
            GROUP BY timeframe
        ''')
        timeframe_counts = dict(cursor.fetchall())
        
        # Get date range
        cursor.execute('''
            SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
            FROM market_data
        ''')
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'asset_class_counts': asset_counts,
            'timeframe_counts': timeframe_counts,
            'date_range': {
                'start': date_range[0],
                'end': date_range[1]
            },
            'total_records': sum(asset_counts.values())
        }

# Example usage
if __name__ == "__main__":
    collector = FinancialDataCollector()
    
    # Collect sample data
    print("Collecting sample forex data...")
    forex_data = collector.collect_market_data(
        symbols=['EURUSD=X', 'GBPUSD=X'],
        timeframe='1d',
        period='1y',
        asset_class='forex'
    )
    
    print(f"Collected {len(forex_data)} records")
    print(forex_data.head())
    
    # Get data summary
    summary = collector.get_data_summary()
    print("\nData Summary:")
    print(summary)
