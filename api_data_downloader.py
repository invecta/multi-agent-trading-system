"""
API Data Downloader
Downloads market data from Alpaca and Polygon.io APIs and saves as pickle files
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIDataDownloader:
    """Download market data from various APIs and save as pickle files"""
    
    def __init__(self):
        """Initialize the data downloader with API configurations"""
        
        # Alpaca API Configuration
        self.alpaca_api_key = "PKOEKMI4RY0LHF565WDO"
        self.alpaca_secret_key = "Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb"
        self.alpaca_base_url = "https://paper-api.alpaca.markets"
        self.alpaca_data_url = "https://data.alpaca.markets"
        
        # Polygon.io API Configuration
        self.polygon_api_key = "SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP"
        self.polygon_base_url = "https://api.polygon.io"
        
        # Data storage directory
        self.data_dir = "downloaded_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Symbols to download
        self.symbols = {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F']
        }
        
        # Timeframes
        self.timeframes = ['1d', '1h', '5m', '1m']
        
    def download_alpaca_data(self, symbol: str, timeframe: str = '1Day', 
                           start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Download data from Alpaca API"""
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{self.alpaca_data_url}/v2/stocks/{symbol}/bars"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }
            
            params = {
                'start': start_date,
                'end': end_date,
                'timeframe': timeframe,
                'limit': 5000,
                'asof': None,
                'feed': 'iex',
                'page_token': None,
                'sort': 'asc',
                'order': 'asc'
            }
            
            logger.info(f"Downloading {symbol} data from Alpaca...")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'bars' in data and data['bars']:
                    df = pd.DataFrame(data['bars'])
                    df['timestamp'] = pd.to_datetime(df['t'])
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={
                        'o': 'Open', 'h': 'High', 'l': 'Low', 
                        'c': 'Close', 'v': 'Volume'
                    }, inplace=True)
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                else:
                    logger.warning(f"No data returned for {symbol}")
                    return None
            else:
                logger.error(f"Alpaca API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {symbol} from Alpaca: {str(e)}")
            return None
    
    def download_polygon_data(self, symbol: str, timeframe: str = 'day',
                            start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Download data from Polygon.io API"""
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Convert timeframe
            timeframe_map = {
                '1m': 'minute', '5m': 'minute', '1h': 'hour', '1d': 'day'
            }
            polygon_timeframe = timeframe_map.get(timeframe, 'day')
            
            url = f"{self.polygon_base_url}/v2/aggs/ticker/{symbol}/range/1/{polygon_timeframe}/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apikey': self.polygon_api_key
            }
            
            logger.info(f"Downloading {symbol} data from Polygon.io...")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={
                        'o': 'Open', 'h': 'High', 'l': 'Low', 
                        'c': 'Close', 'v': 'Volume'
                    }, inplace=True)
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                else:
                    logger.warning(f"No data returned for {symbol}")
                    return None
            else:
                logger.error(f"Polygon API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {symbol} from Polygon: {str(e)}")
            return None
    
    def download_yfinance_data(self, symbol: str, timeframe: str = '1d',
                              start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance as fallback"""
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Downloading {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if not df.empty:
                df.columns = [col.title() for col in df.columns]
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {symbol} from Yahoo Finance: {str(e)}")
            return None
    
    def download_symbol_data(self, symbol: str, category: str, 
                           timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Download data for a single symbol across multiple timeframes"""
        
        if timeframes is None:
            timeframes = self.timeframes
        
        symbol_data = {}
        
        for timeframe in timeframes:
            logger.info(f"Downloading {symbol} - {timeframe}")
            
            # Try Alpaca first
            df = self.download_alpaca_data(symbol, timeframe)
            
            # Fallback to Polygon if Alpaca fails
            if df is None:
                df = self.download_polygon_data(symbol, timeframe)
            
            # Fallback to Yahoo Finance if both fail
            if df is None:
                df = self.download_yfinance_data(symbol, timeframe)
            
            if df is not None and not df.empty:
                symbol_data[timeframe] = df
                logger.info(f"Successfully downloaded {symbol} - {timeframe}: {len(df)} records")
            else:
                logger.warning(f"Failed to download {symbol} - {timeframe}")
            
            # Rate limiting
            time.sleep(0.1)
        
        return symbol_data
    
    def save_data_as_pickle(self, data: Dict, filename: str) -> bool:
        """Save data as pickle file"""
        
        try:
            filepath = os.path.join(self.data_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")
            return False
    
    def load_data_from_pickle(self, filename: str) -> Optional[Dict]:
        """Load data from pickle file"""
        
        try:
            filepath = os.path.join(self.data_dir, f"{filename}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Data loaded from {filepath}")
                return data
            else:
                logger.warning(f"File {filepath} not found")
                return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return None
    
    def download_all_data(self, categories: List[str] = None, 
                         timeframes: List[str] = None) -> Dict:
        """Download data for all symbols and categories"""
        
        if categories is None:
            categories = list(self.symbols.keys())
        
        if timeframes is None:
            timeframes = self.timeframes
        
        all_data = {}
        
        for category in categories:
            if category not in self.symbols:
                continue
                
            logger.info(f"Downloading {category} data...")
            category_data = {}
            
            for symbol in self.symbols[category]:
                symbol_data = self.download_symbol_data(symbol, category, timeframes)
                if symbol_data:
                    category_data[symbol] = symbol_data
                
                # Rate limiting between symbols
                time.sleep(0.5)
            
            if category_data:
                all_data[category] = category_data
                # Save category data
                self.save_data_as_pickle(category_data, f"{category}_data")
        
        # Save complete dataset
        if all_data:
            self.save_data_as_pickle(all_data, "complete_market_data")
            logger.info("Complete market data saved successfully")
        
        return all_data
    
    def get_data_summary(self, data: Dict) -> pd.DataFrame:
        """Generate summary of downloaded data"""
        
        summary_data = []
        
        for category, category_data in data.items():
            for symbol, symbol_data in category_data.items():
                for timeframe, df in symbol_data.items():
                    summary_data.append({
                        'Category': category,
                        'Symbol': symbol,
                        'Timeframe': timeframe,
                        'Records': len(df),
                        'Start_Date': df.index.min().strftime('%Y-%m-%d'),
                        'End_Date': df.index.max().strftime('%Y-%m-%d'),
                        'Latest_Price': df['Close'].iloc[-1] if not df.empty else None
                    })
        
        return pd.DataFrame(summary_data)
    
    def test_api_connections(self) -> Dict[str, bool]:
        """Test API connections"""
        
        results = {}
        
        # Test Alpaca
        try:
            url = f"{self.alpaca_base_url}/v2/account"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }
            response = requests.get(url, headers=headers)
            results['Alpaca'] = response.status_code == 200
        except:
            results['Alpaca'] = False
        
        # Test Polygon
        try:
            url = f"{self.polygon_base_url}/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02"
            params = {'apikey': self.polygon_api_key}
            response = requests.get(url, params=params)
            results['Polygon'] = response.status_code == 200
        except:
            results['Polygon'] = False
        
        # Test Yahoo Finance
        try:
            ticker = yf.Ticker("AAPL")
            df = ticker.history(period="1d")
            results['Yahoo Finance'] = not df.empty
        except:
            results['Yahoo Finance'] = False
        
        return results

def main():
    """Main function to run the data downloader"""
    
    print("=== API Data Downloader ===")
    print("Downloading market data from Alpaca, Polygon.io, and Yahoo Finance")
    print("Saving data as pickle files for efficient storage and loading")
    print()
    
    # Initialize downloader
    downloader = APIDataDownloader()
    
    # Test API connections
    print("Testing API connections...")
    api_status = downloader.test_api_connections()
    for api, status in api_status.items():
        print(f"{api}: {'Connected' if status else 'Failed'}")
    print()
    
    # Download data
    print("Starting data download...")
    print("This may take several minutes depending on the amount of data...")
    print()
    
    # Download data for specific categories (you can modify this)
    categories_to_download = ['stocks', 'indices']  # Start with these
    timeframes_to_download = ['1d', '1h']  # Start with daily and hourly
    
    all_data = downloader.download_all_data(
        categories=categories_to_download,
        timeframes=timeframes_to_download
    )
    
    # Generate summary
    if all_data:
        summary = downloader.get_data_summary(all_data)
        print("\n=== Data Download Summary ===")
        print(summary.to_string(index=False))
        
        # Save summary
        summary.to_csv(os.path.join(downloader.data_dir, "data_summary.csv"), index=False)
        print(f"\nSummary saved to {downloader.data_dir}/data_summary.csv")
        
        print(f"\nAll data saved to {downloader.data_dir}/ directory")
        print("Files created:")
        for file in os.listdir(downloader.data_dir):
            if file.endswith('.pkl'):
                filepath = os.path.join(downloader.data_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
    else:
        print("No data was downloaded. Please check your API keys and internet connection.")

if __name__ == "__main__":
    main()
