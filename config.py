"""
Configuration file for Financial Trading Strategy Analysis Project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the financial analysis project"""
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    
    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///financial_data.db')
    
    # Data Collection Settings
    DEFAULT_TIMEFRAMES = ['1min', '5min', '15min', '1h', '4h', '1d']
    DEFAULT_ASSETS = {
        'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
        'indices': ['^GSPC', '^IXIC', '^DJI', '^FTSE'],
        'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F'],
        'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
    }
    
    # Goldbach Trading Parameters
    GOLDBACH_PARAMETERS = {
        'power_of_three_base': 3,
        'goldbach_levels': 6,
        'lookback_period': 9,
        'dealing_range_multiplier': 1.0
    }
    
    # Analysis Settings
    BACKTEST_START_DATE = '2020-01-01'
    BACKTEST_END_DATE = '2024-12-31'
    INITIAL_CAPITAL = 100000  # $100,000
    
    # Risk Management
    MAX_POSITION_SIZE = 0.02  # 2% of portfolio per trade
    STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
    TAKE_PROFIT_RATIO = 2.0  # 2:1 risk-reward ratio
    
    # Dashboard Settings
    DASHBOARD_PORT = 8050
    DASHBOARD_HOST = '127.0.0.1'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'financial_analysis.log'
