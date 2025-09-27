#!/usr/bin/env python3
"""
Collect data for multiple timeframes to support dashboard
"""

import logging
from data_collector import FinancialDataCollector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_multi_timeframe_data():
    """Collect data for multiple timeframes"""
    collector = FinancialDataCollector()
    
    # Symbols to collect
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', '^GSPC', 'GC=F', 'BTC-USD']
    
    # Timeframes to collect (focusing on available ones)
    timeframes = ['1d', '1h', '4h']  # Start with these as they're more likely to work
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logger.info(f"Collecting data for {symbol} ({timeframe})")
                data = collector.get_market_data(symbol, timeframe)
                
                if not data.empty:
                    logger.info(f"Successfully collected {len(data)} records for {symbol} ({timeframe})")
                else:
                    logger.warning(f"No data available for {symbol} ({timeframe})")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol} ({timeframe}): {str(e)}")
    
    # Verify what we have
    logger.info("\nFinal database summary:")
    # Use the verify_data script instead
    import subprocess
    subprocess.run(['python', 'verify_data.py'])

if __name__ == "__main__":
    collect_multi_timeframe_data()
