"""
Load Pickle Data
Example script to load and use the downloaded pickle data
"""

import pickle
import pandas as pd
import os
from datetime import datetime

def load_pickle_data(filename: str):
    """Load data from pickle file"""
    
    filepath = os.path.join("downloaded_data", f"{filename}.pkl")
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {filename}.pkl")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None

def explore_data_structure(data, level=0, max_level=3):
    """Explore the structure of the loaded data"""
    
    if level > max_level:
        return
    
    indent = "  " * level
    
    if isinstance(data, dict):
        print(f"{indent}Dictionary with {len(data)} keys:")
        for key, value in data.items():
            print(f"{indent}  - {key}: {type(value).__name__}")
            if level < max_level:
                explore_data_structure(value, level + 1, max_level)
    elif isinstance(data, pd.DataFrame):
        print(f"{indent}DataFrame: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"{indent}  Columns: {list(data.columns)}")
        print(f"{indent}  Date range: {data.index.min()} to {data.index.max()}")
        if not data.empty:
            print(f"{indent}  Latest price: {data['Close'].iloc[-1]:.2f}")
    else:
        print(f"{indent}{type(data).__name__}: {len(data) if hasattr(data, '__len__') else 'N/A'}")

def get_symbol_data(data, symbol, timeframe='1d'):
    """Get data for a specific symbol and timeframe"""
    
    if 'stocks' in data and symbol in data['stocks']:
        if timeframe in data['stocks'][symbol]:
            return data['stocks'][symbol][timeframe]
    elif 'indices' in data and symbol in data['indices']:
        if timeframe in data['indices'][symbol]:
            return data['indices'][symbol][timeframe]
    
    print(f"Data not found for {symbol} - {timeframe}")
    return None

def calculate_returns(df):
    """Calculate returns from price data"""
    
    if df is None or df.empty:
        return None
    
    returns = df['Close'].pct_change().dropna()
    return returns

def main():
    """Main function to demonstrate data loading and usage"""
    
    print("=== Pickle Data Loader ===")
    print("Loading and exploring downloaded market data")
    print()
    
    # Load complete market data
    print("1. Loading complete market data...")
    complete_data = load_pickle_data("complete_market_data")
    
    if complete_data is None:
        print("Failed to load data. Please run api_data_downloader.py first.")
        return
    
    print("\n2. Exploring data structure...")
    explore_data_structure(complete_data, max_level=2)
    
    # Load specific category data
    print("\n3. Loading stocks data...")
    stocks_data = load_pickle_data("stocks_data")
    
    print("\n4. Loading indices data...")
    indices_data = load_pickle_data("indices_data")
    
    # Example: Get AAPL data
    print("\n5. Getting AAPL daily data...")
    aapl_data = get_symbol_data(complete_data, 'AAPL', '1d')
    
    if aapl_data is not None:
        print(f"AAPL daily data: {len(aapl_data)} records")
        print(f"Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
        print(f"Latest close price: ${aapl_data['Close'].iloc[-1]:.2f}")
        
        # Calculate returns
        returns = calculate_returns(aapl_data)
        if returns is not None:
            print(f"Average daily return: {returns.mean()*100:.2f}%")
            print(f"Daily volatility: {returns.std()*100:.2f}%")
    
    # Example: Get S&P 500 data
    print("\n6. Getting S&P 500 daily data...")
    sp500_data = get_symbol_data(complete_data, '^GSPC', '1d')
    
    if sp500_data is not None:
        print(f"S&P 500 daily data: {len(sp500_data)} records")
        print(f"Latest close: {sp500_data['Close'].iloc[-1]:.2f}")
    
    # Example: Compare multiple stocks
    print("\n7. Comparing multiple stocks...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for symbol in symbols:
        data = get_symbol_data(complete_data, symbol, '1d')
        if data is not None:
            latest_price = data['Close'].iloc[-1]
            print(f"{symbol}: ${latest_price:.2f}")
    
    print("\n=== Data Loading Complete ===")
    print("You can now use this data for:")
    print("- Backtesting trading strategies")
    print("- Technical analysis")
    print("- Portfolio optimization")
    print("- Risk analysis")
    print("- Machine learning models")

if __name__ == "__main__":
    main()
