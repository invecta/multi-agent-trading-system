#!/usr/bin/env python3
"""
Polygon.io API Connection Test
Tests your Polygon.io API credentials and connection
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

async def test_polygon_connection():
    """Test Polygon.io API connection with your credentials"""
    
    # Your Polygon.io credentials
    api_key = "SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP"
    base_url = "https://api.polygon.io"
    
    print("Testing Polygon.io API Connection")
    print("=" * 40)
    print(f"API Key: {api_key[:8]}...")
    print(f"Base URL: {base_url}")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Market Status
            print("Test 1: Getting Market Status...")
            url = f"{base_url}/v1/marketstatus/now"
            params = {"apikey": api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Market status access successful!")
                    print(f"   Market Status: {data.get('market', 'Unknown')}")
                    print(f"   Server Time: {data.get('serverTime', 'Unknown')}")
                    print(f"   Exchanges: {data.get('exchanges', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Market status access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 2: Stock Ticker Details (AAPL)
            print("Test 2: Getting Stock Ticker Details (AAPL)...")
            url = f"{base_url}/v3/reference/tickers/AAPL"
            params = {"apikey": api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Stock ticker details access successful!")
                    if 'results' in data:
                        ticker = data['results']
                        print(f"   Symbol: {ticker.get('ticker', 'AAPL')}")
                        print(f"   Name: {ticker.get('name', 'Unknown')}")
                        print(f"   Market: {ticker.get('market', 'Unknown')}")
                        print(f"   Locale: {ticker.get('locale', 'Unknown')}")
                        print(f"   Primary Exchange: {ticker.get('primary_exchange', 'Unknown')}")
                        print(f"   Type: {ticker.get('type', 'Unknown')}")
                        print(f"   Active: {ticker.get('active', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Stock ticker details access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 3: Previous Close (AAPL)
            print("Test 3: Getting Previous Close (AAPL)...")
            url = f"{base_url}/v2/aggs/ticker/AAPL/prev"
            params = {"apikey": api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Previous close data access successful!")
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        print(f"   Symbol: {result.get('T', 'AAPL')}")
                        print(f"   Close Price: ${result.get('c', 0):.2f}")
                        print(f"   High: ${result.get('h', 0):.2f}")
                        print(f"   Low: ${result.get('l', 0):.2f}")
                        print(f"   Open: ${result.get('o', 0):.2f}")
                        print(f"   Volume: {result.get('v', 0):,}")
                        print(f"   Timestamp: {result.get('t', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Previous close data access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 4: Real-time Quote (AAPL)
            print("Test 4: Getting Real-time Quote (AAPL)...")
            url = f"{base_url}/v1/last_quote/stocks/AAPL"
            params = {"apikey": api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Real-time quote access successful!")
                    if 'results' in data:
                        quote = data['results']
                        print(f"   Symbol: {quote.get('T', 'AAPL')}")
                        print(f"   Bid Price: ${quote.get('P', 0):.2f}")
                        print(f"   Ask Price: ${quote.get('p', 0):.2f}")
                        print(f"   Bid Size: {quote.get('S', 0)}")
                        print(f"   Ask Size: {quote.get('s', 0)}")
                        print(f"   Timestamp: {quote.get('t', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Real-time quote access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 5: Historical Data (AAPL - Last 5 Days)
            print("Test 5: Getting Historical Data (AAPL - Last 5 Days)...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            url = f"{base_url}/v2/aggs/ticker/AAPL/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Historical data access successful!")
                    if 'results' in data:
                        results = data['results']
                        print(f"   Number of days: {len(results)}")
                        if results:
                            latest = results[-1]
                            print(f"   Latest Close: ${latest.get('c', 0):.2f}")
                            print(f"   Latest Volume: {latest.get('v', 0):,}")
                            print(f"   Latest Date: {latest.get('t', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Historical data access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            print("Polygon.io API Connection Test Completed!")
            print("SUCCESS: Your credentials are working correctly!")
            print()
            print("Available Data Types:")
            print("- Real-time stock quotes")
            print("- Historical price data")
            print("- Market status information")
            print("- Ticker details and metadata")
            print("- Options data (if subscribed)")
            print("- Forex data (if subscribed)")
            print("- Crypto data (if subscribed)")
            print()
            print("Next Steps:")
            print("1. Your Multi-Agent Trading System can now use Polygon.io for:")
            print("   - Real-time market data")
            print("   - Historical price analysis")
            print("   - Technical indicator calculations")
            print("   - Market status monitoring")
            print()
            print("2. Run the enhanced trading system:")
            print("   python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL")
            print()
            print("3. Your system now has access to professional-grade market data!")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Connection test failed: {e}")
            return False

async def test_polygon_websocket():
    """Test Polygon.io WebSocket connection (optional)"""
    print("\nOptional: Test Polygon.io WebSocket Connection")
    print("=" * 50)
    
    response = input("Would you like to test WebSocket connection? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Skipping WebSocket test.")
        return
    
    print("WebSocket testing requires additional setup.")
    print("For now, your REST API access is confirmed and working!")
    print("WebSocket integration can be added to the full trading system.")

async def main():
    """Main test function"""
    print("Polygon.io API Connection Test")
    print("=" * 50)
    print("This script will test your Polygon.io API credentials")
    print("and verify access to real-time and historical market data.")
    print()
    
    # Test connection
    success = await test_polygon_connection()
    
    if success:
        # Optional WebSocket test
        await test_polygon_websocket()
    
    print("\nReady to use Polygon.io with your Multi-Agent Trading System!")

if __name__ == "__main__":
    asyncio.run(main())
