#!/usr/bin/env python3
"""
Alpaca API Connection Test
Tests your Alpaca API credentials and connection
"""
import asyncio
import aiohttp
import json
from datetime import datetime

async def test_alpaca_connection():
    """Test Alpaca API connection with your credentials"""
    
    # Your Alpaca credentials
    api_key = "PKOEKMI4RY0LHF565WDO"
    secret_key = "Dq14y4AJpsIqFfJ33FWKWvdJw9zqrAPsaLtJhdDb"
    account_id = "PA3TE0S55RX2"
    base_url = "https://paper-api.alpaca.markets/v2"
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }
    
    print("Testing Alpaca API Connection")
    print("=" * 40)
    print(f"API Key: {api_key[:8]}...")
    print(f"Account ID: {account_id}")
    print(f"Base URL: {base_url}")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Account Information
            print("Test 1: Getting Account Information...")
            async with session.get(f"{base_url}/account", headers=headers) as response:
                if response.status == 200:
                    account_data = await response.json()
                    print("SUCCESS: Account connection successful!")
                    print(f"   Account Status: {account_data.get('status', 'Unknown')}")
                    print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
                    print(f"   Cash: ${float(account_data.get('cash', 0)):,.2f}")
                    print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                    print(f"   Equity: ${float(account_data.get('equity', 0)):,.2f}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Account connection failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
            
            print()
            
            # Test 2: Market Data (AAPL)
            print("Test 2: Getting Market Data (AAPL)...")
            async with session.get(f"{base_url}/stocks/AAPL/bars/latest", headers=headers) as response:
                if response.status == 200:
                    market_data = await response.json()
                    print("SUCCESS: Market data access successful!")
                    if 'bar' in market_data:
                        bar = market_data['bar']
                        print(f"   Symbol: {bar.get('S', 'AAPL')}")
                        print(f"   Close Price: ${bar.get('c', 0):.2f}")
                        print(f"   Volume: {bar.get('v', 0):,}")
                        print(f"   Timestamp: {bar.get('t', 'Unknown')}")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Market data access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 3: Positions
            print("Test 3: Getting Current Positions...")
            async with session.get(f"{base_url}/positions", headers=headers) as response:
                if response.status == 200:
                    positions = await response.json()
                    print("SUCCESS: Positions access successful!")
                    if positions:
                        print(f"   Number of positions: {len(positions)}")
                        for pos in positions[:3]:  # Show first 3 positions
                            print(f"   - {pos.get('symbol', 'Unknown')}: {pos.get('qty', 0)} shares")
                    else:
                        print("   No current positions")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Positions access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            
            # Test 4: Orders
            print("Test 4: Getting Recent Orders...")
            async with session.get(f"{base_url}/orders?status=all&limit=5", headers=headers) as response:
                if response.status == 200:
                    orders = await response.json()
                    print("SUCCESS: Orders access successful!")
                    if orders:
                        print(f"   Number of recent orders: {len(orders)}")
                        for order in orders[:3]:  # Show first 3 orders
                            print(f"   - {order.get('symbol', 'Unknown')}: {order.get('side', 'Unknown')} {order.get('qty', 0)} @ ${order.get('limit_price', order.get('filled_avg_price', 0)):.2f}")
                    else:
                        print("   No recent orders")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Orders access failed: {response.status}")
                    print(f"   Error: {error_text}")
            
            print()
            print("Alpaca API Connection Test Completed!")
            print("SUCCESS: Your credentials are working correctly!")
            print()
            print("Next Steps:")
            print("1. Run the Multi-Agent Trading System:")
            print("   python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL")
            print()
            print("2. Or run the interactive demo:")
            print("   python demo_multi_agent_system.py")
            print()
            print("3. Your system will now use Alpaca for real paper trading!")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Connection test failed: {e}")
            return False

async def test_paper_order():
    """Test placing a paper order (optional)"""
    print("\nOptional: Test Paper Order Placement")
    print("=" * 40)
    
    response = input("Would you like to test placing a small paper order? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Skipping paper order test.")
        return
    
    # Your Alpaca credentials
    api_key = "PKOEKMI4RY0LHF565WDO"
    secret_key = "Dq14y4AJpsIqFfJ33FWKWvdJw9zqrAPsaLtJhdDb"
    base_url = "https://paper-api.alpaca.markets/v2"
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }
    
    # Small test order
    order_data = {
        'symbol': 'AAPL',
        'qty': '1',
        'side': 'buy',
        'type': 'market',
        'time_in_force': 'day',
        'client_order_id': f'test_order_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            print("Placing test order: Buy 1 share of AAPL...")
            async with session.post(f"{base_url}/orders", headers=headers, json=order_data) as response:
                if response.status == 201:
                    order_result = await response.json()
                    print("SUCCESS: Test order placed successfully!")
                    print(f"   Order ID: {order_result.get('id', 'Unknown')}")
                    print(f"   Status: {order_result.get('status', 'Unknown')}")
                    print(f"   Symbol: {order_result.get('symbol', 'Unknown')}")
                    print(f"   Quantity: {order_result.get('qty', 'Unknown')}")
                    print(f"   Side: {order_result.get('side', 'Unknown')}")
                    
                    # Cancel the test order
                    order_id = order_result.get('id')
                    if order_id:
                        print(f"\nCancelling test order {order_id}...")
                        async with session.delete(f"{base_url}/orders/{order_id}", headers=headers) as cancel_response:
                            if cancel_response.status == 204:
                                print("SUCCESS: Test order cancelled successfully!")
                            else:
                                print("WARNING: Could not cancel test order (it may have already filled)")
                else:
                    error_text = await response.text()
                    print(f"ERROR: Test order failed: {response.status}")
                    print(f"   Error: {error_text}")
                    
        except Exception as e:
            print(f"ERROR: Test order error: {e}")

async def main():
    """Main test function"""
    print("Alpaca API Connection Test")
    print("=" * 50)
    print("This script will test your Alpaca API credentials")
    print("and verify that your paper trading account is accessible.")
    print()
    
    # Test connection
    success = await test_alpaca_connection()
    
    if success:
        # Optional paper order test
        await test_paper_order()
    
    print("\nReady to use your Multi-Agent Trading System with Alpaca!")

if __name__ == "__main__":
    asyncio.run(main())