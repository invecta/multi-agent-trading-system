#!/usr/bin/env python3
"""
Enhanced Market Data Agent with Polygon.io Integration
Provides real-time and historical market data using Polygon.io API
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PolygonMarketDataAgent:
    """
    Enhanced Market Data Agent using Polygon.io
    Provides real-time quotes, historical data, and market status
    """
    
    def __init__(self, agent_id: str = "PolygonMarketDataAgent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.api_key = "SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP"
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.logger.info(f"{self.agent_id} initialized with Polygon.io API")
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/v1/marketstatus/now"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info("Market status retrieved successfully")
                    return {
                        "market": data.get('market', 'Unknown'),
                        "server_time": data.get('serverTime', 'Unknown'),
                        "exchanges": data.get('exchanges', 'Unknown'),
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Market status request failed: {response.status} - {error_text}")
                    return {"status": "error", "error": error_text}
        except Exception as e:
            self.logger.error(f"Error getting market status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a ticker"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data:
                        ticker = data['results']
                        self.logger.info(f"Ticker details retrieved for {symbol}")
                        return {
                            "symbol": ticker.get('ticker', symbol),
                            "name": ticker.get('name', 'Unknown'),
                            "market": ticker.get('market', 'Unknown'),
                            "locale": ticker.get('locale', 'Unknown'),
                            "primary_exchange": ticker.get('primary_exchange', 'Unknown'),
                            "type": ticker.get('type', 'Unknown'),
                            "active": ticker.get('active', False),
                            "status": "success"
                        }
                    else:
                        return {"status": "error", "error": "No ticker data found"}
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ticker details request failed for {symbol}: {response.status} - {error_text}")
                    return {"status": "error", "error": error_text}
        except Exception as e:
            self.logger.error(f"Error getting ticker details for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/v1/last_quote/stocks/{symbol}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data:
                        quote = data['results']
                        self.logger.info(f"Real-time quote retrieved for {symbol}")
                        return {
                            "symbol": quote.get('T', symbol),
                            "bid_price": quote.get('P', 0),
                            "ask_price": quote.get('p', 0),
                            "bid_size": quote.get('S', 0),
                            "ask_size": quote.get('s', 0),
                            "timestamp": quote.get('t', 'Unknown'),
                            "status": "success"
                        }
                    else:
                        return {"status": "error", "error": "No quote data found"}
                else:
                    error_text = await response.text()
                    self.logger.error(f"Real-time quote request failed for {symbol}: {response.status} - {error_text}")
                    return {"status": "error", "error": error_text}
        except Exception as e:
            self.logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_previous_close(self, symbol: str) -> Dict[str, Any]:
        """Get previous close data for a symbol"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        self.logger.info(f"Previous close data retrieved for {symbol}")
                        return {
                            "symbol": result.get('T', symbol),
                            "close": result.get('c', 0),
                            "high": result.get('h', 0),
                            "low": result.get('l', 0),
                            "open": result.get('o', 0),
                            "volume": result.get('v', 0),
                            "timestamp": result.get('t', 'Unknown'),
                            "status": "success"
                        }
                    else:
                        return {"status": "error", "error": "No previous close data found"}
                else:
                    error_text = await response.text()
                    self.logger.error(f"Previous close request failed for {symbol}: {response.status} - {error_text}")
                    return {"status": "error", "error": error_text}
        except Exception as e:
            self.logger.error(f"Error getting previous close for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        try:
            session = await self._get_session()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data and len(data['results']) > 0:
                        results = data['results']
                        
                        # Convert to DataFrame
                        df_data = []
                        for result in results:
                            df_data.append({
                                'Date': pd.to_datetime(result['t'], unit='ms'),
                                'Open': result.get('o', 0),
                                'High': result.get('h', 0),
                                'Low': result.get('l', 0),
                                'Close': result.get('c', 0),
                                'Volume': result.get('v', 0)
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        self.logger.info(f"Historical data retrieved for {symbol}: {len(df)} days")
                        return df
                    else:
                        self.logger.warning(f"No historical data found for {symbol}")
                        return None
                else:
                    error_text = await response.text()
                    self.logger.error(f"Historical data request failed for {symbol}: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for multiple symbols"""
        self.logger.info(f"Getting real-time quotes for {len(symbols)} symbols")
        
        # Get quotes for all symbols in parallel
        tasks = [self.get_real_time_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quotes = {}
        for i, symbol in enumerate(symbols):
            if isinstance(results[i], Exception):
                self.logger.error(f"Error getting quote for {symbol}: {results[i]}")
                quotes[symbol] = {"status": "error", "error": str(results[i])}
            else:
                quotes[symbol] = results[i]
        
        return {"quotes": quotes, "timestamp": datetime.now().isoformat()}
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Polygon Market Data Agent's logic"""
        symbols = data.get("symbols", ["AAPL", "GOOGL"])
        include_historical = data.get("include_historical", True)
        include_market_status = data.get("include_market_status", True)
        
        self.logger.info(f"Executing Polygon Market Data Agent for symbols: {symbols}")
        
        output_data = {}
        
        # Get market status
        if include_market_status:
            market_status = await self.get_market_status()
            output_data["market_status"] = market_status
        
        # Get real-time quotes for all symbols
        quotes_data = await self.get_multiple_quotes(symbols)
        output_data["real_time_quotes"] = quotes_data
        
        # Get previous close data for all symbols
        prev_close_tasks = [self.get_previous_close(symbol) for symbol in symbols]
        prev_close_results = await asyncio.gather(*prev_close_tasks, return_exceptions=True)
        
        prev_close_data = {}
        for i, symbol in enumerate(symbols):
            if isinstance(prev_close_results[i], Exception):
                prev_close_data[symbol] = {"status": "error", "error": str(prev_close_results[i])}
            else:
                prev_close_data[symbol] = prev_close_results[i]
        
        output_data["previous_close"] = prev_close_data
        
        # Get historical data if requested
        if include_historical:
            historical_data = {}
            for symbol in symbols:
                hist_df = await self.get_historical_data(symbol, days=30)
                if hist_df is not None:
                    historical_data[symbol] = hist_df.to_dict()
            
            output_data["historical_data"] = historical_data
        
        # Get ticker details for all symbols
        ticker_tasks = [self.get_ticker_details(symbol) for symbol in symbols]
        ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
        
        ticker_data = {}
        for i, symbol in enumerate(symbols):
            if isinstance(ticker_results[i], Exception):
                ticker_data[symbol] = {"status": "error", "error": str(ticker_results[i])}
            else:
                ticker_data[symbol] = ticker_results[i]
        
        output_data["ticker_details"] = ticker_data
        
        self.logger.info(f"Polygon Market Data Agent completed for {len(symbols)} symbols")
        return {"polygon_market_data": output_data}
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.logger.info("Polygon Market Data Agent session closed")

# Example usage and testing
async def main():
    """Example usage of the Polygon Market Data Agent"""
    
    # Initialize agent
    agent = PolygonMarketDataAgent()
    
    # Prepare input data
    input_data = {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "include_historical": True,
        "include_market_status": True
    }
    
    try:
        # Execute agent
        results = await agent.execute(input_data)
        
        print("Polygon Market Data Agent Results:")
        print("=" * 50)
        
        # Market Status
        if "market_status" in results["polygon_market_data"]:
            status = results["polygon_market_data"]["market_status"]
            print(f"Market Status: {status.get('market', 'Unknown')}")
            print(f"Server Time: {status.get('server_time', 'Unknown')}")
        
        # Real-time Quotes
        quotes = results["polygon_market_data"]["real_time_quotes"]["quotes"]
        print(f"\nReal-time Quotes:")
        for symbol, quote in quotes.items():
            if quote.get("status") == "success":
                print(f"  {symbol}: Bid ${quote.get('bid_price', 0):.2f} / Ask ${quote.get('ask_price', 0):.2f}")
        
        # Previous Close
        prev_close = results["polygon_market_data"]["previous_close"]
        print(f"\nPrevious Close:")
        for symbol, data in prev_close.items():
            if data.get("status") == "success":
                print(f"  {symbol}: ${data.get('close', 0):.2f} (Vol: {data.get('volume', 0):,})")
        
        # Ticker Details
        tickers = results["polygon_market_data"]["ticker_details"]
        print(f"\nTicker Details:")
        for symbol, data in tickers.items():
            if data.get("status") == "success":
                print(f"  {symbol}: {data.get('name', 'Unknown')} ({data.get('market', 'Unknown')})")
        
        print(f"\nStatus: SUCCESS")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
