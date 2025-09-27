#!/usr/bin/env python3
import asyncio
import aiohttp

async def test_polygon():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.polygon.io/v1/marketstatus/now', params={'apikey': 'SWbaiH7zZIQRj04sFUfWzVLXT4VeKCkP'}) as response:
            data = await response.json()
            print('SUCCESS: Market Status:', data.get('market', 'Unknown'))
            print('Server Time:', data.get('serverTime', 'Unknown'))
            print('Exchanges:', data.get('exchanges', 'Unknown'))

asyncio.run(test_polygon())
