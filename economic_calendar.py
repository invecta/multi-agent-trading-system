"""
Economic Calendar Integration
Provides economic events, earnings calendar, and market impact analysis
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EconomicCalendar:
    """Economic calendar and earnings data integration"""
    
    def __init__(self):
        self.economic_events = []
        self.earnings_events = []
        self.impact_levels = {
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
    
    def generate_economic_events(self, days_ahead: int = 30) -> List[Dict]:
        """Generate simulated economic events"""
        events = []
        base_date = datetime.now()
        
        # Major economic indicators
        economic_indicators = [
            {'name': 'Non-Farm Payrolls', 'impact': 'High', 'frequency': 'Monthly'},
            {'name': 'CPI (Consumer Price Index)', 'impact': 'High', 'frequency': 'Monthly'},
            {'name': 'GDP Growth Rate', 'impact': 'High', 'frequency': 'Quarterly'},
            {'name': 'Federal Funds Rate', 'impact': 'High', 'frequency': 'Quarterly'},
            {'name': 'Unemployment Rate', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'name': 'Retail Sales', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'name': 'Industrial Production', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'name': 'Consumer Confidence', 'impact': 'Medium', 'frequency': 'Monthly'},
            {'name': 'Housing Starts', 'impact': 'Low', 'frequency': 'Monthly'},
            {'name': 'Trade Balance', 'impact': 'Low', 'frequency': 'Monthly'}
        ]
        
        for i in range(days_ahead):
            event_date = base_date + timedelta(days=i)
            
            # Randomly select events for each day
            if np.random.random() < 0.3:  # 30% chance of economic event
                indicator = np.random.choice(economic_indicators)
                
                # Generate forecast and previous values
                if 'Rate' in indicator['name'] or 'CPI' in indicator['name']:
                    forecast = round(np.random.uniform(2.0, 4.0), 2)
                    previous = round(forecast + np.random.uniform(-0.5, 0.5), 2)
                    unit = '%'
                elif 'Payrolls' in indicator['name']:
                    forecast = int(np.random.uniform(150000, 300000))
                    previous = int(forecast + np.random.uniform(-50000, 50000))
                    unit = 'K'
                else:
                    forecast = round(np.random.uniform(0.5, 3.0), 1)
                    previous = round(forecast + np.random.uniform(-0.5, 0.5), 1)
                    unit = '%'
                
                event = {
                    'date': event_date.strftime('%Y-%m-%d'),
                    'time': f"{np.random.randint(8, 16):02d}:30",
                    'event': indicator['name'],
                    'impact': indicator['impact'],
                    'forecast': f"{forecast}{unit}",
                    'previous': f"{previous}{unit}",
                    'currency': 'USD',
                    'description': f"{indicator['name']} release - {indicator['frequency']} data"
                }
                events.append(event)
        
        return events
    
    def generate_earnings_calendar(self, symbols: List[str], days_ahead: int = 30) -> List[Dict]:
        """Generate earnings calendar for given symbols"""
        earnings = []
        base_date = datetime.now()
        
        for symbol in symbols:
            # Randomly assign earnings dates
            if np.random.random() < 0.4:  # 40% chance of earnings in next 30 days
                earnings_date = base_date + timedelta(days=np.random.randint(1, days_ahead))
                
                # Generate earnings estimates
                eps_forecast = round(np.random.uniform(0.5, 5.0), 2)
                eps_previous = round(eps_forecast + np.random.uniform(-0.5, 0.5), 2)
                revenue_forecast = round(np.random.uniform(50, 200), 1)
                revenue_previous = round(revenue_forecast + np.random.uniform(-20, 20), 1)
                
                # Determine impact based on company size (simplified)
                if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
                    impact = 'High'
                elif symbol in ['NVDA', 'META', 'NFLX', 'AMD', 'INTC']:
                    impact = 'Medium'
                else:
                    impact = 'Low'
                
                earning = {
                    'date': earnings_date.strftime('%Y-%m-%d'),
                    'time': '16:00',  # After market close
                    'symbol': symbol,
                    'company': f"{symbol} Inc.",
                    'eps_forecast': eps_forecast,
                    'eps_previous': eps_previous,
                    'revenue_forecast': revenue_forecast,
                    'revenue_previous': revenue_previous,
                    'impact': impact,
                    'description': f"{symbol} Q4 2024 Earnings Release"
                }
                earnings.append(earning)
        
        return sorted(earnings, key=lambda x: x['date'])
    
    def analyze_market_impact(self, events: List[Dict], symbol: str) -> Dict:
        """Analyze potential market impact of economic events"""
        high_impact_events = [e for e in events if e['impact'] == 'High']
        medium_impact_events = [e for e in events if e['impact'] == 'Medium']
        low_impact_events = [e for e in events if e['impact'] == 'Low']
        
        # Calculate impact score
        impact_score = (
            len(high_impact_events) * 3 +
            len(medium_impact_events) * 2 +
            len(low_impact_events) * 1
        )
        
        # Determine market sentiment
        if impact_score >= 10:
            sentiment = 'High Volatility Expected'
        elif impact_score >= 5:
            sentiment = 'Moderate Volatility Expected'
        else:
            sentiment = 'Low Volatility Expected'
        
        # Generate recommendations
        recommendations = []
        if high_impact_events:
            recommendations.append("Consider reducing position size before high-impact events")
        if medium_impact_events:
            recommendations.append("Monitor medium-impact events for trading opportunities")
        if len(events) > 5:
            recommendations.append("High event density - consider defensive positioning")
        
        return {
            'total_events': len(events),
            'high_impact_count': len(high_impact_events),
            'medium_impact_count': len(medium_impact_events),
            'low_impact_count': len(low_impact_events),
            'impact_score': impact_score,
            'market_sentiment': sentiment,
            'recommendations': recommendations,
            'next_high_impact': high_impact_events[0] if high_impact_events else None
        }
    
    def get_economic_calendar_data(self, symbols: List[str], days_ahead: int = 30) -> Dict:
        """Get comprehensive economic calendar data"""
        economic_events = self.generate_economic_events(days_ahead)
        earnings_events = self.generate_earnings_calendar(symbols, days_ahead)
        
        # Combine all events
        all_events = economic_events + earnings_events
        all_events.sort(key=lambda x: x['date'])
        
        # Analyze market impact
        market_impact = self.analyze_market_impact(all_events, symbols[0] if symbols else 'SPY')
        
        return {
            'economic_events': economic_events,
            'earnings_events': earnings_events,
            'all_events': all_events,
            'market_impact_analysis': market_impact,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def demo_economic_calendar():
    """Demo function for economic calendar"""
    calendar = EconomicCalendar()
    
    # Test with sample symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    data = calendar.get_economic_calendar_data(symbols, 30)
    
    print("=== ECONOMIC CALENDAR DATA ===")
    print(f"Generated at: {data['generated_at']}")
    print(f"Total Events: {len(data['all_events'])}")
    print(f"Economic Events: {len(data['economic_events'])}")
    print(f"Earnings Events: {len(data['earnings_events'])}")
    
    print("\n=== MARKET IMPACT ANALYSIS ===")
    impact = data['market_impact_analysis']
    print(f"Impact Score: {impact['impact_score']}")
    print(f"Market Sentiment: {impact['market_sentiment']}")
    print("Recommendations:")
    for rec in impact['recommendations']:
        print(f"  - {rec}")
    
    print("\n=== UPCOMING HIGH-IMPACT EVENTS ===")
    high_impact = [e for e in data['all_events'] if e['impact'] == 'High'][:5]
    for event in high_impact:
        if 'symbol' in event:
            print(f"{event['date']} {event['time']} - {event['symbol']} Earnings")
        else:
            print(f"{event['date']} {event['time']} - {event['event']}")

if __name__ == "__main__":
    demo_economic_calendar()
