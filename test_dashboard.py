#!/usr/bin/env python3
"""
Quick test to verify Tesla 369 dashboard is working
"""

import requests
import time

def test_dashboard():
    """Test if the dashboard is responding"""
    try:
        # Wait a moment for dashboard to start
        time.sleep(2)
        
        # Test dashboard response
        response = requests.get('http://127.0.0.1:8050', timeout=10)
        
        if response.status_code == 200:
            print("Dashboard is running successfully!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Length: {len(response.text)} characters")
            
            # Check if Tesla content is present
            if "Tesla 369" in response.text:
                print("Tesla 369 content detected in dashboard!")
            else:
                print("Tesla 369 content not found in dashboard")
                
            if "Strategy Controls" in response.text:
                print("Strategy controls detected!")
            else:
                print("Strategy controls not found")
                
        else:
            print(f"Dashboard returned status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to dashboard. Is it running?")
    except requests.exceptions.Timeout:
        print("Dashboard request timed out")
    except Exception as e:
        print(f"Error testing dashboard: {str(e)}")

if __name__ == "__main__":
    print("Testing Tesla 369 Dashboard...")
    test_dashboard()
