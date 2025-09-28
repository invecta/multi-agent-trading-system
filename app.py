#!/usr/bin/env python3
"""
Cloud deployment entry point for Enhanced Multi-Agent Trading System Dashboard
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_dashboard_v2 import app
    server = app.server
    print("Successfully imported enhanced_dashboard_v2")
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8059))
    
    print(f"Starting server on port {port}")
    print(f"Server object: {server}")
    
    # Run the app
    try:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"Error running app: {e}")
        sys.exit(1)
