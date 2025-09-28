#!/usr/bin/env python3
"""
Cloud deployment entry point for Enhanced Multi-Agent Trading System Dashboard
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('DASH_DEBUG', 'False')

# Import the Dash app
from enhanced_dashboard_v2 import app

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8059))
    
    print(f"Starting Enhanced Multi-Agent Trading System Dashboard on port {port}")
    
    # Run the app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )