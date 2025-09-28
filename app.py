#!/usr/bin/env python3
"""
Cloud deployment entry point for Enhanced Multi-Agent Trading System Dashboard
"""

import os
import sys
from enhanced_dashboard_v2 import app

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8059))
    
    # Run the app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
