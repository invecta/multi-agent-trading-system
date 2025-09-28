#!/usr/bin/env python3
"""
WSGI entry point for production deployment
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

# Gunicorn expects a callable named 'application'
application = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8059))
    app.run(host="0.0.0.0", port=port, debug=False)
