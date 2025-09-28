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

try:
    from enhanced_dashboard_v2 import app
    application = app.server
except ImportError as e:
    print(f"Error importing app: {e}")
    import traceback
    traceback.print_exc()
    raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8059))
    app.run(host="0.0.0.0", port=port, debug=False)
