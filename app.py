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

try:
    print("Attempting to import enhanced_dashboard_v2...")
    from enhanced_dashboard_v2 import app
    print("Successfully imported enhanced_dashboard_v2")
    
    # Get the server object
    server = app.server
    print(f"Server object created: {type(server)}")
    
except ImportError as e:
    print(f"Error importing app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8059))
    
    print(f"Starting server on port {port}")
    print(f"Server object: {server}")
    print(f"App object: {app}")
    
    # Run the app
    try:
        print("Starting Flask application...")
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"Error running app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
