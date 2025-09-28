#!/usr/bin/env python3
"""
Test WSGI configuration
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('DASH_DEBUG', 'False')

print("Testing WSGI import...")

try:
    from enhanced_dashboard_v2 import app
    print(f"Successfully imported app: {type(app)}")
    
    application = app.server
    print(f"Application server: {type(application)}")
    
    # Test if application is callable
    if callable(application):
        print("Application is callable")
    else:
        print("Application is not callable")
        
    print("WSGI test completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
