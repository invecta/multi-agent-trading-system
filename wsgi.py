#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WSGI configuration file for PythonAnywhere deployment
"""

import sys
import os

# Add your project directory to the Python path
path = '/home/hindaouihani/pdf_analysis'  # Updated with actual PythonAnywhere username
if path not in sys.path:
    sys.path.append(path)

# Import your Flask application
from PRODUCTION_DASHBOARD import app as application

# Set environment variables for production
os.environ['FLASK_ENV'] = 'production'
os.environ['DEBUG'] = 'False'

if __name__ == "__main__":
    application.run()
