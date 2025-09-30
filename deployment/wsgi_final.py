import sys
import os

# Add your project directory to the sys.path
path = '/home/hindaouihani/'
if path not in sys.path:
    sys.path.insert(0, path)

from PRODUCTION_DASHBOARD import app as application
