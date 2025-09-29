import sys
path = '/home/hindaouihani'
if path not in sys.path:
    sys.path.append(path)

from pythonanywhere_alpaca_dashboard import application
