import sys
path = '/home/hindaouihani'
if path not in sys.path:
    sys.path.append(path)

from pythonanywhere_minimal_dashboard import application