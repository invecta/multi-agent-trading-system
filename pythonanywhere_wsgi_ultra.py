import sys
path = '/home/hindaouihani'
if path not in sys.path:
    sys.path.append(path)

try:
    from pythonanywhere_ultra_simple import application
except Exception as e:
    print(f"Import error: {e}")
    # Fallback to a basic Flask app if Dash fails
    from flask import Flask
    application = Flask(__name__)
    
    @application.route('/')
    def index():
        return '''
        <h1>Trading Dashboard</h1>
        <p>Minimal trading dashboard is running!</p>
        <p>Dash functionality will be available soon.</p>
        '''
