#!/usr/bin/env python3
"""
Minimal test app for Railway deployment
"""

import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Railway! Test deployment is working."

@app.route('/health')
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8059))
    print(f"Starting test app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
