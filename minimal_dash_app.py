#!/usr/bin/env python3
"""
Minimal Dash app for Railway deployment testing
"""

import os
import dash
from dash import html

# Create Dash app
app = dash.Dash(__name__)

# Simple layout
app.layout = html.Div([
    html.H1("Minimal Dash App"),
    html.P("This is a test deployment."),
    html.P(f"Port: {os.environ.get('PORT', '8059')}")
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8059))
    print(f"Starting minimal Dash app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
