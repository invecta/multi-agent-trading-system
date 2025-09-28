"""
Mobile-responsive design and PWA capabilities
Provides mobile optimization, PWA features, and responsive design components
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime, timedelta

class MobileResponsiveLayout:
    """Mobile-responsive layout components"""
    
    def __init__(self):
        self.breakpoints = {
            'xs': 0,
            'sm': 576,
            'md': 768,
            'lg': 992,
            'xl': 1200,
            'xxl': 1400
        }
        
    def create_responsive_header(self) -> html.Div:
        """Create responsive header with mobile menu"""
        
        return html.Div([
            # Mobile menu button
            dbc.Button(
                html.Span("☰", style={'fontSize': '24px'}),
                id="mobile-menu-button",
                className="d-lg-none",
                color="primary",
                outline=True,
                style={'position': 'fixed', 'top': '10px', 'left': '10px', 'zIndex': 1000}
            ),
            
            # Main header
            html.Div([
                html.H1("Multi-Agent Trading System", 
                       className="text-center mb-0",
                       style={'fontSize': 'clamp(1.5rem, 4vw, 2.5rem)'}),
                html.P("Advanced Portfolio Analytics Dashboard", 
                      className="text-center text-muted mb-0",
                      style={'fontSize': 'clamp(0.8rem, 2vw, 1.2rem)'})
            ], style={'padding': '20px 0'})
            
        ], className="bg-primary text-white", style={'position': 'relative'})
        
    def create_responsive_navigation(self) -> html.Div:
        """Create responsive navigation menu"""
        
        return html.Div([
            # Desktop navigation
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Dashboard", href="#", id="nav-dashboard")),
                dbc.NavItem(dbc.NavLink("Portfolio", href="#", id="nav-portfolio")),
                dbc.NavItem(dbc.NavLink("Analysis", href="#", id="nav-analysis")),
                dbc.NavItem(dbc.NavLink("Reports", href="#", id="nav-reports")),
                dbc.NavItem(dbc.NavLink("Settings", href="#", id="nav-settings"))
            ], className="d-none d-lg-flex", pills=True, fill=True),
            
            # Mobile navigation (collapsible)
            dbc.Collapse([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Dashboard", href="#", className="text-center")),
                    dbc.NavItem(dbc.NavLink("Portfolio", href="#", className="text-center")),
                    dbc.NavItem(dbc.NavLink("Analysis", href="#", className="text-center")),
                    dbc.NavItem(dbc.NavLink("Reports", href="#", className="text-center")),
                    dbc.NavItem(dbc.NavLink("Settings", href="#", className="text-center"))
                ], vertical=True, pills=True, className="w-100")
            ], id="mobile-nav-collapse", className="d-lg-none")
            
        ], className="bg-light border-bottom")
        
    def create_responsive_controls(self) -> html.Div:
        """Create responsive control panel"""
        
        return html.Div([
            # Desktop controls (horizontal)
            html.Div([
                html.Div([
                    html.Label("Symbol:", className="form-label"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[
                            # Technology Sector
                            {'label': 'Technology', 'value': 'TECH_HEADER', 'disabled': True},
                            {'label': '  AAPL - Apple Inc.', 'value': 'AAPL'},
                            {'label': '  MSFT - Microsoft Corporation', 'value': 'MSFT'},
                            {'label': '  GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
                            {'label': '  AMZN - Amazon.com Inc.', 'value': 'AMZN'},
                            {'label': '  META - Meta Platforms Inc.', 'value': 'META'},
                            {'label': '  NVDA - NVIDIA Corporation', 'value': 'NVDA'},
                            {'label': '  NFLX - Netflix Inc.', 'value': 'NFLX'},
                            {'label': '  PYPL - PayPal Holdings Inc.', 'value': 'PYPL'},
                            {'label': '  CRM - Salesforce Inc.', 'value': 'CRM'},
                            {'label': '  ADBE - Adobe Inc.', 'value': 'ADBE'},
                            {'label': '  INTC - Intel Corporation', 'value': 'INTC'},
                            {'label': '  CSCO - Cisco Systems Inc.', 'value': 'CSCO'},
                            {'label': '  ORCL - Oracle Corporation', 'value': 'ORCL'},
                            {'label': '  IBM - IBM Corporation', 'value': 'IBM'},
                            {'label': '  SNOW - Snowflake Inc.', 'value': 'SNOW'},
                            
                            # Healthcare Sector
                            {'label': 'Healthcare', 'value': 'HEALTH_HEADER', 'disabled': True},
                            {'label': '  JNJ - Johnson & Johnson', 'value': 'JNJ'},
                            {'label': '  PFE - Pfizer Inc.', 'value': 'PFE'},
                            {'label': '  MRNA - Moderna Inc.', 'value': 'MRNA'},
                            {'label': '  UNH - UnitedHealth Group', 'value': 'UNH'},
                            {'label': '  ABT - Abbott Laboratories', 'value': 'ABT'},
                            
                            # Consumer Sector
                            {'label': 'Consumer', 'value': 'CONSUMER_HEADER', 'disabled': True},
                            {'label': '  PG - Procter & Gamble', 'value': 'PG'},
                            {'label': '  KO - Coca-Cola Company', 'value': 'KO'},
                            {'label': '  PEP - PepsiCo Inc.', 'value': 'PEP'},
                            {'label': '  WMT - Walmart Inc.', 'value': 'WMT'},
                            {'label': '  TSLA - Tesla Inc.', 'value': 'TSLA'},
                            
                            # Finance Sector
                            {'label': 'Finance', 'value': 'FINANCE_HEADER', 'disabled': True},
                            {'label': '  JPM - JPMorgan Chase & Co.', 'value': 'JPM'},
                            {'label': '  BAC - Bank of America Corp.', 'value': 'BAC'},
                            {'label': '  WFC - Wells Fargo & Co.', 'value': 'WFC'},
                            {'label': '  GS - Goldman Sachs Group Inc.', 'value': 'GS'},
                            {'label': '  BRK-B - Berkshire Hathaway Inc.', 'value': 'BRK-B'},
                            {'label': '  V - Visa Inc.', 'value': 'V'},
                            {'label': '  MA - Mastercard Inc.', 'value': 'MA'},
                            
                            # Energy Sector
                            {'label': 'Energy', 'value': 'ENERGY_HEADER', 'disabled': True},
                            {'label': '  XOM - Exxon Mobil Corp.', 'value': 'XOM'},
                            {'label': '  CVX - Chevron Corporation', 'value': 'CVX'},
                            {'label': '  COP - ConocoPhillips', 'value': 'COP'},
                            {'label': '  EOG - EOG Resources Inc.', 'value': 'EOG'}
                        ],
                        value='AAPL',
                        className="mb-2"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Time Period:", className="form-label"),
                    dcc.Dropdown(
                        id='time-period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': 30},
                            {'label': '3 Months', 'value': 90},
                            {'label': '6 Months', 'value': 180},
                            {'label': '1 Year', 'value': 365},
                            {'label': '2 Years', 'value': 730},
                            {'label': 'Custom Date Range', 'value': 'custom'}
                        ],
                        value=365,
                        className="mb-2"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Date Range:", className="form-label"),
                    html.Div([
                        html.Div([
                            html.Label("From:", className="form-label-small"),
                            dcc.DatePickerSingle(
                                id='start-date-picker',
                                date=datetime.now() - timedelta(days=365),
                                display_format='MM/DD/YYYY',
                                className="date-picker-input",
                                style={'width': '100%', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px'}
                            )
                        ], className="col-6"),
                        html.Div([
                            html.Label("To:", className="form-label-small"),
                            dcc.DatePickerSingle(
                                id='end-date-picker',
                                date=datetime.now(),
                                display_format='MM/DD/YYYY',
                                className="date-picker-input",
                                style={'width': '100%', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px'}
                            )
                        ], className="col-6")
                    ], className="row", style={'margin': '0'})
                ], className="col-md-6", id="date-range-container", style={'display': 'none'}),
                
                html.Div([
                    html.Label("Timeframe:", className="form-label"),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '1 Minute', 'value': '1m'},
                            {'label': '5 Minutes', 'value': '5m'},
                            {'label': '15 Minutes', 'value': '15m'},
                            {'label': '1 Hour', 'value': '1h'},
                            {'label': '4 Hours', 'value': '4h'},
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '1 Week', 'value': '1w'}
                        ],
                        value='1d',
                        className="mb-2"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Sector:", className="form-label"),
                    dcc.Dropdown(
                        id='sector-dropdown',
                        options=[
                            {'label': 'Technology', 'value': 'Technology'},
                            {'label': 'Healthcare', 'value': 'Healthcare'},
                            {'label': 'Finance', 'value': 'Finance'},
                            {'label': 'Energy', 'value': 'Energy'},
                            {'label': 'Consumer', 'value': 'Consumer'},
                            {'label': 'Indices', 'value': 'Indices'},
                            {'label': 'Forex', 'value': 'Forex'}
                        ],
                        value='Technology',
                        className="mb-2"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Capital:", className="form-label"),
                    dcc.Input(
                        id='capital-input',
                        type='number',
                        value=100000,
                        className="form-control mb-2"
                    )
                ], className="col-md-3")
                
            ], className="row d-none d-lg-flex"),
            
            # Mobile controls (vertical)
            html.Div([
                html.Div([
                    html.Label("Symbol:", className="form-label"),
                    dcc.Dropdown(
                        id='symbol-dropdown-mobile',
                        options=[
                            # Technology Sector
                            {'label': 'Technology', 'value': 'TECH_HEADER', 'disabled': True},
                            {'label': '  AAPL - Apple Inc.', 'value': 'AAPL'},
                            {'label': '  MSFT - Microsoft Corporation', 'value': 'MSFT'},
                            {'label': '  GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
                            {'label': '  AMZN - Amazon.com Inc.', 'value': 'AMZN'},
                            {'label': '  META - Meta Platforms Inc.', 'value': 'META'},
                            {'label': '  NVDA - NVIDIA Corporation', 'value': 'NVDA'},
                            {'label': '  NFLX - Netflix Inc.', 'value': 'NFLX'},
                            {'label': '  PYPL - PayPal Holdings Inc.', 'value': 'PYPL'},
                            {'label': '  CRM - Salesforce Inc.', 'value': 'CRM'},
                            {'label': '  ADBE - Adobe Inc.', 'value': 'ADBE'},
                            {'label': '  INTC - Intel Corporation', 'value': 'INTC'},
                            {'label': '  CSCO - Cisco Systems Inc.', 'value': 'CSCO'},
                            {'label': '  ORCL - Oracle Corporation', 'value': 'ORCL'},
                            {'label': '  IBM - IBM Corporation', 'value': 'IBM'},
                            {'label': '  SNOW - Snowflake Inc.', 'value': 'SNOW'},
                            
                            # Healthcare Sector
                            {'label': 'Healthcare', 'value': 'HEALTH_HEADER', 'disabled': True},
                            {'label': '  JNJ - Johnson & Johnson', 'value': 'JNJ'},
                            {'label': '  PFE - Pfizer Inc.', 'value': 'PFE'},
                            {'label': '  MRNA - Moderna Inc.', 'value': 'MRNA'},
                            {'label': '  UNH - UnitedHealth Group', 'value': 'UNH'},
                            {'label': '  ABT - Abbott Laboratories', 'value': 'ABT'},
                            
                            # Consumer Sector
                            {'label': 'Consumer', 'value': 'CONSUMER_HEADER', 'disabled': True},
                            {'label': '  PG - Procter & Gamble', 'value': 'PG'},
                            {'label': '  KO - Coca-Cola Company', 'value': 'KO'},
                            {'label': '  PEP - PepsiCo Inc.', 'value': 'PEP'},
                            {'label': '  WMT - Walmart Inc.', 'value': 'WMT'},
                            {'label': '  TSLA - Tesla Inc.', 'value': 'TSLA'},
                            
                            # Finance Sector
                            {'label': 'Finance', 'value': 'FINANCE_HEADER', 'disabled': True},
                            {'label': '  JPM - JPMorgan Chase & Co.', 'value': 'JPM'},
                            {'label': '  BAC - Bank of America Corp.', 'value': 'BAC'},
                            {'label': '  WFC - Wells Fargo & Co.', 'value': 'WFC'},
                            {'label': '  GS - Goldman Sachs Group Inc.', 'value': 'GS'},
                            {'label': '  BRK-B - Berkshire Hathaway Inc.', 'value': 'BRK-B'},
                            {'label': '  V - Visa Inc.', 'value': 'V'},
                            {'label': '  MA - Mastercard Inc.', 'value': 'MA'},
                            
                            # Energy Sector
                            {'label': 'Energy', 'value': 'ENERGY_HEADER', 'disabled': True},
                            {'label': '  XOM - Exxon Mobil Corp.', 'value': 'XOM'},
                            {'label': '  CVX - Chevron Corporation', 'value': 'CVX'},
                            {'label': '  COP - ConocoPhillips', 'value': 'COP'},
                            {'label': '  EOG - EOG Resources Inc.', 'value': 'EOG'}
                        ],
                        value='AAPL',
                        className="mb-3"
                    )
                ]),
                
                html.Div([
                    html.Label("Time Period:", className="form-label"),
                    dcc.Dropdown(
                        id='time-period-dropdown-mobile',
                        options=[
                            {'label': '1 Month', 'value': 30},
                            {'label': '3 Months', 'value': 90},
                            {'label': '6 Months', 'value': 180},
                            {'label': '1 Year', 'value': 365},
                            {'label': '2 Years', 'value': 730},
                            {'label': 'Custom Date Range', 'value': 'custom'}
                        ],
                        value=365,
                        className="mb-3"
                    )
                ]),
                
                html.Div([
                    html.Label("Date Range:", className="form-label"),
                    html.Div([
                        html.Div([
                            html.Label("From:", className="form-label-small"),
                            dcc.DatePickerSingle(
                                id='start-date-picker-mobile',
                                date=datetime.now() - timedelta(days=365),
                                display_format='MM/DD/YYYY',
                                className="date-picker-input",
                                style={'width': '100%', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px'}
                            )
                        ], className="col-6"),
                        html.Div([
                            html.Label("To:", className="form-label-small"),
                            dcc.DatePickerSingle(
                                id='end-date-picker-mobile',
                                date=datetime.now(),
                                display_format='MM/DD/YYYY',
                                className="date-picker-input",
                                style={'width': '100%', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px'}
                            )
                        ], className="col-6")
                    ], className="row", style={'margin': '0'})
                ], id="date-range-container-mobile", style={'display': 'none'}),
                
                html.Div([
                    html.Label("Timeframe:", className="form-label"),
                    dcc.Dropdown(
                        id='timeframe-dropdown-mobile',
                        options=[
                            {'label': '1 Minute', 'value': '1m'},
                            {'label': '5 Minutes', 'value': '5m'},
                            {'label': '15 Minutes', 'value': '15m'},
                            {'label': '1 Hour', 'value': '1h'},
                            {'label': '4 Hours', 'value': '4h'},
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '1 Week', 'value': '1w'}
                        ],
                        value='1d',
                        className="mb-3"
                    )
                ]),
                
                html.Div([
                    html.Label("Sector:", className="form-label"),
                    dcc.Dropdown(
                        id='sector-dropdown-mobile',
                        options=[
                            {'label': 'Technology', 'value': 'Technology'},
                            {'label': 'Healthcare', 'value': 'Healthcare'},
                            {'label': 'Finance', 'value': 'Finance'},
                            {'label': 'Energy', 'value': 'Energy'},
                            {'label': 'Consumer', 'value': 'Consumer'},
                            {'label': 'Indices', 'value': 'Indices'}
                        ],
                        value='Technology',
                        className="mb-3"
                    )
                ]),
                
                html.Div([
                    html.Label("Capital:", className="form-label"),
                    dcc.Input(
                        id='capital-input-mobile',
                        type='number',
                        value=100000,
                        className="form-control mb-3"
                    )
                ])
                
            ], className="d-lg-none"),
            
            # Run button
            html.Div([
                dbc.Button(
                    "Run Analysis",
                    id="run-analysis-button-mobile",
                    color="primary",
                    size="lg",
                    className="w-100",
                    style={'fontSize': 'clamp(1rem, 3vw, 1.2rem)'}
                )
            ], className="text-center mt-3")
            
        ], className="container-fluid p-3 bg-light")
        
    def create_responsive_tabs(self) -> html.Div:
        """Create responsive tab navigation"""
        
        return html.Div([
            dcc.Tabs(
                id="main-tabs",
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview", className="responsive-tab"),
                    dcc.Tab(label="Charts", value="charts", className="responsive-tab"),
                    dcc.Tab(label="Analysis", value="analysis", className="responsive-tab"),
                    dcc.Tab(label="Risk", value="risk", className="responsive-tab"),
                    dcc.Tab(label="Portfolio", value="portfolio", className="responsive-tab"),
                    dcc.Tab(label="Reports", value="reports", className="responsive-tab")
                ],
                className="responsive-tabs"
            )
        ], className="container-fluid")
        
    def create_responsive_chart_container(self, chart_id: str, title: str) -> html.Div:
        """Create responsive chart container"""
        
        return html.Div([
            html.H4(title, className="text-center mb-3", 
                   style={'fontSize': 'clamp(1.2rem, 3vw, 1.5rem)'}),
            dcc.Graph(
                id=chart_id,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'responsive': True
                },
                style={'height': '400px', 'width': '100%'}
            )
        ], className="chart-container mb-4")
        
    def create_responsive_metrics_grid(self, metrics: List[Dict]) -> html.Div:
        """Create responsive metrics grid"""
        
        metric_cards = []
        
        for metric in metrics:
            card = dbc.Card([
                dbc.CardBody([
                    html.H5(metric['title'], className="card-title text-center",
                           style={'fontSize': 'clamp(0.9rem, 2.5vw, 1.1rem)'}),
                    html.H3(metric['value'], className="card-text text-center text-primary",
                           style={'fontSize': 'clamp(1.5rem, 4vw, 2rem)'}),
                    html.P(metric['change'], className="card-text text-center text-muted",
                          style={'fontSize': 'clamp(0.8rem, 2vw, 1rem)'})
                ])
            ], className="h-100")
            
            metric_cards.append(html.Div([card], className="col-6 col-md-3 mb-3"))
            
        return html.Div(metric_cards, className="row")

class PWAFeatures:
    """Progressive Web App features"""
    
    def __init__(self):
        self.manifest = self._create_manifest()
        self.service_worker = self._create_service_worker()
        
    def _create_manifest(self) -> Dict:
        """Create PWA manifest"""
        
        return {
            "name": "Multi-Agent Trading System",
            "short_name": "TradingApp",
            "description": "Advanced portfolio analytics and trading dashboard",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#007bff",
            "orientation": "portrait-primary",
            "icons": [
                {
                    "src": "icons/icon-72x72.png",
                    "sizes": "72x72",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-96x96.png",
                    "sizes": "96x96",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-128x128.png",
                    "sizes": "128x128",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-144x144.png",
                    "sizes": "144x144",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-152x152.png",
                    "sizes": "152x152",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-384x384.png",
                    "sizes": "384x384",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        }
        
    def _create_service_worker(self) -> str:
        """Create service worker for offline functionality"""
        
        return """
        const CACHE_NAME = 'trading-app-v1';
        const urlsToCache = [
            '/',
            '/static/css/main.css',
            '/static/js/main.js',
            '/manifest.json'
        ];

        self.addEventListener('install', function(event) {
            event.waitUntil(
                caches.open(CACHE_NAME)
                    .then(function(cache) {
                        return cache.addAll(urlsToCache);
                    })
            );
        });

        self.addEventListener('fetch', function(event) {
            event.respondWith(
                caches.match(event.request)
                    .then(function(response) {
                        if (response) {
                            return response;
                        }
                        return fetch(event.request);
                    }
                )
            );
        });
        """
        
    def create_pwa_meta_tags(self) -> List[html.Meta]:
        """Create PWA meta tags"""
        
        return [
            html.Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            html.Meta(name="theme-color", content="#007bff"),
            html.Meta(name="apple-mobile-web-app-capable", content="yes"),
            html.Meta(name="apple-mobile-web-app-status-bar-style", content="default"),
            html.Meta(name="apple-mobile-web-app-title", content="TradingApp"),
            html.Link(rel="manifest", href="/manifest.json"),
            html.Link(rel="apple-touch-icon", href="/icons/icon-192x192.png")
        ]
        
    def create_offline_indicator(self) -> html.Div:
        """Create offline/online indicator"""
        
        return html.Div([
            html.Div(
                "You are offline",
                id="offline-indicator",
                className="alert alert-warning text-center",
                style={'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0', 'right': '0', 'zIndex': 9999}
            )
        ])
        
    def create_install_prompt(self) -> html.Div:
        """Create PWA install prompt"""
        
        return html.Div([
            dbc.Modal([
                dbc.ModalHeader("Install Trading App"),
                dbc.ModalBody([
                    html.P("Install this app on your device for a better experience!"),
                    html.Ul([
                        html.Li("Access from your home screen"),
                        html.Li("Faster loading times"),
                        html.Li("Offline functionality"),
                        html.Li("Push notifications")
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Install", id="install-app-button", color="primary"),
                    dbc.Button("Not Now", id="dismiss-install-button", color="secondary")
                ])
            ], id="install-modal", is_open=False)
        ])

class MobileOptimizations:
    """Mobile-specific optimizations"""
    
    def __init__(self):
        self.touch_gestures = self._create_touch_gestures()
        self.mobile_charts = self._create_mobile_chart_configs()
        
    def _create_touch_gestures(self) -> Dict:
        """Create touch gesture configurations"""
        
        return {
            'swipe_threshold': 50,
            'tap_threshold': 10,
            'long_press_duration': 500
        }
        
    def _create_mobile_chart_configs(self) -> Dict:
        """Create mobile-optimized chart configurations"""
        
        return {
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'chart',
                'height': 500,
                'width': 700,
                'scale': 2
            }
        }
        
    def create_mobile_chart(self, figure: go.Figure, chart_id: str) -> dcc.Graph:
        """Create mobile-optimized chart"""
        
        # Optimize figure for mobile
        figure.update_layout(
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return dcc.Graph(
            id=chart_id,
            figure=figure,
            config=self.mobile_charts,
            style={'height': '300px', 'width': '100%'}
        )
        
    def create_mobile_table(self, data: List[Dict], columns: List[str]) -> html.Div:
        """Create mobile-optimized table"""
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H6(col, className="text-center font-weight-bold")
                ], className="col")
                for col in columns
            ], className="row border-bottom")
        ] + [
            html.Div([
                html.Div([
                    html.Span(str(row.get(col, '')), className="text-center")
                ], className="col")
                for col in columns
            ], className="row border-bottom py-2")
            for row in data
        ], className="container-fluid", style={'fontSize': '0.9rem'})

class ResponsiveCSS:
    """Responsive CSS styles"""
    
    def __init__(self):
        self.styles = self._create_responsive_styles()
        
    def _create_responsive_styles(self) -> str:
        """Create responsive CSS styles"""
        
        return """
        /* Mobile-first responsive design */
        .responsive-tabs .tab {
            font-size: clamp(0.8rem, 2vw, 1rem);
            padding: 8px 12px;
        }
        
        .responsive-tab {
            min-width: 80px;
        }
        
        .chart-container {
            padding: 10px;
        }
        
        @media (max-width: 768px) {
            .chart-container {
                padding: 5px;
            }
            
            .responsive-tabs {
                font-size: 0.8rem;
            }
            
            .card {
                margin-bottom: 10px;
            }
            
            .btn {
                font-size: 0.9rem;
                padding: 8px 16px;
            }
        }
        
        @media (max-width: 576px) {
            .container-fluid {
                padding: 5px;
            }
            
            .chart-container {
                padding: 2px;
            }
            
            .responsive-tabs .tab {
                padding: 6px 8px;
                font-size: 0.7rem;
            }
        }
        
        /* Touch-friendly elements */
        .touch-target {
            min-height: 44px;
            min-width: 44px;
        }
        
        /* PWA styles */
        .pwa-install-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        /* Offline indicator */
        .offline-indicator {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            text-align: center;
            padding: 10px;
            background-color: #ffc107;
            color: #000;
        }
        
        /* Mobile menu */
        .mobile-menu {
            position: fixed;
            top: 0;
            left: -250px;
            width: 250px;
            height: 100vh;
            background-color: #f8f9fa;
            transition: left 0.3s ease;
            z-index: 1000;
        }
        
        .mobile-menu.open {
            left: 0;
        }
        
        .mobile-menu-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 999;
            display: none;
        }
        
        .mobile-menu-overlay.show {
            display: block;
        }
        """
        
    def get_css(self) -> str:
        """Get CSS styles"""
        return self.styles

class MobileDashboard:
    """Complete mobile-optimized dashboard"""
    
    def __init__(self):
        self.responsive_layout = MobileResponsiveLayout()
        self.pwa_features = PWAFeatures()
        self.mobile_optimizations = MobileOptimizations()
        self.responsive_css = ResponsiveCSS()
        
    def create_mobile_dashboard(self) -> html.Div:
        """Create complete mobile-optimized dashboard"""
        
        return html.Div([
            # PWA meta tags
            *self.pwa_features.create_pwa_meta_tags(),
            
            # CSS styles
            html.Style(self.responsive_css.get_css()),
            
            # Offline indicator
            self.pwa_features.create_offline_indicator(),
            
            # Install prompt
            self.pwa_features.create_install_prompt(),
            
            # Main layout
            html.Div([
                # Header
                self.responsive_layout.create_responsive_header(),
                
                # Navigation
                self.responsive_layout.create_responsive_navigation(),
                
                # Controls
                self.responsive_layout.create_responsive_controls(),
                
                # Tabs
                self.responsive_layout.create_responsive_tabs(),
                
                # Content area
                html.Div(id="main-content", className="container-fluid p-3")
                
            ], className="mobile-dashboard"),
            
        # PWA install button
        html.Div([
            dbc.Button(
                "Install App",
                id="pwa-install-button",
                color="primary",
                size="sm",
                className="pwa-install-button"
            )
        ]),
        
        # Dark theme toggle button
        html.Div([
            dbc.Button(
                "🌙 Dark Mode",
                id="dark-theme-toggle",
                color="secondary",
                size="sm",
                className="dark-theme-toggle"
            )
        ], className="theme-toggle-container")
            
        ], className="app-container")
        
    def create_mobile_callbacks(self, app):
        """Create mobile-specific callbacks"""
        
        @app.callback(
            Output("mobile-nav-collapse", "is_open"),
            [Input("mobile-menu-button", "n_clicks")],
            [State("mobile-nav-collapse", "is_open")]
        )
        def toggle_mobile_nav(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open
            
        @app.callback(
            [Output("install-modal", "is_open"),
             Output("pwa-install-button", "style")],
            [Input("install-app-button", "n_clicks"),
             Input("dismiss-install-button", "n_clicks")]
        )
        def handle_install_prompt(install_clicks, dismiss_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return False, {'display': 'none'}
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == "install-app-button":
                return False, {'display': 'none'}
            elif button_id == "dismiss-install-button":
                return False, {'display': 'none'}
                
            return False, {'display': 'none'}
            
        # Date picker visibility callbacks
        @app.callback(
            Output("date-range-container", "style"),
            [Input("time-period-dropdown", "value")]
        )
        def toggle_date_picker_desktop(time_period):
            if time_period == 'custom':
                return {'display': 'block'}
            else:
                return {'display': 'none'}
                
        @app.callback(
            Output("date-range-container-mobile", "style"),
            [Input("time-period-dropdown-mobile", "value")]
        )
        def toggle_date_picker_mobile(time_period):
            if time_period == 'custom':
                return {'display': 'block'}
            else:
                return {'display': 'none'}

# Global instances
mobile_responsive_layout = MobileResponsiveLayout()
pwa_features = PWAFeatures()
mobile_optimizations = MobileOptimizations()
responsive_css = ResponsiveCSS()
mobile_dashboard = MobileDashboard()
