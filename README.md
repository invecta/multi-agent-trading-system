# Professional Trading Dashboard

A comprehensive, production-ready trading dashboard with advanced analytics, real-time market data, and professional-grade features.

## 🚀 Features

### Core Trading Features
- **Real-time Market Data** - Live price feeds from Yahoo Finance
- **Technical Analysis** - RSI, MACD, Bollinger Bands, Moving Averages
- **Pattern Recognition** - Support/Resistance, Chart Patterns
- **Sentiment Analysis** - Market sentiment indicators
- **Volume Profile** - Advanced volume analysis
- **Backtesting Engine** - Strategy performance testing
- **Walk-Forward Analysis** - Out-of-sample testing

### Advanced Analytics
- **Options Analysis** - Greeks, volatility, pricing models
- **Strategy Builder** - Visual strategy creation
- **Compliance Reporting** - Regulatory requirements
- **Audit Trail** - Complete trading history logging
- **Risk Management** - Position sizing, portfolio optimization
- **Economic Calendar** - Market events and announcements

### Professional Features
- **Alpaca Integration** - Live trading capabilities
- **Portfolio Management** - Real-time portfolio tracking
- **Performance Analytics** - Comprehensive reporting
- **Market Scanner** - Multi-timeframe analysis
- **Chart Controls** - Advanced charting with multiple types

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── PRODUCTION_DASHBOARD.py   # Main Flask application
│   ├── config.py                 # Configuration settings
│   ├── goldbach_calculator.py    # Goldbach analysis tools
│   ├── tesla_369_calculator.py   # Tesla 369 analysis
│   ├── pdf_reader.py             # PDF data extraction
│   └── GoldbachLevels.mq5        # MetaTrader 5 indicator
├── deployment/                   # Deployment files
│   ├── wsgi_final.py            # WSGI configuration
│   ├── requirements_production.txt # Production dependencies
│   └── env_example.txt          # Environment variables template
├── docs/                        # Documentation
│   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   ├── PROJECT_SUMMARY.md       # Project overview
│   └── TRADING_DASHBOARD_SHOWCASE.html # Feature showcase
├── requirements.txt             # Development dependencies
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/trading-dashboard.git
cd trading-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/PRODUCTION_DASHBOARD.py
```

### Production Deployment
```bash
# Install production dependencies
pip install -r deployment/requirements_production.txt

# Configure environment variables
cp deployment/env_example.txt .env
# Edit .env with your API keys

# Deploy using WSGI
# See docs/DEPLOYMENT_GUIDE.md for detailed instructions
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:
```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
DEBUG=False
HOST=0.0.0.0
PORT=5000
SECRET_KEY=your_secret_key
```

### API Keys
- **Alpaca API** - For live trading (paper trading recommended)
- **Yahoo Finance** - For market data (no API key required)

## 📊 Usage

### Dashboard Tabs
1. **Dashboard** - Overview and quick stats
2. **Market Scanner** - Multi-timeframe analysis
3. **Volume Profile** - Volume analysis
4. **Backtesting Engine** - Strategy testing
5. **Strategy Builder** - Visual strategy creation
6. **Walk-Forward Analysis** - Out-of-sample testing
7. **Pattern Recognition** - Chart pattern analysis
8. **Sentiment Analysis** - Market sentiment
9. **Compliance Reporting** - Regulatory compliance
10. **Options Analysis** - Options Greeks and volatility
11. **Audit Trail** - Trading history logging

### Key Features
- **Real-time Updates** - Live market data and portfolio updates
- **Interactive Charts** - Chart.js powered visualizations
- **Responsive Design** - Works on desktop and mobile
- **Professional UI** - Clean, modern interface
- **Export Capabilities** - CSV export for all data

## 🚀 Deployment

### PythonAnywhere
1. Upload files to your PythonAnywhere account
2. Configure WSGI file: `deployment/wsgi_final.py`
3. Install dependencies: `pip3.10 install --user -r deployment/requirements_production.txt`
4. Reload web app

### Docker (Optional)
```bash
# Build image
docker build -t trading-dashboard .

# Run container
docker run -p 5000:5000 trading-dashboard
```

## 📈 Features Overview

### Technical Analysis
- RSI, MACD, Bollinger Bands
- Moving Averages (SMA, EMA, WMA)
- Support and Resistance levels
- Chart pattern recognition

### Options Analysis
- Black-Scholes pricing
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Volatility analysis
- Options chain visualization
- Strategy analysis

### Risk Management
- Position sizing calculator
- Portfolio optimization
- Risk metrics (VaR, Sharpe ratio)
- Drawdown analysis

### Compliance & Audit
- Complete audit trail
- Regulatory reporting
- Risk monitoring
- Data integrity checks

## 🔒 Security

- Environment variable configuration
- Secure API key management
- Audit trail logging
- Data encryption
- Access control

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the deployment guide

## 🎯 Roadmap

- [ ] Mobile app version
- [ ] Advanced AI/ML integration
- [ ] Multi-broker support
- [ ] Social trading features
- [ ] Advanced backtesting
- [ ] Real-time alerts

---

**Professional Trading Dashboard** - Built for serious traders and financial professionals.