# Multi-Agent Trading System - Data Analyst Portfolio Showcase

A comprehensive AI-powered trading system demonstrating advanced data analysis, machine learning, and financial modeling capabilities. This project showcases expertise in quantitative finance, risk management, and interactive data visualization.

## 🎯 Data Analyst Portfolio Highlights

### 📊 Advanced Analytics & Visualization
- **Interactive Dashboards** - Professional-grade financial dashboards with real-time updates
- **Statistical Analysis** - Comprehensive risk metrics, performance attribution, and correlation analysis
- **Sector Analysis** - Cross-sector performance comparison with industry benchmarking
- **Technical Indicators** - Advanced charting with Bollinger Bands, MACD, RSI, and custom signals

### 🔬 Machine Learning & AI
- **Multi-Agent System** - AI-powered decision making with specialized agents
- **Signal Generation** - Machine learning models for trade signal prediction
- **Risk Management** - Automated position sizing and drawdown protection
- **Sentiment Analysis** - NLP-based market sentiment integration

### 💼 Professional Features
- **Comprehensive Reporting** - Executive-level data analysis reports
- **API Integration** - Real-time market data from multiple sources
- **Backtesting Framework** - Historical performance validation
- **Risk Metrics** - Sharpe ratio, Sortino ratio, VaR, and Calmar ratio calculations

## 🚀 Technical Features

### 🤖 Multi-Agent Architecture
- **Market Data Agent** - Real-time data collection and streaming
- **Technical Analysis Agent** - Advanced trading strategies and indicators
- **Sentiment Agent** - NLP-powered market sentiment analysis
- **Risk Manager Agent** - VaR, CVaR, and drawdown calculations
- **Portfolio Manager Agent** - Trade execution and portfolio optimization
- **Fundamentals Agent** - Financial statement and macroeconomic analysis

### 📊 Professional Dashboard Suite
- **Interactive Analytics Dashboard** - Real-time backtesting with advanced visualizations
- **Sector Analysis Dashboard** - Cross-sector performance comparison
- **Risk Management Dashboard** - Comprehensive risk metrics and monitoring
- **Technical Analysis Dashboard** - Advanced charting with multiple indicators
- **Performance Attribution Dashboard** - Detailed return analysis and reporting

### 🔌 API Integration
- **Alpaca Trading API** - Paper and live trading execution
- **Polygon.io** - Real-time and historical market data
- **OpenAI Swarm** - Agent orchestration and coordination
- **LangChain** - Enhanced reasoning and workflow management

## 📋 Prerequisites

- Python 3.8+
- Git
- Web browser (for dashboard)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-trading-system.git
cd multi-agent-trading-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
cp config.example.json config.json
# Edit config.json with your API keys
```

### 4. Run the Dashboard
```bash
python ultra_simple_dashboard.py
```

## 🎯 Quick Start

### Interactive Dashboard
1. **Start the Dashboard:**
   ```bash
   python ultra_simple_dashboard.py
   ```
2. **Open Browser:** Navigate to `http://localhost:8055`
3. **Run Backtest:** Select symbol, set capital, click "Run Backtest"
4. **Analyze Results:** View performance metrics, charts, and trade history

### Command Line Backtesting
```bash
# Simple backtest
python simple_backtest.py

# Multi-agent backtest
python multi_agent_backtest.py

# Test multiple symbols
python test_multiple_symbols.py
```

### Multi-Agent System Demo
```bash
# Run the complete multi-agent workflow
python demo_multi_agent_system.py

# Simple demo without external dependencies
python simple_demo.py
```

## 📊 Dashboard Features

### Performance Metrics
- **Total Return** - Overall profit/loss percentage
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Largest loss from peak
- **Total Trades** - Number of transactions
- **Final Value** - Ending portfolio value

### Interactive Charts
- **Portfolio Performance** - Growth over time
- **Drawdown Analysis** - Risk visualization
- **Risk Metrics** - Comparative analysis
- **Trade History** - Transaction details

### Supported Symbols
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla Inc.
- **NVDA** - NVIDIA Corporation
- **AMZN** - Amazon.com Inc.
- **META** - Meta Platforms Inc.
- **NFLX** - Netflix Inc.

## 🔧 Configuration

### API Keys Setup
1. **Alpaca Trading:**
   - Get API keys from [Alpaca Markets](https://alpaca.markets/)
   - Add to `config.json` under `api_keys`

2. **Polygon.io:**
   - Get API key from [Polygon.io](https://polygon.io/)
   - Add to `config.json` under `api_keys`

3. **OpenAI:**
   - Get API key from [OpenAI](https://openai.com/)
   - Add to `config.json` under `api_keys`

### Trading Configuration
- **Execution Mode:** Paper trading (default) or live trading
- **Initial Capital:** Starting portfolio value
- **Risk Limits:** Maximum drawdown, position sizes
- **Symbols:** Assets to trade
- **Timeframes:** Data granularity

## 📁 Project Structure

```
multi-agent-trading-system/
├── 📊 Dashboard
│   ├── ultra_simple_dashboard.py      # Working dashboard
│   ├── simulated_dashboard.py         # Simulated data version
│   └── fixed_dashboard.py             # Fixed version
├── 🤖 Multi-Agent Framework
│   ├── multi_agent_framework.py       # Core framework
│   ├── langchain_integration.py       # LangChain integration
│   └── enhanced_*_agent.py           # Individual agents
├── 📈 Backtesting
│   ├── multi_agent_backtest.py        # Advanced backtesting
│   ├── simple_backtest.py            # Basic backtesting
│   └── test_*.py                     # Testing scripts
├── 🔌 API Integration
│   ├── test_alpaca_connection.py      # Alpaca API test
│   ├── test_polygon_connection.py     # Polygon API test
│   └── polygon_market_data_agent.py   # Polygon integration
├── 📋 Configuration
│   ├── config.json                    # Main configuration
│   ├── config.example.json           # Example configuration
│   └── requirements.txt              # Dependencies
├── 📚 Documentation
│   ├── PROJECT_SUMMARY.md            # Project overview
│   ├── BACKTESTING_GUIDE.md          # Backtesting guide
│   └── DASHBOARD_GUIDE.md            # Dashboard guide
└── 🎮 Demos
    ├── demo_multi_agent_system.py     # Full system demo
    └── simple_demo.py                # Simple demo
```

## 🧪 Testing

### Test API Connections
```bash
# Test Alpaca connection
python test_alpaca_connection.py

# Test Polygon.io connection
python test_polygon_connection.py
```

### Test Trading Strategies
```bash
# Test different time periods
python test_time_periods.py

# Test custom strategies
python test_custom_strategy.py
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Total Return** - Overall portfolio performance
- **Annualized Return** - Yearly return rate
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- **Average Trade** - Mean profit/loss per trade

## 🔒 Security

- **API Keys:** Store in environment variables or secure config
- **Paper Trading:** Default mode for safe testing
- **Risk Limits:** Built-in position and drawdown controls
- **Validation:** Input validation and error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/multi-agent-trading-system/issues)
- **Documentation:** Check the guides in the `docs/` folder
- **Examples:** See the `examples/` folder for usage examples

## 🎉 Acknowledgments

- **Alpaca Markets** - Trading API and paper trading
- **Polygon.io** - Market data provider
- **OpenAI** - AI orchestration and reasoning
- **LangChain** - Workflow management
- **Dash/Plotly** - Interactive dashboard framework

---

**Built with ❤️ for the trading community**