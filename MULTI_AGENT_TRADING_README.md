# Multi-Agent Trading System

A comprehensive, enterprise-grade multi-agent trading system that implements advanced AI-driven trading strategies with real-time execution capabilities.

## 🚀 System Overview

This multi-agent trading system represents a paradigm shift in algorithmic trading, combining:

- **Six Specialized Agents**: Market Data, Technical Analysis, Fundamentals, Sentiment, Risk Management, and Portfolio Management
- **Real-Time Execution**: Low-latency trade execution with multiple broker integrations
- **Advanced AI Integration**: LangChain orchestration and OpenAI Swarm coordination
- **Comprehensive Risk Management**: VaR, CVaR, drawdown analysis, and stress testing
- **Parallel Processing**: Async workflow orchestration for maximum efficiency
- **Multi-Broker Support**: Alpaca, Interactive Brokers, and demo trading capabilities

## 🏗️ Architecture

### Core Components

1. **Multi-Agent Framework** (`multi_agent_framework.py`)
   - Base agent classes and orchestration
   - Dependency management and workflow coordination
   - Standardized data structures and interfaces

2. **Enhanced Market Data Agent** (`enhanced_market_data_agent.py`)
   - Real-time streaming with Kafka/Flink integration
   - WebSocket connections for live data feeds
   - Order book and tick data processing

3. **Advanced Technical Analysis Agent** (`enhanced_technical_analysis_agent.py`)
   - Trend Following (EMAs, ADX, Ichimoku)
   - Mean Reversion (Bollinger Bands, RSI, MACD)
   - Momentum strategies (price/volume dynamics)
   - Volatility analysis (ATR, regime detection)
   - Statistical Arbitrage (correlation, mean reversion)

4. **Sentiment Analysis Agent** (`enhanced_sentiment_agent.py`)
   - NLP models with Hugging Face Transformers
   - Vector database integration (Pinecone)
   - News and social media sentiment analysis
   - Real-time sentiment tracking

5. **Risk Manager Agent** (`enhanced_risk_manager_agent.py`)
   - Value at Risk (VaR) calculations
   - Conditional VaR (CVaR) analysis
   - Maximum drawdown monitoring
   - Portfolio optimization and stress testing

6. **Portfolio Manager Agent** (`enhanced_portfolio_manager_agent.py`)
   - Advanced portfolio optimization
   - Trade execution and position management
   - Stop-loss and take-profit automation
   - Performance tracking and analytics

7. **LangChain Integration** (`langchain_integration.py`)
   - Enhanced decision-making workflows
   - Context-aware reasoning
   - Multi-agent coordination
   - Advanced prompt engineering

8. **Async Workflow Orchestrator** (`async_workflow_orchestrator.py`)
   - Parallel agent execution
   - Dependency management
   - Circuit breaker patterns
   - Performance monitoring

9. **Real-Time Execution Engine** (`real_time_execution_engine.py`)
   - Multi-broker integration
   - Low-latency order execution
   - Pre/post-trade validation
   - Execution monitoring and analytics

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git (for cloning the repository)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi-agent-trading-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp config.json config_local.json
   # Edit config_local.json with your API keys
   ```

4. **Initialize the system**
   ```bash
   python multi_agent_trading_system.py --mode single --config config_local.json
   ```

## 🎯 Usage

### Command Line Interface

```bash
# Run single trading cycle
python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL BTC-USD

# Run continuous trading (5-minute intervals)
python multi_agent_trading_system.py --mode continuous --interval 5

# Use custom configuration
python multi_agent_trading_system.py --config my_config.json --execution-mode paper
```

### Programmatic Usage

```python
import asyncio
from multi_agent_trading_system import MultiAgentTradingSystem

async def main():
    # Load configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'BTC-USD'],
        'execution_mode': 'paper',
        'initial_capital': 100000.0
    }
    
    # Create and start system
    trading_system = MultiAgentTradingSystem(config)
    await trading_system.start_system()
    
    # Run trading cycle
    result = await trading_system.run_trading_cycle()
    print(f"Trading cycle completed: {result['workflow_result'].status.value}")
    
    # Stop system
    await trading_system.stop_system()

asyncio.run(main())
```

## 📊 Key Features

### Advanced Trading Strategies

- **Trend Following**: EMA crossovers, ADX trend strength, Ichimoku analysis
- **Mean Reversion**: Bollinger Bands, RSI oversold/overbought, MACD divergence
- **Momentum**: Price/volume dynamics, breakout detection, rate of change
- **Volatility**: ATR analysis, volatility regime detection, volatility breakout
- **Statistical Arbitrage**: Correlation analysis, mean reversion, z-score signals

### Risk Management

- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall calculations
- **Maximum Drawdown**: Real-time drawdown monitoring
- **Stress Testing**: Scenario analysis and market shock simulation
- **Portfolio Optimization**: Mean-variance optimization and risk parity

### Real-Time Execution

- **Multi-Broker Support**: Alpaca, Interactive Brokers, demo trading
- **Low-Latency Execution**: Sub-second order processing
- **Pre-Trade Validation**: Risk checks and position sizing
- **Post-Trade Validation**: Execution quality monitoring
- **Slippage Control**: Tolerance-based execution filtering

### AI Integration

- **LangChain Workflows**: Context-aware decision making
- **OpenAI Swarm**: Multi-agent coordination
- **Sentiment Analysis**: NLP-powered market sentiment
- **Vector Databases**: Historical sentiment storage and retrieval
- **Advanced Reasoning**: Chain-of-thought decision processes

## 🔧 Configuration

### System Configuration

```json
{
  "trading_config": {
    "symbols": ["EURUSD=X", "AAPL", "BTC-USD"],
    "execution_mode": "paper",
    "initial_capital": 100000.0,
    "max_position_size": 0.05,
    "max_drawdown": 0.10
  },
  "agent_config": {
    "max_concurrent_agents": 5,
    "agent_timeout": 30.0,
    "enable_circuit_breaker": true
  },
  "execution_config": {
    "max_order_size": 100000.0,
    "slippage_tolerance": 0.001,
    "commission_rate": 0.001
  }
}
```

### API Keys Configuration

```json
{
  "api_keys": {
    "openai_api_key": "your_openai_api_key_here",
    "alpaca_api_key": "your_alpaca_api_key_here",
    "alpaca_secret_key": "your_alpaca_secret_key_here",
    "newsapi_key": "your_newsapi_key_here",
    "pinecone_api_key": "your_pinecone_api_key_here"
  }
}
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Execution Statistics**: Order success rate, average execution time, slippage
- **Risk Metrics**: VaR, CVaR, maximum drawdown, Sharpe ratio
- **Portfolio Performance**: Total return, volatility, win rate
- **System Performance**: Agent execution times, workflow completion rates
- **Trading Statistics**: Total trades, volume, commission costs

## 🚨 Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results.

**Always**:
- Use proper risk management
- Test strategies thoroughly with paper trading
- Start with small position sizes
- Never risk more than you can afford to lose
- Comply with all applicable financial regulations

## 🔍 Monitoring and Alerts

The system includes comprehensive monitoring capabilities:

- **Real-Time Monitoring**: Live execution tracking and performance metrics
- **Alert System**: Configurable alerts for risk violations and system issues
- **Performance Dashboard**: Interactive visualization of system performance
- **Logging**: Detailed logging for debugging and analysis
- **Circuit Breakers**: Automatic system protection against failures

## 🛡️ Security and Compliance

- **API Key Management**: Secure storage and rotation of API keys
- **Data Encryption**: Encrypted storage of sensitive data
- **Audit Trails**: Complete transaction and decision logging
- **Regulatory Compliance**: Built-in compliance checks and reporting
- **Access Control**: Role-based access to system components

## 📚 Documentation

### API Reference

- `multi_agent_framework.py`: Core agent framework and orchestration
- `enhanced_market_data_agent.py`: Real-time data collection and streaming
- `enhanced_technical_analysis_agent.py`: Advanced technical analysis strategies
- `enhanced_sentiment_agent.py`: NLP-powered sentiment analysis
- `enhanced_risk_manager_agent.py`: Comprehensive risk management
- `enhanced_portfolio_manager_agent.py`: Portfolio optimization and execution
- `langchain_integration.py`: AI-powered decision making
- `async_workflow_orchestrator.py`: Parallel workflow execution
- `real_time_execution_engine.py`: Multi-broker trade execution

### Examples

- `examples/single_cycle.py`: Single trading cycle execution
- `examples/continuous_trading.py`: Continuous trading implementation
- `examples/custom_strategy.py`: Custom strategy development
- `examples/risk_analysis.py`: Risk analysis and reporting

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the documentation and examples
2. Review the logs in `multi_agent_trading.log`
3. Check the configuration in `config.json`
4. Ensure all dependencies are installed correctly
5. Verify API keys are configured properly

## 🎉 Acknowledgments

This system builds upon the work of many open-source projects and research in:

- Multi-agent systems and AI orchestration
- Financial market analysis and trading strategies
- Real-time data processing and streaming
- Risk management and portfolio optimization
- Natural language processing and sentiment analysis

---

**Happy Trading! 📈**

*Remember: This is a sophisticated trading system. Always test thoroughly and trade responsibly.*
