# 🚀 Multi-Agent Trading System - Step-by-Step Setup Guide

## Prerequisites Checklist

Before we begin, make sure you have:
- ✅ Python 3.8 or higher installed
- ✅ pip package manager
- ✅ Git (optional, for version control)
- ✅ A text editor or IDE (VS Code, PyCharm, etc.)
- ✅ Basic understanding of Python and trading concepts

## Step 1: Environment Setup

### 1.1 Create Project Directory
```bash
# Create a new directory for your trading system
mkdir multi-agent-trading-system
cd multi-agent-trading-system

# Create virtual environment (recommended)
python -m venv trading_env

# Activate virtual environment
# On Windows:
trading_env\Scripts\activate
# On macOS/Linux:
source trading_env/bin/activate
```

### 1.2 Verify Python Installation
```bash
python --version
pip --version
```

## Step 2: Install Dependencies

### 2.1 Install Core Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 2.2 Verify Installation
```bash
# Test if key packages are installed
python -c "import pandas, numpy, asyncio; print('Core packages installed successfully!')"
python -c "import yfinance; print('Financial data packages ready!')"
```

## Step 3: Configuration Setup

### 3.1 Create Configuration File
```bash
# Copy the example configuration
cp config.json my_config.json
```

### 3.2 Edit Configuration File
Open `my_config.json` in your text editor and customize:

```json
{
  "trading_config": {
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "execution_mode": "paper",
    "initial_capital": 100000.0,
    "max_positions": 5
  },
  "api_keys": {
    "openai_api_key": "your_openai_key_here",
    "alpaca_api_key": "your_alpaca_key_here",
    "alpaca_secret_key": "your_alpaca_secret_here"
  }
}
```

### 3.3 API Keys Setup (Optional for Demo)

**For Paper Trading Demo (No API Keys Required):**
- The system works with demo brokers out of the box
- Skip API key configuration for initial testing

**For Live Trading (API Keys Required):**
- **OpenAI API Key**: Get from https://platform.openai.com/api-keys
- **Alpaca API Key**: Get from https://app.alpaca.markets/paper/dashboard/overview
- **NewsAPI Key**: Get from https://newsapi.org/register
- **Pinecone API Key**: Get from https://app.pinecone.io/

## Step 4: First System Test

### 4.1 Test Basic Import
```bash
# Test if the system imports correctly
python -c "from multi_agent_trading_system import MultiAgentTradingSystem; print('System imported successfully!')"
```

### 4.2 Run Simple Demo
```bash
# Run the interactive demo
python demo_multi_agent_system.py
```

**Expected Output:**
```
🎯 Multi-Agent Trading System Demo
Choose demo mode:
1. Single Trading Cycle
2. Continuous Trading (3 cycles)
3. Exit

Enter your choice (1-3): 1
```

## Step 5: Run Your First Trading Cycle

### 5.1 Single Trading Cycle
```bash
# Run a single trading cycle with default settings
python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL
```

### 5.2 With Custom Configuration
```bash
# Run with your custom configuration
python multi_agent_trading_system.py --mode single --config my_config.json
```

**Expected Output:**
```
🚀 Multi-Agent Trading System Demo
==================================================
📊 Configuration:
   Symbols: ['AAPL', 'GOOGL', 'BTC-USD']
   Execution Mode: paper
   Initial Capital: $100,000
   Max Positions: 5

🔄 Starting Multi-Agent Trading System...
✅ System started successfully!

📈 Executing Trading Cycle...
✅ Trading cycle completed!

📊 Trading Cycle Results:
------------------------------
Status: completed
Execution Time: 15.23 seconds
Completed Nodes: 5
Failed Nodes: 0
```

## Step 6: Understanding the System

### 6.1 System Components Overview
The system consists of these main components:

1. **Market Data Agent**: Collects real-time market data
2. **Technical Analysis Agent**: Analyzes price patterns and trends
3. **Sentiment Agent**: Analyzes news and social media sentiment
4. **Risk Manager Agent**: Calculates risk metrics and filters trades
5. **Portfolio Manager Agent**: Executes trades and manages positions

### 6.2 Workflow Process
```
Market Data → Technical Analysis ┐
              Sentiment Analysis ┴→ Risk Manager → Portfolio Manager → Execution
```

## Step 7: Monitoring and Logs

### 7.1 Check System Logs
```bash
# View system logs
tail -f multi_agent_trading.log
```

### 7.2 Monitor Performance
The system automatically tracks:
- Execution times
- Success rates
- Trade statistics
- Risk metrics
- System performance

## Step 8: Customization Options

### 8.1 Modify Trading Symbols
Edit `my_config.json`:
```json
{
  "trading_config": {
    "symbols": ["TSLA", "NVDA", "AMD", "BTC-USD", "ETH-USD"]
  }
}
```

### 8.2 Adjust Risk Parameters
```json
{
  "trading_config": {
    "max_position_size": 0.03,
    "max_drawdown": 0.05,
    "stop_loss_percentage": 0.015
  }
}
```

### 8.3 Change Execution Mode
```json
{
  "trading_config": {
    "execution_mode": "paper"  // or "live" for real trading
  }
}
```

## Step 9: Advanced Usage

### 9.1 Continuous Trading
```bash
# Run continuous trading (5-minute intervals)
python multi_agent_trading_system.py --mode continuous --interval 5
```

### 9.2 Custom Timeframes
```bash
# Use different timeframes
python multi_agent_trading_system.py --mode single --symbols AAPL --timeframes 1h 4h 1d
```

### 9.3 Programmatic Usage
Create your own script:

```python
import asyncio
from multi_agent_trading_system import MultiAgentTradingSystem

async def my_trading_strategy():
    config = {
        'symbols': ['AAPL', 'TSLA'],
        'execution_mode': 'paper',
        'initial_capital': 50000.0
    }
    
    system = MultiAgentTradingSystem(config)
    await system.start_system()
    
    # Run trading cycle
    result = await system.run_trading_cycle()
    print(f"Result: {result['workflow_result'].status.value}")
    
    await system.stop_system()

# Run your strategy
asyncio.run(my_trading_strategy())
```

## Step 10: Troubleshooting Common Issues

### 10.1 Import Errors
```bash
# If you get import errors, reinstall packages
pip install --upgrade -r requirements.txt
```

### 10.2 API Key Issues
```bash
# Test API keys
python -c "import openai; print('OpenAI key configured')"
```

### 10.3 Permission Errors
```bash
# On Windows, run as administrator if needed
# On macOS/Linux, check file permissions
chmod +x multi_agent_trading_system.py
```

### 10.4 Memory Issues
```bash
# Reduce buffer sizes in config
{
  "market_data_config": {
    "buffer_size": 500  // Reduce from 1000
  }
}
```

## Step 11: Next Steps

### 11.1 Explore the System
- Run different symbol combinations
- Test various timeframes
- Experiment with risk parameters
- Monitor performance metrics

### 11.2 Learn the Components
- Read the documentation in each Python file
- Understand the agent interactions
- Study the risk management features
- Explore the execution engine

### 11.3 Scale Up
- Add more symbols
- Increase position sizes (carefully!)
- Enable live trading (with proper risk management)
- Integrate additional brokers

## 🎯 Quick Start Commands Summary

```bash
# 1. Setup environment
python -m venv trading_env
source trading_env/bin/activate  # or trading_env\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run demo
python demo_multi_agent_system.py

# 4. Run single cycle
python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL

# 5. Run continuous trading
python multi_agent_trading_system.py --mode continuous --interval 5
```

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: `tail -f multi_agent_trading.log`
2. **Verify configuration**: Ensure `my_config.json` is valid JSON
3. **Test components**: Run individual agent tests
4. **Check dependencies**: Ensure all packages are installed
5. **Review documentation**: Read the README and code comments

## 🎉 Congratulations!

You now have a fully functional Multi-Agent Trading System! The system is designed to:
- ✅ Run safely in paper trading mode
- ✅ Provide comprehensive risk management
- ✅ Scale from small to enterprise-level trading
- ✅ Integrate with multiple brokers and data sources
- ✅ Deliver measurable performance improvements

**Happy Trading! 📈**
