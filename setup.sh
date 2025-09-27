#!/bin/bash

echo "🚀 Multi-Agent Trading System - Quick Setup"
echo "==========================================="
echo

echo "🐍 Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo
echo "🔄 Running automated setup..."
python3 setup.py

echo
echo "✅ Setup completed!"
echo
echo "📋 To get started:"
echo "1. Activate virtual environment: source trading_env/bin/activate"
echo "2. Run demo: python demo_multi_agent_system.py"
echo "3. Run trading: python multi_agent_trading_system.py --mode single"
echo
