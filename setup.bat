@echo off
echo 🚀 Multi-Agent Trading System - Quick Setup
echo ===========================================
echo.

echo 🐍 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo 🔄 Running automated setup...
python setup.py

echo.
echo ✅ Setup completed! 
echo.
echo 📋 To get started:
echo 1. Activate virtual environment: trading_env\Scripts\activate
echo 2. Run demo: python demo_multi_agent_system.py
echo 3. Run trading: python multi_agent_trading_system.py --mode single
echo.
pause
