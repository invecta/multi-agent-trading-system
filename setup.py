#!/usr/bin/env python3
"""
Multi-Agent Trading System Setup Script
Automates the initial setup process
"""
import os
import sys
import subprocess
import json
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    if os.path.exists("trading_env"):
        print("📁 Virtual environment already exists, skipping creation...")
        return True
    
    return run_command("python -m venv trading_env", "Creating virtual environment")

def get_activation_command():
    """Get the correct activation command based on OS"""
    if platform.system() == "Windows":
        return "trading_env\\Scripts\\activate"
    else:
        return "source trading_env/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    # Determine pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = "trading_env\\Scripts\\pip"
    else:
        pip_cmd = "trading_env/bin/pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def create_user_config():
    """Create user configuration file"""
    config_file = "my_config.json"
    
    if os.path.exists(config_file):
        print(f"📄 {config_file} already exists, skipping creation...")
        return True
    
    # Create a basic configuration
    config = {
        "trading_config": {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "timeframes": ["1d", "1h"],
            "execution_mode": "paper",
            "initial_capital": 100000.0,
            "max_positions": 5,
            "max_position_size": 0.03,
            "max_drawdown": 0.08
        },
        "agent_config": {
            "max_concurrent_agents": 3,
            "agent_timeout": 30.0,
            "enable_circuit_breaker": True
        },
        "execution_config": {
            "max_order_size": 50000.0,
            "slippage_tolerance": 0.001,
            "commission_rate": 0.001
        },
        "api_keys": {
            "openai_api_key": "demo-key",
            "alpaca_api_key": "demo-key",
            "alpaca_secret_key": "demo-key",
            "newsapi_key": "demo-key",
            "pinecone_api_key": "demo-key"
        },
        "broker_config": {
            "demo": {
                "enabled": True,
                "simulate_execution": True
            }
        }
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created {config_file} with demo configuration!")
        return True
    except Exception as e:
        print(f"❌ Failed to create {config_file}: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    # Test basic imports
    test_script = """
try:
    import pandas as pd
    import numpy as np
    import asyncio
    import yfinance as yf
    print("✅ Core packages imported successfully!")
    
    # Test system import
    from multi_agent_trading_system import MultiAgentTradingSystem
    print("✅ Multi-Agent Trading System imported successfully!")
    
    print("🎉 All tests passed! System is ready to use.")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)
"""
    
    # Determine python command based on OS
    if platform.system() == "Windows":
        python_cmd = "trading_env\\Scripts\\python"
    else:
        python_cmd = "trading_env/bin/python"
    
    try:
        result = subprocess.run([python_cmd, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    activation_cmd = get_activation_command()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    
    print("\n2. Run the interactive demo:")
    print("   python demo_multi_agent_system.py")
    
    print("\n3. Run a single trading cycle:")
    print("   python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL")
    
    print("\n4. Run continuous trading:")
    print("   python multi_agent_trading_system.py --mode continuous --interval 5")
    
    print("\n5. Customize your configuration:")
    print("   Edit my_config.json to adjust settings")
    
    print("\n📚 Documentation:")
    print("   - README: MULTI_AGENT_TRADING_README.md")
    print("   - Setup Guide: STEP_BY_STEP_SETUP_GUIDE.md")
    print("   - System Summary: MULTI_AGENT_SYSTEM_SUMMARY.md")
    
    print("\n🆘 Need Help?")
    print("   - Check the logs: tail -f multi_agent_trading.log")
    print("   - Review configuration files")
    print("   - Read the documentation")
    
    print("\n🚨 Important:")
    print("   - System starts in PAPER TRADING mode (safe)")
    print("   - Configure API keys for live trading")
    print("   - Always test thoroughly before live trading")
    
    print("\n🎯 Happy Trading! 📈")

def main():
    """Main setup function"""
    print("🚀 Multi-Agent Trading System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create user configuration
    if not create_user_config():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
