#!/usr/bin/env python3
"""
Update Personal Configuration with Alpaca API Keys
Creates a personal config file with your Alpaca credentials
"""
import json
import os
from datetime import datetime

def create_personal_config():
    """Create personal configuration file with Alpaca API keys"""
    
    # Your Alpaca credentials
    alpaca_config = {
        "system_config": {
            "name": "Multi-Agent Trading System - Personal",
            "version": "1.0.0",
            "description": "Personal trading system with Alpaca integration",
            "created": datetime.now().isoformat()
        },
        
        "trading_config": {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "timeframes": ["1d", "1h", "4h"],
            "execution_mode": "paper",
            "initial_capital": 100000.0,
            "max_positions": 5,
            "max_position_size": 0.03,
            "max_portfolio_var": 0.02,
            "max_drawdown": 0.08,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.04,
            "enable_alpaca": True,
            "enable_demo": True
        },
        
        "agent_config": {
            "max_concurrent_agents": 3,
            "agent_timeout": 30.0,
            "enable_circuit_breaker": True,
            "retry_delay": 1.0
        },
        
        "market_data_config": {
            "data_sources": ["rest_api"],
            "update_frequency": 5.0,
            "buffer_size": 1000,
            "enable_orderbook": True,
            "enable_tick_data": True
        },
        
        "execution_config": {
            "max_order_size": 50000.0,
            "max_daily_trades": 100,
            "slippage_tolerance": 0.001,
            "commission_rate": 0.001,
            "enable_pre_trade_checks": True,
            "enable_post_trade_validation": True
        },
        
        "risk_config": {
            "risk_free_rate": 0.02,
            "lookback_periods": 252,
            "confidence_levels": [0.95, 0.99],
            "max_correlation": 0.7,
            "max_sector_exposure": 0.3
        },
        
        "api_keys": {
            "openai_api_key": "your_openai_api_key_here",
            "alpaca_api_key": "PKOEKMI4RY0LHF565WDO",
            "alpaca_secret_key": "Dq14y4AJpsIqFfJ33FWKWvdJw9zqrAPsaLtJhdDb",
            "alpaca_account_id": "PA3TE0S55RX2",
            "newsapi_key": "your_newsapi_key_here",
            "alpha_vantage_key": "your_alpha_vantage_key_here",
            "pinecone_api_key": "your_pinecone_api_key_here"
        },
        
        "broker_config": {
            "alpaca": {
                "base_url": "https://paper-api.alpaca.markets/v2",
                "websocket_url": "wss://paper-api.alpaca.markets/stream",
                "sandbox": True,
                "rate_limit": 100,
                "timeout": 5.0,
                "account_id": "PA3TE0S55RX2"
            },
            "demo": {
                "enabled": True,
                "simulate_execution": True
            }
        },
        
        "logging_config": {
            "level": "INFO",
            "file": "multi_agent_trading.log",
            "max_size": "10MB",
            "backup_count": 5
        },
        
        "performance_config": {
            "enable_monitoring": True,
            "metrics_interval": 60,
            "max_history_days": 30,
            "enable_alerts": True
        },
        
        "workflow_config": {
            "default_workflow": "trading_workflow",
            "workflows": {
                "trading_workflow": {
                    "agents": [
                        {
                            "agent_id": "market_data_agent",
                            "dependencies": [],
                            "priority": "medium",
                            "timeout": 30.0,
                            "max_retries": 3
                        },
                        {
                            "agent_id": "technical_analysis_agent",
                            "dependencies": ["market_data_agent"],
                            "priority": "high",
                            "timeout": 45.0,
                            "max_retries": 2
                        },
                        {
                            "agent_id": "sentiment_agent",
                            "dependencies": ["market_data_agent"],
                            "priority": "high",
                            "timeout": 60.0,
                            "max_retries": 2
                        },
                        {
                            "agent_id": "risk_manager_agent",
                            "dependencies": ["technical_analysis_agent", "sentiment_agent"],
                            "priority": "critical",
                            "timeout": 30.0,
                            "max_retries": 1
                        },
                        {
                            "agent_id": "portfolio_manager_agent",
                            "dependencies": ["risk_manager_agent"],
                            "priority": "critical",
                            "timeout": 30.0,
                            "max_retries": 1
                        }
                    ]
                }
            }
        }
    }
    
    # Save personal configuration
    config_file = "my_personal_config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(alpaca_config, f, indent=2)
        
        print("✅ Personal configuration created successfully!")
        print(f"📄 Configuration saved to: {config_file}")
        print()
        print("🔑 Your Alpaca API credentials have been configured:")
        print(f"   API Key: PKOEKMI4RY0LHF565WDO")
        print(f"   Account ID: PA3TE0S55RX2")
        print(f"   Endpoint: https://paper-api.alpaca.markets/v2")
        print()
        print("📋 Next Steps:")
        print("1. Test your Alpaca connection:")
        print("   python test_alpaca_connection.py")
        print()
        print("2. Run the trading system with your personal config:")
        print(f"   python multi_agent_trading_system.py --config {config_file} --mode single")
        print()
        print("3. Or run the demo:")
        print("   python demo_multi_agent_system.py")
        print()
        print("🎯 Your system is now configured for Alpaca paper trading!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating personal configuration: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Multi-Agent Trading System - Personal Configuration Setup")
    print("=" * 60)
    print("This script will create a personal configuration file")
    print("with your Alpaca API credentials.")
    print()
    
    success = create_personal_config()
    
    if success:
        print("\n🎉 Configuration setup completed!")
        print("You can now use your Multi-Agent Trading System with Alpaca!")
    else:
        print("\n❌ Configuration setup failed!")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()
