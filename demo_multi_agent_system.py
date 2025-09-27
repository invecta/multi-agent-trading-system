"""
Multi-Agent Trading System Demo
Demonstrates the capabilities of the comprehensive trading system
"""
import asyncio
import logging
import json
from datetime import datetime
from multi_agent_trading_system import MultiAgentTradingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_single_trading_cycle():
    """Demo single trading cycle execution"""
    print("🚀 Multi-Agent Trading System Demo")
    print("=" * 50)
    
    # Configuration for demo
    config = {
        'symbols': ['AAPL', 'GOOGL', 'BTC-USD'],
        'timeframes': ['1d', '1h'],
        'execution_mode': 'paper',
        'initial_capital': 100000.0,
        'max_positions': 5,
        'max_position_size': 0.03,
        'max_portfolio_var': 0.02,
        'max_drawdown': 0.08,
        'enable_alpaca': False,  # Use demo broker for demo
        'enable_demo': True
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbols: {config['symbols']}")
    print(f"   Execution Mode: {config['execution_mode']}")
    print(f"   Initial Capital: ${config['initial_capital']:,}")
    print(f"   Max Positions: {config['max_positions']}")
    print()
    
    # Create trading system
    trading_system = MultiAgentTradingSystem(config)
    
    try:
        # Start system
        print("🔄 Starting Multi-Agent Trading System...")
        await trading_system.start_system()
        print("✅ System started successfully!")
        print()
        
        # Run trading cycle
        print("📈 Executing Trading Cycle...")
        start_time = datetime.now()
        
        result = await trading_system.run_trading_cycle()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("✅ Trading cycle completed!")
        print()
        
        # Display results
        print("📊 Trading Cycle Results:")
        print("-" * 30)
        print(f"Status: {result['workflow_result'].status.value}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Completed Nodes: {len(result['workflow_result'].completed_nodes)}")
        print(f"Failed Nodes: {len(result['workflow_result'].failed_nodes)}")
        
        if result['workflow_result'].failed_nodes:
            print(f"Failed Nodes: {list(result['workflow_result'].failed_nodes)}")
        
        print()
        
        # System statistics
        stats = result['system_stats']
        print("📈 System Statistics:")
        print("-" * 30)
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Successful Executions: {stats['successful_executions']}")
        print(f"Failed Executions: {stats['failed_executions']}")
        print(f"Total Trades Executed: {stats['total_trades_executed']}")
        
        if stats['total_executions'] > 0:
            success_rate = stats['successful_executions'] / stats['total_executions']
            print(f"Success Rate: {success_rate:.2%}")
        
        print()
        
        # Execution statistics
        exec_stats = result['execution_stats']
        print("⚡ Execution Statistics:")
        print("-" * 30)
        print(f"Total Orders: {exec_stats['total_orders']}")
        print(f"Success Rate: {exec_stats['success_rate']:.2%}")
        print(f"Total Volume: ${exec_stats['total_volume']:.2f}")
        print(f"Total Commission: ${exec_stats['total_commission']:.2f}")
        print(f"Average Execution Time: {exec_stats['average_execution_time']:.3f}s")
        
        print()
        
        # Portfolio summary
        portfolio_summary = trading_system.get_portfolio_summary()
        if portfolio_summary:
            print("💼 Portfolio Summary:")
            print("-" * 30)
            portfolio = portfolio_summary.get('portfolio', {})
            if portfolio:
                print(f"Total Value: ${portfolio.get('total_value', 0):.2f}")
                print(f"Cash: ${portfolio.get('cash', 0):.2f}")
                print(f"Total P&L: ${portfolio.get('total_pnl', 0):.2f}")
                print(f"Total Return: {portfolio.get('total_return', 0):.2%}")
                print(f"Positions: {len(portfolio.get('positions', {}))}")
        
        print()
        
        # Workflow details
        workflow_result = result['workflow_result']
        print("🔧 Workflow Details:")
        print("-" * 30)
        for node_id, node in workflow_result.nodes.items():
            status_emoji = "✅" if node.status.value == "completed" else "❌"
            print(f"{status_emoji} {node_id}: {node.status.value} ({node.execution_time:.2f}s)")
        
        print()
        
        # Performance insights
        print("💡 Performance Insights:")
        print("-" * 30)
        
        if execution_time < 10:
            print("⚡ Excellent performance - sub-10 second execution!")
        elif execution_time < 30:
            print("🚀 Good performance - under 30 seconds")
        else:
            print("⏱️  Consider optimizing - execution took over 30 seconds")
        
        if exec_stats['success_rate'] > 0.9:
            print("🎯 High success rate - system performing well!")
        elif exec_stats['success_rate'] > 0.7:
            print("👍 Good success rate - minor optimizations possible")
        else:
            print("⚠️  Low success rate - review configuration and agents")
        
        if stats['total_trades_executed'] > 0:
            print("💰 Trades executed successfully!")
        else:
            print("📊 No trades executed - check signal generation and risk filters")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")
    
    finally:
        # Stop system
        print("\n🛑 Stopping Multi-Agent Trading System...")
        await trading_system.stop_system()
        print("✅ System stopped successfully!")
        print()
        print("🎉 Demo completed! Thank you for exploring the Multi-Agent Trading System.")

async def demo_continuous_trading():
    """Demo continuous trading (simplified)"""
    print("🔄 Continuous Trading Demo")
    print("=" * 50)
    print("This demo would run continuous trading cycles.")
    print("For safety, this demo is limited to 3 cycles.")
    print()
    
    config = {
        'symbols': ['AAPL'],
        'execution_mode': 'paper',
        'initial_capital': 50000.0,
        'max_positions': 3
    }
    
    trading_system = MultiAgentTradingSystem(config)
    
    try:
        await trading_system.start_system()
        
        for cycle in range(3):
            print(f"🔄 Running cycle {cycle + 1}/3...")
            result = await trading_system.run_trading_cycle()
            print(f"✅ Cycle {cycle + 1} completed: {result['workflow_result'].status.value}")
            
            if cycle < 2:  # Don't wait after last cycle
                print("⏳ Waiting 10 seconds before next cycle...")
                await asyncio.sleep(10)
        
        print("🎉 Continuous trading demo completed!")
        
    except Exception as e:
        print(f"❌ Continuous trading demo failed: {e}")
    
    finally:
        await trading_system.stop_system()

async def main():
    """Main demo function"""
    print("🎯 Multi-Agent Trading System Demo")
    print("Choose demo mode:")
    print("1. Single Trading Cycle")
    print("2. Continuous Trading (3 cycles)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            await demo_single_trading_cycle()
        elif choice == "2":
            await demo_continuous_trading()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Running single cycle demo...")
            await demo_single_trading_cycle()
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
