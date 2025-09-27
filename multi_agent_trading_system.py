"""
Multi-Agent Trading System - Main Integration Module
Brings together all components into a comprehensive trading system
"""
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import sys
import os

# Import all components
from multi_agent_framework import (
    MultiAgentOrchestrator, 
    MarketDataAgent, 
    TechnicalAnalysisAgent, 
    FundamentalsAgent, 
    SentimentAgent, 
    RiskManagerAgent, 
    PortfolioManagerAgent
)

from langchain_integration import EnhancedMultiAgentOrchestrator
from enhanced_market_data_agent import EnhancedMarketDataAgent, MarketDataConfig
from enhanced_technical_analysis_agent import AdvancedTechnicalAnalysisAgent
from enhanced_sentiment_agent import EnhancedSentimentAgent
from enhanced_risk_manager_agent import EnhancedRiskManagerAgent
from enhanced_portfolio_manager_agent import EnhancedPortfolioManagerAgent
from async_workflow_orchestrator import AsyncWorkflowOrchestrator, WorkflowBuilder, ExecutionPriority
from real_time_execution_engine import (
    RealTimeExecutionEngine, 
    MultiBrokerExecutionEngine, 
    ExecutionConfig, 
    BrokerConfig, 
    BrokerType, 
    ExecutionMode,
    AlpacaBroker,
    DemoBroker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultiAgentTradingSystem:
    """Main trading system integrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.async_orchestrator = None
        self.langchain_orchestrator = None
        self.execution_engine = None
        self.workflow_orchestrator = None
        
        # System status
        self.is_running = False
        self.system_stats = {
            'start_time': None,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_signals_generated': 0,
            'total_trades_executed': 0
        }
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing Multi-Agent Trading System components...")
        
        # Initialize async workflow orchestrator
        self.workflow_orchestrator = AsyncWorkflowOrchestrator({
            'max_concurrent_agents': self.config.get('max_concurrent_agents', 5),
            'default_timeout': self.config.get('agent_timeout', 30.0),
            'enable_circuit_breaker': True
        })
        
        # Initialize enhanced agents
        self._initialize_enhanced_agents()
        
        # Initialize execution engine
        self._initialize_execution_engine()
        
        # Initialize LangChain orchestrator
        self._initialize_langchain_orchestrator()
        
        logger.info("All components initialized successfully")
    
    def _initialize_enhanced_agents(self):
        """Initialize enhanced agents"""
        # Market Data Agent
        market_data_config = MarketDataConfig(
            symbols=self.config.get('symbols', ['EURUSD=X', 'GBPUSD=X', 'AAPL', 'BTC-USD']),
            data_sources=self.config.get('data_sources', ['rest_api']),
            timeframes=self.config.get('timeframes', ['1m', '5m', '1h', '1d']),
            update_frequency=self.config.get('update_frequency', 5.0),
            buffer_size=self.config.get('buffer_size', 1000),
            enable_orderbook=True,
            enable_tick_data=True
        )
        
        self.market_data_agent = EnhancedMarketDataAgent(market_data_config)
        self.workflow_orchestrator.register_agent(self.market_data_agent)
        
        # Technical Analysis Agent
        self.technical_analysis_agent = AdvancedTechnicalAnalysisAgent()
        self.workflow_orchestrator.register_agent(self.technical_analysis_agent)
        
        # Sentiment Agent
        sentiment_config = {
            'pinecone_api_key': self.config.get('pinecone_api_key', 'demo-key'),
            'newsapi_key': self.config.get('newsapi_key', 'demo-key'),
            'alpha_vantage_key': self.config.get('alpha_vantage_key', 'demo-key')
        }
        self.sentiment_agent = EnhancedSentimentAgent(sentiment_config)
        self.workflow_orchestrator.register_agent(self.sentiment_agent)
        
        # Risk Manager Agent
        risk_config = {
            'max_position_size': self.config.get('max_position_size', 0.05),
            'max_portfolio_var': self.config.get('max_portfolio_var', 0.02),
            'max_drawdown': self.config.get('max_drawdown', 0.10),
            'risk_free_rate': self.config.get('risk_free_rate', 0.02)
        }
        self.risk_manager_agent = EnhancedRiskManagerAgent(risk_config)
        self.workflow_orchestrator.register_agent(self.risk_manager_agent)
        
        # Portfolio Manager Agent
        portfolio_config = {
            'initial_capital': self.config.get('initial_capital', 100000.0),
            'max_positions': self.config.get('max_positions', 10),
            'commission_rate': self.config.get('commission_rate', 0.001),
            'slippage_rate': self.config.get('slippage_rate', 0.0005),
            'stop_loss_percentage': self.config.get('stop_loss_percentage', 0.02),
            'take_profit_percentage': self.config.get('take_profit_percentage', 0.04)
        }
        self.portfolio_manager_agent = EnhancedPortfolioManagerAgent(portfolio_config)
        self.workflow_orchestrator.register_agent(self.portfolio_manager_agent)
    
    def _initialize_execution_engine(self):
        """Initialize execution engine"""
        execution_config = ExecutionConfig(
            mode=ExecutionMode(self.config.get('execution_mode', 'paper')),
            max_order_size=self.config.get('max_order_size', 100000.0),
            max_daily_trades=self.config.get('max_daily_trades', 1000),
            slippage_tolerance=self.config.get('slippage_tolerance', 0.001),
            commission_rate=self.config.get('commission_rate', 0.001)
        )
        
        self.execution_engine = RealTimeExecutionEngine(execution_config)
        
        # Register brokers
        if self.config.get('enable_alpaca', False):
            alpaca_config = BrokerConfig(
                broker_type=BrokerType.ALPACA,
                api_key=self.config.get('alpaca_api_key', 'demo'),
                secret_key=self.config.get('alpaca_secret_key', 'demo'),
                base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets'),
                websocket_url=self.config.get('alpaca_websocket_url', 'wss://paper-api.alpaca.markets/stream'),
                sandbox=True
            )
            alpaca_broker = AlpacaBroker(alpaca_config)
            self.execution_engine.register_broker("alpaca", alpaca_broker)
        
        # Always register demo broker for testing
        demo_config = BrokerConfig(
            broker_type=BrokerType.DEMO,
            api_key="demo",
            secret_key="demo",
            base_url="demo",
            websocket_url="demo"
        )
        demo_broker = DemoBroker(demo_config)
        self.execution_engine.register_broker("demo", demo_broker)
    
    def _initialize_langchain_orchestrator(self):
        """Initialize LangChain orchestrator"""
        langchain_config = {
            'openai_api_key': self.config.get('openai_api_key', 'demo-key')
        }
        self.langchain_orchestrator = EnhancedMultiAgentOrchestrator(
            openai_api_key=langchain_config['openai_api_key'],
            config=langchain_config
        )
    
    async def start_system(self):
        """Start the trading system"""
        logger.info("Starting Multi-Agent Trading System...")
        
        try:
            # Start execution engine
            await self.execution_engine.start()
            
            # Start market data streaming
            await self.market_data_agent.start_real_time_streaming()
            
            self.is_running = True
            self.system_stats['start_time'] = datetime.now()
            
            logger.info("Multi-Agent Trading System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            raise
    
    async def stop_system(self):
        """Stop the trading system"""
        logger.info("Stopping Multi-Agent Trading System...")
        
        try:
            self.is_running = False
            
            # Stop execution engine
            await self.execution_engine.stop()
            
            # Stop market data streaming
            await self.market_data_agent.stop_real_time_streaming()
            
            logger.info("Multi-Agent Trading System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Run a complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # Create workflow
            builder = WorkflowBuilder(self.workflow_orchestrator)
            execution_id = builder.add_agent('market_data_agent') \
                                .add_agent('technical_analysis_agent', ['market_data_agent'], ExecutionPriority.HIGH) \
                                .add_agent('sentiment_agent', ['market_data_agent'], ExecutionPriority.HIGH) \
                                .add_agent('risk_manager_agent', ['technical_analysis_agent', 'sentiment_agent'], ExecutionPriority.CRITICAL) \
                                .add_agent('portfolio_manager_agent', ['risk_manager_agent'], ExecutionPriority.CRITICAL) \
                                .build('trading_workflow')
            
            # Prepare input data
            input_data = {
                'symbols': self.config.get('symbols', ['EURUSD=X', 'AAPL', 'BTC-USD']),
                'timeframes': self.config.get('timeframes', ['1d', '1h']),
                'execution_mode': self.config.get('execution_mode', 'paper')
            }
            
            # Execute workflow
            workflow_result = await self.workflow_orchestrator.execute_workflow(execution_id, input_data)
            
            # Process portfolio manager output for execution
            if workflow_result.status.value == 'completed':
                portfolio_output = workflow_result.output_data.get('portfolio_manager_agent')
                if portfolio_output and portfolio_output.get('executed_trades'):
                    await self._execute_trades(portfolio_output['executed_trades'])
            
            # Update system stats
            self.system_stats['total_executions'] += 1
            if workflow_result.status.value == 'completed':
                self.system_stats['successful_executions'] += 1
            else:
                self.system_stats['failed_executions'] += 1
            
            logger.info(f"Trading cycle completed: {workflow_result.status.value}")
            
            return {
                'workflow_result': workflow_result,
                'system_stats': self.system_stats.copy(),
                'execution_stats': self.execution_engine.get_execution_stats()
            }
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            self.system_stats['failed_executions'] += 1
            raise
    
    async def _execute_trades(self, executed_trades: List[Dict[str, Any]]):
        """Execute trades from portfolio manager"""
        try:
            for trade in executed_trades:
                # Convert trade to trading signal
                from multi_agent_framework import TradingSignal, SignalType
                
                signal_type = SignalType.BUY if trade['signal_type'] == 'buy' else SignalType.SELL
                
                signal = TradingSignal(
                    symbol=trade['symbol'],
                    signal_type=signal_type,
                    confidence=trade['confidence'],
                    price=trade['price'],
                    timestamp=datetime.now(),
                    agent_id='portfolio_manager_agent',
                    metadata={
                        'strategy': 'portfolio_optimization',
                        'position_size': trade['position_size'],
                        'trade_id': trade.get('trade_id', 'unknown')
                    }
                )
                
                # Execute signal
                execution_report = await self.execution_engine.execute_signal(signal)
                
                if execution_report.status.value == 'filled':
                    self.system_stats['total_trades_executed'] += 1
                    logger.info(f"Trade executed: {trade['symbol']} {trade['signal_type']} @ ${execution_report.executed_price:.2f}")
                else:
                    logger.warning(f"Trade failed: {trade['symbol']} - {execution_report.error_message}")
                
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    async def run_continuous_trading(self, interval_minutes: int = 5):
        """Run continuous trading with specified interval"""
        logger.info(f"Starting continuous trading with {interval_minutes} minute intervals")
        
        while self.is_running:
            try:
                # Run trading cycle
                result = await self.run_trading_cycle()
                
                # Log results
                logger.info(f"Trading cycle completed - Status: {result['workflow_result'].status.value}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Continuous trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous trading: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'is_running': self.is_running,
            'system_stats': self.system_stats.copy(),
            'execution_stats': self.execution_engine.get_execution_stats() if self.execution_engine else {},
            'workflow_stats': self.workflow_orchestrator.get_performance_stats() if self.workflow_orchestrator else {},
            'uptime': (datetime.now() - self.system_stats['start_time']).total_seconds() if self.system_stats['start_time'] else 0
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        if hasattr(self, 'portfolio_manager_agent'):
            return self.portfolio_manager_agent.get_portfolio_summary()
        return {}

async def main():
    """Main function for running the trading system"""
    parser = argparse.ArgumentParser(description='Multi-Agent Trading System')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                       help='Execution mode: single cycle or continuous')
    parser.add_argument('--interval', type=int, default=5,
                       help='Interval in minutes for continuous mode')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD=X', 'AAPL', 'BTC-USD'],
                       help='Trading symbols')
    parser.add_argument('--execution-mode', choices=['paper', 'live'], default='paper',
                       help='Execution mode: paper trading or live trading')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    config.update({
        'symbols': args.symbols,
        'execution_mode': args.execution_mode,
        'mode': args.mode,
        'interval': args.interval
    })
    
    # Create and start trading system
    trading_system = MultiAgentTradingSystem(config)
    
    try:
        # Start system
        await trading_system.start_system()
        
        if args.mode == 'single':
            # Run single trading cycle
            result = await trading_system.run_trading_cycle()
            
            print("Trading Cycle Results:")
            print(f"Status: {result['workflow_result'].status.value}")
            print(f"Execution Time: {result['workflow_result'].total_execution_time:.2f}s")
            print(f"Completed Nodes: {len(result['workflow_result'].completed_nodes)}")
            print(f"Failed Nodes: {len(result['workflow_result'].failed_nodes)}")
            
            # Print system stats
            stats = result['system_stats']
            print(f"\nSystem Statistics:")
            print(f"Total Executions: {stats['total_executions']}")
            print(f"Successful Executions: {stats['successful_executions']}")
            print(f"Failed Executions: {stats['failed_executions']}")
            print(f"Total Trades Executed: {stats['total_trades_executed']}")
            
            # Print execution stats
            exec_stats = result['execution_stats']
            print(f"\nExecution Statistics:")
            print(f"Total Orders: {exec_stats['total_orders']}")
            print(f"Success Rate: {exec_stats['success_rate']:.2%}")
            print(f"Total Volume: ${exec_stats['total_volume']:.2f}")
            print(f"Total Commission: ${exec_stats['total_commission']:.2f}")
            
        else:
            # Run continuous trading
            await trading_system.run_continuous_trading(args.interval)
    
    except KeyboardInterrupt:
        print("\nShutting down trading system...")
    
    finally:
        # Stop system
        await trading_system.stop_system()
        print("Trading system stopped.")

if __name__ == "__main__":
    asyncio.run(main())
