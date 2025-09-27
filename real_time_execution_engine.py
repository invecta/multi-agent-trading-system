"""
Real-Time Execution Engine with Trading API Integration
Implements high-frequency trading execution with multiple broker integrations
"""
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import websockets
import ssl
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import uuid
import hashlib
import hmac
import base64

from multi_agent_framework import TradingSignal, SignalType
from enhanced_portfolio_manager_agent import Order, OrderType, OrderStatus, PositionSide, ExecutionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    ROBINHOOD = "robinhood"
    BINANCE = "binance"
    COINBASE = "coinbase"
    DEMO = "demo"

class ExecutionMode(Enum):
    """Execution modes"""
    PAPER_TRADING = "paper"
    LIVE_TRADING = "live"
    SIMULATION = "simulation"

@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_type: BrokerType
    api_key: str
    secret_key: str
    base_url: str
    websocket_url: str
    sandbox: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: float = 5.0

@dataclass
class ExecutionConfig:
    """Execution configuration"""
    mode: ExecutionMode
    max_order_size: float = 100000.0
    max_daily_trades: int = 1000
    max_position_size: float = 0.1
    slippage_tolerance: float = 0.001
    commission_rate: float = 0.001
    enable_pre_trade_checks: bool = True
    enable_post_trade_validation: bool = True

@dataclass
class MarketOrder:
    """Market order for execution"""
    order_id: str
    symbol: str
    side: PositionSide
    quantity: float
    order_type: OrderType
    time_in_force: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ExecutionReport:
    """Execution report"""
    order_id: str
    symbol: str
    executed_quantity: float
    executed_price: float
    execution_time: datetime
    broker: str
    commission: float
    slippage: float
    status: OrderStatus
    error_message: Optional[str]

class BaseBroker:
    """Base broker interface"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.session = None
        self.websocket = None
        self.is_connected = False
        self.order_cache = {}
        self.position_cache = {}
    
    async def connect(self):
        """Connect to broker"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from broker"""
        raise NotImplementedError
    
    async def place_order(self, order: MarketOrder) -> ExecutionReport:
        """Place an order"""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        raise NotImplementedError
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        raise NotImplementedError
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        raise NotImplementedError

class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation"""
    
    async def connect(self):
        """Connect to Alpaca"""
        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'APCA-API-KEY-ID': self.config.api_key,
                    'APCA-API-SECRET-KEY': self.config.secret_key,
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            async with self.session.get(f"{self.config.base_url}/v2/account") as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to Alpaca")
                else:
                    raise Exception(f"Alpaca connection failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info("Disconnected from Alpaca")
    
    async def place_order(self, order: MarketOrder) -> ExecutionReport:
        """Place order with Alpaca"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Prepare order data
            order_data = {
                'symbol': order.symbol,
                'qty': str(int(order.quantity)),
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force,
                'client_order_id': order.order_id
            }
            
            # Place order
            async with self.session.post(
                f"{self.config.base_url}/v2/orders",
                json=order_data
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    
                    return ExecutionReport(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        executed_quantity=float(result.get('qty', 0)),
                        executed_price=float(result.get('filled_avg_price', 0)),
                        execution_time=datetime.now(),
                        broker='alpaca',
                        commission=float(result.get('commission', 0)),
                        slippage=0.0,  # Would calculate from market data
                        status=OrderStatus.FILLED if result.get('status') == 'filled' else OrderStatus.PENDING,
                        error_message=None
                    )
                else:
                    error_text = await response.text()
                    return ExecutionReport(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        executed_quantity=0,
                        executed_price=0,
                        execution_time=datetime.now(),
                        broker='alpaca',
                        commission=0,
                        slippage=0,
                        status=OrderStatus.REJECTED,
                        error_message=f"Order rejected: {error_text}"
                    )
                    
        except Exception as e:
            logger.error(f"Error placing order with Alpaca: {e}")
            return ExecutionReport(
                order_id=order.order_id,
                symbol=order.symbol,
                executed_quantity=0,
                executed_price=0,
                execution_time=datetime.now(),
                broker='alpaca',
                commission=0,
                slippage=0,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca"""
        try:
            async with self.session.delete(f"{self.config.base_url}/v2/orders/{order_id}") as response:
                return response.status == 204
        except Exception as e:
            logger.error(f"Error cancelling order with Alpaca: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca"""
        try:
            async with self.session.get(f"{self.config.base_url}/v2/orders/{order_id}") as response:
                if response.status == 200:
                    result = await response.json()
                    status = result.get('status', 'unknown')
                    
                    if status == 'filled':
                        return OrderStatus.FILLED
                    elif status == 'partially_filled':
                        return OrderStatus.PARTIALLY_FILLED
                    elif status == 'canceled':
                        return OrderStatus.CANCELLED
                    else:
                        return OrderStatus.PENDING
                else:
                    return OrderStatus.REJECTED
        except Exception as e:
            logger.error(f"Error getting order status from Alpaca: {e}")
            return OrderStatus.REJECTED
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get positions from Alpaca"""
        try:
            async with self.session.get(f"{self.config.base_url}/v2/positions") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting positions from Alpaca: {e}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Alpaca"""
        try:
            async with self.session.get(f"{self.config.base_url}/v2/account") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting account info from Alpaca: {e}")
            return {}

class DemoBroker(BaseBroker):
    """Demo broker for testing"""
    
    async def connect(self):
        """Connect to demo broker"""
        self.is_connected = True
        logger.info("Connected to Demo Broker")
    
    async def disconnect(self):
        """Disconnect from demo broker"""
        self.is_connected = False
        logger.info("Disconnected from Demo Broker")
    
    async def place_order(self, order: MarketOrder) -> ExecutionReport:
        """Place order with demo broker"""
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Simulate execution
        executed_price = 100.0 + (hash(order.order_id) % 100) / 100  # Simulate price variation
        commission = order.quantity * executed_price * 0.001
        
        return ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            executed_quantity=order.quantity,
            executed_price=executed_price,
            execution_time=datetime.now(),
            broker='demo',
            commission=commission,
            slippage=0.001,
            status=OrderStatus.FILLED,
            error_message=None
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with demo broker"""
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from demo broker"""
        return OrderStatus.FILLED
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get positions from demo broker"""
        return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from demo broker"""
        return {
            'buying_power': 100000.0,
            'cash': 100000.0,
            'portfolio_value': 100000.0
        }

class RealTimeExecutionEngine:
    """Real-time execution engine with multiple broker support"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.brokers: Dict[str, BaseBroker] = {}
        self.execution_queue = asyncio.Queue()
        self.execution_results = {}
        self.is_running = False
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'average_execution_time': 0.0
        }
        
        # Pre-trade validation
        self.pre_trade_validators = []
        
        # Post-trade validation
        self.post_trade_validators = []
        
        # Execution monitoring
        self.execution_monitor = None
    
    def register_broker(self, name: str, broker: BaseBroker):
        """Register a broker"""
        self.brokers[name] = broker
        logger.info(f"Registered broker: {name}")
    
    def add_pre_trade_validator(self, validator: Callable):
        """Add pre-trade validator"""
        self.pre_trade_validators.append(validator)
    
    def add_post_trade_validator(self, validator: Callable):
        """Add post-trade validator"""
        self.post_trade_validators.append(validator)
    
    async def start(self):
        """Start the execution engine"""
        logger.info("Starting real-time execution engine")
        
        # Connect to all brokers
        for name, broker in self.brokers.items():
            try:
                await broker.connect()
            except Exception as e:
                logger.error(f"Failed to connect to broker {name}: {e}")
        
        self.is_running = True
        
        # Start execution monitor
        self.execution_monitor = asyncio.create_task(self._monitor_executions())
        
        # Start order processor
        self.order_processor = asyncio.create_task(self._process_orders())
        
        logger.info("Real-time execution engine started")
    
    async def stop(self):
        """Stop the execution engine"""
        logger.info("Stopping real-time execution engine")
        
        self.is_running = False
        
        # Cancel tasks
        if self.execution_monitor:
            self.execution_monitor.cancel()
        if self.order_processor:
            self.order_processor.cancel()
        
        # Disconnect from all brokers
        for name, broker in self.brokers.items():
            try:
                await broker.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from broker {name}: {e}")
        
        logger.info("Real-time execution engine stopped")
    
    async def execute_signal(self, signal: TradingSignal, broker_name: str = None) -> ExecutionReport:
        """Execute a trading signal"""
        try:
            # Convert signal to market order
            order = self._signal_to_order(signal)
            
            # Pre-trade validation
            if self.config.enable_pre_trade_checks:
                validation_result = await self._validate_pre_trade(order)
                if not validation_result['valid']:
                    return ExecutionReport(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        executed_quantity=0,
                        executed_price=0,
                        execution_time=datetime.now(),
                        broker=broker_name or 'unknown',
                        commission=0,
                        slippage=0,
                        status=OrderStatus.REJECTED,
                        error_message=validation_result['error']
                    )
            
            # Select broker
            broker = self._select_broker(broker_name)
            if not broker:
                return ExecutionReport(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    executed_quantity=0,
                    executed_price=0,
                    execution_time=datetime.now(),
                    broker='unknown',
                    commission=0,
                    slippage=0,
                    status=OrderStatus.REJECTED,
                    error_message="No broker available"
                )
            
            # Execute order
            start_time = time.time()
            execution_report = await broker.place_order(order)
            execution_time = time.time() - start_time
            
            # Update execution stats
            self._update_execution_stats(execution_report, execution_time)
            
            # Post-trade validation
            if self.config.enable_post_trade_validation:
                validation_result = await self._validate_post_trade(execution_report)
                if not validation_result['valid']:
                    logger.warning(f"Post-trade validation failed: {validation_result['error']}")
            
            # Store execution result
            self.execution_results[order.order_id] = execution_report
            
            logger.info(f"Executed signal: {signal.symbol} {signal.signal_type.value} - Status: {execution_report.status.value}")
            
            return execution_report
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return ExecutionReport(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                executed_quantity=0,
                executed_price=0,
                execution_time=datetime.now(),
                broker='unknown',
                commission=0,
                slippage=0,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
    
    def _signal_to_order(self, signal: TradingSignal) -> MarketOrder:
        """Convert trading signal to market order"""
        # Determine order side
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            side = PositionSide.LONG
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            side = PositionSide.SHORT
        else:
            raise ValueError(f"Invalid signal type for execution: {signal.signal_type}")
        
        # Calculate quantity
        position_size = signal.metadata.get('position_size', 0.01)
        quantity = position_size * 1000  # Simplified quantity calculation
        
        return MarketOrder(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force="DAY",
            timestamp=datetime.now(),
            metadata={
                'signal_confidence': signal.confidence,
                'strategy': signal.metadata.get('strategy', 'unknown'),
                'original_signal': signal
            }
        )
    
    async def _validate_pre_trade(self, order: MarketOrder) -> Dict[str, Any]:
        """Validate order before execution"""
        try:
            # Check order size
            if order.quantity <= 0:
                return {'valid': False, 'error': 'Invalid quantity'}
            
            if order.quantity * 100 > self.config.max_order_size:  # Assuming $100 per share
                return {'valid': False, 'error': 'Order size exceeds maximum'}
            
            # Check daily trade limit
            if self.execution_stats['total_orders'] >= self.config.max_daily_trades:
                return {'valid': False, 'error': 'Daily trade limit exceeded'}
            
            # Run custom validators
            for validator in self.pre_trade_validators:
                try:
                    result = await validator(order)
                    if not result.get('valid', True):
                        return result
                except Exception as e:
                    logger.error(f"Pre-trade validator error: {e}")
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Pre-trade validation error: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _validate_post_trade(self, execution_report: ExecutionReport) -> Dict[str, Any]:
        """Validate execution after trade"""
        try:
            # Check execution success
            if execution_report.status != OrderStatus.FILLED:
                return {'valid': False, 'error': 'Order not filled'}
            
            # Check slippage
            if execution_report.slippage > self.config.slippage_tolerance:
                return {'valid': False, 'error': f'Slippage exceeds tolerance: {execution_report.slippage}'}
            
            # Run custom validators
            for validator in self.post_trade_validators:
                try:
                    result = await validator(execution_report)
                    if not result.get('valid', True):
                        return result
                except Exception as e:
                    logger.error(f"Post-trade validator error: {e}")
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Post-trade validation error: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _select_broker(self, broker_name: str = None) -> Optional[BaseBroker]:
        """Select broker for execution"""
        if broker_name and broker_name in self.brokers:
            return self.brokers[broker_name]
        
        # Select first available broker
        for broker in self.brokers.values():
            if broker.is_connected:
                return broker
        
        return None
    
    def _update_execution_stats(self, execution_report: ExecutionReport, execution_time: float):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if execution_report.status == OrderStatus.FILLED:
            self.execution_stats['successful_orders'] += 1
            self.execution_stats['total_volume'] += execution_report.executed_quantity * execution_report.executed_price
        else:
            self.execution_stats['failed_orders'] += 1
        
        self.execution_stats['total_commission'] += execution_report.commission
        
        # Update average execution time
        total_time = self.execution_stats['average_execution_time'] * (self.execution_stats['total_orders'] - 1)
        total_time += execution_time
        self.execution_stats['average_execution_time'] = total_time / self.execution_stats['total_orders']
    
    async def _process_orders(self):
        """Process orders from queue"""
        while self.is_running:
            try:
                # Get order from queue
                order_data = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                
                # Process order
                signal = order_data['signal']
                broker_name = order_data.get('broker_name')
                
                execution_report = await self.execute_signal(signal, broker_name)
                
                # Notify callback if provided
                callback = order_data.get('callback')
                if callback:
                    try:
                        await callback(execution_report)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing order: {e}")
    
    async def _monitor_executions(self):
        """Monitor execution performance"""
        while self.is_running:
            try:
                # Log execution stats periodically
                logger.info(f"Execution Stats: {self.execution_stats}")
                
                # Check for stuck orders
                await self._check_stuck_orders()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in execution monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_stuck_orders(self):
        """Check for stuck orders"""
        # Implementation would check for orders that have been pending too long
        pass
    
    async def queue_signal(self, signal: TradingSignal, broker_name: str = None, callback: Callable = None):
        """Queue a signal for execution"""
        await self.execution_queue.put({
            'signal': signal,
            'broker_name': broker_name,
            'callback': callback
        })
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        stats['success_rate'] = stats['successful_orders'] / max(stats['total_orders'], 1)
        stats['is_running'] = self.is_running
        stats['registered_brokers'] = list(self.brokers.keys())
        return stats
    
    def get_execution_result(self, order_id: str) -> Optional[ExecutionReport]:
        """Get execution result by order ID"""
        return self.execution_results.get(order_id)

class MultiBrokerExecutionEngine:
    """Multi-broker execution engine with load balancing"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.execution_engines: Dict[str, RealTimeExecutionEngine] = {}
        self.load_balancer = LoadBalancer()
    
    def add_execution_engine(self, name: str, engine: RealTimeExecutionEngine):
        """Add execution engine"""
        self.execution_engines[name] = engine
        self.load_balancer.add_engine(name, engine)
    
    async def start_all(self):
        """Start all execution engines"""
        for name, engine in self.execution_engines.items():
            await engine.start()
            logger.info(f"Started execution engine: {name}")
    
    async def stop_all(self):
        """Stop all execution engines"""
        for name, engine in self.execution_engines.items():
            await engine.stop()
            logger.info(f"Stopped execution engine: {name}")
    
    async def execute_signal(self, signal: TradingSignal, preferred_broker: str = None) -> ExecutionReport:
        """Execute signal with load balancing"""
        # Select best execution engine
        engine_name = self.load_balancer.select_engine(preferred_broker)
        engine = self.execution_engines[engine_name]
        
        # Execute signal
        return await engine.execute_signal(signal, preferred_broker)
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all engines"""
        combined_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'engines': {}
        }
        
        for name, engine in self.execution_engines.items():
            stats = engine.get_execution_stats()
            combined_stats['engines'][name] = stats
            
            combined_stats['total_orders'] += stats['total_orders']
            combined_stats['successful_orders'] += stats['successful_orders']
            combined_stats['failed_orders'] += stats['failed_orders']
            combined_stats['total_volume'] += stats['total_volume']
            combined_stats['total_commission'] += stats['total_commission']
        
        combined_stats['overall_success_rate'] = combined_stats['successful_orders'] / max(combined_stats['total_orders'], 1)
        
        return combined_stats

class LoadBalancer:
    """Load balancer for execution engines"""
    
    def __init__(self):
        self.engines = {}
        self.engine_stats = {}
    
    def add_engine(self, name: str, engine: RealTimeExecutionEngine):
        """Add engine to load balancer"""
        self.engines[name] = engine
        self.engine_stats[name] = {
            'total_orders': 0,
            'success_rate': 1.0,
            'average_execution_time': 0.0,
            'last_used': datetime.now()
        }
    
    def select_engine(self, preferred: str = None) -> str:
        """Select best engine for execution"""
        if preferred and preferred in self.engines:
            return preferred
        
        # Simple round-robin selection
        # In production, would use more sophisticated load balancing
        engine_names = list(self.engines.keys())
        if not engine_names:
            raise ValueError("No execution engines available")
        
        # Select engine with lowest load
        best_engine = min(engine_names, key=lambda name: self.engine_stats[name]['total_orders'])
        
        # Update stats
        self.engine_stats[best_engine]['total_orders'] += 1
        self.engine_stats[best_engine]['last_used'] = datetime.now()
        
        return best_engine

# Example usage and testing
async def main():
    """Example usage of the real-time execution engine"""
    
    # Create execution config
    execution_config = ExecutionConfig(
        mode=ExecutionMode.PAPER_TRADING,
        max_order_size=50000.0,
        max_daily_trades=100,
        slippage_tolerance=0.002,
        commission_rate=0.001
    )
    
    # Create brokers
    alpaca_config = BrokerConfig(
        broker_type=BrokerType.ALPACA,
        api_key="demo_key",
        secret_key="demo_secret",
        base_url="https://paper-api.alpaca.markets",
        websocket_url="wss://paper-api.alpaca.markets/stream",
        sandbox=True
    )
    
    demo_config = BrokerConfig(
        broker_type=BrokerType.DEMO,
        api_key="demo",
        secret_key="demo",
        base_url="demo",
        websocket_url="demo"
    )
    
    # Create brokers
    alpaca_broker = AlpacaBroker(alpaca_config)
    demo_broker = DemoBroker(demo_config)
    
    # Create execution engine
    execution_engine = RealTimeExecutionEngine(execution_config)
    execution_engine.register_broker("alpaca", alpaca_broker)
    execution_engine.register_broker("demo", demo_broker)
    
    # Start execution engine
    await execution_engine.start()
    
    # Create sample trading signal
    signal = TradingSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.8,
        price=150.0,
        timestamp=datetime.now(),
        agent_id="test_agent",
        metadata={'position_size': 0.02, 'strategy': 'test'}
    )
    
    try:
        # Execute signal
        execution_report = await execution_engine.execute_signal(signal, "demo")
        
        print("Execution Results:")
        print(f"Order ID: {execution_report.order_id}")
        print(f"Symbol: {execution_report.symbol}")
        print(f"Executed Quantity: {execution_report.executed_quantity}")
        print(f"Executed Price: ${execution_report.executed_price:.2f}")
        print(f"Status: {execution_report.status.value}")
        print(f"Commission: ${execution_report.commission:.2f}")
        print(f"Slippage: {execution_report.slippage:.4f}")
        
        # Print execution stats
        stats = execution_engine.get_execution_stats()
        print(f"\nExecution Stats:")
        print(f"Total Orders: {stats['total_orders']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Total Volume: ${stats['total_volume']:.2f}")
        print(f"Total Commission: ${stats['total_commission']:.2f}")
        
    except Exception as e:
        print(f"Execution failed: {e}")
    
    finally:
        # Stop execution engine
        await execution_engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
