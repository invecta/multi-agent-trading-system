"""
Enhanced Portfolio Manager Agent for Trade Execution and Portfolio Optimization
Implements advanced portfolio management, trade execution, and optimization strategies
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from multi_agent_framework import BaseAgent, AgentOutput, TradingSignal, SignalType, RiskMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """Trading order structure"""
    order_id: str
    symbol: str
    side: PositionSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    status: OrderStatus
    filled_quantity: float
    average_price: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Portfolio position structure"""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    cost_basis: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class Portfolio:
    """Portfolio structure"""
    total_value: float
    cash: float
    positions: Dict[str, Position]
    total_pnl: float
    total_return: float
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class ExecutionResult:
    """Trade execution result"""
    order_id: str
    symbol: str
    executed_quantity: float
    executed_price: float
    execution_time: datetime
    slippage: float
    commission: float
    success: bool
    error_message: Optional[str]

class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.05)
    
    def optimize_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                          current_weights: np.ndarray = None) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            if current_weights is None:
                current_weights = np.ones(n_assets) / n_assets
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_risk
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]
            
            # Bounds: weights between 0 and max_position_size
            bounds = [(0, self.max_position_size) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(objective, current_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                return {
                    'optimal_weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'expected_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'success': True
                }
            else:
                logger.warning("Portfolio optimization failed")
                return {
                    'optimal_weights': current_weights,
                    'expected_return': np.dot(current_weights, expected_returns),
                    'expected_risk': np.sqrt(np.dot(current_weights, np.dot(cov_matrix, current_weights))),
                    'sharpe_ratio': 0.0,
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {
                'optimal_weights': np.ones(len(expected_returns)) / len(expected_returns),
                'expected_return': 0.0,
                'expected_risk': 0.0,
                'sharpe_ratio': 0.0,
                'success': False
            }
    
    def calculate_rebalance_trades(self, current_weights: np.ndarray, 
                                 target_weights: np.ndarray, 
                                 portfolio_value: float) -> List[Dict[str, Any]]:
        """Calculate trades needed for rebalancing"""
        try:
            trades = []
            weight_diff = target_weights - current_weights
            
            for i, diff in enumerate(weight_diff):
                if abs(diff) > self.rebalance_threshold:
                    trade_value = diff * portfolio_value
                    trades.append({
                        'asset_index': i,
                        'trade_value': trade_value,
                        'weight_change': diff,
                        'current_weight': current_weights[i],
                        'target_weight': target_weights[i]
                    })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalance trades: {e}")
            return []

class TradeExecutor:
    """Trade execution engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.commission_rate = self.config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = self.config.get('slippage_rate', 0.0005)    # 0.05%
        self.max_order_size = self.config.get('max_order_size', 100000)   # $100k
        self.execution_history = []
    
    async def execute_order(self, order: Order, market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute a trading order"""
        try:
            symbol = order.symbol
            current_price = market_data.get(symbol, {}).get('1d', {}).get('close', 0)
            
            if current_price <= 0:
                return ExecutionResult(
                    order_id=order.order_id,
                    symbol=symbol,
                    executed_quantity=0,
                    executed_price=0,
                    execution_time=datetime.now(),
                    slippage=0,
                    commission=0,
                    success=False,
                    error_message="Invalid market price"
                )
            
            # Simulate execution with slippage
            if order.order_type == OrderType.MARKET:
                executed_price = current_price * (1 + np.random.normal(0, self.slippage_rate))
            elif order.order_type == OrderType.LIMIT:
                if order.side == PositionSide.LONG and order.price >= current_price:
                    executed_price = order.price
                elif order.side == PositionSide.SHORT and order.price <= current_price:
                    executed_price = order.price
                else:
                    return ExecutionResult(
                        order_id=order.order_id,
                        symbol=symbol,
                        executed_quantity=0,
                        executed_price=0,
                        execution_time=datetime.now(),
                        slippage=0,
                        commission=0,
                        success=False,
                        error_message="Limit order not executable"
                    )
            else:
                executed_price = current_price
            
            # Calculate slippage
            slippage = abs(executed_price - current_price) / current_price
            
            # Calculate commission
            trade_value = order.quantity * executed_price
            commission = trade_value * self.commission_rate
            
            # Record execution
            execution_result = ExecutionResult(
                order_id=order.order_id,
                symbol=symbol,
                executed_quantity=order.quantity,
                executed_price=executed_price,
                execution_time=datetime.now(),
                slippage=slippage,
                commission=commission,
                success=True,
                error_message=None
            )
            
            self.execution_history.append(execution_result)
            
            logger.info(f"Executed order {order.order_id}: {order.quantity} {symbol} @ ${executed_price:.2f}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                symbol=symbol,
                executed_quantity=0,
                executed_price=0,
                execution_time=datetime.now(),
                slippage=0,
                commission=0,
                success=False,
                error_message=str(e)
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                'total_trades': 0,
                'success_rate': 0.0,
                'average_slippage': 0.0,
                'total_commission': 0.0
            }
        
        successful_trades = [t for t in self.execution_history if t.success]
        
        return {
            'total_trades': len(self.execution_history),
            'successful_trades': len(successful_trades),
            'success_rate': len(successful_trades) / len(self.execution_history),
            'average_slippage': np.mean([t.slippage for t in successful_trades]),
            'total_commission': sum([t.commission for t in successful_trades]),
            'total_executed_value': sum([t.executed_quantity * t.executed_price for t in successful_trades])
        }

class PositionManager:
    """Position management engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.positions = {}
        self.position_history = []
        self.stop_loss_percentage = self.config.get('stop_loss_percentage', 0.02)
        self.take_profit_percentage = self.config.get('take_profit_percentage', 0.04)
    
    def update_position(self, symbol: str, execution_result: ExecutionResult, 
                       side: PositionSide) -> Optional[Position]:
        """Update position after trade execution"""
        try:
            if not execution_result.success:
                return None
            
            if symbol not in self.positions:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=execution_result.executed_quantity,
                    average_price=execution_result.executed_price,
                    current_price=execution_result.executed_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    market_value=execution_result.executed_quantity * execution_result.executed_price,
                    cost_basis=execution_result.executed_quantity * execution_result.executed_price,
                    timestamp=execution_result.execution_time,
                    metadata={'orders': [execution_result.order_id]}
                )
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if position.side == side:
                    # Add to position
                    total_cost = (position.quantity * position.average_price + 
                                execution_result.executed_quantity * execution_result.executed_price)
                    total_quantity = position.quantity + execution_result.executed_quantity
                    
                    position.quantity = total_quantity
                    position.average_price = total_cost / total_quantity
                    position.cost_basis = total_cost
                    position.metadata['orders'].append(execution_result.order_id)
                else:
                    # Close or reduce position
                    if execution_result.executed_quantity >= position.quantity:
                        # Close position
                        realized_pnl = self._calculate_realized_pnl(position, execution_result)
                        position.realized_pnl += realized_pnl
                        del self.positions[symbol]
                        return None
                    else:
                        # Reduce position
                        realized_pnl = self._calculate_realized_pnl(position, execution_result)
                        position.realized_pnl += realized_pnl
                        position.quantity -= execution_result.executed_quantity
                        position.cost_basis = position.quantity * position.average_price
            
            # Update position metadata
            position.timestamp = execution_result.execution_time
            position.current_price = execution_result.executed_price
            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = self._calculate_unrealized_pnl(position)
            
            return position
            
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            return None
    
    def _calculate_realized_pnl(self, position: Position, execution_result: ExecutionResult) -> float:
        """Calculate realized P&L"""
        if position.side == PositionSide.LONG:
            return execution_result.executed_quantity * (execution_result.executed_price - position.average_price)
        else:
            return execution_result.executed_quantity * (position.average_price - execution_result.executed_price)
    
    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L"""
        if position.side == PositionSide.LONG:
            return position.quantity * (position.current_price - position.average_price)
        else:
            return position.quantity * (position.average_price - position.current_price)
    
    def update_market_prices(self, market_data: Dict[str, Any]):
        """Update position prices with current market data"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('1d', {}).get('close', position.current_price)
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)
    
    def get_portfolio_summary(self) -> Portfolio:
        """Get portfolio summary"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_cost_basis = sum(pos.cost_basis for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        total_pnl = total_unrealized_pnl + total_realized_pnl
        total_return = total_pnl / total_cost_basis if total_cost_basis > 0 else 0.0
        
        return Portfolio(
            total_value=total_market_value,
            cash=100000.0 - total_cost_basis,  # Simplified cash calculation
            positions=self.positions.copy(),
            total_pnl=total_pnl,
            total_return=total_return,
            last_updated=datetime.now(),
            metadata={
                'total_positions': len(self.positions),
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl
            }
        )
    
    def check_stop_loss_take_profit(self, symbol: str) -> List[Order]:
        """Check for stop loss and take profit triggers"""
        orders = []
        
        if symbol not in self.positions:
            return orders
        
        position = self.positions[symbol]
        
        # Check stop loss
        if position.side == PositionSide.LONG:
            stop_loss_price = position.average_price * (1 - self.stop_loss_percentage)
            if position.current_price <= stop_loss_price:
                orders.append(Order(
                    order_id=f"stop_loss_{symbol}_{datetime.now().timestamp()}",
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    price=None,
                    stop_price=stop_loss_price,
                    time_in_force="GTC",
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    average_price=None,
                    timestamp=datetime.now(),
                    metadata={'reason': 'stop_loss'}
                ))
            
            # Check take profit
            take_profit_price = position.average_price * (1 + self.take_profit_percentage)
            if position.current_price >= take_profit_price:
                orders.append(Order(
                    order_id=f"take_profit_{symbol}_{datetime.now().timestamp()}",
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    price=None,
                    stop_price=take_profit_price,
                    time_in_force="GTC",
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    average_price=None,
                    timestamp=datetime.now(),
                    metadata={'reason': 'take_profit'}
                ))
        
        return orders

class EnhancedPortfolioManagerAgent(BaseAgent):
    """Enhanced Portfolio Manager Agent with advanced portfolio management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("enhanced_portfolio_manager_agent", config)
        self.add_dependency("risk_manager_agent")
        
        # Initialize components
        self.portfolio_optimizer = PortfolioOptimizer(config)
        self.trade_executor = TradeExecutor(config)
        self.position_manager = PositionManager(config)
        
        # Portfolio configuration
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.max_positions = config.get('max_positions', 10)
        self.rebalance_frequency = config.get('rebalance_frequency', 7)  # days
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.rebalance_history = []
        
        # Initialize portfolio
        self.current_portfolio = Portfolio(
            total_value=self.initial_capital,
            cash=self.initial_capital,
            positions={},
            total_pnl=0.0,
            total_return=0.0,
            last_updated=datetime.now(),
            metadata={}
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process portfolio management and trade execution"""
        start_time = datetime.now()
        self.status = "running"
        
        try:
            # Get risk manager output
            risk_output = input_data.get('risk_output')
            market_data = input_data.get('market_data', {})
            
            if not risk_output:
                raise ValueError("No risk manager output provided")
            
            # Update market prices
            self.position_manager.update_market_prices(market_data)
            
            # Process risk-adjusted signals
            risk_adjusted_signals = risk_output.signals
            risk_metrics = risk_output.risk_metrics
            
            # Generate trading orders
            orders = await self._generate_trading_orders(risk_adjusted_signals, risk_metrics)
            
            # Execute orders
            execution_results = await self._execute_orders(orders, market_data)
            
            # Update positions
            await self._update_positions(execution_results)
            
            # Check for stop loss/take profit
            stop_orders = await self._check_stop_loss_take_profit()
            
            # Execute stop orders
            if stop_orders:
                stop_execution_results = await self._execute_orders(stop_orders, market_data)
                await self._update_positions(stop_execution_results)
            
            # Optimize portfolio if needed
            optimization_results = await self._optimize_portfolio_if_needed(risk_metrics)
            
            # Update portfolio
            self.current_portfolio = self.position_manager.get_portfolio_summary()
            
            # Track performance
            self._track_performance()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = AgentOutput(
                agent_id=self.agent_id,
                status="completed",
                data={
                    'portfolio': self.current_portfolio,
                    'execution_results': execution_results,
                    'optimization_results': optimization_results,
                    'performance_metrics': self._calculate_performance_metrics(),
                    'trade_statistics': self.trade_executor.get_execution_stats()
                },
                signals=[],
                risk_metrics=risk_metrics,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Portfolio Manager Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status="error",
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
    
    async def _generate_trading_orders(self, signals: List[TradingSignal], 
                                     risk_metrics: RiskMetrics) -> List[Order]:
        """Generate trading orders from signals"""
        orders = []
        
        try:
            for signal in signals:
                # Determine order side
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    side = PositionSide.LONG
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    side = PositionSide.SHORT
                else:
                    continue
                
                # Calculate position size
                position_size = signal.metadata.get('position_size', 0.01)
                portfolio_value = self.current_portfolio.total_value
                position_value = position_size * portfolio_value
                
                # Get current price
                current_price = signal.price
                if current_price <= 0:
                    continue
                
                # Calculate quantity
                quantity = position_value / current_price
                
                # Check if we already have a position
                existing_position = self.position_manager.positions.get(signal.symbol)
                
                if existing_position:
                    # Adjust quantity based on existing position
                    if existing_position.side == side:
                        # Add to position
                        target_value = position_value
                        current_value = existing_position.market_value
                        additional_value = target_value - current_value
                        
                        if additional_value > 0:
                            quantity = additional_value / current_price
                        else:
                            continue  # Position already at target size
                    else:
                        # Close or reduce opposite position
                        quantity = min(quantity, existing_position.quantity)
                
                # Create order
                order = Order(
                    order_id=f"{signal.symbol}_{side.value}_{datetime.now().timestamp()}",
                    symbol=signal.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    price=None,
                    stop_price=None,
                    time_in_force="DAY",
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    average_price=None,
                    timestamp=datetime.now(),
                    metadata={
                        'signal_confidence': signal.confidence,
                        'strategy': signal.metadata.get('strategy', 'unknown'),
                        'risk_adjusted': True
                    }
                )
                
                orders.append(order)
            
        except Exception as e:
            logger.error(f"Error generating trading orders: {e}")
        
        return orders
    
    async def _execute_orders(self, orders: List[Order], market_data: Dict[str, Any]) -> List[ExecutionResult]:
        """Execute trading orders"""
        execution_results = []
        
        try:
            for order in orders:
                # Check if we have enough cash for the order
                if order.side == PositionSide.LONG:
                    required_cash = order.quantity * order.price if order.price else order.quantity * market_data.get(order.symbol, {}).get('1d', {}).get('close', 0)
                    if required_cash > self.current_portfolio.cash:
                        logger.warning(f"Insufficient cash for order {order.order_id}")
                        continue
                
                # Execute order
                execution_result = await self.trade_executor.execute_order(order, market_data)
                execution_results.append(execution_result)
                
                # Update order status
                if execution_result.success:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = execution_result.executed_quantity
                    order.average_price = execution_result.executed_price
                else:
                    order.status = OrderStatus.REJECTED
                
                # Record trade
                self.trade_history.append({
                    'order': order,
                    'execution_result': execution_result,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error executing orders: {e}")
        
        return execution_results
    
    async def _update_positions(self, execution_results: List[ExecutionResult]):
        """Update positions after trade execution"""
        try:
            for execution_result in execution_results:
                if not execution_result.success:
                    continue
                
                # Determine position side based on order
                # This is simplified - in practice, you'd track this from the order
                side = PositionSide.LONG  # Simplified
                
                # Update position
                position = self.position_manager.update_position(
                    execution_result.symbol,
                    execution_result,
                    side
                )
                
                if position:
                    logger.info(f"Updated position for {execution_result.symbol}: {position.quantity} @ ${position.average_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _check_stop_loss_take_profit(self) -> List[Order]:
        """Check for stop loss and take profit triggers"""
        stop_orders = []
        
        try:
            for symbol in self.position_manager.positions.keys():
                orders = self.position_manager.check_stop_loss_take_profit(symbol)
                stop_orders.extend(orders)
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {e}")
        
        return stop_orders
    
    async def _optimize_portfolio_if_needed(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Optimize portfolio if needed"""
        try:
            # Check if rebalancing is needed
            last_rebalance = self.rebalance_history[-1] if self.rebalance_history else None
            days_since_rebalance = (datetime.now() - last_rebalance['timestamp']).days if last_rebalance else 999
            
            if days_since_rebalance < self.rebalance_frequency:
                return {'action': 'no_rebalance_needed', 'reason': 'too_soon'}
            
            # Get current positions
            positions = self.position_manager.positions
            
            if len(positions) < 2:
                return {'action': 'no_rebalance_needed', 'reason': 'insufficient_positions'}
            
            # Calculate current weights
            total_value = self.current_portfolio.total_value
            current_weights = np.array([pos.market_value / total_value for pos in positions.values()])
            
            # Simulate expected returns and covariance matrix
            symbols = list(positions.keys())
            n_assets = len(symbols)
            
            # Generate simulated expected returns and covariance matrix
            expected_returns = np.random.normal(0.05, 0.02, n_assets)  # 5% expected return
            cov_matrix = np.random.rand(n_assets, n_assets)
            cov_matrix = cov_matrix @ cov_matrix.T  # Make it positive definite
            cov_matrix *= 0.01  # Scale down variance
            
            # Optimize portfolio
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                expected_returns, cov_matrix, current_weights
            )
            
            if optimization_result['success']:
                # Calculate rebalance trades
                target_weights = optimization_result['optimal_weights']
                rebalance_trades = self.portfolio_optimizer.calculate_rebalance_trades(
                    current_weights, target_weights, total_value
                )
                
                # Record rebalancing
                self.rebalance_history.append({
                    'timestamp': datetime.now(),
                    'current_weights': current_weights.tolist(),
                    'target_weights': target_weights.tolist(),
                    'rebalance_trades': rebalance_trades,
                    'expected_return': optimization_result['expected_return'],
                    'expected_risk': optimization_result['expected_risk'],
                    'sharpe_ratio': optimization_result['sharpe_ratio']
                })
                
                return {
                    'action': 'rebalanced',
                    'rebalance_trades': rebalance_trades,
                    'optimization_result': optimization_result
                }
            else:
                return {'action': 'optimization_failed', 'reason': 'optimization_error'}
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {'action': 'error', 'error': str(e)}
    
    def _track_performance(self):
        """Track portfolio performance"""
        try:
            performance_metrics = self._calculate_performance_metrics()
            
            self.performance_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': self.current_portfolio.total_value,
                'total_return': self.current_portfolio.total_return,
                'total_pnl': self.current_portfolio.total_pnl,
                'cash': self.current_portfolio.cash,
                'positions_count': len(self.current_portfolio.positions),
                'metrics': performance_metrics
            })
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            if not self.performance_history:
                return {}
            
            # Calculate returns
            portfolio_values = [p['portfolio_value'] for p in self.performance_history]
            returns = pd.Series(portfolio_values).pct_change().dropna()
            
            if len(returns) < 2:
                return {}
            
            # Calculate metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            running_max = pd.Series(portfolio_values).expanding().max()
            drawdown = (pd.Series(portfolio_values) - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
                'total_trades': len(self.trade_history),
                'win_rate': self._calculate_win_rate()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Simplified win rate calculation
            # In practice, you'd track individual trade P&L
            return 0.6  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate portfolio manager input"""
        return 'risk_output' in input_data
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'portfolio': self.current_portfolio,
            'performance_history': self.performance_history[-10:],  # Last 10 entries
            'trade_statistics': self.trade_executor.get_execution_stats(),
            'rebalance_history': self.rebalance_history[-5:],  # Last 5 rebalances
            'total_positions': len(self.current_portfolio.positions),
            'total_trades': len(self.trade_history),
            'last_update': datetime.now()
        }

# Example usage and testing
async def main():
    """Example usage of the enhanced portfolio manager agent"""
    
    # Create sample risk manager output
    from multi_agent_framework import RiskMetrics
    
    sample_risk_metrics = RiskMetrics(
        var_95=1000.0,
        var_99=1500.0,
        cvar_95=1200.0,
        cvar_99=1800.0,
        max_drawdown=0.05,
        volatility=0.15,
        sharpe_ratio=1.2,
        portfolio_value=100000.0
    )
    
    sample_risk_output = AgentOutput(
        agent_id="risk_manager_agent",
        status="completed",
        data={},
        signals=[
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=150.0,
                timestamp=datetime.now(),
                agent_id="risk_manager_agent",
                metadata={'position_size': 0.03, 'strategy': 'risk_adjusted'}
            )
        ],
        risk_metrics=sample_risk_metrics,
        timestamp=datetime.now(),
        execution_time=1.0
    )
    
    # Create sample market data
    sample_market_data = {
        'AAPL': {
            '1d': {
                'symbol': 'AAPL',
                'timestamp': datetime.now(),
                'open': 150.0,
                'high': 152.5,
                'low': 149.5,
                'close': 151.2,
                'volume': 50000000,
                'timeframe': '1d'
            }
        }
    }
    
    # Initialize portfolio manager agent
    config = {
        'initial_capital': 100000.0,
        'max_positions': 10,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005,
        'stop_loss_percentage': 0.02,
        'take_profit_percentage': 0.04
    }
    
    agent = EnhancedPortfolioManagerAgent(config)
    
    # Process portfolio management
    input_data = {
        'risk_output': sample_risk_output,
        'market_data': sample_market_data
    }
    
    result = await agent.process(input_data)
    
    print("Portfolio Management Results:")
    print(f"Status: {result.status}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    if result.data.get('portfolio'):
        portfolio = result.data['portfolio']
        print(f"\nPortfolio Summary:")
        print(f"  Total Value: ${portfolio.total_value:.2f}")
        print(f"  Cash: ${portfolio.cash:.2f}")
        print(f"  Total P&L: ${portfolio.total_pnl:.2f}")
        print(f"  Total Return: {portfolio.total_return:.2%}")
        print(f"  Positions: {len(portfolio.positions)}")
    
    # Print trade statistics
    trade_stats = result.data.get('trade_statistics', {})
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {trade_stats.get('total_trades', 0)}")
    print(f"  Success Rate: {trade_stats.get('success_rate', 0):.2%}")
    print(f"  Average Slippage: {trade_stats.get('average_slippage', 0):.4f}")
    print(f"  Total Commission: ${trade_stats.get('total_commission', 0):.2f}")
    
    # Print portfolio summary
    summary = agent.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"Total Positions: {summary['total_positions']}")
    print(f"Total Trades: {summary['total_trades']}")

if __name__ == "__main__":
    asyncio.run(main())
