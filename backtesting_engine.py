"""
Backtesting Engine for Goldbach Trading Strategies
Validates strategy performance using historical data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from tesla_369_calculator import Tesla369Calculator
from config import Config

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """Backtesting engine for Tesla 369 trading strategies"""
    
    def __init__(self, initial_capital: float = None):
        self.config = Config()
        self.initial_capital = initial_capital or self.config.INITIAL_CAPITAL
        self.tesla_calc = Tesla369Calculator()
        
        # Trading parameters
        self.max_position_size = self.config.MAX_POSITION_SIZE
        self.stop_loss_pct = self.config.STOP_LOSS_PERCENTAGE
        self.take_profit_ratio = self.config.TAKE_PROFIT_RATIO
        
        # Performance tracking
        self.trades = []
        self.portfolio_value = []
        self.drawdowns = []
        
    def run_backtest(self, data: pd.DataFrame, strategy_params: Dict = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data with OHLC columns
            strategy_params: Strategy parameters to test
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(data)} periods")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'position': 0,
            'position_value': 0,
            'total_value': self.initial_capital
        }
        
        # Reset tracking
        self.trades = []
        self.portfolio_value = []
        self.drawdowns = []
        
        # Analyze market structure
        market_analysis = self.tesla_calc.analyze_market_structure(data)
        signals_df = market_analysis['signals']
        
        # Process each period
        for i in range(len(signals_df)):
            current_data = signals_df.iloc[i]
            current_price = current_data['close']
            
            # Check for exit signals first
            if portfolio['position'] != 0:
                exit_signal = self._check_exit_conditions(
                    current_data, portfolio, i, signals_df
                )
                if exit_signal:
                    portfolio = self._execute_exit(portfolio, current_price, exit_signal, i)
            
            # Check for entry signals
            if portfolio['position'] == 0 and current_data['signal'] != 0:
                logger.debug(f"Found signal {current_data['signal']} at index {i}")
                entry_signal = self._check_entry_conditions(current_data, portfolio)
                if entry_signal:
                    portfolio = self._execute_entry(
                        portfolio, current_price, entry_signal, i
                    )
                else:
                    logger.debug(f"Entry conditions not met for signal {current_data['signal']}")
            
            # Update portfolio value
            portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
            self.portfolio_value.append({
                'timestamp': signals_df.index[i],
                'total_value': portfolio['total_value'],
                'cash': portfolio['cash'],
                'position_value': portfolio['position_value'],
                'position': portfolio['position']
            })
            
            # Calculate drawdown
            peak_value = max([pv['total_value'] for pv in self.portfolio_value])
            current_drawdown = (peak_value - portfolio['total_value']) / peak_value
            self.drawdowns.append(current_drawdown)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(data)
        
        return {
            'trades': self.trades,
            'portfolio_history': self.portfolio_value,
            'drawdowns': self.drawdowns,
            'performance_metrics': performance_metrics,
            'market_analysis': market_analysis
        }
    
    def _check_entry_conditions(self, current_data: pd.Series, portfolio: Dict) -> Optional[Dict]:
        """Check if entry conditions are met"""
        if current_data['signal'] == 0:
            return None
        
        # Calculate position size (number of shares/units)
        max_position_value = portfolio['cash'] * self.max_position_size  # 2% of cash
        position_size = max_position_value / current_data['close']  # Number of shares/units
        
        # Adjust minimum position size based on asset type
        min_position_size = 0.001 if current_data['close'] > 1000 else 1  # Crypto vs stocks
        if position_size < min_position_size:
            logger.debug(f"Position size too small: {position_size} (min: {min_position_size})")
            return None
        
        logger.debug(f"Entry conditions met: signal={current_data['signal']}, position_size={position_size}")
        return {
            'signal': current_data['signal'],
            'entry_price': current_data['entry_price'],
            'stop_loss': current_data['stop_loss'],
            'take_profit': current_data['take_profit'],
            'position_size': position_size,
            'signal_strength': current_data['signal_strength']
        }
    
    def _check_exit_conditions(self, current_data: pd.Series, portfolio: Dict, 
                             current_index: int, signals_df: pd.DataFrame) -> Optional[Dict]:
        """Check if exit conditions are met"""
        if portfolio['position'] == 0:
            return None
        
        current_price = current_data['close']
        position_value = portfolio['position_value']
        position = portfolio['position']
        
        # Calculate current P&L
        if position > 0:  # Long position
            pnl_pct = (current_price - portfolio['entry_price']) / portfolio['entry_price']
            if current_price <= portfolio['stop_loss']:
                return {'reason': 'stop_loss', 'price': current_price}
            elif current_price >= portfolio['take_profit']:
                return {'reason': 'take_profit', 'price': current_price}
        else:  # Short position
            pnl_pct = (portfolio['entry_price'] - current_price) / portfolio['entry_price']
            if current_price >= portfolio['stop_loss']:
                return {'reason': 'stop_loss', 'price': current_price}
            elif current_price <= portfolio['take_profit']:
                return {'reason': 'take_profit', 'price': current_price}
        
        # Check for opposite signal
        if current_data['signal'] != 0 and current_data['signal'] != np.sign(position):
            return {'reason': 'opposite_signal', 'price': current_price}
        
        return None
    
    def _execute_entry(self, portfolio: Dict, price: float, entry_signal: Dict, 
                      index: int) -> Dict:
        """Execute entry trade"""
        position_size = entry_signal['position_size']
        trade_value = position_size * price
        
        if trade_value > portfolio['cash']:
            position_size = portfolio['cash'] / price
            trade_value = portfolio['cash']
        
        portfolio['cash'] -= trade_value
        portfolio['position'] = position_size if entry_signal['signal'] > 0 else -position_size
        portfolio['position_value'] = abs(portfolio['position']) * price
        portfolio['entry_price'] = price
        portfolio['stop_loss'] = entry_signal['stop_loss']
        portfolio['take_profit'] = entry_signal['take_profit']
        
        # Record trade
        trade = {
            'entry_time': index,
            'entry_price': price,
            'position_size': portfolio['position'],
            'stop_loss': entry_signal['stop_loss'],
            'take_profit': entry_signal['take_profit'],
            'signal_strength': entry_signal['signal_strength']
        }
        
        self.trades.append(trade)
        logger.info(f"Entry trade: {portfolio['position']:.2f} @ {price:.2f}")
        return portfolio
    
    def _execute_exit(self, portfolio: Dict, price: float, exit_signal: Dict, 
                     index: int) -> Dict:
        """Execute exit trade"""
        if portfolio['position'] == 0:
            return portfolio
        
        # Calculate P&L
        position_value = abs(portfolio['position']) * price
        pnl = position_value - (abs(portfolio['position']) * portfolio['entry_price'])
        
        if portfolio['position'] < 0:  # Short position
            pnl = (abs(portfolio['position']) * portfolio['entry_price']) - position_value
        
        # Update cash
        portfolio['cash'] += position_value
        
        # Record trade completion
        trade = {
            'exit_time': index,
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': pnl / (abs(portfolio['position']) * portfolio['entry_price']),
            'exit_reason': exit_signal['reason']
        }
        
        # Find the corresponding entry trade and update it
        for i in range(len(self.trades) - 1, -1, -1):
            if 'exit_time' not in self.trades[i]:
                self.trades[i].update(trade)
                break
        
        logger.info(f"Exit trade: {portfolio['position']:.2f} @ {price:.2f}, P&L: {pnl:.2f}")
        
        # Reset position
        portfolio['position'] = 0
        portfolio['position_value'] = 0
        
        return portfolio
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_value:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        
        # Risk metrics
        returns = portfolio_df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        peak = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        completed_trades = [t for t in self.trades if 'exit_time' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Buy and hold comparison
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return
        }
    
    def optimize_strategy(self, data: pd.DataFrame, param_ranges: Dict) -> Dict:
        """
        Optimize strategy parameters using walk-forward analysis
        
        Args:
            data: Historical price data
            param_ranges: Parameter ranges to test
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting strategy optimization")
        
        best_params = None
        best_performance = -np.inf
        optimization_results = []
        
        # Test different parameter combinations
        for po3_mult in param_ranges.get('po3_multipliers', [0.5, 1.0, 1.5, 2.0]):
            for lookback in param_ranges.get('lookback_periods', [6, 9, 12, 18]):
                for max_pos in param_ranges.get('max_position_sizes', [0.01, 0.02, 0.03]):
                    
                    # Update parameters
                    self.max_position_size = max_pos
                    self.goldbach_calc.lookback_period = lookback
                    
                    # Run backtest
                    results = self.run_backtest(data)
                    
                    if results['performance_metrics']:
                        performance = results['performance_metrics']['sharpe_ratio']
                        
                        optimization_results.append({
                            'po3_multiplier': po3_mult,
                            'lookback_period': lookback,
                            'max_position_size': max_pos,
                            'sharpe_ratio': performance,
                            'total_return': results['performance_metrics']['total_return'],
                            'max_drawdown': results['performance_metrics']['max_drawdown']
                        })
                        
                        if performance > best_performance:
                            best_performance = performance
                            best_params = {
                                'po3_multiplier': po3_mult,
                                'lookback_period': lookback,
                                'max_position_size': max_pos,
                                'performance': performance
                            }
        
        return {
            'best_params': best_params,
            'optimization_results': optimization_results,
            'best_performance': best_performance
        }

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 252)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 252))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 252))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Initialize backtesting engine
    backtester = BacktestingEngine(initial_capital=100000)
    
    # Run backtest
    results = backtester.run_backtest(sample_data)
    
    # Display results
    metrics = results['performance_metrics']
    print("Backtest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Run optimization
    param_ranges = {
        'po3_multipliers': [0.5, 1.0, 1.5],
        'lookback_periods': [6, 9, 12],
        'max_position_sizes': [0.01, 0.02]
    }
    
    optimization = backtester.optimize_strategy(sample_data, param_ranges)
    print(f"\nOptimization Results:")
    print(f"Best Parameters: {optimization['best_params']}")
