"""
Enhanced Risk Manager Agent with Advanced Risk Calculations
Implements VaR, CVaR, drawdown analysis, and comprehensive risk management
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from multi_agent_framework import BaseAgent, AgentOutput, TradingSignal, SignalType, RiskMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float  # Maximum position size as % of portfolio
    max_portfolio_var: float  # Maximum portfolio VaR as % of portfolio
    max_drawdown: float  # Maximum allowed drawdown
    max_correlation: float  # Maximum correlation between positions
    max_sector_exposure: float  # Maximum exposure to single sector
    stop_loss_percentage: float  # Stop loss percentage
    take_profit_ratio: float  # Risk-reward ratio

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_size: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_with_portfolio: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_var_95: float
    total_var_99: float
    total_cvar_95: float
    total_cvar_99: float
    portfolio_volatility: float
    portfolio_sharpe: float
    portfolio_beta: float
    max_drawdown: float
    current_drawdown: float
    diversification_ratio: float
    concentration_risk: float
    sector_exposure: Dict[str, float]

class RiskCalculator:
    """Advanced risk calculation engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual
        self.lookback_periods = self.config.get('lookback_periods', 252)  # 1 year
        self.confidence_levels = [0.95, 0.99]  # VaR confidence levels
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Historical VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            
            return abs(var_value)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            if len(returns) < 30:
                return 0.0
            
            var_value = self.calculate_var(returns, confidence_level)
            
            # CVaR is the expected value of returns below VaR
            tail_returns = returns[returns <= -var_value]
            
            if len(tail_returns) == 0:
                return var_value
            
            cvar_value = abs(tail_returns.mean())
            return cvar_value
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, datetime, datetime]:
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 2:
                return 0.0, datetime.now(), datetime.now()
            
            # Calculate running maximum
            running_max = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - running_max) / running_max
            
            # Find maximum drawdown
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin()
            
            # Find recovery date
            recovery_date = None
            if max_dd_date < prices.index[-1]:
                post_dd_prices = prices[max_dd_date:]
                post_dd_max = post_dd_prices.expanding().max()
                recovery_mask = post_dd_prices >= prices[max_dd_date]
                if recovery_mask.any():
                    recovery_date = recovery_mask.idxmax()
            
            return abs(max_dd), max_dd_date, recovery_date or datetime.now()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, datetime.now(), datetime.now()
    
    def calculate_current_drawdown(self, prices: pd.Series) -> float:
        """Calculate current drawdown from peak"""
        try:
            if len(prices) < 2:
                return 0.0
            
            current_price = prices.iloc[-1]
            peak_price = prices.expanding().max().iloc[-1]
            
            current_dd = (current_price - peak_price) / peak_price
            return abs(current_dd)
            
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility"""
        try:
            if len(returns) < 30:
                return 0.0
            
            volatility = returns.std()
            
            if annualized:
                # Annualize volatility (assuming daily returns)
                volatility *= np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 30:
                return 0.0
            
            risk_free_rate = risk_free_rate or self.risk_free_rate
            
            # Calculate excess returns
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate Sharpe ratio
            sharpe = excess_returns.mean() / returns.std()
            
            # Annualize
            sharpe *= np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            if len(asset_returns) < 30 or len(market_returns) < 30:
                return 1.0
            
            # Align returns
            aligned_returns = pd.concat([asset_returns, market_returns], axis=1, join='inner')
            asset_aligned = aligned_returns.iloc[:, 0]
            market_aligned = aligned_returns.iloc[:, 1]
            
            # Calculate covariance and variance
            covariance = np.cov(asset_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        try:
            return returns_df.corr()
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_var(self, weights: np.ndarray, cov_matrix: np.ndarray, 
                              confidence_level: float = 0.95) -> float:
        """Calculate portfolio VaR using parametric method"""
        try:
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # VaR calculation
            z_score = stats.norm.ppf(1 - confidence_level)
            var_value = abs(z_score * portfolio_std)
            
            return var_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Individual asset volatilities
            individual_vols = np.sqrt(np.diag(cov_matrix))
            
            # Weighted average volatility
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Diversification ratio
            div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            
            return div_ratio
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0

class RiskOptimizer:
    """Portfolio risk optimization engine"""
    
    def __init__(self, risk_calculator: RiskCalculator):
        self.risk_calculator = risk_calculator
    
    def optimize_portfolio_weights(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                 risk_tolerance: float = 0.5) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective function: minimize risk-adjusted return
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -(portfolio_return - risk_tolerance * portfolio_risk)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                logger.warning("Portfolio optimization failed, using equal weights")
                return initial_weights
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio weights: {e}")
            return np.ones(len(expected_returns)) / len(expected_returns)
    
    def calculate_risk_parity_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk parity weights"""
        try:
            # Calculate inverse volatility weights
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
            
            return inv_vol_weights
            
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return np.ones(len(cov_matrix)) / len(cov_matrix)

class StressTester:
    """Stress testing and scenario analysis"""
    
    def __init__(self, risk_calculator: RiskCalculator):
        self.risk_calculator = risk_calculator
    
    def run_stress_tests(self, portfolio_returns: pd.Series, 
                        scenarios: Dict[str, float] = None) -> Dict[str, float]:
        """Run stress tests on portfolio"""
        try:
            if scenarios is None:
                scenarios = {
                    'market_crash': -0.20,  # 20% market decline
                    'recession': -0.15,     # 15% decline
                    'volatility_spike': 0.30,  # 30% volatility increase
                    'interest_rate_shock': 0.05  # 5% rate increase
                }
            
            stress_results = {}
            
            for scenario_name, shock in scenarios.items():
                if 'decline' in scenario_name or 'crash' in scenario_name:
                    # Apply negative shock
                    stressed_returns = portfolio_returns + shock
                else:
                    # Apply volatility shock
                    stressed_returns = portfolio_returns * (1 + shock)
                
                # Calculate stressed VaR
                stressed_var = self.risk_calculator.calculate_var(stressed_returns)
                stress_results[scenario_name] = stressed_var
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}

class EnhancedRiskManagerAgent(BaseAgent):
    """Enhanced Risk Manager Agent with comprehensive risk analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("enhanced_risk_manager_agent", config)
        self.add_dependency("technical_analysis_agent")
        self.add_dependency("fundamentals_agent")
        self.add_dependency("sentiment_agent")
        
        # Initialize risk components
        self.risk_calculator = RiskCalculator(config)
        self.risk_optimizer = RiskOptimizer(self.risk_calculator)
        self.stress_tester = StressTester(self.risk_calculator)
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_position_size=config.get('max_position_size', 0.05),  # 5%
            max_portfolio_var=config.get('max_portfolio_var', 0.02),   # 2%
            max_drawdown=config.get('max_drawdown', 0.10),            # 10%
            max_correlation=config.get('max_correlation', 0.7),      # 70%
            max_sector_exposure=config.get('max_sector_exposure', 0.3), # 30%
            stop_loss_percentage=config.get('stop_loss_percentage', 0.02), # 2%
            take_profit_ratio=config.get('take_profit_ratio', 2.0)    # 2:1
        )
        
        # Risk tracking
        self.portfolio_history = []
        self.risk_violations = []
        self.risk_metrics_history = []
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process risk management analysis"""
        start_time = datetime.now()
        self.status = "running"
        
        try:
            # Collect signals from analysis agents
            agent_outputs = input_data.get('agent_outputs', [])
            market_data = input_data.get('market_data', {})
            
            # Analyze risk for each signal
            risk_analysis = await self._analyze_portfolio_risk(agent_outputs, market_data)
            
            # Filter signals based on risk criteria
            filtered_signals = await self._filter_signals_by_risk(agent_outputs, risk_analysis)
            
            # Calculate comprehensive risk metrics
            risk_metrics = await self._calculate_comprehensive_risk_metrics(risk_analysis)
            
            # Run stress tests
            stress_test_results = await self._run_stress_tests(risk_analysis)
            
            # Generate risk recommendations
            risk_recommendations = await self._generate_risk_recommendations(risk_analysis, risk_metrics)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = AgentOutput(
                agent_id=self.agent_id,
                status="completed",
                data={
                    'risk_analysis': risk_analysis,
                    'filtered_signals': filtered_signals,
                    'risk_metrics': risk_metrics,
                    'stress_test_results': stress_test_results,
                    'risk_recommendations': risk_recommendations,
                    'risk_violations': self.risk_violations
                },
                signals=filtered_signals,
                risk_metrics=risk_metrics,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Risk Manager Agent error: {e}")
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
    
    async def _analyze_portfolio_risk(self, agent_outputs: List[AgentOutput], 
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk"""
        risk_analysis = {
            'position_risks': {},
            'portfolio_risk': None,
            'correlation_matrix': None,
            'risk_violations': []
        }
        
        try:
            # Collect all signals
            all_signals = []
            for output in agent_outputs:
                all_signals.extend(output.signals)
            
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in all_signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []
                signals_by_symbol[signal.symbol].append(signal)
            
            # Analyze risk for each symbol
            for symbol, signals in signals_by_symbol.items():
                position_risk = await self._calculate_position_risk(symbol, signals, market_data)
                risk_analysis['position_risks'][symbol] = position_risk
            
            # Calculate portfolio-level risk
            portfolio_risk = await self._calculate_portfolio_risk(risk_analysis['position_risks'])
            risk_analysis['portfolio_risk'] = portfolio_risk
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(signals_by_symbol, market_data)
            risk_analysis['correlation_matrix'] = correlation_matrix
            
            # Check for risk violations
            violations = await self._check_risk_violations(risk_analysis)
            risk_analysis['risk_violations'] = violations
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {e}")
        
        return risk_analysis
    
    async def _calculate_position_risk(self, symbol: str, signals: List[TradingSignal], 
                                     market_data: Dict[str, Any]) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            # Get market data for symbol
            symbol_data = market_data.get(symbol, {})
            daily_data = symbol_data.get('1d', {})
            
            # Simulate price history for risk calculations
            current_price = daily_data.get('close', 100.0)
            price_history = self._simulate_price_history(current_price, 252)  # 1 year
            
            # Calculate returns
            returns = price_history.pct_change().dropna()
            
            # Calculate risk metrics
            var_95 = self.risk_calculator.calculate_var(returns, 0.95)
            var_99 = self.risk_calculator.calculate_var(returns, 0.99)
            cvar_95 = self.risk_calculator.calculate_cvar(returns, 0.95)
            cvar_99 = self.risk_calculator.calculate_cvar(returns, 0.99)
            volatility = self.risk_calculator.calculate_volatility(returns)
            sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(returns)
            
            # Calculate max drawdown
            max_dd, _, _ = self.risk_calculator.calculate_max_drawdown(price_history)
            
            # Calculate position size based on signals
            position_size = self._calculate_position_size(signals)
            
            # Calculate beta (simplified)
            beta = 1.0  # Would calculate against market index
            
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_dd,
                correlation_with_portfolio=0.5  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol,
                position_size=0.0,
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                volatility=0.0,
                beta=1.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                correlation_with_portfolio=0.0
            )
    
    def _simulate_price_history(self, current_price: float, periods: int) -> pd.Series:
        """Simulate price history for risk calculations"""
        # Generate random walk with drift
        returns = np.random.normal(0.0005, 0.02, periods)  # Daily returns
        prices = [current_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        return pd.Series(prices, index=dates)
    
    def _calculate_position_size(self, signals: List[TradingSignal]) -> float:
        """Calculate position size based on signals"""
        if not signals:
            return 0.0
        
        # Use average confidence as position size indicator
        avg_confidence = np.mean([signal.confidence for signal in signals])
        
        # Cap position size based on risk limits
        max_size = self.risk_limits.max_position_size
        position_size = min(max_size, avg_confidence * max_size)
        
        return position_size
    
    async def _calculate_portfolio_risk(self, position_risks: Dict[str, PositionRisk]) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        try:
            if not position_risks:
                return PortfolioRisk(
                    total_var_95=0.0,
                    total_var_99=0.0,
                    total_cvar_95=0.0,
                    total_cvar_99=0.0,
                    portfolio_volatility=0.0,
                    portfolio_sharpe=0.0,
                    portfolio_beta=1.0,
                    max_drawdown=0.0,
                    current_drawdown=0.0,
                    diversification_ratio=1.0,
                    concentration_risk=0.0,
                    sector_exposure={}
                )
            
            # Calculate portfolio metrics
            total_var_95 = sum(pos.var_95 * pos.position_size for pos in position_risks.values())
            total_var_99 = sum(pos.var_99 * pos.position_size for pos in position_risks.values())
            total_cvar_95 = sum(pos.cvar_95 * pos.position_size for pos in position_risks.values())
            total_cvar_99 = sum(pos.cvar_99 * pos.position_size for pos in position_risks.values())
            
            # Calculate weighted portfolio metrics
            total_position_size = sum(pos.position_size for pos in position_risks.values())
            
            if total_position_size > 0:
                portfolio_volatility = sum(pos.volatility * pos.position_size for pos in position_risks.values()) / total_position_size
                portfolio_sharpe = sum(pos.sharpe_ratio * pos.position_size for pos in position_risks.values()) / total_position_size
                portfolio_beta = sum(pos.beta * pos.position_size for pos in position_risks.values()) / total_position_size
                max_drawdown = max(pos.max_drawdown for pos in position_risks.values())
            else:
                portfolio_volatility = 0.0
                portfolio_sharpe = 0.0
                portfolio_beta = 1.0
                max_drawdown = 0.0
            
            # Calculate concentration risk (Herfindahl index)
            position_sizes = [pos.position_size for pos in position_risks.values()]
            concentration_risk = sum(size**2 for size in position_sizes) if position_sizes else 0.0
            
            # Calculate diversification ratio (simplified)
            diversification_ratio = 1.0 / (1.0 + concentration_risk) if concentration_risk > 0 else 1.0
            
            return PortfolioRisk(
                total_var_95=total_var_95,
                total_var_99=total_var_99,
                total_cvar_95=total_cvar_95,
                total_cvar_99=total_cvar_99,
                portfolio_volatility=portfolio_volatility,
                portfolio_sharpe=portfolio_sharpe,
                portfolio_beta=portfolio_beta,
                max_drawdown=max_drawdown,
                current_drawdown=0.0,  # Would calculate from current portfolio value
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                sector_exposure={}  # Would calculate from sector classifications
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return PortfolioRisk(
                total_var_95=0.0,
                total_var_99=0.0,
                total_cvar_95=0.0,
                total_cvar_99=0.0,
                portfolio_volatility=0.0,
                portfolio_sharpe=0.0,
                portfolio_beta=1.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                diversification_ratio=1.0,
                concentration_risk=0.0,
                sector_exposure={}
            )
    
    async def _calculate_correlation_matrix(self, signals_by_symbol: Dict[str, List[TradingSignal]], 
                                          market_data: Dict[str, Any]) -> pd.DataFrame:
        """Calculate correlation matrix between symbols"""
        try:
            symbols = list(signals_by_symbol.keys())
            if len(symbols) < 2:
                return pd.DataFrame()
            
            # Simulate returns for correlation calculation
            returns_data = {}
            for symbol in symbols:
                symbol_data = market_data.get(symbol, {})
                daily_data = symbol_data.get('1d', {})
                current_price = daily_data.get('close', 100.0)
                
                # Generate simulated returns
                returns = np.random.normal(0, 0.02, 252)  # 1 year of daily returns
                returns_data[symbol] = returns
            
            # Create DataFrame and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    async def _check_risk_violations(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for risk limit violations"""
        violations = []
        
        try:
            portfolio_risk = risk_analysis.get('portfolio_risk')
            if not portfolio_risk:
                return violations
            
            # Check portfolio VaR limit
            if portfolio_risk.total_var_95 > self.risk_limits.max_portfolio_var:
                violations.append({
                    'type': 'portfolio_var_exceeded',
                    'current_value': portfolio_risk.total_var_95,
                    'limit': self.risk_limits.max_portfolio_var,
                    'severity': 'high'
                })
            
            # Check max drawdown limit
            if portfolio_risk.max_drawdown > self.risk_limits.max_drawdown:
                violations.append({
                    'type': 'max_drawdown_exceeded',
                    'current_value': portfolio_risk.max_drawdown,
                    'limit': self.risk_limits.max_drawdown,
                    'severity': 'critical'
                })
            
            # Check concentration risk
            if portfolio_risk.concentration_risk > 0.5:  # 50% concentration threshold
                violations.append({
                    'type': 'concentration_risk_high',
                    'current_value': portfolio_risk.concentration_risk,
                    'limit': 0.5,
                    'severity': 'medium'
                })
            
            # Check individual position limits
            for symbol, position_risk in risk_analysis.get('position_risks', {}).items():
                if position_risk.position_size > self.risk_limits.max_position_size:
                    violations.append({
                        'type': 'position_size_exceeded',
                        'symbol': symbol,
                        'current_value': position_risk.position_size,
                        'limit': self.risk_limits.max_position_size,
                        'severity': 'high'
                    })
            
        except Exception as e:
            logger.error(f"Error checking risk violations: {e}")
        
        return violations
    
    async def _filter_signals_by_risk(self, agent_outputs: List[AgentOutput], 
                                    risk_analysis: Dict[str, Any]) -> List[TradingSignal]:
        """Filter signals based on risk criteria"""
        filtered_signals = []
        
        try:
            # Collect all signals
            all_signals = []
            for output in agent_outputs:
                all_signals.extend(output.signals)
            
            # Filter signals based on risk criteria
            for signal in all_signals:
                # Check if signal passes risk filters
                if await self._signal_passes_risk_filters(signal, risk_analysis):
                    # Adjust position size based on risk
                    adjusted_signal = await self._adjust_signal_for_risk(signal, risk_analysis)
                    filtered_signals.append(adjusted_signal)
            
        except Exception as e:
            logger.error(f"Error filtering signals by risk: {e}")
        
        return filtered_signals
    
    async def _signal_passes_risk_filters(self, signal: TradingSignal, 
                                        risk_analysis: Dict[str, Any]) -> bool:
        """Check if signal passes risk filters"""
        try:
            # Check if there are risk violations
            violations = risk_analysis.get('risk_violations', [])
            critical_violations = [v for v in violations if v.get('severity') == 'critical']
            
            # Block all signals if critical violations exist
            if critical_violations:
                return False
            
            # Check signal-specific risk criteria
            symbol = signal.symbol
            position_risks = risk_analysis.get('position_risks', {})
            
            if symbol in position_risks:
                position_risk = position_risks[symbol]
                
                # Check volatility limit
                if position_risk.volatility > 0.5:  # 50% annual volatility limit
                    return False
                
                # Check VaR limit
                if position_risk.var_95 > 0.05:  # 5% daily VaR limit
                    return False
            
            # Check signal confidence
            if signal.confidence < 0.6:  # Minimum confidence threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal risk filters: {e}")
            return False
    
    async def _adjust_signal_for_risk(self, signal: TradingSignal, 
                                    risk_analysis: Dict[str, Any]) -> TradingSignal:
        """Adjust signal based on risk analysis"""
        try:
            # Adjust position size based on risk
            symbol = signal.symbol
            position_risks = risk_analysis.get('position_risks', {})
            
            if symbol in position_risks:
                position_risk = position_risks[symbol]
                
                # Reduce position size based on volatility
                volatility_adjustment = max(0.5, 1.0 - position_risk.volatility)
                
                # Reduce position size based on VaR
                var_adjustment = max(0.5, 1.0 - (position_risk.var_95 / 0.05))
                
                # Apply adjustments
                adjusted_position_size = signal.metadata.get('position_size', 0.01)
                adjusted_position_size *= volatility_adjustment * var_adjustment
                
                # Cap at risk limit
                adjusted_position_size = min(adjusted_position_size, self.risk_limits.max_position_size)
                
                # Update signal metadata
                signal.metadata['position_size'] = adjusted_position_size
                signal.metadata['risk_adjusted'] = True
                signal.metadata['volatility_adjustment'] = volatility_adjustment
                signal.metadata['var_adjustment'] = var_adjustment
            
            return signal
            
        except Exception as e:
            logger.error(f"Error adjusting signal for risk: {e}")
            return signal
    
    async def _calculate_comprehensive_risk_metrics(self, risk_analysis: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            portfolio_risk = risk_analysis.get('portfolio_risk')
            if not portfolio_risk:
                return RiskMetrics(
                    var_95=0.0,
                    var_99=0.0,
                    cvar_95=0.0,
                    cvar_99=0.0,
                    max_drawdown=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    portfolio_value=100000.0
                )
            
            return RiskMetrics(
                var_95=portfolio_risk.total_var_95,
                var_99=portfolio_risk.total_var_99,
                cvar_95=portfolio_risk.total_cvar_95,
                cvar_99=portfolio_risk.total_cvar_99,
                max_drawdown=portfolio_risk.max_drawdown,
                volatility=portfolio_risk.portfolio_volatility,
                sharpe_ratio=portfolio_risk.portfolio_sharpe,
                portfolio_value=100000.0  # Would get from actual portfolio
            )
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk metrics: {e}")
            return RiskMetrics(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                portfolio_value=100000.0
            )
    
    async def _run_stress_tests(self, risk_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Run stress tests on portfolio"""
        try:
            # Simulate portfolio returns for stress testing
            portfolio_returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
            
            # Run stress tests
            stress_results = self.stress_tester.run_stress_tests(portfolio_returns)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    async def _generate_risk_recommendations(self, risk_analysis: Dict[str, Any], 
                                           risk_metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            violations = risk_analysis.get('risk_violations', [])
            
            for violation in violations:
                if violation['type'] == 'portfolio_var_exceeded':
                    recommendations.append({
                        'type': 'reduce_position_sizes',
                        'priority': 'high',
                        'description': f"Portfolio VaR ({violation['current_value']:.3f}) exceeds limit ({violation['limit']:.3f})",
                        'action': 'Reduce position sizes across portfolio'
                    })
                
                elif violation['type'] == 'max_drawdown_exceeded':
                    recommendations.append({
                        'type': 'emergency_stop',
                        'priority': 'critical',
                        'description': f"Max drawdown ({violation['current_value']:.3f}) exceeds limit ({violation['limit']:.3f})",
                        'action': 'Stop all trading and reassess strategy'
                    })
                
                elif violation['type'] == 'concentration_risk_high':
                    recommendations.append({
                        'type': 'diversify_portfolio',
                        'priority': 'medium',
                        'description': f"Portfolio concentration ({violation['current_value']:.3f}) is too high",
                        'action': 'Add more diversified positions'
                    })
            
            # General recommendations based on risk metrics
            if risk_metrics.sharpe_ratio < 1.0:
                recommendations.append({
                    'type': 'improve_risk_adjusted_returns',
                    'priority': 'medium',
                    'description': f"Sharpe ratio ({risk_metrics.sharpe_ratio:.2f}) is below target",
                    'action': 'Optimize portfolio for better risk-adjusted returns'
                })
            
            if risk_metrics.volatility > 0.2:
                recommendations.append({
                    'type': 'reduce_volatility',
                    'priority': 'medium',
                    'description': f"Portfolio volatility ({risk_metrics.volatility:.2f}) is high",
                    'action': 'Consider adding defensive positions'
                })
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
        
        return recommendations
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate risk manager input"""
        return 'agent_outputs' in input_data or 'market_data' in input_data
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        return {
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_portfolio_var': self.risk_limits.max_portfolio_var,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_correlation': self.risk_limits.max_correlation
            },
            'total_violations': len(self.risk_violations),
            'critical_violations': len([v for v in self.risk_violations if v.get('severity') == 'critical']),
            'last_analysis': datetime.now()
        }

# Example usage and testing
async def main():
    """Example usage of the enhanced risk manager agent"""
    
    # Create sample agent outputs
    sample_outputs = [
        AgentOutput(
            agent_id="technical_analysis_agent",
            status="completed",
            data={},
            signals=[
                TradingSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY,
                    confidence=0.8,
                    price=150.0,
                    timestamp=datetime.now(),
                    agent_id="technical_analysis_agent",
                    metadata={'strategy': 'trend_following', 'position_size': 0.03}
                )
            ],
            risk_metrics=None,
            timestamp=datetime.now(),
            execution_time=1.0
        )
    ]
    
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
    
    # Initialize risk manager agent
    config = {
        'max_position_size': 0.05,
        'max_portfolio_var': 0.02,
        'max_drawdown': 0.10,
        'risk_free_rate': 0.02
    }
    
    agent = EnhancedRiskManagerAgent(config)
    
    # Process risk analysis
    input_data = {
        'agent_outputs': sample_outputs,
        'market_data': sample_market_data
    }
    
    result = await agent.process(input_data)
    
    print("Risk Management Results:")
    print(f"Status: {result.status}")
    print(f"Filtered Signals: {len(result.signals)}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    if result.risk_metrics:
        print(f"\nRisk Metrics:")
        print(f"  VaR 95%: ${result.risk_metrics.var_95:.2f}")
        print(f"  VaR 99%: ${result.risk_metrics.var_99:.2f}")
        print(f"  CVaR 95%: ${result.risk_metrics.cvar_95:.2f}")
        print(f"  Max Drawdown: {result.risk_metrics.max_drawdown:.2%}")
        print(f"  Volatility: {result.risk_metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.2f}")
    
    # Print risk summary
    summary = agent.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Total Violations: {summary['total_violations']}")
    print(f"Critical Violations: {summary['critical_violations']}")

if __name__ == "__main__":
    asyncio.run(main())
