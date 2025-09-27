"""
Portfolio Optimization Module
Modern Portfolio Theory implementation for optimal asset allocation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Modern Portfolio Theory implementation for optimal portfolio allocation"""
    
    def __init__(self):
        self.returns = None
        self.covariance_matrix = None
        self.expected_returns = None
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.optimization_results = {}
        
    def calculate_returns(self, prices):
        """Calculate returns from price data"""
        if isinstance(prices, pd.DataFrame):
            returns = prices.pct_change().dropna()
        else:
            returns = pd.Series(prices).pct_change().dropna()
        
        return returns
    
    def calculate_expected_returns(self, returns):
        """Calculate expected returns (mean returns)"""
        if isinstance(returns, pd.DataFrame):
            expected_returns = returns.mean() * 252  # Annualized
        else:
            # Single asset case - return as array
            expected_returns = np.array([returns.mean() * 252])
        
        return expected_returns
    
    def calculate_covariance_matrix(self, returns):
        """Calculate covariance matrix of returns"""
        if isinstance(returns, pd.DataFrame):
            covariance_matrix = returns.cov() * 252  # Annualized
        else:
            # Single asset case
            variance = returns.var() * 252
            covariance_matrix = np.array([[variance]])
        
        return covariance_matrix
    
    def portfolio_performance(self, weights, expected_returns, covariance_matrix):
        """Calculate portfolio performance metrics"""
        # Ensure arrays are 1D
        weights = np.array(weights).flatten()
        expected_returns = np.array(expected_returns).flatten()
        
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def negative_sharpe_ratio(self, weights, expected_returns, covariance_matrix, risk_free_rate):
        """Negative Sharpe ratio for minimization"""
        portfolio_return, portfolio_volatility = self.portfolio_performance(weights, expected_returns, covariance_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    def portfolio_variance(self, weights, covariance_matrix):
        """Portfolio variance for minimization"""
        weights = np.array(weights).flatten()
        return np.dot(weights.T, np.dot(covariance_matrix, weights))
    
    def optimize_portfolio(self, returns, optimization_type='max_sharpe', target_return=None, target_volatility=None):
        """Optimize portfolio allocation"""
        self.returns = returns
        self.expected_returns = self.calculate_expected_returns(returns)
        self.covariance_matrix = self.calculate_covariance_matrix(returns)
        
        num_assets = len(self.expected_returns)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long-only portfolio)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/num_assets] * num_assets)
        
        if optimization_type == 'max_sharpe':
            # Maximize Sharpe ratio
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(self.expected_returns, self.covariance_matrix, self.risk_free_rate),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
        elif optimization_type == 'min_variance':
            # Minimize portfolio variance
            result = minimize(
                self.portfolio_variance,
                initial_weights,
                args=(self.covariance_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
        elif optimization_type == 'target_return':
            # Target return optimization
            if target_return is None:
                target_return = np.mean(self.expected_returns)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}
            ]
            
            result = minimize(
                self.portfolio_variance,
                initial_weights,
                args=(self.covariance_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
        elif optimization_type == 'target_volatility':
            # Target volatility optimization
            if target_volatility is None:
                target_volatility = np.sqrt(np.mean(np.diag(self.covariance_matrix)))
            
            def target_volatility_constraint(weights):
                portfolio_return, portfolio_volatility = self.portfolio_performance(weights, self.expected_returns, self.covariance_matrix)
                return portfolio_volatility - target_volatility
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': target_volatility_constraint}
            ]
            
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(self.expected_returns, self.covariance_matrix, self.risk_free_rate),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        else:
            raise ValueError("Invalid optimization type")
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self.portfolio_performance(
                optimal_weights, self.expected_returns, self.covariance_matrix
            )
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            self.optimization_results[optimization_type] = {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
            
            return self.optimization_results[optimization_type]
        else:
            print(f"Optimization failed: {result.message}")
            return None
    
    def efficient_frontier(self, returns, num_portfolios=100):
        """Generate efficient frontier"""
        self.returns = returns
        self.expected_returns = self.calculate_expected_returns(returns)
        self.covariance_matrix = self.calculate_covariance_matrix(returns)
        
        num_assets = len(self.expected_returns)
        
        # Generate random portfolios
        portfolio_returns = []
        portfolio_volatilities = []
        portfolio_weights = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Calculate portfolio performance
            portfolio_return, portfolio_volatility = self.portfolio_performance(
                weights, self.expected_returns, self.covariance_matrix
            )
            
            portfolio_returns.append(portfolio_return)
            portfolio_volatilities.append(portfolio_volatility)
            portfolio_weights.append(weights)
        
        # Find optimal portfolios
        min_vol_portfolio = self.optimize_portfolio(returns, 'min_variance')
        max_sharpe_portfolio = self.optimize_portfolio(returns, 'max_sharpe')
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_volatilities': portfolio_volatilities,
            'portfolio_weights': portfolio_weights,
            'min_volatility_portfolio': min_vol_portfolio,
            'max_sharpe_portfolio': max_sharpe_portfolio
        }
    
    def calculate_portfolio_metrics(self, weights, returns):
        """Calculate comprehensive portfolio metrics"""
        if isinstance(returns, pd.DataFrame):
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns * weights[0]
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Downside risk
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def monte_carlo_simulation(self, returns, num_simulations=1000, time_horizon=252):
        """Monte Carlo simulation for portfolio returns"""
        if isinstance(returns, pd.DataFrame):
            expected_returns = self.calculate_expected_returns(returns)
            covariance_matrix = self.calculate_covariance_matrix(returns)
        else:
            expected_returns = np.array([returns.mean() * 252])
            covariance_matrix = np.array([[returns.var() * 252]])
        
        # Generate random returns
        # Ensure expected_returns is 1D and covariance_matrix is 2D
        expected_returns = np.array(expected_returns).flatten()
        covariance_matrix = np.array(covariance_matrix)
        
        simulated_returns = np.random.multivariate_normal(
            expected_returns, covariance_matrix, (num_simulations, time_horizon)
        )
        
        # Calculate portfolio values
        initial_value = 100000  # $100,000 initial investment
        portfolio_values = np.zeros((num_simulations, time_horizon + 1))
        portfolio_values[:, 0] = initial_value
        
        for i in range(time_horizon):
            # Ensure simulated_returns[:, i] is 1D
            returns_i = simulated_returns[:, i].flatten()
            portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + returns_i)
        
        # Calculate statistics
        final_values = portfolio_values[:, -1]
        returns_simulation = (final_values - initial_value) / initial_value
        
        return {
            'simulated_returns': returns_simulation,
            'portfolio_values': portfolio_values,
            'mean_return': np.mean(returns_simulation),
            'std_return': np.std(returns_simulation),
            'percentile_5': np.percentile(returns_simulation, 5),
            'percentile_95': np.percentile(returns_simulation, 95),
            'probability_of_loss': np.mean(returns_simulation < 0)
        }
    
    def generate_optimization_report(self, returns, asset_names=None):
        """Generate comprehensive optimization report"""
        if asset_names is None:
            if isinstance(returns, pd.DataFrame):
                asset_names = returns.columns.tolist()
            else:
                asset_names = ['Asset']
        
        # Run different optimizations
        optimizations = ['max_sharpe', 'min_variance', 'target_return']
        results = {}
        
        for opt_type in optimizations:
            if opt_type == 'target_return':
                target_return = np.mean(self.calculate_expected_returns(returns))
                results[opt_type] = self.optimize_portfolio(returns, opt_type, target_return=target_return)
            else:
                results[opt_type] = self.optimize_portfolio(returns, opt_type)
        
        # Generate efficient frontier
        frontier = self.efficient_frontier(returns)
        
        # Monte Carlo simulation
        monte_carlo = self.monte_carlo_simulation(returns)
        
        # Create report
        report = {
            'asset_names': asset_names,
            'optimization_results': results,
            'efficient_frontier': frontier,
            'monte_carlo_simulation': monte_carlo,
            'risk_free_rate': self.risk_free_rate,
            'analysis_date': pd.Timestamp.now()
        }
        
        return report

def demo_portfolio_optimization():
    """Demo function for portfolio optimization"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate correlated returns for 5 assets
    num_assets = 5
    asset_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.7, 0.6, 0.5, 0.4],
        [0.7, 1.0, 0.8, 0.6, 0.5],
        [0.6, 0.8, 1.0, 0.7, 0.6],
        [0.5, 0.6, 0.7, 1.0, 0.8],
        [0.4, 0.5, 0.6, 0.8, 1.0]
    ])
    
    # Generate returns
    returns_data = np.random.multivariate_normal(
        [0.0008, 0.0009, 0.001, 0.0011, 0.0012],  # Daily expected returns
        correlation_matrix * 0.0001,  # Daily covariance
        252
    )
    
    returns_df = pd.DataFrame(returns_data, index=dates, columns=asset_names)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Generate optimization report
    report = optimizer.generate_optimization_report(returns_df, asset_names)
    
    print("\n=== PORTFOLIO OPTIMIZATION REPORT ===")
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Risk-Free Rate: {report['risk_free_rate']:.2%}")
    print(f"Assets: {', '.join(report['asset_names'])}")
    
    print("\n=== OPTIMIZATION RESULTS ===")
    for opt_type, result in report['optimization_results'].items():
        if result:
            print(f"\n{opt_type.upper().replace('_', ' ')} PORTFOLIO:")
            print(f"Expected Return: {result['expected_return']:.2%}")
            print(f"Volatility: {result['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print("Weights:")
            for i, (asset, weight) in enumerate(zip(asset_names, result['weights'])):
                print(f"  {asset}: {weight:.2%}")
    
    print("\n=== EFFICIENT FRONTIER ===")
    frontier = report['efficient_frontier']
    print(f"Min Volatility Portfolio Return: {frontier['min_volatility_portfolio']['expected_return']:.2%}")
    print(f"Min Volatility Portfolio Volatility: {frontier['min_volatility_portfolio']['volatility']:.2%}")
    print(f"Max Sharpe Portfolio Return: {frontier['max_sharpe_portfolio']['expected_return']:.2%}")
    print(f"Max Sharpe Portfolio Volatility: {frontier['max_sharpe_portfolio']['volatility']:.2%}")
    
    print("\n=== MONTE CARLO SIMULATION ===")
    mc = report['monte_carlo_simulation']
    print(f"Mean Return (1 year): {mc['mean_return']:.2%}")
    print(f"Standard Deviation: {mc['std_return']:.2%}")
    print(f"5th Percentile: {mc['percentile_5']:.2%}")
    print(f"95th Percentile: {mc['percentile_95']:.2%}")
    print(f"Probability of Loss: {mc['probability_of_loss']:.2%}")

if __name__ == "__main__":
    demo_portfolio_optimization()
