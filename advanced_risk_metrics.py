"""
Advanced risk metrics module
Implements VaR, CVaR, Monte Carlo simulations, and other sophisticated risk measures
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskMetrics:
    """Advanced risk metrics calculator"""
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.time_horizons = [1, 5, 10, 30]  # days
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = 'historical') -> Dict:
        """Calculate Value at Risk (VaR)"""
        
        if method == 'historical':
            # Historical simulation
            var_value = np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # Parametric (normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            var_value = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean_return = returns.mean()
            std_return = returns.std()
            simulations = np.random.normal(mean_return, std_return, 10000)
            var_value = np.percentile(simulations, (1 - confidence_level) * 100)
            
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
            
        return {
            'var': var_value,
            'confidence_level': confidence_level,
            'method': method,
            'interpretation': f"With {confidence_level*100}% confidence, losses will not exceed {abs(var_value)*100:.2f}%"
        }
        
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Dict:
        """Calculate Conditional Value at Risk (CVaR) - Expected Shortfall"""
        
        var_value = self.calculate_var(returns, confidence_level)['var']
        cvar_value = returns[returns <= var_value].mean()
        
        return {
            'cvar': cvar_value,
            'var': var_value,
            'confidence_level': confidence_level,
            'interpretation': f"Expected loss when VaR is exceeded: {abs(cvar_value)*100:.2f}%"
        }
        
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict:
        """Calculate maximum drawdown and related metrics"""
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Calculate drawdown duration
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        return {
            'maximum_drawdown': max_dd,
            'max_dd_date': max_dd_date,
            'current_drawdown': drawdown.iloc[-1],
            'average_drawdown': drawdown[drawdown < 0].mean(),
            'drawdown_duration': dd_duration,
            'interpretation': f"Maximum loss from peak: {abs(max_dd)*100:.2f}%"
        }
        
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict:
        """Calculate drawdown duration statistics"""
        
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
                
        if current_period > 0:
            drawdown_periods.append(current_period)
            
        if drawdown_periods:
            return {
                'max_duration': max(drawdown_periods),
                'avg_duration': np.mean(drawdown_periods),
                'current_duration': current_period
            }
        else:
            return {
                'max_duration': 0,
                'avg_duration': 0,
                'current_duration': 0
            }
            
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
        """Calculate Sharpe ratio and related metrics"""
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        max_dd = abs(self.calculate_maximum_drawdown(returns)['maximum_drawdown'])
        calmar_ratio = returns.mean() * 252 / max_dd if max_dd > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'risk_free_rate': risk_free_rate
        }
        
    def monte_carlo_simulation(self, returns: pd.Series, num_simulations: int = 10000, 
                             time_horizon: int = 252) -> Dict:
        """Monte Carlo simulation for portfolio returns"""
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulations = np.random.normal(mean_return, std_return, (num_simulations, time_horizon))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulations, axis=1)
        
        # Calculate statistics
        final_returns = cumulative_returns[:, -1] - 1
        
        return {
            'simulations': simulations,
            'cumulative_returns': cumulative_returns,
            'final_returns': final_returns,
            'mean_final_return': final_returns.mean(),
            'std_final_return': final_returns.std(),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'probability_of_loss': (final_returns < 0).mean(),
            'expected_return': mean_return * time_horizon,
            'expected_volatility': std_return * np.sqrt(time_horizon)
        }
        
    def stress_testing(self, returns: pd.Series, stress_scenarios: List[Dict] = None) -> Dict:
        """Stress testing with various market scenarios"""
        
        if stress_scenarios is None:
            stress_scenarios = [
                {'name': 'Market Crash', 'return_multiplier': -3.0, 'volatility_multiplier': 2.0},
                {'name': 'High Volatility', 'return_multiplier': 0.0, 'volatility_multiplier': 3.0},
                {'name': 'Bear Market', 'return_multiplier': -1.5, 'volatility_multiplier': 1.5},
                {'name': 'Bull Market', 'return_multiplier': 1.5, 'volatility_multiplier': 0.8}
            ]
            
        results = {}
        
        for scenario in stress_scenarios:
            # Apply stress scenario
            stressed_returns = returns * scenario['return_multiplier']
            stressed_volatility = returns.std() * scenario['volatility_multiplier']
            
            # Calculate metrics under stress
            var_95 = self.calculate_var(stressed_returns, 0.95)['var']
            cvar_95 = self.calculate_cvar(stressed_returns, 0.95)['cvar']
            max_dd = self.calculate_maximum_drawdown(stressed_returns)['maximum_drawdown']
            
            results[scenario['name']] = {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_dd,
                'expected_return': stressed_returns.mean() * 252,
                'volatility': stressed_volatility * np.sqrt(252)
            }
            
        return results
        
    def calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> Dict:
        """Calculate beta and related metrics"""
        
        # Align returns
        common_dates = portfolio_returns.index.intersection(market_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_dates]
        market_aligned = market_returns.loc[common_dates]
        
        # Calculate beta
        covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
        market_variance = market_aligned.var()
        beta = covariance / market_variance
        
        # Calculate alpha
        portfolio_mean = portfolio_aligned.mean()
        market_mean = market_aligned.mean()
        alpha = portfolio_mean - beta * market_mean
        
        # Calculate R-squared
        correlation = portfolio_aligned.corr(market_aligned)
        r_squared = correlation ** 2
        
        return {
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'correlation': correlation,
            'interpretation': f"Portfolio is {'more' if beta > 1 else 'less'} volatile than market"
        }
        
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Dict:
        """Calculate tracking error and related metrics"""
        
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate excess returns
        excess_returns = portfolio_aligned - benchmark_aligned
        
        # Calculate tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Calculate information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'excess_return': excess_returns.mean() * 252,
            'excess_volatility': excess_returns.std() * np.sqrt(252)
        }

class RiskVisualization:
    """Risk visualization tools"""
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
        
    def create_var_chart(self, returns: pd.Series, confidence_levels: List[float] = None) -> go.Figure:
        """Create VaR visualization chart"""
        
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        var_values = []
        for cl in confidence_levels:
            var_value = self.risk_metrics.calculate_var(returns, cl)['var']
            var_values.append(var_value)
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'{cl*100}%' for cl in confidence_levels],
            y=var_values,
            name='VaR',
            marker_color=['lightcoral', 'coral', 'darkred'],
            text=[f'{abs(v)*100:.2f}%' for v in var_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Value at Risk (VaR) Analysis',
            xaxis_title='Confidence Level',
            yaxis_title='VaR Value',
            height=400,
            template='plotly_dark'
        )
        
        return fig
        
    def create_drawdown_chart(self, returns: pd.Series) -> go.Figure:
        """Create drawdown visualization chart"""
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Cumulative Returns', 'Drawdown')
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name='Cumulative Returns',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Drawdown Analysis',
            height=600,
            template='plotly_dark'
        )
        
        return fig
        
    def create_monte_carlo_chart(self, returns: pd.Series, num_simulations: int = 1000) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        
        mc_results = self.risk_metrics.monte_carlo_simulation(returns, num_simulations)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monte Carlo Simulations', 'Final Returns Distribution')
        )
        
        # Sample of simulation paths
        sample_paths = mc_results['cumulative_returns'][:100]  # Show first 100 paths
        
        for i, path in enumerate(sample_paths):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(path))),
                    y=path,
                    mode='lines',
                    line=dict(width=1, opacity=0.3),
                    showlegend=False
                ),
                row=1, col=1
            )
            
        # Final returns distribution
        fig.add_trace(
            go.Histogram(
                x=mc_results['final_returns'],
                nbinsx=50,
                name='Final Returns',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Monte Carlo Simulation Results',
            height=600,
            template='plotly_dark'
        )
        
        return fig
        
    def create_risk_heatmap(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series = None) -> go.Figure:
        """Create risk metrics heatmap"""
        
        # Calculate various risk metrics
        var_95 = self.risk_metrics.calculate_var(portfolio_returns, 0.95)['var']
        cvar_95 = self.risk_metrics.calculate_cvar(portfolio_returns, 0.95)['cvar']
        max_dd = self.risk_metrics.calculate_maximum_drawdown(portfolio_returns)['maximum_drawdown']
        sharpe = self.risk_metrics.calculate_sharpe_ratio(portfolio_returns)['sharpe_ratio']
        
        metrics = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Sharpe Ratio']
        values = [abs(var_95), abs(cvar_95), abs(max_dd), sharpe]
        
        # Normalize values for heatmap
        normalized_values = []
        for i, value in enumerate(values):
            if i < 3:  # Risk metrics (lower is better)
                normalized_values.append(1 - min(value, 1))  # Invert for heatmap
            else:  # Sharpe ratio (higher is better)
                normalized_values.append(min(value, 3) / 3)
                
        fig = go.Figure(data=go.Heatmap(
            z=[normalized_values],
            x=metrics,
            y=['Risk Metrics'],
            colorscale='RdYlGn',
            text=[[f'{v:.3f}' for v in values]],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Risk Metrics Heatmap',
            height=200,
            template='plotly_dark'
        )
        
        return fig

class PortfolioRiskAnalyzer:
    """Comprehensive portfolio risk analysis"""
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
        self.risk_viz = RiskVisualization()
        
    def analyze_portfolio_risk(self, returns: pd.Series, 
                             benchmark_returns: pd.Series = None) -> Dict:
        """Comprehensive portfolio risk analysis"""
        
        analysis = {}
        
        # Basic risk metrics
        analysis['var_analysis'] = {
            'var_90': self.risk_metrics.calculate_var(returns, 0.90),
            'var_95': self.risk_metrics.calculate_var(returns, 0.95),
            'var_99': self.risk_metrics.calculate_var(returns, 0.99)
        }
        
        analysis['cvar_analysis'] = {
            'cvar_95': self.risk_metrics.calculate_cvar(returns, 0.95),
            'cvar_99': self.risk_metrics.calculate_cvar(returns, 0.99)
        }
        
        analysis['drawdown_analysis'] = self.risk_metrics.calculate_maximum_drawdown(returns)
        analysis['sharpe_analysis'] = self.risk_metrics.calculate_sharpe_ratio(returns)
        
        # Monte Carlo simulation
        analysis['monte_carlo'] = self.risk_metrics.monte_carlo_simulation(returns)
        
        # Stress testing
        analysis['stress_testing'] = self.risk_metrics.stress_testing(returns)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            analysis['beta_analysis'] = self.risk_metrics.calculate_beta(returns, benchmark_returns)
            analysis['tracking_error'] = self.risk_metrics.calculate_tracking_error(returns, benchmark_returns)
            
        return analysis
        
    def generate_risk_report(self, returns: pd.Series, 
                           benchmark_returns: pd.Series = None) -> Dict:
        """Generate comprehensive risk report"""
        
        analysis = self.analyze_portfolio_risk(returns, benchmark_returns)
        
        # Create visualizations
        charts = {
            'var_chart': self.risk_viz.create_var_chart(returns),
            'drawdown_chart': self.risk_viz.create_drawdown_chart(returns),
            'monte_carlo_chart': self.risk_viz.create_monte_carlo_chart(returns),
            'risk_heatmap': self.risk_viz.create_risk_heatmap(returns, benchmark_returns)
        }
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(analysis)
        
        return {
            'analysis': analysis,
            'charts': charts,
            'recommendations': recommendations,
            'summary': self._create_risk_summary(analysis)
        }
        
    def _generate_risk_recommendations(self, analysis: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # VaR recommendations
        var_95 = analysis['var_analysis']['var_95']['var']
        if abs(var_95) > 0.05:  # 5% daily VaR
            recommendations.append("High VaR detected - consider reducing position sizes or increasing diversification")
            
        # Drawdown recommendations
        max_dd = analysis['drawdown_analysis']['maximum_drawdown']
        if abs(max_dd) > 0.20:  # 20% max drawdown
            recommendations.append("High maximum drawdown - implement stop-loss mechanisms")
            
        # Sharpe ratio recommendations
        sharpe = analysis['sharpe_analysis']['sharpe_ratio']
        if sharpe < 1.0:
            recommendations.append("Low Sharpe ratio - optimize risk-return profile")
            
        # Monte Carlo recommendations
        prob_loss = analysis['monte_carlo']['probability_of_loss']
        if prob_loss > 0.4:  # 40% probability of loss
            recommendations.append("High probability of loss in simulations - review strategy")
            
        return recommendations
        
    def _create_risk_summary(self, analysis: Dict) -> Dict:
        """Create risk summary"""
        
        return {
            'overall_risk_level': 'High' if abs(analysis['var_analysis']['var_95']['var']) > 0.03 else 'Medium' if abs(analysis['var_analysis']['var_95']['var']) > 0.02 else 'Low',
            'key_risks': [
                f"VaR (95%): {abs(analysis['var_analysis']['var_95']['var'])*100:.2f}%",
                f"Max Drawdown: {abs(analysis['drawdown_analysis']['maximum_drawdown'])*100:.2f}%",
                f"Sharpe Ratio: {analysis['sharpe_analysis']['sharpe_ratio']:.2f}"
            ],
            'risk_score': self._calculate_risk_score(analysis)
        }
        
    def _calculate_risk_score(self, analysis: Dict) -> float:
        """Calculate overall risk score (0-100)"""
        
        var_score = min(abs(analysis['var_analysis']['var_95']['var']) * 1000, 50)
        dd_score = min(abs(analysis['drawdown_analysis']['maximum_drawdown']) * 200, 30)
        sharpe_score = max(0, 20 - analysis['sharpe_analysis']['sharpe_ratio'] * 10)
        
        return min(var_score + dd_score + sharpe_score, 100)

# Global instances
risk_metrics = RiskMetrics()
risk_visualization = RiskVisualization()
portfolio_risk_analyzer = PortfolioRiskAnalyzer()
