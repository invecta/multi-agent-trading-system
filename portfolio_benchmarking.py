"""
Portfolio benchmarking and comparison module
Provides portfolio performance analysis, benchmarking against indices, and comparison tools
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class PortfolioBenchmark:
    """Portfolio benchmarking and comparison engine"""
    
    def __init__(self):
        self.benchmark_indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication': 'XLC'
        }
        
    def get_benchmark_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get benchmark data for comparison"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching benchmark data for {symbol}: {e}")
            return pd.DataFrame()
            
    def calculate_portfolio_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Risk metrics
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
            metrics['alpha'] = metrics['annualized_return'] - (metrics['beta'] * benchmark_returns.mean() * 252)
            metrics['information_ratio'] = (returns.mean() - benchmark_returns.mean()) / (returns - benchmark_returns).std()
            metrics['tracking_error'] = (returns - benchmark_returns).std() * np.sqrt(252)
            
        # Additional metrics
        metrics['win_rate'] = (returns > 0).mean()
        metrics['avg_win'] = returns[returns > 0].mean() if (returns > 0).any() else 0
        metrics['avg_loss'] = returns[returns < 0].mean() if (returns < 0).any() else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
        
        return metrics
        
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    def benchmark_portfolio(self, portfolio_returns: pd.Series, 
                          benchmark_symbol: str = '^GSPC',
                          start_date: str = None, end_date: str = None) -> Dict:
        """Benchmark portfolio against a market index"""
        
        if start_date is None:
            start_date = portfolio_returns.index[0].strftime('%Y-%m-%d')
        if end_date is None:
            end_date = portfolio_returns.index[-1].strftime('%Y-%m-%d')
            
        # Get benchmark data
        benchmark_data = self.get_benchmark_data(benchmark_symbol, start_date, end_date)
        
        if benchmark_data.empty:
            return {'error': f'Could not fetch benchmark data for {benchmark_symbol}'}
            
        # Calculate benchmark returns
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_aligned, benchmark_aligned)
        benchmark_metrics = self.calculate_portfolio_metrics(benchmark_aligned)
        
        return {
            'portfolio': portfolio_metrics,
            'benchmark': benchmark_metrics,
            'comparison': {
                'excess_return': portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return'],
                'relative_volatility': portfolio_metrics['volatility'] / benchmark_metrics['volatility'],
                'information_ratio': portfolio_metrics.get('information_ratio', 0),
                'tracking_error': portfolio_metrics.get('tracking_error', 0)
            }
        }
        
    def create_benchmark_chart(self, portfolio_returns: pd.Series, 
                             benchmark_symbol: str = '^GSPC',
                             start_date: str = None, end_date: str = None) -> go.Figure:
        """Create benchmark comparison chart"""
        
        if start_date is None:
            start_date = portfolio_returns.index[0].strftime('%Y-%m-%d')
        if end_date is None:
            end_date = portfolio_returns.index[-1].strftime('%Y-%m-%d')
            
        # Get benchmark data
        benchmark_data = self.get_benchmark_data(benchmark_symbol, start_date, end_date)
        
        if benchmark_data.empty:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(text="No benchmark data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
            
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_cumulative.loc[common_dates]
        benchmark_aligned = benchmark_cumulative.loc[common_dates]
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_aligned.index,
            y=portfolio_aligned.values,
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_aligned.index,
            y=benchmark_aligned.values,
            name=f'Benchmark ({benchmark_symbol})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio vs Benchmark Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_risk_return_scatter(self, portfolios: Dict[str, pd.Series]) -> go.Figure:
        """Create risk-return scatter plot for multiple portfolios"""
        
        data = []
        
        for name, returns in portfolios.items():
            metrics = self.calculate_portfolio_metrics(returns)
            data.append({
                'Portfolio': name,
                'Return': metrics['annualized_return'],
                'Risk': metrics['volatility'],
                'Sharpe': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown']
            })
            
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Risk'],
            y=df['Return'],
            mode='markers+text',
            text=df['Portfolio'],
            textposition='top center',
            marker=dict(
                size=df['Sharpe'] * 20,
                color=df['Sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Return: %{y:.2%}<br>' +
                         'Risk: %{x:.2%}<br>' +
                         'Sharpe: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Annualized Return',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_sector_allocation_chart(self, portfolio_weights: Dict[str, float]) -> go.Figure:
        """Create sector allocation pie chart"""
        
        # Map stocks to sectors (simplified mapping)
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
            'META': 'Communication', 'NVDA': 'Technology', 'NFLX': 'Communication',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples'
        }
        
        # Aggregate by sector
        sector_weights = {}
        for stock, weight in portfolio_weights.items():
            sector = sector_mapping.get(stock, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(sector_weights.keys()),
            values=list(sector_weights.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title='Portfolio Sector Allocation',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_performance_attribution(self, portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series) -> go.Figure:
        """Create performance attribution analysis"""
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Cumulative Excess Returns', 'Rolling 30-Day Excess Returns')
        )
        
        # Cumulative excess returns
        cumulative_excess = excess_returns.cumsum()
        fig.add_trace(
            go.Scatter(
                x=cumulative_excess.index,
                y=cumulative_excess.values,
                name='Cumulative Excess',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Rolling excess returns
        rolling_excess = excess_returns.rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(
                x=rolling_excess.index,
                y=rolling_excess.values,
                name='30-Day Rolling Excess',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Performance Attribution Analysis',
            height=600,
            template='plotly_dark'
        )
        
        return fig
        
    def generate_benchmark_report(self, portfolio_returns: pd.Series, 
                                benchmark_symbol: str = '^GSPC') -> Dict:
        """Generate comprehensive benchmark report"""
        
        benchmark_results = self.benchmark_portfolio(portfolio_returns, benchmark_symbol)
        
        if 'error' in benchmark_results:
            return benchmark_results
            
        portfolio_metrics = benchmark_results['portfolio']
        benchmark_metrics = benchmark_results['benchmark']
        comparison = benchmark_results['comparison']
        
        report = {
            'summary': {
                'portfolio_return': portfolio_metrics['annualized_return'],
                'benchmark_return': benchmark_metrics['annualized_return'],
                'excess_return': comparison['excess_return'],
                'outperformance': comparison['excess_return'] > 0
            },
            'risk_metrics': {
                'portfolio_volatility': portfolio_metrics['volatility'],
                'benchmark_volatility': benchmark_metrics['volatility'],
                'portfolio_sharpe': portfolio_metrics['sharpe_ratio'],
                'benchmark_sharpe': benchmark_metrics['sharpe_ratio'],
                'portfolio_max_drawdown': portfolio_metrics['max_drawdown'],
                'benchmark_max_drawdown': benchmark_metrics['max_drawdown']
            },
            'relative_metrics': {
                'beta': portfolio_metrics.get('beta', 0),
                'alpha': portfolio_metrics.get('alpha', 0),
                'information_ratio': comparison['information_ratio'],
                'tracking_error': comparison['tracking_error']
            },
            'recommendations': self._generate_recommendations(portfolio_metrics, benchmark_metrics, comparison)
        }
        
        return report
        
    def _generate_recommendations(self, portfolio_metrics: Dict, 
                                benchmark_metrics: Dict, comparison: Dict) -> List[str]:
        """Generate investment recommendations based on analysis"""
        
        recommendations = []
        
        # Return analysis
        if comparison['excess_return'] > 0.02:
            recommendations.append("Portfolio significantly outperforms benchmark - consider maintaining strategy")
        elif comparison['excess_return'] < -0.02:
            recommendations.append("Portfolio underperforms benchmark - consider strategy review")
            
        # Risk analysis
        if portfolio_metrics['volatility'] > benchmark_metrics['volatility'] * 1.2:
            recommendations.append("Portfolio risk is high relative to benchmark - consider diversification")
            
        # Sharpe ratio analysis
        if portfolio_metrics['sharpe_ratio'] < benchmark_metrics['sharpe_ratio']:
            recommendations.append("Risk-adjusted returns below benchmark - optimize risk-return profile")
            
        # Drawdown analysis
        if portfolio_metrics['max_drawdown'] < -0.2:
            recommendations.append("High maximum drawdown - implement risk management measures")
            
        return recommendations

# Global instance
benchmark_engine = PortfolioBenchmark()
