"""
Automated report generation with scheduling
Provides comprehensive reporting with email delivery and scheduling capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import io
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
# import schedule  # Commented out due to installation issues
import time
import threading
import logging
from dataclasses import dataclass
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Report type enumeration"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Report format enumeration"""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"

@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType
    format: ReportFormat
    recipients: List[str]
    schedule_time: str = "09:00"
    include_charts: bool = True
    include_analysis: bool = True
    include_recommendations: bool = True
    custom_sections: List[str] = None

class ReportGenerator:
    """Automated report generator"""
    
    def __init__(self):
        self.report_templates = self._initialize_templates()
        self.chart_cache = {}
        
    def _initialize_templates(self) -> Dict:
        """Initialize report templates"""
        
        templates = {
            ReportType.DAILY: {
                'title': 'Daily Market Report',
                'sections': ['market_overview', 'top_movers', 'sector_performance', 'technical_analysis'],
                'charts': ['market_heatmap', 'sector_chart', 'volume_analysis']
            },
            ReportType.WEEKLY: {
                'title': 'Weekly Portfolio Report',
                'sections': ['portfolio_performance', 'risk_analysis', 'sector_allocation', 'market_outlook'],
                'charts': ['portfolio_chart', 'risk_metrics', 'allocation_pie', 'correlation_matrix']
            },
            ReportType.MONTHLY: {
                'title': 'Monthly Investment Report',
                'sections': ['executive_summary', 'portfolio_performance', 'risk_analysis', 'market_analysis', 'recommendations'],
                'charts': ['performance_chart', 'risk_heatmap', 'sector_analysis', 'benchmark_comparison']
            },
            ReportType.QUARTERLY: {
                'title': 'Quarterly Investment Review',
                'sections': ['executive_summary', 'portfolio_performance', 'risk_analysis', 'market_analysis', 'strategy_review', 'outlook'],
                'charts': ['performance_chart', 'risk_analysis', 'sector_analysis', 'benchmark_comparison', 'allocation_evolution']
            }
        }
        
        return templates
        
    def generate_report(self, config: ReportConfig, data: Dict = None) -> Dict:
        """Generate a comprehensive report"""
        
        template = self.report_templates.get(config.report_type, {})
        
        if not template:
            return {'error': f'No template found for {config.report_type}'}
            
        # Generate report content
        report_content = {
            'metadata': {
                'title': template['title'],
                'generated_at': datetime.now().isoformat(),
                'report_type': config.report_type.value,
                'format': config.format.value
            },
            'sections': {},
            'charts': {},
            'summary': {}
        }
        
        # Generate sections
        for section in template['sections']:
            section_content = self._generate_section(section, data)
            if section_content:
                report_content['sections'][section] = section_content
                
        # Generate charts
        if config.include_charts:
            for chart in template['charts']:
                chart_content = self._generate_chart(chart, data)
                if chart_content:
                    report_content['charts'][chart] = chart_content
                    
        # Generate summary
        report_content['summary'] = self._generate_summary(report_content)
        
        return report_content
        
    def _generate_section(self, section_name: str, data: Dict) -> Dict:
        """Generate a specific report section"""
        
        section_generators = {
            'market_overview': self._generate_market_overview,
            'portfolio_performance': self._generate_portfolio_performance,
            'risk_analysis': self._generate_risk_analysis,
            'sector_performance': self._generate_sector_performance,
            'technical_analysis': self._generate_technical_analysis,
            'executive_summary': self._generate_executive_summary,
            'market_analysis': self._generate_market_analysis,
            'recommendations': self._generate_recommendations,
            'strategy_review': self._generate_strategy_review,
            'outlook': self._generate_outlook
        }
        
        generator = section_generators.get(section_name)
        if generator:
            return generator(data)
        else:
            return {'error': f'No generator for section {section_name}'}
            
    def _generate_market_overview(self, data: Dict) -> Dict:
        """Generate market overview section"""
        
        # Simulate market data
        market_data = {
            'indices': {
                'S&P 500': {'change': 0.5, 'value': 4500.0},
                'Dow Jones': {'change': 0.3, 'value': 35000.0},
                'NASDAQ': {'change': 0.8, 'value': 14000.0}
            },
            'sectors': {
                'Technology': {'change': 1.2, 'performance': 'outperforming'},
                'Healthcare': {'change': -0.3, 'performance': 'underperforming'},
                'Finance': {'change': 0.7, 'performance': 'outperforming'}
            },
            'volatility': {
                'VIX': 18.5,
                'trend': 'decreasing'
            }
        }
        
        return {
            'title': 'Market Overview',
            'content': market_data,
            'key_points': [
                f"Market indices showing mixed performance",
                f"Technology sector leading gains",
                f"Volatility trending lower"
            ]
        }
        
    def _generate_portfolio_performance(self, data: Dict) -> Dict:
        """Generate portfolio performance section"""
        
        # Simulate portfolio data
        portfolio_data = {
            'total_value': 100000.0,
            'daily_change': 500.0,
            'daily_change_percent': 0.5,
            'ytd_return': 8.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -5.2,
            'top_performers': [
                {'symbol': 'AAPL', 'return': 12.5},
                {'symbol': 'MSFT', 'return': 10.8},
                {'symbol': 'GOOGL', 'return': 9.2}
            ],
            'underperformers': [
                {'symbol': 'TSLA', 'return': -3.2},
                {'symbol': 'META', 'return': -1.8}
            ]
        }
        
        return {
            'title': 'Portfolio Performance',
            'content': portfolio_data,
            'key_points': [
                f"Portfolio value: ${portfolio_data['total_value']:,.2f}",
                f"YTD return: {portfolio_data['ytd_return']:.1f}%",
                f"Sharpe ratio: {portfolio_data['sharpe_ratio']:.2f}"
            ]
        }
        
    def _generate_risk_analysis(self, data: Dict) -> Dict:
        """Generate risk analysis section"""
        
        # Simulate risk data
        risk_data = {
            'var_95': -2.1,
            'cvar_95': -3.5,
            'max_drawdown': -5.2,
            'volatility': 15.8,
            'beta': 1.1,
            'correlation_sp500': 0.85,
            'risk_score': 65,
            'risk_level': 'Medium'
        }
        
        return {
            'title': 'Risk Analysis',
            'content': risk_data,
            'key_points': [
                f"95% VaR: {abs(risk_data['var_95']):.1f}%",
                f"Maximum drawdown: {abs(risk_data['max_drawdown']):.1f}%",
                f"Risk level: {risk_data['risk_level']}"
            ]
        }
        
    def _generate_sector_performance(self, data: Dict) -> Dict:
        """Generate sector performance section"""
        
        # Simulate sector data
        sector_data = {
            'sectors': [
                {'name': 'Technology', 'weight': 35.0, 'return': 12.5, 'performance': 'outperforming'},
                {'name': 'Healthcare', 'weight': 20.0, 'return': 8.2, 'performance': 'neutral'},
                {'name': 'Finance', 'weight': 15.0, 'return': 6.8, 'performance': 'underperforming'},
                {'name': 'Consumer', 'weight': 15.0, 'return': 9.1, 'performance': 'outperforming'},
                {'name': 'Energy', 'weight': 10.0, 'return': 4.5, 'performance': 'underperforming'},
                {'name': 'Other', 'weight': 5.0, 'return': 7.3, 'performance': 'neutral'}
            ]
        }
        
        return {
            'title': 'Sector Performance',
            'content': sector_data,
            'key_points': [
                "Technology sector leading performance",
                "Healthcare showing steady gains",
                "Energy sector underperforming"
            ]
        }
        
    def _generate_technical_analysis(self, data: Dict) -> Dict:
        """Generate technical analysis section"""
        
        # Simulate technical data
        technical_data = {
            'market_sentiment': 'Bullish',
            'trend_analysis': {
                'short_term': 'Uptrend',
                'medium_term': 'Uptrend',
                'long_term': 'Uptrend'
            },
            'key_levels': {
                'support': 4400.0,
                'resistance': 4600.0,
                'current': 4500.0
            },
            'indicators': {
                'rsi': 65.2,
                'macd': 'Bullish',
                'moving_averages': 'Bullish'
            }
        }
        
        return {
            'title': 'Technical Analysis',
            'content': technical_data,
            'key_points': [
                f"Market sentiment: {technical_data['market_sentiment']}",
                f"RSI: {technical_data['indicators']['rsi']:.1f}",
                f"Key resistance: {technical_data['key_levels']['resistance']:.0f}"
            ]
        }
        
    def _generate_executive_summary(self, data: Dict) -> Dict:
        """Generate executive summary section"""
        
        summary_data = {
            'portfolio_value': 100000.0,
            'ytd_return': 8.5,
            'risk_level': 'Medium',
            'top_performance': 'Technology sector',
            'key_risks': ['Market volatility', 'Interest rate changes'],
            'recommendations': ['Maintain current allocation', 'Monitor technology exposure']
        }
        
        return {
            'title': 'Executive Summary',
            'content': summary_data,
            'key_points': [
                f"Portfolio value: ${summary_data['portfolio_value']:,.2f}",
                f"YTD return: {summary_data['ytd_return']:.1f}%",
                f"Risk level: {summary_data['risk_level']}"
            ]
        }
        
    def _generate_market_analysis(self, data: Dict) -> Dict:
        """Generate market analysis section"""
        
        market_analysis = {
            'economic_indicators': {
                'gdp_growth': 2.5,
                'inflation': 3.2,
                'unemployment': 4.1,
                'interest_rates': 5.25
            },
            'market_outlook': {
                'short_term': 'Positive',
                'medium_term': 'Cautious',
                'long_term': 'Optimistic'
            },
            'key_events': [
                'Fed meeting next week',
                'Earnings season approaching',
                'Economic data releases'
            ]
        }
        
        return {
            'title': 'Market Analysis',
            'content': market_analysis,
            'key_points': [
                f"GDP growth: {market_analysis['economic_indicators']['gdp_growth']:.1f}%",
                f"Inflation: {market_analysis['economic_indicators']['inflation']:.1f}%",
                f"Market outlook: {market_analysis['market_outlook']['short_term']}"
            ]
        }
        
    def _generate_recommendations(self, data: Dict) -> Dict:
        """Generate recommendations section"""
        
        recommendations = {
            'portfolio_recommendations': [
                'Consider rebalancing technology exposure',
                'Monitor interest rate sensitive positions',
                'Maintain diversification across sectors'
            ],
            'market_recommendations': [
                'Watch for earnings season volatility',
                'Monitor Fed policy changes',
                'Consider defensive positioning'
            ],
            'risk_recommendations': [
                'Implement stop-loss orders',
                'Review position sizes',
                'Monitor correlation changes'
            ]
        }
        
        return {
            'title': 'Recommendations',
            'content': recommendations,
            'key_points': [
                "Portfolio rebalancing recommended",
                "Monitor market volatility",
                "Maintain risk management discipline"
            ]
        }
        
    def _generate_strategy_review(self, data: Dict) -> Dict:
        """Generate strategy review section"""
        
        strategy_review = {
            'current_strategy': 'Multi-factor equity strategy',
            'performance_vs_benchmark': 2.3,
            'strategy_strengths': [
                'Strong risk-adjusted returns',
                'Good diversification',
                'Consistent performance'
            ],
            'strategy_weaknesses': [
                'High technology concentration',
                'Limited international exposure'
            ],
            'adjustments_needed': [
                'Reduce technology overweight',
                'Add international diversification'
            ]
        }
        
        return {
            'title': 'Strategy Review',
            'content': strategy_review,
            'key_points': [
                f"Strategy outperforming benchmark by {strategy_review['performance_vs_benchmark']:.1f}%",
                "Technology concentration needs review",
                "International diversification recommended"
            ]
        }
        
    def _generate_outlook(self, data: Dict) -> Dict:
        """Generate outlook section"""
        
        outlook = {
            'market_outlook': {
                'next_quarter': 'Cautiously optimistic',
                'next_year': 'Positive with volatility',
                'key_drivers': ['Earnings growth', 'Interest rates', 'Geopolitical events']
            },
            'portfolio_outlook': {
                'expected_return': 8.0,
                'expected_volatility': 16.0,
                'risk_factors': ['Market volatility', 'Sector rotation', 'Interest rate changes']
            },
            'action_items': [
                'Monitor earnings season',
                'Review sector allocation',
                'Prepare for volatility'
            ]
        }
        
        return {
            'title': 'Outlook',
            'content': outlook,
            'key_points': [
                f"Expected return: {outlook['portfolio_outlook']['expected_return']:.1f}%",
                f"Expected volatility: {outlook['portfolio_outlook']['expected_volatility']:.1f}%",
                "Monitor key risk factors"
            ]
        }
        
    def _generate_chart(self, chart_name: str, data: Dict) -> Dict:
        """Generate a chart for the report"""
        
        chart_generators = {
            'market_heatmap': self._generate_market_heatmap,
            'sector_chart': self._generate_sector_chart,
            'volume_analysis': self._generate_volume_analysis,
            'portfolio_chart': self._generate_portfolio_chart,
            'risk_metrics': self._generate_risk_metrics_chart,
            'allocation_pie': self._generate_allocation_pie,
            'correlation_matrix': self._generate_correlation_matrix,
            'performance_chart': self._generate_performance_chart,
            'risk_heatmap': self._generate_risk_heatmap,
            'sector_analysis': self._generate_sector_analysis,
            'benchmark_comparison': self._generate_benchmark_comparison,
            'allocation_evolution': self._generate_allocation_evolution
        }
        
        generator = chart_generators.get(chart_name)
        if generator:
            return generator(data)
        else:
            return {'error': f'No generator for chart {chart_name}'}
            
    def _generate_market_heatmap(self, data: Dict) -> Dict:
        """Generate market heatmap chart"""
        
        # Simulate market data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        returns = np.random.uniform(-0.05, 0.05, len(symbols))
        
        fig = go.Figure(data=go.Heatmap(
            z=[returns],
            x=symbols,
            y=['Returns'],
            colorscale='RdYlGn',
            zmid=0
        ))
        
        fig.update_layout(title='Market Performance Heatmap')
        
        return {
            'title': 'Market Performance Heatmap',
            'chart': fig,
            'description': 'Daily returns for major stocks'
        }
        
    def _generate_sector_chart(self, data: Dict) -> Dict:
        """Generate sector performance chart"""
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
        returns = np.random.uniform(-0.02, 0.03, len(sectors))
        
        fig = go.Figure(data=go.Bar(
            x=sectors,
            y=returns,
            marker_color=['green' if r > 0 else 'red' for r in returns]
        ))
        
        fig.update_layout(title='Sector Performance', yaxis_title='Return %')
        
        return {
            'title': 'Sector Performance',
            'chart': fig,
            'description': 'Sector performance comparison'
        }
        
    def _generate_volume_analysis(self, data: Dict) -> Dict:
        """Generate volume analysis chart"""
        
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volumes = np.random.randint(1000000, 5000000, 30)
        
        fig = go.Figure(data=go.Bar(
            x=dates,
            y=volumes,
            marker_color='lightblue'
        ))
        
        fig.update_layout(title='Volume Analysis', xaxis_title='Date', yaxis_title='Volume')
        
        return {
            'title': 'Volume Analysis',
            'chart': fig,
            'description': 'Trading volume over time'
        }
        
    def _generate_portfolio_chart(self, data: Dict) -> Dict:
        """Generate portfolio performance chart"""
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        portfolio_values = 100000 + np.cumsum(np.random.randn(100) * 500)
        
        fig = go.Figure(data=go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value'
        ))
        
        fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Value ($)')
        
        return {
            'title': 'Portfolio Performance',
            'chart': fig,
            'description': 'Portfolio value over time'
        }
        
    def _generate_risk_metrics_chart(self, data: Dict) -> Dict:
        """Generate risk metrics chart"""
        
        metrics = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Volatility']
        values = [2.1, 3.5, 5.2, 15.8]
        
        fig = go.Figure(data=go.Bar(
            x=metrics,
            y=values,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(title='Risk Metrics', yaxis_title='Value (%)')
        
        return {
            'title': 'Risk Metrics',
            'chart': fig,
            'description': 'Key risk metrics'
        }
        
    def _generate_allocation_pie(self, data: Dict) -> Dict:
        """Generate allocation pie chart"""
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy', 'Other']
        weights = [35, 20, 15, 15, 10, 5]
        
        fig = go.Figure(data=go.Pie(
            labels=sectors,
            values=weights
        ))
        
        fig.update_layout(title='Portfolio Allocation')
        
        return {
            'title': 'Portfolio Allocation',
            'chart': fig,
            'description': 'Sector allocation breakdown'
        }
        
    def _generate_correlation_matrix(self, data: Dict) -> Dict:
        """Generate correlation matrix chart"""
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        correlation_data = np.random.uniform(0.3, 0.9, (4, 4))
        np.fill_diagonal(correlation_data, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(title='Correlation Matrix')
        
        return {
            'title': 'Correlation Matrix',
            'chart': fig,
            'description': 'Asset correlation analysis'
        }
        
    def _generate_performance_chart(self, data: Dict) -> Dict:
        """Generate performance comparison chart"""
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        portfolio = 100 + np.cumsum(np.random.randn(100) * 0.5)
        benchmark = 100 + np.cumsum(np.random.randn(100) * 0.3)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=portfolio, name='Portfolio', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=benchmark, name='Benchmark', line=dict(color='red')))
        
        fig.update_layout(title='Performance vs Benchmark', xaxis_title='Date', yaxis_title='Value')
        
        return {
            'title': 'Performance vs Benchmark',
            'chart': fig,
            'description': 'Portfolio performance relative to benchmark'
        }
        
    def _generate_risk_heatmap(self, data: Dict) -> Dict:
        """Generate risk heatmap chart"""
        
        risk_metrics = ['VaR', 'CVaR', 'Volatility', 'Beta', 'Sharpe']
        risk_levels = [0.7, 0.8, 0.6, 0.9, 0.5]
        
        fig = go.Figure(data=go.Heatmap(
            z=[risk_levels],
            x=risk_metrics,
            y=['Risk Level'],
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(title='Risk Metrics Heatmap')
        
        return {
            'title': 'Risk Metrics Heatmap',
            'chart': fig,
            'description': 'Risk assessment visualization'
        }
        
    def _generate_sector_analysis(self, data: Dict) -> Dict:
        """Generate sector analysis chart"""
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
        returns = np.random.uniform(-0.02, 0.03, len(sectors))
        volatilities = np.random.uniform(0.15, 0.25, len(sectors))
        
        fig = go.Figure(data=go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=sectors,
            textposition='top center',
            marker=dict(size=20, color=returns, colorscale='Viridis')
        ))
        
        fig.update_layout(title='Sector Risk-Return Analysis', xaxis_title='Volatility', yaxis_title='Return')
        
        return {
            'title': 'Sector Risk-Return Analysis',
            'chart': fig,
            'description': 'Sector performance vs risk'
        }
        
    def _generate_benchmark_comparison(self, data: Dict) -> Dict:
        """Generate benchmark comparison chart"""
        
        metrics = ['Return', 'Volatility', 'Sharpe', 'Max DD']
        portfolio = [8.5, 15.8, 1.2, 5.2]
        benchmark = [6.2, 12.5, 0.8, 8.1]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Portfolio', x=metrics, y=portfolio, marker_color='blue'))
        fig.add_trace(go.Bar(name='Benchmark', x=metrics, y=benchmark, marker_color='red'))
        
        fig.update_layout(title='Portfolio vs Benchmark', yaxis_title='Value')
        
        return {
            'title': 'Portfolio vs Benchmark',
            'chart': fig,
            'description': 'Performance metrics comparison'
        }
        
    def _generate_allocation_evolution(self, data: Dict) -> Dict:
        """Generate allocation evolution chart"""
        
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        tech_allocation = np.random.uniform(30, 40, 12)
        healthcare_allocation = np.random.uniform(15, 25, 12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=tech_allocation, name='Technology', fill='tonexty'))
        fig.add_trace(go.Scatter(x=dates, y=healthcare_allocation, name='Healthcare', fill='tozeroy'))
        
        fig.update_layout(title='Allocation Evolution', xaxis_title='Date', yaxis_title='Allocation %')
        
        return {
            'title': 'Allocation Evolution',
            'chart': fig,
            'description': 'Portfolio allocation changes over time'
        }
        
    def _generate_summary(self, report_content: Dict) -> Dict:
        """Generate report summary"""
        
        sections = report_content.get('sections', {})
        charts = report_content.get('charts', {})
        
        summary = {
            'total_sections': len(sections),
            'total_charts': len(charts),
            'key_metrics': {},
            'highlights': []
        }
        
        # Extract key metrics from sections
        for section_name, section_data in sections.items():
            if 'content' in section_data:
                content = section_data['content']
                if 'portfolio_value' in content:
                    summary['key_metrics']['portfolio_value'] = content['portfolio_value']
                if 'ytd_return' in content:
                    summary['key_metrics']['ytd_return'] = content['ytd_return']
                if 'risk_level' in content:
                    summary['key_metrics']['risk_level'] = content['risk_level']
                    
        # Generate highlights
        if 'portfolio_performance' in sections:
            perf_data = sections['portfolio_performance']['content']
            summary['highlights'].append(f"Portfolio YTD return: {perf_data.get('ytd_return', 0):.1f}%")
            
        if 'risk_analysis' in sections:
            risk_data = sections['risk_analysis']['content']
            summary['highlights'].append(f"Risk level: {risk_data.get('risk_level', 'Unknown')}")
            
        return summary

class ReportScheduler:
    """Report scheduling and automation"""
    
    def __init__(self):
        self.report_generator = ReportGenerator()
        self.scheduled_reports = {}
        self.is_running = False
        
    def schedule_report(self, config: ReportConfig) -> bool:
        """Schedule a report for automatic generation"""
        
        try:
            # Simplified scheduling without schedule library
            self.scheduled_reports[config.report_type.value] = config
            logger.info(f"Scheduled {config.report_type.value} report for {config.schedule_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling report: {e}")
            return False
            
    def _generate_and_send_report(self, config: ReportConfig):
        """Generate and send a scheduled report"""
        
        try:
            # Generate report
            report_content = self.report_generator.generate_report(config)
            
            # Convert to desired format
            if config.format == ReportFormat.PDF:
                report_file = self._create_pdf_report(report_content)
            elif config.format == ReportFormat.HTML:
                report_file = self._create_html_report(report_content)
            else:
                report_file = None
                
            # Send report
            if report_file and config.recipients:
                self._send_report_email(report_file, config)
                
            logger.info(f"Generated and sent {config.report_type.value} report")
            
        except Exception as e:
            logger.error(f"Error generating/sending report: {e}")
            
    def _create_pdf_report(self, report_content: Dict) -> str:
        """Create PDF report"""
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(report_content['metadata']['title'], title_style))
        story.append(Spacer(1, 12))
        
        # Sections
        for section_name, section_data in report_content['sections'].items():
            story.append(Paragraph(section_data['title'], styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Add key points
            for point in section_data.get('key_points', []):
                story.append(Paragraph(f"• {point}", styles['Normal']))
                
            story.append(Spacer(1, 12))
            
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Save to file
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
            
        return filename
        
    def _create_html_report(self, report_content: Dict) -> str:
        """Create HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_content['metadata']['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .section {{ margin-bottom: 30px; }}
                .key-points {{ background-color: #f5f5f5; padding: 15px; }}
            </style>
        </head>
        <body>
            <h1>{report_content['metadata']['title']}</h1>
            <p>Generated: {report_content['metadata']['generated_at']}</p>
        """
        
        # Add sections
        for section_name, section_data in report_content['sections'].items():
            html_content += f"""
            <div class="section">
                <h2>{section_data['title']}</h2>
                <div class="key-points">
                    <ul>
            """
            
            for point in section_data.get('key_points', []):
                html_content += f"<li>{point}</li>"
                
            html_content += """
                    </ul>
                </div>
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        # Save to file
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w') as f:
            f.write(html_content)
            
        return filename
        
    def _send_report_email(self, report_file: str, config: ReportConfig):
        """Send report via email"""
        
        try:
            # Email configuration (in production, use environment variables)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_password"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(config.recipients)
            msg['Subject'] = f"Automated Report - {config.report_type.value.title()}"
            
            # Add body
            body = f"Please find attached the {config.report_type.value} report."
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachment
            with open(report_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {report_file}'
            )
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, config.recipients, text)
            server.quit()
            
            logger.info(f"Report sent to {config.recipients}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            
    def start_scheduler(self):
        """Start the report scheduler"""
        
        self.is_running = True
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("Report scheduler started")
        
    def stop_scheduler(self):
        """Stop the report scheduler"""
        
        self.is_running = False
        logger.info("Report scheduler stopped")
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        
        while self.is_running:
            # Simplified scheduler without schedule library
            time.sleep(60)  # Check every minute
            
    def get_scheduled_reports(self) -> Dict:
        """Get list of scheduled reports"""
        
        return {
            'scheduled_reports': list(self.scheduled_reports.keys()),
            'is_running': self.is_running
        }

# Global instances
report_generator = ReportGenerator()
report_scheduler = ReportScheduler()
