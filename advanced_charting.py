"""
Advanced charting module for enhanced dashboard
Provides candlestick patterns, volume profile, technical analysis overlays
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib
from datetime import datetime, timedelta

class AdvancedCharting:
    """Advanced charting capabilities with technical analysis"""
    
    def __init__(self):
        self.chart_types = {
            'candlestick': self.create_candlestick_chart,
            'line': self.create_line_chart,
            'volume': self.create_volume_chart,
            'heatmap': self.create_heatmap_chart,
            'scatter': self.create_scatter_chart
        }
        
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str, 
                                indicators: List[str] = None) -> go.Figure:
        """Create advanced candlestick chart with technical indicators"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI'),
            row_width=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol,
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
        # Add Bollinger Bands
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )
            
        # Volume chart
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI chart
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
        # Update layout
        fig.update_layout(
            title=f'{symbol} Advanced Chart Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
        
    def create_volume_profile_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volume profile chart"""
        
        # Calculate volume profile
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
        volume_profile = []
        
        for i in range(len(price_bins) - 1):
            mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i + 1])
            volume_sum = df[mask]['Volume'].sum()
            volume_profile.append(volume_sum)
            
        # Create figure
        fig = go.Figure()
        
        # Volume profile bars
        fig.add_trace(
            go.Bar(
                x=volume_profile,
                y=price_bins[:-1],
                orientation='h',
                name='Volume Profile',
                marker_color='lightblue',
                opacity=0.7
            )
        )
        
        # Current price line
        current_price = df['Close'].iloc[-1]
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{symbol} Volume Profile',
            xaxis_title='Volume',
            yaxis_title='Price',
            height=600,
            template='plotly_dark'
        )
        
        return fig
        
    def create_heatmap_chart(self, data: Dict, title: str) -> go.Figure:
        """Create correlation heatmap"""
        
        # Prepare data for heatmap
        symbols = list(data.keys())
        correlation_matrix = np.corrcoef([data[symbol]['returns'] for symbol in symbols])
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_scatter_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: str = None) -> go.Figure:
        """Create scatter plot with optional color coding"""
        
        fig = go.Figure()
        
        if color_col and color_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    marker=dict(
                        color=df[color_col],
                        colorscale='Viridis',
                        size=8,
                        opacity=0.7,
                        colorbar=dict(title=color_col)
                    ),
                    text=df.index,
                    hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Date: %{{text}}'
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    text=df.index,
                    hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Date: %{{text}}'
                )
            )
            
        fig.update_layout(
            title=f'{y_col} vs {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_line_chart(self, df: pd.DataFrame, columns: List[str], 
                         title: str) -> go.Figure:
        """Create multi-line chart"""
        
        fig = go.Figure()
        
        for col in columns:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col,
                        line=dict(width=2)
                    )
                )
                
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            height=500,
            template='plotly_dark'
        )
        
        return fig
        
    def create_volume_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volume analysis chart"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume Analysis')
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Volume moving average
        if 'Volume_SMA' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Volume_SMA'],
                    name='Volume SMA',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title=f'{symbol} Volume Analysis',
            height=600,
            template='plotly_dark'
        )
        
        return fig
        
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        
        patterns = {}
        
        # Convert to numpy arrays for talib
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        
        # Common patterns
        pattern_functions = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'ENGULFING': talib.CDLENGULFING,
            'MORNING_STAR': talib.CDLMORNINGSTAR,
            'EVENING_STAR': talib.CDLEVENINGSTAR,
            'HARAMI': talib.CDLHARAMI,
            'SPINNING_TOP': talib.CDLSPINNINGTOP
        }
        
        for pattern_name, pattern_func in pattern_functions.items():
            try:
                pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                pattern_dates = df.index[pattern_result != 0].tolist()
                patterns[pattern_name] = {
                    'count': len(pattern_dates),
                    'dates': pattern_dates,
                    'signals': pattern_result[pattern_result != 0].tolist()
                }
            except Exception as e:
                patterns[pattern_name] = {'error': str(e)}
                
        return patterns
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Stochastic
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
        
        return df
        
    def generate_chart_insights(self, df: pd.DataFrame, patterns: Dict) -> List[str]:
        """Generate insights from chart analysis"""
        
        insights = []
        
        # Price trend analysis
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                insights.append("Bullish trend: 20-day SMA above 50-day SMA")
            else:
                insights.append("Bearish trend: 20-day SMA below 50-day SMA")
                
        # RSI analysis
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi > 70:
                insights.append(f"Overbought condition: RSI at {rsi:.1f}")
            elif rsi < 30:
                insights.append(f"Oversold condition: RSI at {rsi:.1f}")
            else:
                insights.append(f"Neutral RSI: {rsi:.1f}")
                
        # Volume analysis
        if 'Volume_Ratio' in df.columns:
            vol_ratio = df['Volume_Ratio'].iloc[-1]
            if vol_ratio > 1.5:
                insights.append(f"High volume activity: {vol_ratio:.1f}x average")
            elif vol_ratio < 0.5:
                insights.append(f"Low volume activity: {vol_ratio:.1f}x average")
                
        # Pattern analysis
        for pattern, data in patterns.items():
            if isinstance(data, dict) and 'count' in data and data['count'] > 0:
                insights.append(f"{pattern} pattern detected: {data['count']} occurrences")
                
        return insights

# Global instance
charting_engine = AdvancedCharting()
