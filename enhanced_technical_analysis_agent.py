"""
Enhanced Technical Analysis Agent with Advanced Trading Strategies
Implements trend following, mean reversion, momentum, volatility, and statistical arbitrage strategies
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from multi_agent_framework import BaseAgent, AgentOutput, TradingSignal, SignalType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicator:
    """Technical indicator result"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    timestamp: datetime

@dataclass
class StrategyResult:
    """Strategy analysis result"""
    strategy_name: str
    signal: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    indicators: List[TechnicalIndicator]

class AdvancedTechnicalAnalysisAgent(BaseAgent):
    """Enhanced Technical Analysis Agent with multiple strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("advanced_technical_analysis_agent", config)
        self.add_dependency("market_data_agent")
        
        # Strategy configurations
        self.strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'momentum': MomentumStrategy(),
            'volatility': VolatilityStrategy(),
            'statistical_arbitrage': StatisticalArbitrageStrategy()
        }
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        
        # Performance tracking
        self.strategy_performance = {}
        self.signal_history = []
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process technical analysis with multiple strategies"""
        start_time = datetime.now()
        self.status = "running"
        
        try:
            market_data = input_data.get('market_data', {})
            all_signals = []
            strategy_results = {}
            
            for symbol, data in market_data.items():
                logger.info(f"Analyzing {symbol} with technical strategies")
                
                # Convert to DataFrame for analysis
                df = await self._prepare_dataframe(data)
                if df.empty:
                    continue
                
                # Run all strategies
                symbol_signals = []
                symbol_results = {}
                
                for strategy_name, strategy in self.strategies.items():
                    try:
                        result = await strategy.analyze(df, symbol)
                        if result:
                            symbol_results[strategy_name] = result
                            signal = self._convert_to_signal(result, symbol)
                            if signal:
                                symbol_signals.append(signal)
                                all_signals.append(signal)
                    except Exception as e:
                        logger.error(f"Error in {strategy_name} for {symbol}: {e}")
                
                strategy_results[symbol] = symbol_results
                
                # Update performance tracking
                self._update_strategy_performance(symbol, symbol_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = AgentOutput(
                agent_id=self.agent_id,
                status="completed",
                data={
                    'strategy_results': strategy_results,
                    'total_signals': len(all_signals),
                    'strategy_performance': self.strategy_performance
                },
                signals=all_signals,
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Technical Analysis Agent error: {e}")
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
    
    async def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare DataFrame from market data"""
        try:
            # Use daily data for analysis
            daily_data = data.get('1d', {})
            if not daily_data:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame([daily_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Add OHLCV columns
            df['open'] = daily_data.get('open', 0)
            df['high'] = daily_data.get('high', 0)
            df['low'] = daily_data.get('low', 0)
            df['close'] = daily_data.get('close', 0)
            df['volume'] = daily_data.get('volume', 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {e}")
            return pd.DataFrame()
    
    def _convert_to_signal(self, result: StrategyResult, symbol: str) -> Optional[TradingSignal]:
        """Convert strategy result to trading signal"""
        try:
            return TradingSignal(
                symbol=symbol,
                signal_type=result.signal,
                confidence=result.confidence,
                price=result.entry_price,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={
                    'strategy': result.strategy_name,
                    'stop_loss': result.stop_loss,
                    'take_profit': result.take_profit,
                    'position_size': result.position_size,
                    'reasoning': result.reasoning,
                    'indicators': [ind.name for ind in result.indicators]
                }
            )
        except Exception as e:
            logger.error(f"Error converting to signal: {e}")
            return None
    
    def _update_strategy_performance(self, symbol: str, results: Dict[str, StrategyResult]):
        """Update strategy performance tracking"""
        for strategy_name, result in results.items():
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'total_signals': 0,
                    'avg_confidence': 0,
                    'symbols_analyzed': set()
                }
            
            perf = self.strategy_performance[strategy_name]
            perf['total_signals'] += 1
            perf['avg_confidence'] = (
                (perf['avg_confidence'] * (perf['total_signals'] - 1) + result.confidence) 
                / perf['total_signals']
            )
            perf['symbols_analyzed'].add(symbol)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate technical analysis input"""
        return 'market_data' in input_data

class TechnicalIndicators:
    """Collection of technical indicators"""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        tr = TechnicalIndicators.calculate_atr(high, low, close, period)
        plus_dm = high.diff().where(high.diff() > low.diff().abs(), 0)
        minus_dm = low.diff().abs().where(low.diff().abs() > high.diff(), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx

class BaseStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze data and return strategy result"""
        raise NotImplementedError

class TrendFollowingStrategy(BaseStrategy):
    """Trend Following Strategy using EMAs, ADX, and Ichimoku"""
    
    def __init__(self):
        super().__init__("trend_following")
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze trend following signals"""
        try:
            if len(df) < 50:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Calculate indicators
            ema_12 = self.indicators.calculate_ema(close, 12)
            ema_26 = self.indicators.calculate_ema(close, 26)
            ema_50 = self.indicators.calculate_ema(close, 50)
            adx = self.indicators.calculate_adx(high, low, close)
            
            # Current values
            current_price = close.iloc[-1]
            current_ema_12 = ema_12.iloc[-1]
            current_ema_26 = ema_26.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]
            current_adx = adx.iloc[-1]
            
            # Trend analysis
            trend_up = (current_ema_12 > current_ema_26 > current_ema_50)
            trend_down = (current_ema_12 < current_ema_26 < current_ema_50)
            strong_trend = current_adx > 25
            
            # Generate signals
            if trend_up and strong_trend:
                signal = SignalType.BUY
                confidence = min(0.9, (current_adx / 50) * 0.8)
                reasoning = f"Uptrend confirmed: EMA12({current_ema_12:.2f}) > EMA26({current_ema_26:.2f}) > EMA50({current_ema_50:.2f}), ADX({current_adx:.2f})"
            elif trend_down and strong_trend:
                signal = SignalType.SELL
                confidence = min(0.9, (current_adx / 50) * 0.8)
                reasoning = f"Downtrend confirmed: EMA12({current_ema_12:.2f}) < EMA26({current_ema_26:.2f}) < EMA50({current_ema_50:.2f}), ADX({current_adx:.2f})"
            else:
                signal = SignalType.HOLD
                confidence = 0.3
                reasoning = f"No clear trend: ADX({current_adx:.2f}) < 25 or conflicting EMAs"
            
            # Calculate stop loss and take profit
            atr = self.indicators.calculate_atr(high, low, close).iloc[-1]
            stop_loss = current_price - (2 * atr) if signal == SignalType.BUY else current_price + (2 * atr)
            take_profit = current_price + (3 * atr) if signal == SignalType.BUY else current_price - (3 * atr)
            
            # Position sizing based on volatility
            position_size = min(0.02, 0.01 / (atr / current_price))
            
            return StrategyResult(
                strategy_name=self.name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                indicators=[
                    TechnicalIndicator("EMA12", current_ema_12, "trend", 0.8, datetime.now()),
                    TechnicalIndicator("EMA26", current_ema_26, "trend", 0.8, datetime.now()),
                    TechnicalIndicator("ADX", current_adx, "strength", 0.7, datetime.now())
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return None

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy using Bollinger Bands, RSI, and MACD"""
    
    def __init__(self):
        super().__init__("mean_reversion")
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze mean reversion signals"""
        try:
            if len(df) < 30:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Calculate indicators
            bb = self.indicators.calculate_bollinger_bands(close)
            rsi = self.indicators.calculate_rsi(close)
            macd = self.indicators.calculate_macd(close)
            
            # Current values
            current_price = close.iloc[-1]
            current_bb_upper = bb['upper'].iloc[-1]
            current_bb_lower = bb['lower'].iloc[-1]
            current_bb_middle = bb['middle'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd['macd'].iloc[-1]
            current_signal = macd['signal'].iloc[-1]
            
            # Mean reversion analysis
            oversold = current_price <= current_bb_lower and current_rsi < 30
            overbought = current_price >= current_bb_upper and current_rsi > 70
            macd_bullish = current_macd > current_signal
            macd_bearish = current_macd < current_signal
            
            # Generate signals
            if oversold and macd_bullish:
                signal = SignalType.BUY
                confidence = min(0.85, (30 - current_rsi) / 30 * 0.7 + 0.3)
                reasoning = f"Oversold condition: Price({current_price:.2f}) <= BB Lower({current_bb_lower:.2f}), RSI({current_rsi:.2f}) < 30, MACD bullish"
            elif overbought and macd_bearish:
                signal = SignalType.SELL
                confidence = min(0.85, (current_rsi - 70) / 30 * 0.7 + 0.3)
                reasoning = f"Overbought condition: Price({current_price:.2f}) >= BB Upper({current_bb_upper:.2f}), RSI({current_rsi:.2f}) > 70, MACD bearish"
            else:
                signal = SignalType.HOLD
                confidence = 0.3
                reasoning = f"No mean reversion signal: Price between BB bands, RSI({current_rsi:.2f}) neutral"
            
            # Calculate stop loss and take profit
            bb_width = current_bb_upper - current_bb_lower
            if signal == SignalType.BUY:
                stop_loss = current_bb_lower - (bb_width * 0.1)
                take_profit = current_bb_middle
            elif signal == SignalType.SELL:
                stop_loss = current_bb_upper + (bb_width * 0.1)
                take_profit = current_bb_middle
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Position sizing
            position_size = min(0.015, confidence * 0.02)
            
            return StrategyResult(
                strategy_name=self.name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                indicators=[
                    TechnicalIndicator("RSI", current_rsi, "momentum", 0.8, datetime.now()),
                    TechnicalIndicator("BB_Position", (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower), "mean_reversion", 0.7, datetime.now()),
                    TechnicalIndicator("MACD", current_macd - current_signal, "momentum", 0.6, datetime.now())
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return None

class MomentumStrategy(BaseStrategy):
    """Momentum Strategy using price and volume dynamics"""
    
    def __init__(self):
        super().__init__("momentum")
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze momentum signals"""
        try:
            if len(df) < 20:
                return None
            
            close = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']
            
            # Calculate momentum indicators
            price_momentum = close.pct_change(5)  # 5-period price momentum
            volume_momentum = volume.pct_change(5)  # 5-period volume momentum
            roc = close.pct_change(10)  # Rate of Change
            
            # Current values
            current_price = close.iloc[-1]
            current_price_momentum = price_momentum.iloc[-1]
            current_volume_momentum = volume_momentum.iloc[-1]
            current_roc = roc.iloc[-1]
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            volume_spike = volume.iloc[-1] > avg_volume * 1.5
            
            # Momentum analysis
            strong_up_momentum = current_price_momentum > 0.02 and current_roc > 0.05 and volume_spike
            strong_down_momentum = current_price_momentum < -0.02 and current_roc < -0.05 and volume_spike
            
            # Generate signals
            if strong_up_momentum:
                signal = SignalType.STRONG_BUY
                confidence = min(0.9, abs(current_price_momentum) * 10 + 0.5)
                reasoning = f"Strong upward momentum: Price momentum({current_price_momentum:.3f}), ROC({current_roc:.3f}), Volume spike({volume_spike})"
            elif strong_down_momentum:
                signal = SignalType.STRONG_SELL
                confidence = min(0.9, abs(current_price_momentum) * 10 + 0.5)
                reasoning = f"Strong downward momentum: Price momentum({current_price_momentum:.3f}), ROC({current_roc:.3f}), Volume spike({volume_spike})"
            else:
                signal = SignalType.HOLD
                confidence = 0.3
                reasoning = f"No strong momentum: Price momentum({current_price_momentum:.3f}), ROC({current_roc:.3f})"
            
            # Calculate stop loss and take profit
            atr = self.indicators.calculate_atr(high, low, close).iloc[-1]
            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                stop_loss = current_price - (1.5 * atr)
                take_profit = current_price + (2.5 * atr)
            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                stop_loss = current_price + (1.5 * atr)
                take_profit = current_price - (2.5 * atr)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Position sizing based on momentum strength
            position_size = min(0.025, confidence * 0.03)
            
            return StrategyResult(
                strategy_name=self.name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                indicators=[
                    TechnicalIndicator("Price_Momentum", current_price_momentum, "momentum", 0.8, datetime.now()),
                    TechnicalIndicator("ROC", current_roc, "momentum", 0.7, datetime.now()),
                    TechnicalIndicator("Volume_Spike", 1 if volume_spike else 0, "volume", 0.6, datetime.now())
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return None

class VolatilityStrategy(BaseStrategy):
    """Volatility Strategy using ATR, Bollinger Bands, and volatility regimes"""
    
    def __init__(self):
        super().__init__("volatility")
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze volatility signals"""
        try:
            if len(df) < 30:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Calculate volatility indicators
            atr = self.indicators.calculate_atr(high, low, close)
            bb = self.indicators.calculate_bollinger_bands(close)
            volatility = close.pct_change().rolling(window=20).std()
            
            # Current values
            current_price = close.iloc[-1]
            current_atr = atr.iloc[-1]
            current_bb_width = (bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / current_price
            current_volatility = volatility.iloc[-1]
            
            # Volatility regime analysis
            avg_volatility = volatility.rolling(window=50).mean().iloc[-1]
            high_volatility = current_volatility > avg_volatility * 1.5
            low_volatility = current_volatility < avg_volatility * 0.7
            
            # Volatility breakout analysis
            bb_position = (current_price - bb['lower'].iloc[-1]) / (bb['upper'].iloc[-1] - bb['lower'].iloc[-1])
            
            # Generate signals
            if high_volatility and bb_position > 0.8:
                signal = SignalType.SELL
                confidence = min(0.8, current_volatility * 20)
                reasoning = f"High volatility sell: Volatility({current_volatility:.3f}) > avg({avg_volatility:.3f}), BB position({bb_position:.2f}) > 0.8"
            elif high_volatility and bb_position < 0.2:
                signal = SignalType.BUY
                confidence = min(0.8, current_volatility * 20)
                reasoning = f"High volatility buy: Volatility({current_volatility:.3f}) > avg({avg_volatility:.3f}), BB position({bb_position:.2f}) < 0.2"
            elif low_volatility:
                signal = SignalType.HOLD
                confidence = 0.4
                reasoning = f"Low volatility: Volatility({current_volatility:.3f}) < avg({avg_volatility:.3f}), waiting for breakout"
            else:
                signal = SignalType.HOLD
                confidence = 0.3
                reasoning = f"Normal volatility: Volatility({current_volatility:.3f}) within normal range"
            
            # Calculate stop loss and take profit
            if signal in [SignalType.BUY, SignalType.SELL]:
                stop_loss = current_price - (2 * current_atr) if signal == SignalType.BUY else current_price + (2 * current_atr)
                take_profit = current_price + (1.5 * current_atr) if signal == SignalType.BUY else current_price - (1.5 * current_atr)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Position sizing based on volatility
            position_size = min(0.02, 0.01 / max(current_volatility, 0.01))
            
            return StrategyResult(
                strategy_name=self.name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                indicators=[
                    TechnicalIndicator("ATR", current_atr, "volatility", 0.8, datetime.now()),
                    TechnicalIndicator("Volatility", current_volatility, "volatility", 0.7, datetime.now()),
                    TechnicalIndicator("BB_Width", current_bb_width, "volatility", 0.6, datetime.now())
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in volatility strategy: {e}")
            return None

class StatisticalArbitrageStrategy(BaseStrategy):
    """Statistical Arbitrage Strategy using correlation and mean reversion"""
    
    def __init__(self):
        super().__init__("statistical_arbitrage")
    
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze statistical arbitrage signals"""
        try:
            if len(df) < 50:
                return None
            
            close = df['close']
            
            # Calculate statistical measures
            returns = close.pct_change().dropna()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            z_score = (close.iloc[-1] - close.mean()) / close.std()
            
            # Moving average convergence/divergence
            ma_short = close.rolling(window=10).mean()
            ma_long = close.rolling(window=30).mean()
            ma_spread = (ma_short - ma_long) / ma_long
            
            # Current values
            current_price = close.iloc[-1]
            current_z_score = z_score
            current_ma_spread = ma_spread.iloc[-1]
            
            # Statistical arbitrage analysis
            extreme_z_score = abs(current_z_score) > 2
            ma_divergence = abs(current_ma_spread) > 0.02
            
            # Generate signals
            if extreme_z_score and current_z_score > 2 and ma_divergence:
                signal = SignalType.SELL
                confidence = min(0.8, abs(current_z_score) / 3)
                reasoning = f"Statistical sell: Z-score({current_z_score:.2f}) > 2, MA spread({current_ma_spread:.3f}), Skewness({skewness:.2f})"
            elif extreme_z_score and current_z_score < -2 and ma_divergence:
                signal = SignalType.BUY
                confidence = min(0.8, abs(current_z_score) / 3)
                reasoning = f"Statistical buy: Z-score({current_z_score:.2f}) < -2, MA spread({current_ma_spread:.3f}), Skewness({skewness:.2f})"
            else:
                signal = SignalType.HOLD
                confidence = 0.3
                reasoning = f"No statistical signal: Z-score({current_z_score:.2f}), MA spread({current_ma_spread:.3f})"
            
            # Calculate stop loss and take profit
            price_std = close.std()
            if signal == SignalType.BUY:
                stop_loss = current_price - (1.5 * price_std)
                take_profit = current_price + (2 * price_std)
            elif signal == SignalType.SELL:
                stop_loss = current_price + (1.5 * price_std)
                take_profit = current_price - (2 * price_std)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Position sizing based on statistical confidence
            position_size = min(0.015, confidence * 0.02)
            
            return StrategyResult(
                strategy_name=self.name,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                indicators=[
                    TechnicalIndicator("Z_Score", current_z_score, "statistical", 0.8, datetime.now()),
                    TechnicalIndicator("MA_Spread", current_ma_spread, "mean_reversion", 0.7, datetime.now()),
                    TechnicalIndicator("Skewness", skewness, "distribution", 0.6, datetime.now()),
                    TechnicalIndicator("Kurtosis", kurtosis, "distribution", 0.5, datetime.now())
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in statistical arbitrage strategy: {e}")
            return None

# Example usage and testing
async def main():
    """Example usage of the enhanced technical analysis agent"""
    
    # Create sample market data
    sample_data = {
        'EURUSD=X': {
            '1d': {
                'symbol': 'EURUSD=X',
                'timestamp': datetime.now(),
                'open': 1.0850,
                'high': 1.0875,
                'low': 1.0840,
                'close': 1.0865,
                'volume': 1000000,
                'timeframe': '1d'
            }
        },
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
    
    # Initialize technical analysis agent
    agent = AdvancedTechnicalAnalysisAgent()
    
    # Process analysis
    input_data = {'market_data': sample_data}
    result = await agent.process(input_data)
    
    print("Technical Analysis Results:")
    print(f"Status: {result.status}")
    print(f"Total Signals: {len(result.signals)}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    for signal in result.signals:
        print(f"\nSignal: {signal.symbol}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Price: ${signal.price:.2f}")
        print(f"  Strategy: {signal.metadata['strategy']}")
        print(f"  Reasoning: {signal.metadata['reasoning']}")

if __name__ == "__main__":
    asyncio.run(main())
