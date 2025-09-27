"""
Multi-Agent Trading System Framework
Implements the six-agent orchestration system with OpenAI Swarm and LangChain integration
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pandas as pd
import numpy as np
from enum import Enum
import threading
from queue import Queue, Empty
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    asset_class: str

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    agent_id: str
    metadata: Dict[str, Any]

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    cvar_99: float  # Conditional VaR 99%
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    portfolio_value: float

@dataclass
class AgentOutput:
    """Standardized agent output"""
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    signals: List[TradingSignal]
    risk_metrics: Optional[RiskMetrics]
    timestamp: datetime
    execution_time: float
    error: Optional[str] = None

class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.last_output = None
        self.execution_history = []
        self.dependencies = []
        self.output_queue = Queue()
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process input data and return agent output"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data format and completeness"""
        pass
    
    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status
    
    def add_dependency(self, agent_id: str):
        """Add a dependency agent"""
        if agent_id not in self.dependencies:
            self.dependencies.append(agent_id)
    
    def can_execute(self, completed_agents: List[str]) -> bool:
        """Check if agent can execute based on dependencies"""
        return all(dep in completed_agents for dep in self.dependencies)

class MarketDataAgent(BaseAgent):
    """Market Data Agent - Ingests real-time market data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("market_data_agent", config)
        self.data_streams = {}
        self.historical_data = {}
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process market data ingestion"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            # Simulate real-time data ingestion
            symbols = input_data.get('symbols', ['EURUSD=X', 'GBPUSD=X', 'AAPL'])
            timeframes = input_data.get('timeframes', ['1m', '5m', '1h', '1d'])
            
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = await self._fetch_market_data(symbol, timeframes)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={'market_data': market_data},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Market Data Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _fetch_market_data(self, symbol: str, timeframes: List[str]) -> Dict[str, Any]:
        """Fetch market data for a symbol across timeframes"""
        # This would integrate with real data sources like Kafka, WebSocket feeds
        # For now, we'll simulate the data structure
        data = {}
        for tf in timeframes:
            data[tf] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'open': 100.0 + np.random.normal(0, 1),
                'high': 101.0 + np.random.normal(0, 1),
                'low': 99.0 + np.random.normal(0, 1),
                'close': 100.5 + np.random.normal(0, 1),
                'volume': int(1000000 * np.random.uniform(0.5, 2.0)),
                'timeframe': tf
            }
        return data
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate market data input"""
        required_keys = ['symbols', 'timeframes']
        return all(key in input_data for key in required_keys)

class TechnicalAnalysisAgent(BaseAgent):
    """Technical Analysis Agent - Generates technical trading signals"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("technical_analysis_agent", config)
        self.add_dependency("market_data_agent")
        self.strategies = [
            'trend_following',
            'mean_reversion', 
            'momentum',
            'volatility',
            'statistical_arbitrage'
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process technical analysis"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            market_data = input_data.get('market_data', {})
            signals = []
            
            for symbol, data in market_data.items():
                for timeframe, price_data in data.items():
                    # Generate signals for each strategy
                    symbol_signals = await self._generate_signals(symbol, price_data, timeframe)
                    signals.extend(symbol_signals)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={'signals': signals, 'strategies_used': self.strategies},
                signals=signals,
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Technical Analysis Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _generate_signals(self, symbol: str, price_data: Dict, timeframe: str) -> List[TradingSignal]:
        """Generate trading signals using multiple strategies"""
        signals = []
        
        # Trend Following Strategy
        trend_signal = await self._trend_following_strategy(symbol, price_data, timeframe)
        if trend_signal:
            signals.append(trend_signal)
        
        # Mean Reversion Strategy
        mean_reversion_signal = await self._mean_reversion_strategy(symbol, price_data, timeframe)
        if mean_reversion_signal:
            signals.append(mean_reversion_signal)
        
        # Momentum Strategy
        momentum_signal = await self._momentum_strategy(symbol, price_data, timeframe)
        if momentum_signal:
            signals.append(momentum_signal)
        
        return signals
    
    async def _trend_following_strategy(self, symbol: str, price_data: Dict, timeframe: str) -> Optional[TradingSignal]:
        """Trend following using EMAs, ADX, and Ichimoku"""
        # Simplified implementation - would use actual technical indicators
        price = price_data['close']
        
        # Simulate trend analysis
        if price > 100.0:  # Simplified trend detection
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.75,
                price=price,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={'strategy': 'trend_following', 'timeframe': timeframe}
            )
        return None
    
    async def _mean_reversion_strategy(self, symbol: str, price_data: Dict, timeframe: str) -> Optional[TradingSignal]:
        """Mean reversion using Bollinger Bands, RSI, MACD"""
        price = price_data['close']
        
        # Simulate mean reversion analysis
        if price < 99.0:  # Simplified oversold detection
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.65,
                price=price,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={'strategy': 'mean_reversion', 'timeframe': timeframe}
            )
        return None
    
    async def _momentum_strategy(self, symbol: str, price_data: Dict, timeframe: str) -> Optional[TradingSignal]:
        """Momentum strategy using price and volume dynamics"""
        price = price_data['close']
        volume = price_data['volume']
        
        # Simulate momentum analysis
        if volume > 1500000:  # High volume breakout
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.STRONG_BUY,
                confidence=0.85,
                price=price,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={'strategy': 'momentum', 'timeframe': timeframe}
            )
        return None
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate technical analysis input"""
        return 'market_data' in input_data

class FundamentalsAgent(BaseAgent):
    """Fundamentals Agent - Analyzes financial statements and macroeconomic data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("fundamentals_agent", config)
        self.add_dependency("market_data_agent")
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process fundamental analysis"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            market_data = input_data.get('market_data', {})
            fundamental_signals = []
            
            for symbol, data in market_data.items():
                # Analyze fundamentals for each symbol
                fundamental_analysis = await self._analyze_fundamentals(symbol, data)
                if fundamental_analysis:
                    fundamental_signals.append(fundamental_analysis)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={'fundamental_signals': fundamental_signals},
                signals=fundamental_signals,
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Fundamentals Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _analyze_fundamentals(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Analyze fundamental metrics like P/E, EV/EBITDA, dividend yield"""
        # Simplified fundamental analysis
        # In production, this would scrape financial statements, earnings reports
        
        # Simulate fundamental analysis
        if 'AAPL' in symbol or 'EURUSD' in symbol:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.70,
                price=data.get('1d', {}).get('close', 100.0),
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={'strategy': 'fundamentals', 'metrics': {'pe_ratio': 15.2, 'dividend_yield': 0.025}}
            )
        return None
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate fundamentals input"""
        return 'market_data' in input_data

class SentimentAgent(BaseAgent):
    """Sentiment Agent - Analyzes news, social media, and analyst sentiment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("sentiment_agent", config)
        self.add_dependency("market_data_agent")
        self.sentiment_model = None  # Would load NLP model
        self.vector_db = None  # Would connect to Pinecone
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process sentiment analysis"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            market_data = input_data.get('market_data', {})
            sentiment_signals = []
            
            for symbol, data in market_data.items():
                # Analyze sentiment for each symbol
                sentiment_analysis = await self._analyze_sentiment(symbol, data)
                if sentiment_analysis:
                    sentiment_signals.append(sentiment_analysis)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={'sentiment_signals': sentiment_signals},
                signals=sentiment_signals,
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Sentiment Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _analyze_sentiment(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Analyze sentiment using NLP models"""
        # Simplified sentiment analysis
        # In production, this would use Hugging Face transformers, Pinecone vector DB
        
        # Simulate sentiment analysis
        sentiment_score = np.random.uniform(-1, 1)  # -1 (bearish) to 1 (bullish)
        
        if sentiment_score > 0.3:
            signal_type = SignalType.BUY
            confidence = min(0.9, abs(sentiment_score))
        elif sentiment_score < -0.3:
            signal_type = SignalType.SELL
            confidence = min(0.9, abs(sentiment_score))
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=data.get('1d', {}).get('close', 100.0),
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            metadata={'strategy': 'sentiment', 'sentiment_score': sentiment_score}
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate sentiment input"""
        return 'market_data' in input_data

class RiskManagerAgent(BaseAgent):
    """Risk Manager Agent - Calculates risk metrics and manages position sizes"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("risk_manager_agent", config)
        self.add_dependency("technical_analysis_agent")
        self.add_dependency("fundamentals_agent")
        self.add_dependency("sentiment_agent")
        self.portfolio_value = 100000  # Initial portfolio value
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process risk management"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            # Collect signals from all analysis agents
            all_signals = []
            for agent_output in input_data.get('agent_outputs', []):
                all_signals.extend(agent_output.signals)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(all_signals)
            
            # Filter signals based on risk criteria
            filtered_signals = await self._filter_signals_by_risk(all_signals, risk_metrics)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={'filtered_signals': filtered_signals, 'risk_metrics': risk_metrics},
                signals=filtered_signals,
                risk_metrics=risk_metrics,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Risk Manager Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _calculate_risk_metrics(self, signals: List[TradingSignal]) -> RiskMetrics:
        """Calculate VaR, CVaR, drawdown, and other risk metrics"""
        # Simplified risk calculation
        # In production, this would use historical returns, Monte Carlo simulation
        
        portfolio_value = self.portfolio_value
        volatility = 0.15  # 15% annual volatility
        sharpe_ratio = 1.2
        
        # Calculate VaR (simplified)
        var_95 = portfolio_value * 0.05  # 5% VaR
        var_99 = portfolio_value * 0.02  # 2% VaR
        cvar_95 = portfolio_value * 0.07  # 7% CVaR
        cvar_99 = portfolio_value * 0.03  # 3% CVaR
        
        max_drawdown = portfolio_value * 0.10  # 10% max drawdown
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            portfolio_value=portfolio_value
        )
    
    async def _filter_signals_by_risk(self, signals: List[TradingSignal], risk_metrics: RiskMetrics) -> List[TradingSignal]:
        """Filter signals based on risk criteria"""
        filtered_signals = []
        
        for signal in signals:
            # Risk filtering logic
            if signal.confidence > 0.6:  # Only high-confidence signals
                # Adjust position size based on risk
                adjusted_signal = TradingSignal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    price=signal.price,
                    timestamp=signal.timestamp,
                    agent_id=signal.agent_id,
                    metadata={
                        **signal.metadata,
                        'position_size': min(0.02, signal.confidence * 0.03),  # Max 2% position
                        'risk_adjusted': True
                    }
                )
                filtered_signals.append(adjusted_signal)
        
        return filtered_signals
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate risk manager input"""
        return 'agent_outputs' in input_data

class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager Agent - Executes trades and manages portfolio"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("portfolio_manager_agent", config)
        self.add_dependency("risk_manager_agent")
        self.portfolio = {}
        self.trade_history = []
    
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process portfolio management and trade execution"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            risk_output = input_data.get('risk_output')
            if not risk_output:
                raise ValueError("No risk manager output provided")
            
            signals = risk_output.signals
            risk_metrics = risk_output.risk_metrics
            
            # Execute trades based on risk-adjusted signals
            executed_trades = await self._execute_trades(signals, risk_metrics)
            
            # Update portfolio
            portfolio_update = await self._update_portfolio(executed_trades)
            
            execution_time = time.time() - start_time
            output = AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={
                    'executed_trades': executed_trades,
                    'portfolio_update': portfolio_update,
                    'trade_history': self.trade_history
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
            self.status = AgentStatus.ERROR
            logger.error(f"Portfolio Manager Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _execute_trades(self, signals: List[TradingSignal], risk_metrics: RiskMetrics) -> List[Dict]:
        """Execute trades based on signals"""
        executed_trades = []
        
        for signal in signals:
            # Simulate trade execution
            trade = {
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'price': signal.price,
                'position_size': signal.metadata.get('position_size', 0.01),
                'timestamp': datetime.now(),
                'confidence': signal.confidence,
                'agent_id': signal.agent_id
            }
            
            executed_trades.append(trade)
            self.trade_history.append(trade)
        
        return executed_trades
    
    async def _update_portfolio(self, executed_trades: List[Dict]) -> Dict:
        """Update portfolio based on executed trades"""
        # Simplified portfolio update
        portfolio_value = 100000  # Would calculate from actual positions
        
        return {
            'portfolio_value': portfolio_value,
            'total_trades': len(self.trade_history),
            'active_positions': len(executed_trades),
            'last_update': datetime.now()
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate portfolio manager input"""
        return 'risk_output' in input_data

class MultiAgentOrchestrator:
    """Main orchestrator for the multi-agent trading system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents = {}
        self.execution_order = []
        self.completed_agents = []
        self.agent_outputs = {}
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        self.agents = {
            'market_data_agent': MarketDataAgent(),
            'technical_analysis_agent': TechnicalAnalysisAgent(),
            'fundamentals_agent': FundamentalsAgent(),
            'sentiment_agent': SentimentAgent(),
            'risk_manager_agent': RiskManagerAgent(),
            'portfolio_manager_agent': PortfolioManagerAgent()
        }
        
        # Define execution order
        self.execution_order = [
            'market_data_agent',
            ['technical_analysis_agent', 'fundamentals_agent', 'sentiment_agent'],  # Parallel
            'risk_manager_agent',
            'portfolio_manager_agent'
        ]
    
    async def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete multi-agent workflow"""
        logger.info("Starting multi-agent trading workflow")
        
        try:
            # Execute agents in order
            for step in self.execution_order:
                if isinstance(step, list):
                    # Parallel execution
                    await self._execute_parallel_agents(step, input_data)
                else:
                    # Sequential execution
                    await self._execute_agent(step, input_data)
            
            # Compile final results
            final_output = self._compile_results()
            
            logger.info("Multi-agent workflow completed successfully")
            return final_output
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_parallel_agents(self, agent_ids: List[str], input_data: Dict[str, Any]):
        """Execute multiple agents in parallel"""
        logger.info(f"Executing parallel agents: {agent_ids}")
        
        tasks = []
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            if agent.can_execute(self.completed_agents):
                task = asyncio.create_task(agent.process(input_data))
                tasks.append((agent_id, task))
        
        # Wait for all parallel tasks to complete
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Store results
        for i, (agent_id, _) in enumerate(tasks):
            if isinstance(results[i], Exception):
                logger.error(f"Agent {agent_id} failed: {results[i]}")
            else:
                self.agent_outputs[agent_id] = results[i]
                self.completed_agents.append(agent_id)
    
    async def _execute_agent(self, agent_id: str, input_data: Dict[str, Any]):
        """Execute a single agent"""
        logger.info(f"Executing agent: {agent_id}")
        
        agent = self.agents[agent_id]
        if not agent.can_execute(self.completed_agents):
            logger.warning(f"Agent {agent_id} dependencies not met")
            return
        
        try:
            output = await agent.process(input_data)
            self.agent_outputs[agent_id] = output
            self.completed_agents.append(agent_id)
            
            # Update input data for next agents
            if agent_id == 'market_data_agent':
                input_data['market_data'] = output.data['market_data']
            elif agent_id in ['technical_analysis_agent', 'fundamentals_agent', 'sentiment_agent']:
                if 'agent_outputs' not in input_data:
                    input_data['agent_outputs'] = []
                input_data['agent_outputs'].append(output)
            elif agent_id == 'risk_manager_agent':
                input_data['risk_output'] = output
            
        except Exception as e:
            logger.error(f"Agent {agent_id} execution failed: {e}")
            raise
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results from all agents"""
        return {
            'workflow_status': 'completed',
            'agent_outputs': self.agent_outputs,
            'completed_agents': self.completed_agents,
            'total_execution_time': sum(
                output.execution_time for output in self.agent_outputs.values()
            ),
            'timestamp': datetime.now()
        }
    
    def get_agent_status(self) -> Dict[str, AgentStatus]:
        """Get status of all agents"""
        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
    
    def reset_workflow(self):
        """Reset workflow for new execution"""
        self.completed_agents = []
        self.agent_outputs = {}
        for agent in self.agents.values():
            agent.status = AgentStatus.IDLE

# Example usage and testing
async def main():
    """Example usage of the multi-agent system"""
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Prepare input data
    input_data = {
        'symbols': ['EURUSD=X', 'GBPUSD=X', 'AAPL'],
        'timeframes': ['1m', '5m', '1h', '1d']
    }
    
    # Execute workflow
    try:
        results = await orchestrator.execute_workflow(input_data)
        print("Workflow Results:")
        print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"Workflow failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
