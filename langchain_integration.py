"""
LangChain Integration for Multi-Agent Trading System
Provides enhanced workflow orchestration and decision-making capabilities
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.callbacks import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

from multi_agent_framework import (
    BaseAgent, AgentOutput, TradingSignal, SignalType, 
    RiskMetrics, MarketDataAgent, TechnicalAnalysisAgent,
    FundamentalsAgent, SentimentAgent, RiskManagerAgent, PortfolioManagerAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for trading system"""
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action"""
        logger.info(f"Agent Action: {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes"""
        logger.info(f"Agent Finish: {finish.return_values}")

class LangChainTradingOrchestrator:
    """LangChain-powered orchestrator for enhanced decision making"""
    
    def __init__(self, openai_api_key: str = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.openai_api_key = openai_api_key or "demo-key"
        
        # Initialize LangChain components
        self.llm = OpenAI(
            temperature=0.1,  # Low temperature for consistent trading decisions
            max_tokens=1000,
            api_key=self.openai_api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.callback_handler = TradingCallbackHandler()
        
        # Initialize trading tools
        self.trading_tools = self._create_trading_tools()
        
        # Create specialized agents
        self.decision_agent = self._create_decision_agent()
        self.risk_assessment_agent = self._create_risk_assessment_agent()
        self.portfolio_optimization_agent = self._create_portfolio_optimization_agent()
        
        # Workflow chains
        self.workflow_chains = self._create_workflow_chains()
    
    def _create_trading_tools(self) -> List[Tool]:
        """Create specialized trading tools for LangChain agents"""
        
        def analyze_market_data(query: str) -> str:
            """Analyze market data and return insights"""
            # This would integrate with actual market data
            return f"Market analysis for {query}: Current trend is bullish with strong momentum indicators"
        
        def calculate_risk_metrics(symbol: str) -> str:
            """Calculate risk metrics for a symbol"""
            # This would integrate with actual risk calculations
            return f"Risk metrics for {symbol}: VaR 95% = 2.5%, Sharpe Ratio = 1.2, Max Drawdown = 8%"
        
        def generate_trading_signals(symbol: str) -> str:
            """Generate trading signals for a symbol"""
            # This would integrate with technical analysis
            return f"Trading signals for {symbol}: BUY signal with 75% confidence, target price $105"
        
        def assess_sentiment(symbol: str) -> str:
            """Assess market sentiment for a symbol"""
            # This would integrate with sentiment analysis
            return f"Sentiment analysis for {symbol}: Bullish sentiment (0.7), news impact positive"
        
        def optimize_portfolio(portfolio_data: str) -> str:
            """Optimize portfolio allocation"""
            # This would integrate with portfolio optimization
            return f"Portfolio optimization: Recommended allocation - 40% stocks, 30% bonds, 30% alternatives"
        
        tools = [
            Tool(
                name="analyze_market_data",
                description="Analyze market data and provide insights",
                func=analyze_market_data
            ),
            Tool(
                name="calculate_risk_metrics",
                description="Calculate risk metrics for trading symbols",
                func=calculate_risk_metrics
            ),
            Tool(
                name="generate_trading_signals",
                description="Generate trading signals based on technical analysis",
                func=generate_trading_signals
            ),
            Tool(
                name="assess_sentiment",
                description="Assess market sentiment for trading symbols",
                func=assess_sentiment
            ),
            Tool(
                name="optimize_portfolio",
                description="Optimize portfolio allocation and risk management",
                func=optimize_portfolio
            )
        ]
        
        return tools
    
    def _create_decision_agent(self) -> AgentExecutor:
        """Create the main decision-making agent"""
        
        decision_prompt = PromptTemplate(
            input_variables=["agent_scratchpad", "tools", "tool_names", "input", "chat_history"],
            template="""
You are an expert trading decision agent. Your role is to analyze trading signals from multiple sources and make informed decisions.

Available tools: {tools}
Tool names: {tool_names}

Previous conversation:
{chat_history}

Current situation: {input}

Use the available tools to analyze the situation and provide a clear trading decision.
Consider:
1. Technical analysis signals
2. Fundamental analysis
3. Market sentiment
4. Risk metrics
5. Portfolio constraints

Provide your analysis and final decision in a structured format.

{agent_scratchpad}
"""
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.trading_tools,
            prompt=decision_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.trading_tools,
            memory=self.memory,
            callbacks=[self.callback_handler],
            verbose=True
        )
    
    def _create_risk_assessment_agent(self) -> AgentExecutor:
        """Create specialized risk assessment agent"""
        
        risk_prompt = PromptTemplate(
            input_variables=["agent_scratchpad", "tools", "tool_names", "input", "chat_history"],
            template="""
You are a risk management specialist. Your role is to assess and manage trading risks.

Available tools: {tools}
Tool names: {tool_names}

Previous conversation:
{chat_history}

Risk assessment request: {input}

Analyze the risk profile and provide:
1. Value at Risk (VaR) calculations
2. Conditional VaR (CVaR) analysis
3. Maximum drawdown assessment
4. Portfolio risk metrics
5. Risk mitigation recommendations

{agent_scratchpad}
"""
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.trading_tools,
            prompt=risk_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.trading_tools,
            memory=self.memory,
            callbacks=[self.callback_handler],
            verbose=True
        )
    
    def _create_portfolio_optimization_agent(self) -> AgentExecutor:
        """Create portfolio optimization agent"""
        
        portfolio_prompt = PromptTemplate(
            input_variables=["agent_scratchpad", "tools", "tool_names", "input", "chat_history"],
            template="""
You are a portfolio optimization specialist. Your role is to optimize portfolio allocation and manage positions.

Available tools: {tools}
Tool names: {tool_names}

Previous conversation:
{chat_history}

Portfolio optimization request: {input}

Provide portfolio optimization recommendations:
1. Asset allocation strategy
2. Position sizing recommendations
3. Rebalancing schedule
4. Risk-adjusted returns optimization
5. Diversification analysis

{agent_scratchpad}
"""
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.trading_tools,
            prompt=portfolio_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.trading_tools,
            memory=self.memory,
            callbacks=[self.callback_handler],
            verbose=True
        )
    
    def _create_workflow_chains(self) -> Dict[str, List[str]]:
        """Create workflow chains for different trading scenarios"""
        
        # Market Analysis Chain
        market_analysis_chain = [
            "analyze_market_data",
            "generate_trading_signals",
            "assess_sentiment"
        ]
        
        # Risk Management Chain
        risk_management_chain = [
            "calculate_risk_metrics",
            "optimize_portfolio"
        ]
        
        # Complete Trading Chain
        complete_trading_chain = [
            "analyze_market_data",
            "generate_trading_signals",
            "assess_sentiment",
            "calculate_risk_metrics",
            "optimize_portfolio"
        ]
        
        return {
            "market_analysis": market_analysis_chain,
            "risk_management": risk_management_chain,
            "complete_trading": complete_trading_chain
        }
    
    async def orchestrate_trading_decision(self, 
                                         market_data: Dict[str, Any],
                                         agent_outputs: Dict[str, AgentOutput]) -> Dict[str, Any]:
        """Orchestrate trading decision using LangChain agents"""
        
        logger.info("Starting LangChain trading orchestration")
        
        try:
            # Prepare context for decision making
            context = self._prepare_decision_context(market_data, agent_outputs)
            
            # Use decision agent to analyze and decide
            decision_result = await self._run_decision_agent(context)
            
            # Use risk assessment agent for risk analysis
            risk_result = await self._run_risk_assessment_agent(context)
            
            # Use portfolio optimization agent for allocation
            portfolio_result = await self._run_portfolio_optimization_agent(context)
            
            # Compile final decision
            final_decision = self._compile_final_decision(
                decision_result, risk_result, portfolio_result
            )
            
            logger.info("LangChain orchestration completed successfully")
            return final_decision
            
        except Exception as e:
            logger.error(f"LangChain orchestration failed: {e}")
            raise
    
    def _prepare_decision_context(self, 
                                 market_data: Dict[str, Any],
                                 agent_outputs: Dict[str, AgentOutput]) -> str:
        """Prepare context for decision making"""
        
        context_parts = []
        
        # Market data summary
        context_parts.append("MARKET DATA SUMMARY:")
        for symbol, data in market_data.items():
            context_parts.append(f"- {symbol}: Price ${data.get('close', 'N/A')}")
        
        # Agent outputs summary
        context_parts.append("\nAGENT ANALYSIS SUMMARY:")
        for agent_id, output in agent_outputs.items():
            if output.signals:
                context_parts.append(f"- {agent_id}: {len(output.signals)} signals generated")
                for signal in output.signals[:3]:  # Top 3 signals
                    context_parts.append(f"  * {signal.symbol}: {signal.signal_type.value} (confidence: {signal.confidence:.2f})")
        
        # Risk metrics
        risk_output = agent_outputs.get('risk_manager_agent')
        if risk_output and risk_output.risk_metrics:
            rm = risk_output.risk_metrics
            context_parts.append(f"\nRISK METRICS:")
            context_parts.append(f"- VaR 95%: ${rm.var_95:.2f}")
            context_parts.append(f"- Sharpe Ratio: {rm.sharpe_ratio:.2f}")
            context_parts.append(f"- Max Drawdown: {rm.max_drawdown:.2f}")
        
        return "\n".join(context_parts)
    
    async def _run_decision_agent(self, context: str) -> str:
        """Run the decision agent"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.decision_agent.run,
                {"input": f"Analyze the following trading situation and provide a decision:\n\n{context}"}
            )
            return result
        except Exception as e:
            logger.error(f"Decision agent failed: {e}")
            return f"Decision agent error: {e}"
    
    async def _run_risk_assessment_agent(self, context: str) -> str:
        """Run the risk assessment agent"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.risk_assessment_agent.run,
                {"input": f"Assess the risk for the following trading situation:\n\n{context}"}
            )
            return result
        except Exception as e:
            logger.error(f"Risk assessment agent failed: {e}")
            return f"Risk assessment agent error: {e}"
    
    async def _run_portfolio_optimization_agent(self, context: str) -> str:
        """Run the portfolio optimization agent"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.portfolio_optimization_agent.run,
                {"input": f"Optimize the portfolio for the following trading situation:\n\n{context}"}
            )
            return result
        except Exception as e:
            logger.error(f"Portfolio optimization agent failed: {e}")
            return f"Portfolio optimization agent error: {e}"
    
    def _compile_final_decision(self, 
                               decision_result: str,
                               risk_result: str,
                               portfolio_result: str) -> Dict[str, Any]:
        """Compile final trading decision"""
        
        return {
            "timestamp": datetime.now(),
            "decision_analysis": decision_result,
            "risk_assessment": risk_result,
            "portfolio_optimization": portfolio_result,
            "final_recommendation": self._extract_final_recommendation(decision_result),
            "confidence_score": self._calculate_confidence_score(decision_result),
            "risk_score": self._extract_risk_score(risk_result),
            "execution_priority": self._determine_execution_priority(decision_result, risk_result)
        }
    
    def _extract_final_recommendation(self, decision_result: str) -> str:
        """Extract final recommendation from decision result"""
        # Simple extraction logic - in production, this would be more sophisticated
        if "BUY" in decision_result.upper():
            return "BUY"
        elif "SELL" in decision_result.upper():
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence_score(self, decision_result: str) -> float:
        """Calculate confidence score from decision result"""
        # Simple scoring logic - in production, this would analyze the decision text
        if "high confidence" in decision_result.lower():
            return 0.9
        elif "medium confidence" in decision_result.lower():
            return 0.7
        elif "low confidence" in decision_result.lower():
            return 0.5
        else:
            return 0.6
    
    def _extract_risk_score(self, risk_result: str) -> float:
        """Extract risk score from risk assessment"""
        # Simple risk scoring - in production, this would analyze risk metrics
        if "high risk" in risk_result.lower():
            return 0.8
        elif "medium risk" in risk_result.lower():
            return 0.5
        elif "low risk" in risk_result.lower():
            return 0.2
        else:
            return 0.5
    
    def _determine_execution_priority(self, decision_result: str, risk_result: str) -> str:
        """Determine execution priority based on decision and risk"""
        confidence = self._calculate_confidence_score(decision_result)
        risk = self._extract_risk_score(risk_result)
        
        if confidence > 0.8 and risk < 0.3:
            return "HIGH"
        elif confidence > 0.6 and risk < 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def run_workflow_chain(self, chain_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific workflow chain"""
        
        if chain_name not in self.workflow_chains:
            raise ValueError(f"Unknown workflow chain: {chain_name}")
        
        chain_steps = self.workflow_chains[chain_name]
        
        try:
            # Execute each step in the chain
            results = []
            for step in chain_steps:
                # Find the tool for this step
                tool = next((t for t in self.trading_tools if t.name == step), None)
                if tool:
                    result = tool.func(str(input_data))
                    results.append({"step": step, "result": result})
            
            return {
                "chain_name": chain_name,
                "results": results,
                "timestamp": datetime.now(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Workflow chain {chain_name} failed: {e}")
            return {
                "chain_name": chain_name,
                "error": str(e),
                "timestamp": datetime.now(),
                "status": "failed"
            }
    
    def get_agent_memory(self) -> Dict[str, Any]:
        """Get conversation memory from agents"""
        return {
            "memory_buffer": self.memory.buffer,
            "memory_variables": self.memory.memory_variables,
            "return_messages": self.memory.return_messages
        }
    
    def clear_memory(self):
        """Clear agent memory"""
        self.memory.clear()

class EnhancedMultiAgentOrchestrator:
    """Enhanced orchestrator combining multi-agent framework with LangChain"""
    
    def __init__(self, openai_api_key: str = None, config: Dict[str, Any] = None):
        from multi_agent_framework import MultiAgentOrchestrator
        
        self.base_orchestrator = MultiAgentOrchestrator(config)
        self.langchain_orchestrator = LangChainTradingOrchestrator(openai_api_key, config)
        
    async def execute_enhanced_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced workflow with LangChain integration"""
        
        logger.info("Starting enhanced multi-agent workflow with LangChain")
        
        try:
            # Execute base multi-agent workflow
            base_results = await self.base_orchestrator.execute_workflow(input_data)
            
            # Extract market data and agent outputs for LangChain processing
            market_data = base_results['agent_outputs']['market_data_agent'].data['market_data']
            agent_outputs = base_results['agent_outputs']
            
            # Run LangChain orchestration
            langchain_results = await self.langchain_orchestrator.orchestrate_trading_decision(
                market_data, agent_outputs
            )
            
            # Combine results
            enhanced_results = {
                **base_results,
                "langchain_orchestration": langchain_results,
                "enhanced_workflow": True,
                "total_agents": len(base_results['agent_outputs']) + 3,  # +3 LangChain agents
                "workflow_type": "enhanced_multi_agent_langchain"
            }
            
            logger.info("Enhanced workflow completed successfully")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Enhanced workflow failed: {e}")
            raise

# Example usage and testing
async def main():
    """Example usage of the enhanced LangChain integration"""
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedMultiAgentOrchestrator()
    
    # Prepare input data
    input_data = {
        'symbols': ['EURUSD=X', 'GBPUSD=X', 'AAPL'],
        'timeframes': ['1m', '5m', '1h', '1d']
    }
    
    # Execute enhanced workflow
    try:
        results = await orchestrator.execute_enhanced_workflow(input_data)
        print("Enhanced Workflow Results:")
        print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"Enhanced workflow failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
