#!/usr/bin/env python3
"""
Simplified Multi-Agent Trading System Demo
Demonstrates the core functionality without complex dependencies
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMarketDataAgent:
    """Simplified Market Data Agent"""
    
    def __init__(self, agent_id: str = "MarketDataAgent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market data collection"""
        symbols = data.get("symbols", ["AAPL", "GOOGL"])
        
        self.logger.info(f"Collecting market data for symbols: {symbols}")
        
        # Simulate market data collection
        market_data = {}
        for symbol in symbols:
            # Generate simulated price data
            base_price = 100 + hash(symbol) % 200  # Simulate different base prices
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            current_price = base_price * (1 + price_change)
            
            market_data[symbol] = {
                "symbol": symbol,
                "price": round(current_price, 2),
                "volume": np.random.randint(1000000, 10000000),
                "timestamp": datetime.now().isoformat(),
                "change": round(price_change * 100, 2)
            }
        
        self.logger.info(f"Market data collected for {len(market_data)} symbols")
        return {"market_data": market_data}

class SimpleTechnicalAnalysisAgent:
    """Simplified Technical Analysis Agent"""
    
    def __init__(self, agent_id: str = "TechnicalAnalysisAgent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute technical analysis"""
        market_data = data.get("market_data", {})
        
        self.logger.info(f"Performing technical analysis for {len(market_data)} symbols")
        
        technical_signals = {}
        for symbol, data in market_data.items():
            price = data["price"]
            change = data["change"]
            
            # Simple technical analysis logic
            signals = []
            signal_strength = "NEUTRAL"
            
            if change > 2:
                signals.append("STRONG_UPTREND")
                signal_strength = "BULLISH"
            elif change > 0.5:
                signals.append("UPTREND")
                signal_strength = "BULLISH"
            elif change < -2:
                signals.append("STRONG_DOWNTREND")
                signal_strength = "BEARISH"
            elif change < -0.5:
                signals.append("DOWNTREND")
                signal_strength = "BEARISH"
            else:
                signals.append("SIDEWAYS")
                signal_strength = "NEUTRAL"
            
            # Add momentum signals
            if data["volume"] > 5000000:  # High volume
                signals.append("HIGH_VOLUME")
            
            technical_signals[symbol] = {
                "signals": signals,
                "strength": signal_strength,
                "confidence": min(0.9, abs(change) / 5 + 0.5)  # Higher confidence for larger moves
            }
        
        self.logger.info(f"Technical analysis completed for {len(technical_signals)} symbols")
        return {"technical_signals": technical_signals}

class SimpleRiskManagerAgent:
    """Simplified Risk Manager Agent"""
    
    def __init__(self, agent_id: str = "RiskManagerAgent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk management"""
        market_data = data.get("market_data", {})
        technical_signals = data.get("technical_signals", {})
        
        self.logger.info("Performing risk assessment")
        
        risk_assessment = {
            "portfolio_var": 0.02,  # 2% VaR
            "max_drawdown": 0.05,   # 5% max drawdown
            "risk_score": 0.3,      # Low risk
            "recommendations": []
        }
        
        # Analyze each symbol for risk
        for symbol, tech_data in technical_signals.items():
            confidence = tech_data["confidence"]
            strength = tech_data["strength"]
            
            if confidence > 0.8 and strength in ["BULLISH", "BEARISH"]:
                risk_assessment["recommendations"].append({
                    "symbol": symbol,
                    "action": "APPROVE",
                    "reason": f"High confidence {strength.lower()} signal"
                })
            elif confidence < 0.4:
                risk_assessment["recommendations"].append({
                    "symbol": symbol,
                    "action": "REJECT",
                    "reason": "Low confidence signal"
                })
            else:
                risk_assessment["recommendations"].append({
                    "symbol": symbol,
                    "action": "MONITOR",
                    "reason": "Medium confidence signal"
                })
        
        self.logger.info(f"Risk assessment completed with {len(risk_assessment['recommendations'])} recommendations")
        return {"risk_assessment": risk_assessment}

class SimplePortfolioManagerAgent:
    """Simplified Portfolio Manager Agent"""
    
    def __init__(self, agent_id: str = "PortfolioManagerAgent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio management"""
        market_data = data.get("market_data", {})
        technical_signals = data.get("technical_signals", {})
        risk_assessment = data.get("risk_assessment", {})
        
        self.logger.info("Executing portfolio management")
        
        # Generate trading orders based on signals and risk
        orders = []
        for symbol, tech_data in technical_signals.items():
            recommendations = risk_assessment.get("recommendations", [])
            symbol_rec = next((r for r in recommendations if r["symbol"] == symbol), None)
            
            if symbol_rec and symbol_rec["action"] == "APPROVE":
                price = market_data[symbol]["price"]
                strength = tech_data["strength"]
                confidence = tech_data["confidence"]
                
                if strength == "BULLISH":
                    orders.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": int(1000 * confidence),  # Position size based on confidence
                        "price": price,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
                elif strength == "BEARISH":
                    orders.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": int(1000 * confidence),
                        "price": price,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
        
        portfolio_summary = {
            "total_orders": len(orders),
            "buy_orders": len([o for o in orders if o["side"] == "BUY"]),
            "sell_orders": len([o for o in orders if o["side"] == "SELL"]),
            "total_value": sum(o["quantity"] * o["price"] for o in orders),
            "average_confidence": np.mean([o["confidence"] for o in orders]) if orders else 0
        }
        
        self.logger.info(f"Portfolio management completed: {len(orders)} orders generated")
        return {
            "orders": orders,
            "portfolio_summary": portfolio_summary
        }

class SimpleMultiAgentOrchestrator:
    """Simplified Multi-Agent Orchestrator"""
    
    def __init__(self):
        self.agents = {
            "market_data": SimpleMarketDataAgent(),
            "technical_analysis": SimpleTechnicalAnalysisAgent(),
            "risk_manager": SimpleRiskManagerAgent(),
            "portfolio_manager": SimplePortfolioManagerAgent()
        }
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
    
    async def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete multi-agent workflow"""
        
        self.logger.info("Starting Multi-Agent Trading Workflow")
        start_time = datetime.now()
        
        try:
            # Step 1: Market Data Collection
            self.logger.info("Step 1: Collecting market data...")
            market_data_result = await self.agents["market_data"].execute(input_data)
            
            # Step 2: Technical Analysis
            self.logger.info("Step 2: Performing technical analysis...")
            technical_result = await self.agents["technical_analysis"].execute(market_data_result)
            
            # Step 3: Risk Management
            self.logger.info("Step 3: Assessing risk...")
            risk_result = await self.agents["risk_manager"].execute({
                **market_data_result,
                **technical_result
            })
            
            # Step 4: Portfolio Management
            self.logger.info("Step 4: Managing portfolio...")
            portfolio_result = await self.agents["portfolio_manager"].execute({
                **market_data_result,
                **technical_result,
                **risk_result
            })
            
            # Compile results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            results = {
                "status": "completed",
                "execution_time": execution_time,
                "timestamp": end_time.isoformat(),
                "market_data": market_data_result,
                "technical_analysis": technical_result,
                "risk_assessment": risk_result,
                "portfolio_management": portfolio_result,
                "summary": {
                    "symbols_analyzed": len(market_data_result.get("market_data", {})),
                    "signals_generated": len(technical_result.get("technical_signals", {})),
                    "orders_created": len(portfolio_result.get("orders", [])),
                    "total_execution_time": execution_time
                }
            }
            
            self.logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

async def run_demo():
    """Run the simplified multi-agent trading demo"""
    
    print("=" * 60)
    print("Multi-Agent Trading System Demo")
    print("=" * 60)
    print("This demo showcases a simplified multi-agent trading system")
    print("with Alpaca integration capabilities.")
    print()
    
    # Initialize orchestrator
    orchestrator = SimpleMultiAgentOrchestrator()
    
    # Prepare input data
    input_data = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "timeframes": ["1d"],
        "execution_mode": "paper"
    }
    
    print("Configuration:")
    print(f"  Symbols: {input_data['symbols']}")
    print(f"  Execution Mode: {input_data['execution_mode']}")
    print(f"  Broker: Alpaca (Paper Trading)")
    print()
    
    # Execute workflow
    print("Executing Multi-Agent Trading Workflow...")
    print("-" * 40)
    
    results = await orchestrator.execute_workflow(input_data)
    
    if results["status"] == "completed":
        print("\nWorkflow Results:")
        print("=" * 40)
        
        # Market Data Summary
        market_data = results["market_data"]["market_data"]
        print(f"\nMarket Data ({len(market_data)} symbols):")
        for symbol, data in market_data.items():
            print(f"  {symbol}: ${data['price']:.2f} ({data['change']:+.2f}%)")
        
        # Technical Analysis Summary
        tech_signals = results["technical_analysis"]["technical_signals"]
        print(f"\nTechnical Analysis ({len(tech_signals)} signals):")
        for symbol, signals in tech_signals.items():
            print(f"  {symbol}: {signals['strength']} (confidence: {signals['confidence']:.2f})")
            print(f"    Signals: {', '.join(signals['signals'])}")
        
        # Risk Assessment Summary
        risk_recs = results["risk_assessment"]["risk_assessment"]["recommendations"]
        print(f"\nRisk Assessment ({len(risk_recs)} recommendations):")
        for rec in risk_recs:
            print(f"  {rec['symbol']}: {rec['action']} - {rec['reason']}")
        
        # Portfolio Management Summary
        orders = results["portfolio_management"]["orders"]
        portfolio_summary = results["portfolio_management"]["portfolio_summary"]
        print(f"\nPortfolio Management:")
        print(f"  Total Orders: {portfolio_summary['total_orders']}")
        print(f"  Buy Orders: {portfolio_summary['buy_orders']}")
        print(f"  Sell Orders: {portfolio_summary['sell_orders']}")
        print(f"  Total Value: ${portfolio_summary['total_value']:,.2f}")
        print(f"  Average Confidence: {portfolio_summary['average_confidence']:.2f}")
        
        if orders:
            print(f"\nGenerated Orders:")
            for order in orders:
                print(f"  {order['side']} {order['quantity']} {order['symbol']} @ ${order['price']:.2f} (conf: {order['confidence']:.2f})")
        
        # Summary
        summary = results["summary"]
        print(f"\nExecution Summary:")
        print(f"  Symbols Analyzed: {summary['symbols_analyzed']}")
        print(f"  Signals Generated: {summary['signals_generated']}")
        print(f"  Orders Created: {summary['orders_created']}")
        print(f"  Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        
        print(f"\nStatus: {results['status'].upper()}")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        
    else:
        print(f"\nWorkflow Failed: {results.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Test your Alpaca connection:")
    print("   python test_alpaca_connection.py")
    print()
    print("2. Run the full trading system:")
    print("   python multi_agent_trading_system.py --mode single --symbols AAPL GOOGL")
    print()
    print("3. Your system is ready for paper trading with Alpaca!")

if __name__ == "__main__":
    asyncio.run(run_demo())
