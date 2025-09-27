"""
Main Application for Financial Trading Strategy Analysis
Entry point for the Goldbach Trading Strategy Analyzer
"""
import sys
import logging
from datetime import datetime
import argparse
from data_collector import FinancialDataCollector
from tesla_369_calculator import Tesla369Calculator
from backtesting_engine import BacktestingEngine
from dashboard import TradingDashboard
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinancialAnalysisApp:
    """Main application class for financial analysis"""
    
    def __init__(self):
        self.config = Config()
        self.data_collector = FinancialDataCollector()
        self.tesla_calc = Tesla369Calculator()
        self.backtester = BacktestingEngine()
        self.dashboard = TradingDashboard()
        
    def collect_data(self, symbols: list = None, timeframes: list = None):
        """Collect market data for analysis"""
        logger.info("Starting data collection...")
        
        if symbols is None:
            symbols = []
            for asset_class, syms in self.config.DEFAULT_ASSETS.items():
                symbols.extend(syms)
        
        if timeframes is None:
            timeframes = self.config.DEFAULT_TIMEFRAMES
        
        for timeframe in timeframes:
            logger.info(f"Collecting data for timeframe: {timeframe}")
            self.data_collector.collect_market_data(
                symbols=symbols,
                timeframe=timeframe,
                period='2y',
                asset_class='mixed'
            )
        
        logger.info("Data collection completed")
    
    def run_analysis(self, symbol: str, timeframe: str = '1d'):
        """Run comprehensive analysis for a symbol"""
        logger.info(f"Running analysis for {symbol} ({timeframe})")
        
        # Get market data
        data = self.data_collector.get_market_data(symbol, timeframe)
        
        if data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Run Tesla analysis
        analysis = self.tesla_calc.analyze_market_structure(data)
        
        # Run backtest
        backtest_results = self.backtester.run_backtest(data)
        
        # Combine results
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_date': datetime.now(),
            'market_analysis': analysis,
            'backtest_results': backtest_results
        }
        
        logger.info(f"Analysis completed for {symbol}")
        return results
    
    def run_optimization(self, symbol: str, timeframe: str = '1d'):
        """Run parameter optimization for a symbol"""
        logger.info(f"Running optimization for {symbol} ({timeframe})")
        
        # Get market data
        data = self.data_collector.get_market_data(symbol, timeframe)
        
        if data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Define parameter ranges
        param_ranges = {
            'po3_multipliers': [0.5, 1.0, 1.5, 2.0],
            'lookback_periods': [6, 9, 12, 18],
            'max_position_sizes': [0.01, 0.02, 0.03]
        }
        
        # Run optimization
        optimization_results = self.backtester.optimize_strategy(data, param_ranges)
        
        logger.info(f"Optimization completed for {symbol}")
        return optimization_results
    
    def start_dashboard(self, debug: bool = False, port: int = None):
        """Start the interactive dashboard"""
        logger.info("Starting dashboard...")
        self.dashboard.run(debug=debug, port=port)
    
    def generate_report(self, results: dict, output_file: str = None):
        """Generate analysis report"""
        if output_file is None:
            output_file = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logger.info(f"Generating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("Goldbach Trading Strategy Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {results['analysis_date']}\n")
            f.write(f"Symbol: {results['symbol']}\n")
            f.write(f"Timeframe: {results['timeframe']}\n\n")
            
            # Market Analysis
            f.write("MARKET ANALYSIS\n")
            f.write("-" * 20 + "\n")
            market_stats = results['market_analysis']['statistics']
            f.write(f"Total Periods: {market_stats['total_periods']}\n")
            f.write(f"Price Range: {market_stats['price_range']:.2f}\n")
            f.write(f"Volatility: {market_stats['volatility']:.4f}\n")
            f.write(f"Trend Direction: {market_stats['trend_direction']}\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            metrics = results['backtest_results']['performance_metrics']
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n\n")
            
            # Trade Summary
            f.write("TRADE SUMMARY\n")
            f.write("-" * 20 + "\n")
            trades = results['backtest_results']['trades']
            completed_trades = [t for t in trades if 'exit_time' in t]
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t['pnl'] > 0]
                losing_trades = [t for t in completed_trades if t['pnl'] < 0]
                
                f.write(f"Completed Trades: {len(completed_trades)}\n")
                f.write(f"Winning Trades: {len(winning_trades)}\n")
                f.write(f"Losing Trades: {len(losing_trades)}\n")
                
                if winning_trades:
                    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                    f.write(f"Average Win: ${avg_win:.2f}\n")
                
                if losing_trades:
                    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                    f.write(f"Average Loss: ${avg_loss:.2f}\n")
        
        logger.info(f"Report saved to: {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Goldbach Trading Strategy Analyzer')
    parser.add_argument('--mode', choices=['collect', 'analyze', 'optimize', 'dashboard'], 
                       default='dashboard', help='Application mode')
    parser.add_argument('--symbol', default='EURUSD=X', help='Trading symbol')
    parser.add_argument('--timeframe', default='1d', help='Data timeframe')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    
    args = parser.parse_args()
    
    app = FinancialAnalysisApp()
    
    try:
        if args.mode == 'collect':
            logger.info("Starting data collection mode")
            app.collect_data()
            
        elif args.mode == 'analyze':
            logger.info(f"Starting analysis mode for {args.symbol}")
            results = app.run_analysis(args.symbol, args.timeframe)
            if results:
                app.generate_report(results)
            
        elif args.mode == 'optimize':
            logger.info(f"Starting optimization mode for {args.symbol}")
            results = app.run_optimization(args.symbol, args.timeframe)
            if results:
                logger.info(f"Best parameters: {results['best_params']}")
            
        elif args.mode == 'dashboard':
            logger.info("Starting dashboard mode")
            app.start_dashboard(debug=args.debug, port=args.port)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
