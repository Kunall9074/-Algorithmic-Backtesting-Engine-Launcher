"""
Algorithmic Backtesting Engine
Main entry point for running backtests
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader, generate_sample_data
from engine.backtest import Backtest
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from visualization.charts import BacktestVisualizer, interactive_chart


def run_backtest(
    data_source: str = 'sample',
    symbol: str = 'RELIANCE.NS',
    strategy_name: str = 'sma_crossover',
    initial_cash: float = 100000,
    show_charts: bool = True,
    interactive: bool = False
):
    """
    Run a backtest with specified parameters.
    
    Args:
        data_source: 'sample', 'yahoo', or path to CSV file
        symbol: Stock symbol for Yahoo Finance
        strategy_name: 'sma_crossover' or 'rsi'
        initial_cash: Starting capital
        show_charts: Whether to display charts
        interactive: Use interactive Plotly charts
    """
    print("\n" + "="*60)
    print("       ALGORITHMIC BACKTESTING ENGINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    
    if data_source == 'sample':
        data = generate_sample_data(days=500, volatility=0.02, trend=0.0003)
        print(f"   Generated {len(data)} days of sample data")
    elif data_source == 'yahoo':
        try:
            data = loader.load_yahoo(symbol, period='2y')
            print(f"   Loaded {len(data)} days of {symbol} data from Yahoo Finance")
        except Exception as e:
            print(f"   Error loading from Yahoo: {e}")
            print("   Falling back to sample data...")
            data = generate_sample_data(days=500)
    else:
        # Assume it's a CSV path
        try:
            data = loader.load_csv(data_source)
            print(f"   Loaded {len(data)} days from CSV")
        except Exception as e:
            print(f"   Error loading CSV: {e}")
            return
    
    # Select strategy
    print(f"\nStrategy: {strategy_name.upper()}")
    
    if strategy_name == 'sma_crossover':
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        print("   Fast SMA: 10, Slow SMA: 30")
    elif strategy_name == 'rsi':
        strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
        print("   RSI Period: 14, Oversold: 30, Overbought: 70")
    else:
        print(f"   Unknown strategy: {strategy_name}")
        return
    
    # Run backtest
    print(f"\nRunning backtest...")
    print(f"   Initial Capital: Rs. {initial_cash:,.0f}")
    
    bt = Backtest(
        data=data,
        strategy=strategy,
        initial_cash=initial_cash,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )
    
    results = bt.run()
    
    # Print results
    print(results.summary())
    
    # Final value
    final_value = results.equity_curve['equity'].iloc[-1]
    profit = final_value - initial_cash
    print(f"\nFinal Portfolio Value: Rs. {final_value:,.2f}")
    print(f"   Profit/Loss: Rs. {profit:,.2f} ({profit/initial_cash*100:.2f}%)")
    
    # Show charts
    if show_charts:
        print("\nGenerating charts...")
        
        if interactive:
            interactive_chart(results)
        else:
            viz = BacktestVisualizer(results)
            
            # Show summary dashboard
            fig = viz.plot_summary()
            
            import matplotlib.pyplot as plt
            plt.show()
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Algorithmic Backtesting Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with sample data
  python main.py --symbol RELIANCE.NS --yahoo       # Use Yahoo Finance data
  python main.py --strategy rsi                     # Use RSI strategy
  python main.py --data mydata.csv                  # Use CSV file
  python main.py --interactive                      # Interactive Plotly charts
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        default='sample',
        help='Data source: "sample", "yahoo", or path to CSV file'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        default='RELIANCE.NS',
        help='Stock symbol for Yahoo Finance (default: RELIANCE.NS)'
    )
    
    parser.add_argument(
        '--yahoo', '-y',
        action='store_true',
        help='Use Yahoo Finance data'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['sma_crossover', 'rsi'],
        default='sma_crossover',
        help='Trading strategy to test'
    )
    
    parser.add_argument(
        '--cash', '-c',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Disable chart display'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Use interactive Plotly charts'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple strategies'
    )
    
    args = parser.parse_args()
    
    # Check for compare mode
    if args.compare:
        compare_strategies(
            data_source='yahoo' if args.yahoo else args.data,
            symbol=args.symbol,
            initial_cash=args.cash
        )
        return

    # Determine data source
    data_source = 'yahoo' if args.yahoo else args.data
    
    # Run backtest
    run_backtest(
        data_source=data_source,
        symbol=args.symbol,
        strategy_name=args.strategy,
        initial_cash=args.cash,
        show_charts=not args.no_charts,
        interactive=args.interactive
    )


def compare_strategies(
    data_source: str = 'sample',
    symbol: str = 'RELIANCE.NS',
    initial_cash: float = 100000
):
    """
    Run and compare multiple strategies.
    
    Args:
        data_source: Data source
        symbol: Symbol
        initial_cash: Initial capital
    """
    print("\n" + "="*80)
    print(f"       STRATEGY COMPARISON: {symbol}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    
    if data_source == 'sample':
        data = generate_sample_data(days=500)
    elif data_source == 'yahoo':
        try:
            data = loader.load_yahoo(symbol, period='2y')
        except Exception as e:
            print(f"   Error: {e}, using sample data")
            data = generate_sample_data(days=500)
    else:
        try:
            data = loader.load_csv(data_source)
        except Exception:
            return

    # Define strategies to compare
    strategies = [
        ('SMA Crossover (10/30)', SMACrossoverStrategy(10, 30)),
        ('SMA Crossover (20/50)', SMACrossoverStrategy(20, 50)),
        ('RSI (14, 30/70)', RSIStrategy(14, 30, 70)),
        ('RSI (14, 25/75)', RSIStrategy(14, 25, 75)),
    ]
    
    results_list = []
    
    print(f"\nRunning backtests on {len(data)} bars...")
    print("-" * 105)
    print(f"{'Strategy':<25} | {'Return':<8} | {'CAGR':<7} | {'Win Rate':<8} | {'Drawdown':<8} | {'Trades':<6} | {'Sharpe':<6}")
    print("-" * 105)
    
    # 1. Buy and Hold Benchmark
    bh_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
    bh_dd = 0 # Simplified
    cummax = data['Close'].cummax()
    dd = (data['Close'] - cummax) / cummax
    bh_max_dd = abs(dd.min()) * 100
    
    print(f"{'Buy & Hold':<25} | {bh_return:6.2f}% | {'N/A':<7} | {'100%':<8} | {bh_max_dd:6.2f}% | {'1':<6} | {'N/A':<6}")
    
    for name, strategy in strategies:
        bt = Backtest(data, strategy, initial_cash)
        res = bt.run()
        m = res.metrics
        
        print(f"{name:<25} | {m.total_return*100:6.2f}% | {m.cagr*100:5.1f}%  | {m.win_rate*100:6.1f}%  | {m.max_drawdown*100:6.2f}%  | {m.total_trades:<6} | {m.sharpe_ratio:5.2f}")
        results_list.append((name, res))
        
    print("-" * 105)


if __name__ == '__main__':
    main()
