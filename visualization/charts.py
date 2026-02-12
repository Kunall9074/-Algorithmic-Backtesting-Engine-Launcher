"""
Visualization Module
Charts and plots for backtesting results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class BacktestVisualizer:
    """
    Visualization tools for backtest results.
    """
    
    def __init__(self, results: 'BacktestResults'):
        """
        Initialize visualizer with backtest results.
        
        Args:
            results: BacktestResults from Backtest.run()
        """
        self.results = results
        self.equity_curve = results.equity_curve
        self.trades = results.trades
        self.data = results.data
        self.metrics = results.metrics
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'equity': '#2E86AB',
            'benchmark': '#A23B72',
            'drawdown': '#E94F37',
            'buy': '#2ECC71',
            'sell': '#E74C3C',
            'positive': '#27AE60',
            'negative': '#C0392B'
        }
    
    def plot_equity_curve(
        self, 
        benchmark: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve with optional benchmark.
        
        Args:
            benchmark: Optional benchmark returns series
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity
        equity = self.equity_curve['equity']
        ax.plot(equity.index, equity.values, 
                label='Strategy', color=self.colors['equity'], linewidth=2)
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values,
                    label='Benchmark', color=self.colors['benchmark'], 
                    linewidth=2, linestyle='--')
        
        # Format
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (14, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown over time.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate drawdown
        equity = self.equity_curve['equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0,
                       color=self.colors['drawdown'], alpha=0.5)
        ax.plot(drawdown.index, drawdown.values,
               color=self.colors['drawdown'], linewidth=1)
        
        # Format
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Max drawdown annotation
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.annotate(f'Max: {max_dd_val:.1f}%',
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(max_dd_idx, max_dd_val - 5),
                   fontsize=10, color=self.colors['drawdown'])
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trades(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot price chart with trade markers.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        price = self.data['Close']
        ax.plot(price.index, price.values, 
                label='Price', color='#34495E', linewidth=1.5)
        
        # Plot trades
        if not self.trades.empty:
            buys = self.trades[self.trades['side'] == 'buy']
            sells = self.trades[self.trades['side'] == 'sell']
            
            if not buys.empty:
                ax.scatter(buys['timestamp'], buys['price'],
                          marker='^', s=100, color=self.colors['buy'],
                          label='Buy', zorder=5)
            
            if not sells.empty:
                ax.scatter(sells['timestamp'], sells['price'],
                          marker='v', s=100, color=self.colors['sell'],
                          label='Sell', zorder=5)
        
        # Format
        ax.set_title('Price Chart with Trades', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate monthly returns
        equity = self.equity_curve['equity']
        monthly = equity.resample('ME').last().pct_change() * 100
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color array
        colors = np.where(pivot.values > 0, 
                         self.colors['positive'], 
                         self.colors['negative'])
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                      vmin=-10, vmax=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', fontsize=12)
        
        # Set ticks
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(12))
        ax.set_xticklabels(months[:len(pivot.columns)])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f'{val:.1f}%',
                           ha='center', va='center',
                           color='white' if abs(val) > 5 else 'black',
                           fontsize=9)
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_summary(
        self,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
        
        # 1. Equity Curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        equity = self.equity_curve['equity']
        ax1.plot(equity.index, equity.values, 
                color=self.colors['equity'], linewidth=2)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color=self.colors['drawdown'], alpha=0.5)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if not self.trades.empty:
            buys = len(self.trades[self.trades['side'] == 'buy'])
            sells = len(self.trades[self.trades['side'] == 'sell'])
            ax3.bar(['Buy', 'Sell'], [buys, sells],
                   color=[self.colors['buy'], self.colors['sell']])
        ax3.set_title('Trade Distribution', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count')
        
        # 4. Metrics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        metrics_dict = self.metrics.to_dict()
        metrics_text = "  |  ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
        
        # Split into two rows
        items = list(metrics_dict.items())
        mid = len(items) // 2
        row1 = "  |  ".join([f"{k}: {v}" for k, v in items[:mid]])
        row2 = "  |  ".join([f"{k}: {v}" for k, v in items[mid:]])
        
        ax4.text(0.5, 0.7, row1, fontsize=10, ha='center', va='center',
                family='monospace', transform=ax4.transAxes)
        ax4.text(0.5, 0.3, row2, fontsize=10, ha='center', va='center',
                family='monospace', transform=ax4.transAxes)
        ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        plt.suptitle('Backtest Summary', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def interactive_chart(results: 'BacktestResults') -> None:
    """
    Create interactive chart using Plotly.
    
    Args:
        results: BacktestResults from Backtest.run()
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price & Trades', 'Equity Curve', 'Drawdown')
    )
    
    # Price with candlesticks
    fig.add_trace(
        go.Candlestick(
            x=results.data.index,
            open=results.data['Open'],
            high=results.data['High'],
            low=results.data['Low'],
            close=results.data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add trades
    if not results.trades.empty:
        buys = results.trades[results.trades['side'] == 'buy']
        sells = results.trades[results.trades['side'] == 'sell']
        
        fig.add_trace(
            go.Scatter(
                x=buys['timestamp'],
                y=buys['price'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sells['timestamp'],
                y=sells['price'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell'
            ),
            row=1, col=1
        )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=results.equity_curve.index,
            y=results.equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#2E86AB', width=2)
        ),
        row=2, col=1
    )
    
    # Drawdown
    equity = results.equity_curve['equity']
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#E94F37', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title='Backtest Results (Interactive)',
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.show()
