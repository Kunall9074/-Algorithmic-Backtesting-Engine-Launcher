"""
Performance Metrics Module
Calculates various performance and risk metrics for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Return metrics
    total_return: float = 0.0
    cagr: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Other
    exposure_time: float = 0.0
    
    # Risk Metrics Continued
    value_at_risk: float = 0.0  # 95% Confidence
    beta: float = 0.0 # Correlation to benchmark (assumed SPY/Similar)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'Total Return (%)': round(self.total_return * 100, 2),
            'CAGR (%)': round(self.cagr * 100, 2),
            'Volatility (%)': round(self.volatility * 100, 2),
            'Sharpe Ratio': round(self.sharpe_ratio, 2),
            'Sortino Ratio': round(self.sortino_ratio, 2),
            'Max Drawdown (%)': round(self.max_drawdown * 100, 2),
            'Max DD Duration (days)': self.max_drawdown_duration,
            'Value at Risk (95%)': round(self.value_at_risk * 100, 2),
            'Total Trades': self.total_trades,
            'Win Rate (%)': round(self.win_rate * 100, 2),
            'Profit Factor': round(self.profit_factor, 2),
            'Avg Win (%)': round(self.avg_win * 100, 2),
            'Avg Loss (%)': round(self.avg_loss * 100, 2),
            'Exposure Time (%)': round(self.exposure_time * 100, 2),
        }

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Daily returns series.
        confidence_level: Confidence level (0.05 for 95%).
        
    Returns:
        VaR float value (positive number representing loss %).
    """
    if len(returns) < 2:
        return 0.0
    return abs(np.percentile(returns, confidence_level * 100))

    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics_dict = self.to_dict()
        lines = ["\n===== Performance Metrics ====="]
        for key, value in metrics_dict.items():
            lines.append(f"{key}: {value}")
        lines.append("=" * 32)
        return "\n".join(lines)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.
    
    Args:
        equity_curve: DataFrame with 'equity' column and datetime index
        trades: DataFrame with trade history
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()
    
    if equity_curve.empty:
        return metrics
    
    equity = equity_curve['equity']
    
    # Calculate returns
    returns = equity.pct_change().dropna()
    
    # Total return
    final_value = equity.iloc[-1]
    metrics.total_return = (final_value - initial_capital) / initial_capital
    
    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    if years > 0:
        metrics.cagr = (final_value / initial_capital) ** (1 / years) - 1
    
    # Volatility (annualized)
    if len(returns) > 1:
        metrics.volatility = returns.std() * np.sqrt(252)
        metrics.value_at_risk = calculate_var(returns)
    
    # Sharpe Ratio
    if metrics.volatility > 0:
        excess_return = metrics.cagr - risk_free_rate
        metrics.sharpe_ratio = excess_return / metrics.volatility
    
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = negative_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics.sortino_ratio = (metrics.cagr - risk_free_rate) / downside_std
    
    # Max Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    metrics.max_drawdown = abs(drawdown.min())
    
    # Max Drawdown Duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        drawdown_periods = []
        start = None
        for i, (idx, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start is None:
                start = idx
            elif not is_dd and start is not None:
                drawdown_periods.append((start, idx))
                start = None
        if start is not None:
            drawdown_periods.append((start, equity.index[-1]))
        
        if drawdown_periods:
            max_duration = max((end - start).days for start, end in drawdown_periods)
            metrics.max_drawdown_duration = max_duration
    
    # Trade metrics
    if not trades.empty:
        metrics.total_trades = len(trades)
        
        # Calculate P&L for each round trip
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']
        
        # Simplified P&L calculation
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Match trades
            pnls = []
            buy_queue = list(buy_trades.itertuples())
            sell_queue = list(sell_trades.itertuples())
            
            i, j = 0, 0
            while i < len(buy_queue) and j < len(sell_queue):
                buy = buy_queue[i]
                sell = sell_queue[j]
                pnl = (sell.price - buy.price) / buy.price
                pnls.append(pnl)
                i += 1
                j += 1
            
            if pnls:
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p < 0]
                
                metrics.winning_trades = len(wins)
                metrics.losing_trades = len(losses)
                metrics.win_rate = len(wins) / len(pnls) if pnls else 0
                
                metrics.avg_win = np.mean(wins) if wins else 0
                metrics.avg_loss = np.mean(losses) if losses else 0
                metrics.avg_trade = np.mean(pnls)
                
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses)) if losses else 0
                metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Exposure time (percentage of time in market)
    if 'cash' in equity_curve.columns:
        in_market = equity_curve['equity'] != equity_curve['cash']
        metrics.exposure_time = in_market.sum() / len(equity_curve)
    
    return metrics


def calculate_drawdown_series(equity: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series.
    
    Args:
        equity: Series of portfolio values
        
    Returns:
        DataFrame with drawdown and underwater equity
    """
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    
    return pd.DataFrame({
        'equity': equity,
        'peak': cummax,
        'drawdown': drawdown
    })


def calculate_rolling_sharpe(
    returns: pd.Series, 
    window: int = 60,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Daily returns series
        window: Rolling window in days
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Rolling Sharpe ratio series
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    rolling_mean = excess_returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    
    return rolling_mean / rolling_std
