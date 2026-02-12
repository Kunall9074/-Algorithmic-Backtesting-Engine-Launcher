"""
Core Backtesting Engine
Executes trading strategies on historical data
"""

import pandas as pd
import numpy as np
from typing import Type, Dict, Any, Optional
from datetime import datetime

from .portfolio import Portfolio, OrderSide
from .metrics import PerformanceMetrics, calculate_metrics


class Backtest:
    """
    Main backtesting engine.
    
    Usage:
        bt = Backtest(data, strategy, initial_cash=100000)
        results = bt.run()
        print(results.metrics)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: 'Strategy',
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        symbol: str = 'ASSET'
    ):
        """
        Initialize backtesting engine.
        
        Args:
            data: OHLCV DataFrame with datetime index
            strategy: Strategy instance to test
            initial_cash: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            symbol: Symbol name for the asset
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.symbol = symbol
        
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage
        )
        
        self.results: Optional[BacktestResults] = None
        
    def run(self) -> 'BacktestResults':
        """
        Run the backtest.
        
        Returns:
            BacktestResults object with all results and metrics
        """
        # Reset portfolio
        self.portfolio.reset()
        
        # Initialize strategy
        self.strategy.init(self.data)
        
        # Iterate through each bar
        for i in range(len(self.data)):
            current_bar = self.data.iloc[:i+1]
            current_row = self.data.iloc[i]
            timestamp = self.data.index[i]
            current_price = current_row['Close']
            
            # Get current position
            position = self.portfolio.get_position(self.symbol)
            
            # Get signal from strategy
            signal = self.strategy.on_data(current_bar, position)
            
            # Execute signal
            if signal is not None:
                self._execute_signal(signal, current_price, timestamp, position)
            
            # Record equity
            self.portfolio.record_equity(
                timestamp, 
                {self.symbol: current_price}
            )
        
        # Calculate metrics
        equity_curve = self.portfolio.get_equity_curve()
        trades = self.portfolio.get_trades_df()
        
        metrics = calculate_metrics(
            equity_curve,
            trades,
            self.initial_cash
        )
        
        # Create results
        self.results = BacktestResults(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            data=self.data,
            initial_cash=self.initial_cash,
            portfolio=self.portfolio
        )
        
        return self.results
    
    def _execute_signal(
        self, 
        signal: Dict[str, Any], 
        price: float, 
        timestamp: datetime,
        position
    ) -> None:
        """Execute a trading signal."""
        action = signal.get('action')
        
        if action == 'buy':
            # Calculate position size
            size = signal.get('size', 0.1)  # Default 10% of portfolio
            
            if isinstance(size, float) and size <= 1:
                # Percentage of portfolio
                portfolio_value = self.portfolio.get_portfolio_value({self.symbol: price})
                quantity = int((portfolio_value * size) / price)
            else:
                quantity = int(size)
            
            if quantity > 0 and position.is_flat():
                self.portfolio.buy(self.symbol, quantity, price, timestamp)
                
        elif action == 'sell':
            if position.is_long():
                self.portfolio.sell(
                    self.symbol, 
                    position.quantity, 
                    price, 
                    timestamp
                )


class BacktestResults:
    """Container for backtest results."""
    
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: PerformanceMetrics,
        data: pd.DataFrame,
        initial_cash: float,
        portfolio: Portfolio
    ):
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = metrics
        self.data = data
        self.initial_cash = initial_cash
        self.portfolio = portfolio
    
    def summary(self) -> str:
        """Get summary of backtest results."""
        return str(self.metrics)
    
    def __repr__(self) -> str:
        return f"BacktestResults(trades={len(self.trades)}, return={self.metrics.total_return:.2%})"
