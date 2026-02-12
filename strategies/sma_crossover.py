"""
SMA Crossover Strategy
Classic Moving Average crossover trading strategy
"""

import pandas as pd
from typing import Optional, Dict, Any
from .base import Strategy


class SMACrossoverStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    
    - BUY when fast SMA crosses above slow SMA (Golden Cross)
    - SELL when fast SMA crosses below slow SMA (Death Cross)
    
    Parameters:
        fast_period: Fast moving average period (default: 10)
        slow_period: Slow moving average period (default: 30)
        position_size: Fraction of portfolio to trade (default: 0.95)
    """
    
    def __init__(
        self, 
        fast_period: int = 10, 
        slow_period: int = 30,
        position_size: float = 0.95
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            position_size=position_size
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        
        self.fast_sma = None
        self.slow_sma = None
    
    def init(self, data: pd.DataFrame) -> None:
        """Calculate moving averages."""
        super().init(data)
        
        close = data['Close']
        self.fast_sma = close.rolling(window=self.fast_period).mean()
        self.slow_sma = close.rolling(window=self.slow_period).mean()
    
    def on_data(
        self, 
        data: pd.DataFrame, 
        position
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on SMA crossover.
        """
        idx = len(data) - 1
        
        # Need enough data for both SMAs
        if idx < self.slow_period:
            return None
        
        fast = self.fast_sma.iloc[idx]
        slow = self.slow_sma.iloc[idx]
        fast_prev = self.fast_sma.iloc[idx - 1]
        slow_prev = self.slow_sma.iloc[idx - 1]
        
        # Check for crossover
        if fast_prev <= slow_prev and fast > slow:
            # Golden cross - BUY signal
            if position.is_flat():
                return {'action': 'buy', 'size': self.position_size}
        
        elif fast_prev >= slow_prev and fast < slow:
            # Death cross - SELL signal
            if position.is_long():
                return {'action': 'sell'}
        
        return None
