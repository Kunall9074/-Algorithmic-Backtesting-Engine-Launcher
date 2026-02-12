"""
MACD Strategy
Moving Average Convergence Divergence algorithm
"""

import pandas as pd
from typing import Optional, Dict, Any
from .base import Strategy


class MACDStrategy(Strategy):
    """
    MACD Strategy.
    
    - BUY when MACD line crosses above Signal line
    - SELL when MACD line crosses below Signal line
    - Uses standard settings: Fast=12, Slow=26, Signal=9
    
    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        position_size: Fraction of portfolio to trade (default: 0.95)
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        position_size: float = 0.95
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            position_size=position_size
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_size = position_size
        
        self.macd = None
        self.signal = None
        self.hist = None
    
    def init(self, data: pd.DataFrame) -> None:
        """Calculate MACD indicators."""
        super().init(data)
        
        close = data['Close']
        
        # Calculate EMAs
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD Line
        self.macd = ema_fast - ema_slow
        
        # Calculate Signal Line
        self.signal = self.macd.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate Histogram
        self.hist = self.macd - self.signal
    
    def on_data(
        self,
        data: pd.DataFrame,
        position
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on MACD crossovers.
        """
        idx = len(data) - 1
        
        # Need enough data
        if idx < self.slow_period:
            return None
        
        current_macd = self.macd.iloc[idx]
        current_signal = self.signal.iloc[idx]
        prev_macd = self.macd.iloc[idx - 1]
        prev_signal = self.signal.iloc[idx - 1]
        
        # Bullish Crossover (Buy)
        if prev_macd <= prev_signal and current_macd > current_signal:
            if position.is_flat():
                return {'action': 'buy', 'size': self.position_size}
        
        # Bearish Crossover (Sell)
        elif prev_macd >= prev_signal and current_macd < current_signal:
            if position.is_long():
                return {'action': 'sell'}
        
        return None
