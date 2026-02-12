"""
RSI Strategy
Mean reversion strategy based on Relative Strength Index
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base import Strategy


class RSIStrategy(Strategy):
    """
    RSI-based Mean Reversion Strategy.
    
    - BUY when RSI falls below oversold level (indicates undervaluation)
    - SELL when RSI rises above overbought level (indicates overvaluation)
    
    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        position_size: Fraction of portfolio to trade (default: 0.95)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        position_size: float = 0.95
    ):
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            position_size=position_size
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size
        
        self.rsi = None
    
    def init(self, data: pd.DataFrame) -> None:
        """Calculate RSI indicator."""
        super().init(data)
        self.rsi = self._calculate_rsi(data['Close'], self.rsi_period)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI indicator.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Use exponential moving average for smoother RSI
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def on_data(
        self,
        data: pd.DataFrame,
        position
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on RSI levels.
        """
        idx = len(data) - 1
        
        # Need enough data for RSI
        if idx < self.rsi_period + 1:
            return None
        
        current_rsi = self.rsi.iloc[idx]
        prev_rsi = self.rsi.iloc[idx - 1]
        
        # Check for oversold condition (BUY)
        if prev_rsi >= self.oversold and current_rsi < self.oversold:
            if position.is_flat():
                return {'action': 'buy', 'size': self.position_size}
        
        # Check for overbought condition (SELL)
        elif prev_rsi <= self.overbought and current_rsi > self.overbought:
            if position.is_long():
                return {'action': 'sell'}
        
        return None
    
    def get_rsi_values(self) -> pd.Series:
        """Get RSI series for visualization."""
        return self.rsi
