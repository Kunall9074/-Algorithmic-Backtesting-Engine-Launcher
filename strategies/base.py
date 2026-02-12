"""
Base Strategy Class
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement:
    - init(): Called once before backtesting starts
    - on_data(): Called for each bar during backtesting
    
    Example:
        class MyStrategy(Strategy):
            def init(self, data):
                self.sma = data['Close'].rolling(20).mean()
            
            def on_data(self, data, position):
                if data['Close'].iloc[-1] > self.sma.iloc[len(data)-1]:
                    return {'action': 'buy', 'size': 0.1}
                return None
    """
    
    def __init__(self, **params):
        """
        Initialize strategy with parameters.
        
        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self.data = None
        self.indicators = {}
    
    def init(self, data: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data.
        Called once before backtesting starts.
        Override to compute indicators.
        
        Args:
            data: Full historical OHLCV DataFrame
        """
        self.data = data
    
    @abstractmethod
    def on_data(
        self, 
        data: pd.DataFrame, 
        position: 'Position'
    ) -> Optional[Dict[str, Any]]:
        """
        Process a new bar and generate trading signal.
        Called for each bar during backtesting.
        
        Args:
            data: Historical data up to current bar
            position: Current position in the asset
            
        Returns:
            Signal dict or None:
            - {'action': 'buy', 'size': 0.1} - Buy 10% of portfolio
            - {'action': 'sell'} - Sell entire position
            - None - No action
        """
        pass
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get strategy parameter."""
        return self.params.get(name, default)
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
