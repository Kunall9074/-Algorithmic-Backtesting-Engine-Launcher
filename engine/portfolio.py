"""
Portfolio and Position Management Module
Handles tracking of positions, cash, and trade history
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import pandas as pd


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """Represents a completed trade."""
    timestamp: datetime
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        """Total value of the trade."""
        return self.quantity * self.price
    
    @property
    def cost(self) -> float:
        """Total cost including commission."""
        return self.value + self.commission


@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    entry_date: Any = None
    
    @property
    def market_value(self) -> float:
        """Current market value (requires current price)."""
        return self.quantity * self.avg_price
    
    def update(self, quantity: int, price: float, timestamp: Any = None) -> None:
        """Update position with a new trade."""
        if self.quantity == 0:
            self.avg_price = price
            self.quantity = quantity
            self.entry_date = timestamp
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to position
            total_cost = (self.quantity * self.avg_price) + (quantity * price)
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing position
            self.quantity += quantity
            if self.quantity == 0:
                self.avg_price = 0
                self.entry_date = None
    
    def is_long(self) -> bool:
        return self.quantity > 0
    
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        return self.quantity == 0


class Portfolio:
    """
    Portfolio manager for tracking positions, cash, and trades.
    """
    
    def __init__(
        self, 
        initial_cash: float = 100000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005    # 0.05% slippage
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission
        self.slippage_rate = slippage
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trades_history: List[Dict] = []  # Detailed PnL history
        self.equity_curve: List[Dict] = []
        
    def get_position(self, symbol: str) -> Position:
        """Get or create a position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def execute_order(
        self, 
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        timestamp: datetime
    ) -> Optional[Trade]:
        """
        Execute an order and update portfolio.
        """
        if quantity <= 0:
            return None
        
        # Apply slippage
        if side == OrderSide.BUY:
            exec_price = price * (1 + self.slippage_rate)
            signed_qty = quantity
        else:
            exec_price = price * (1 - self.slippage_rate)
            signed_qty = -quantity
        
        # Calculate costs
        trade_value = quantity * exec_price
        commission = trade_value * self.commission_rate
        
        # Check if we have enough cash for buy orders
        if side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                # Reduce quantity to fit budget
                quantity = int((self.cash - commission) / exec_price)
                if quantity <= 0:
                    return None
                trade_value = quantity * exec_price
                commission = trade_value * self.commission_rate
                signed_qty = quantity
        
        # PnL Calculation Logic
        position = self.get_position(symbol)
        if (side == OrderSide.SELL and position.is_long()) or (side == OrderSide.BUY and position.is_short()):
            # Closing or reducing position
            closed_qty = min(abs(position.quantity), quantity)
            
            # Calculate PnL
            if position.is_long():
                pnl = (exec_price - position.avg_price) * closed_qty - commission
                return_pct = ((exec_price - position.avg_price) / position.avg_price) * 100
            else:
                pnl = (position.avg_price - exec_price) * closed_qty - commission
                return_pct = ((position.avg_price - exec_price) / position.avg_price) * 100
                
            self.trades_history.append({
                'entry_date': position.entry_date,
                'exit_date': timestamp,
                'symbol': symbol,
                'side': 'Long' if position.is_long() else 'Short',
                'entry_price': position.avg_price,
                'exit_price': exec_price,
                'quantity': closed_qty,
                'pnl': pnl,
                'return_pct': return_pct,
                'commission': commission
            })

        # Update cash
        if side == OrderSide.BUY:
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)
        
        # Update position
        position.update(signed_qty, exec_price, timestamp)
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            side=side,
            quantity=quantity,
            price=exec_price,
            commission=commission
        )
        self.trades.append(trade)
        
        return trade
    
    def buy(self, symbol: str, quantity: int, price: float, timestamp: datetime) -> Optional[Trade]:
        """Place a buy order."""
        return self.execute_order(symbol, OrderSide.BUY, quantity, price, timestamp)
    
    def sell(self, symbol: str, quantity: int, price: float, timestamp: datetime) -> Optional[Trade]:
        """Place a sell order."""
        return self.execute_order(symbol, OrderSide.SELL, quantity, price, timestamp)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        """
        positions_value = sum(
            pos.quantity * current_prices.get(symbol, pos.avg_price)
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]) -> None:
        """Record current equity for equity curve."""
        equity = self.get_portfolio_value(current_prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [{
            'timestamp': t.timestamp,
            'side': t.side.value,
            'quantity': t.quantity,
            'price': t.price,
            'value': t.value,
            'commission': t.commission
        } for t in self.trades]
        
        return pd.DataFrame(trades_data)
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.trades_history.clear()
        self.equity_curve.clear()
