"""
Data Loader Module
Handles loading market data from CSV files and Yahoo Finance API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union
import os


class DataLoader:
    """
    Data loader for backtesting engine.
    Supports CSV files and Yahoo Finance API.
    """
    
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self):
        self.data = None
        self.symbol = None
        
    def load_csv(self, filepath: str, date_column: str = 'Date') -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            date_column: Name of the date column
            
        Returns:
            DataFrame with OHLCV data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Parse date column
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        # Validate columns
        self._validate_columns(df)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        self.data = df
        return df
    
    def load_yahoo(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = '1y'
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Alternative to start/end - '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Clean up columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        self.data = df
        self.symbol = symbol
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            # Try case-insensitive match
            col_map = {col.lower(): col for col in df.columns}
            for req_col in missing:
                if req_col.lower() in col_map:
                    df.rename(columns={col_map[req_col.lower()]: req_col}, inplace=True)
                else:
                    raise ValueError(f"Missing required column: {req_col}")
    
    def resample(self, timeframe: str = 'D') -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            timeframe: 'D' for daily, 'W' for weekly, 'M' for monthly
            
        Returns:
            Resampled DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_yahoo() first.")
        
        resampled = self.data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def add_returns(self) -> pd.DataFrame:
        """Add return columns to the data."""
        if self.data is None:
            raise ValueError("No data loaded.")
        
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        return self.data
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded.")
        return self.data.copy()


def generate_sample_data(
    days: int = 252, 
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005
) -> pd.DataFrame:
    """
    Generate synthetic stock data for testing.
    
    Args:
        days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility
        trend: Daily drift/trend
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Generate returns
    returns = np.random.normal(trend, volatility, days)
    
    # Generate prices
    close_prices = start_price * np.cumprod(1 + returns)
    
    # Generate OHLV based on close
    high_prices = close_prices * (1 + np.random.uniform(0, 0.02, days))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.02, days))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    
    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # Generate volume
    volume = np.random.randint(100000, 10000000, days)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return df
