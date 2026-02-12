"""
Machine Learning Module
Simple price prediction using Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class PricePredictor:
    """
    Price predictor using Linear Regression.
    Predicts next day's close price based on recent price action.
    """
    
    def __init__(self, lookback: int = 5):
        """
        Initialize predictor.
        
        Args:
            lookback: Number of past days to use for prediction
        """
        self.lookback = lookback
        self.model = LinearRegression()
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features (X) and target (y) for training.
        """
        df = data.copy()
        df['Target'] = df['Close'].shift(-1)  # Next day's price
        
        # Create features: Past N days closing prices
        for i in range(self.lookback):
            df[f'Lag_{i+1}'] = df['Close'].shift(i+1)
            
        df.dropna(inplace=True)
        
        feature_cols = [f'Lag_{i+1}' for i in range(self.lookback)]
        X = df[feature_cols].values
        y = df['Target'].values
        
        return X, y
    
    def train(self, data: pd.DataFrame) -> float:
        """
        Train the model.
        
        Returns:
            R-squared score of the model
        """
        X, y = self.prepare_data(data)
        
        if len(X) < 10:
            return 0.0
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return self.model.score(X_test, y_test)
    
    def predict_next(self, recent_data: pd.DataFrame) -> Optional[float]:
        """
        Predict next day's closing price.
        
        Args:
            recent_data: DataFrame containing at least 'lookback' rows
            
        Returns:
            Predicted price or None if not enough data
        """
        if not self.is_trained:
            return None
            
        if len(recent_data) < self.lookback:
            return None
            
        # Extract last 'lookback' days to create feature vector
        last_prices = recent_data['Close'].iloc[-self.lookback:].values
        # Feature vector expected order: Lag_1 (most recent), Lag_2, ...
        # last_prices is [Day-4, Day-3, Day-2, Day-1, Day-0]
        # We need to reverse it to match Lag_1, Lag_2...
        feature_vector = last_prices[::-1].reshape(1, -1)
        
        prediction = self.model.predict(feature_vector)[0]
        return prediction
