# 🚀 Algorithmic Backtesting Engine

A professional-grade Python backtesting framework designed for high-performance strategy testing and analysis.

## 🏗️ Core Architecture
This engine uses an **Event-Driven Design**. Unlike simple vectorized backtesters, this simulates a real-world trading loops where each "bar" (price update) is processed sequentially. This prevents "look-ahead bias" and ensures your strategy only uses information available at that specific point in time.

### Key Components:
- **DataLoader**: Handles data ingestion from CSV or Yahoo Finance.
- **Backtest Engine**: The "heart" that iterates through time, keeping track of cash and positions.
- **Portfolio**: Manages your balance, calculates commission, and tracks open trades.
- **Metrics**: Calculates complex risk metrics like Sharpe Ratio and Value at Risk (VaR).

---

## 🛠️ Installation & Setup

1. **Clone & Navigate**:
   ```bash
   cd backtesting_engine
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

---

## 📈 Trading Strategies Explained

### 1. SMA Crossover (Trend Following)
- **Logic**: Uses two Moving Averages (Fast & Slow).
- **Buy Signal**: When the Fast SMA crosses **above** the Slow SMA (Golden Cross).
- **Sell Signal**: When the Fast SMA crosses **below** the Slow SMA (Death Cross).
- *Best for: Trending markets.*

### 2. RSI Mean Reversion (Momentum)
- **Logic**: Measures the speed and change of price movements.
- **Buy Signal**: When RSI falls below **30** (Oversold).
- **Sell Signal**: When RSI rises above **70** (Overbought).
- *Best for: Ranging/Sideways markets.*

### 3. MACD (Momentum + Trend)
- **Logic**: Uses the difference between two EMAs and a signal line.
- **Buy Signal**: MACD line crosses above the Signal line.
- **Sell Signal**: MACD line crosses below the Signal line.

---

## 🤖 Machine Learning Integration
Located in `ml/predictor.py`, we use **Scikit-Learn** to forecast price movements.
- **Model**: Linear Regression.
- **Features**: Uses the last 5 days of closing prices (Lags) to predict the 6th day.
- **Usage**: You can train this model on historical data to filter out bad signals from your technical strategies.

---

## 📊 Understanding Your Results
| Metric | What it tells you |
|--------|-------------------|
| **Sharpe Ratio** | Is your profit worth the risk? (>1.0 is good). |
| **Max Drawdown** | The biggest "dip" your account took from its peak. |
| **Win Rate** | Percentage of trades that were profitable. |
| **VaR (95%)** | The maximum expected loss on a single day with 95% confidence. |

---

## 🚀 How to Run
- **Interactive Dashboard**: `run_dashboard.bat` or `streamlit run dashboard.py`
- **Full Test Suite**: `python test_all.py` (Tests all 9 modules at once)
- **Comparison Mode**: `python main.py --compare`
- **Real Market Data**: `python main.py --yahoo --symbol RELIANCE.NS`

---

## 📁 Project Structure
- `data/`: Ingestion logic.
- `engine/`: Execution logic.
- `strategies/`: Indicator logic.
- `visualization/`: Plotting (Plotly/Matplotlib).
- `ml/`: Predictive logic.

---

> [!IMPORTANT]
> **Disclaimer**: This is for educational use. Trading involves risk. Backtesting results do not guarantee future performance.
