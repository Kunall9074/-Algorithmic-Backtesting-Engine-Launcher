

# 🚀 Algorithmic Backtesting Engine

A Python framework for **backtesting trading strategies** and simulating real-market conditions. Designed to test ideas safely before going live.

---

## 🏗️ How It Works

The engine is **event-driven**, meaning it processes each price update sequentially, just like real markets. No “cheating” with future data — what your strategy sees is exactly what it would see in real trading.

**Main Components:**

* **DataLoader:** Load historical data from CSV or Yahoo Finance.
* **Backtest Engine:** Loops through market data, tracks cash, positions, and trades.
* **Portfolio:** Manages balances, commissions, and open positions.
* **Metrics:** Calculates Sharpe Ratio, Max Drawdown, VaR, and other risk measures.

---

## 🛠️ Setup

```bash
git clone <repo_url>
cd backtesting_engine
pip install -r requirements.txt
streamlit run dashboard.py
```

Open the dashboard in your browser to see results and charts.

---

## 📈 Strategies Included

1. **SMA Crossover (Trend Following)**

   * Buy: Fast SMA crosses above Slow SMA
   * Sell: Fast SMA crosses below Slow SMA

2. **RSI Mean Reversion (Momentum)**

   * Buy: RSI < 30 (oversold)
   * Sell: RSI > 70 (overbought)

3. **MACD (Trend + Momentum)**

   * Buy: MACD crosses above signal line
   * Sell: MACD crosses below signal line

---

## 🤖 Machine Learning

A simple Linear Regression model predicts next-day prices using the last 5 days. Helps **filter trades with low probability** and improve strategy performance.

* File: `ml/predictor.py`
* Library: Scikit-Learn

---

## 📊 Metrics to Track

| Metric       | Meaning                 | Goal             |
| ------------ | ----------------------- | ---------------- |
| Sharpe Ratio | Risk-adjusted returns   | > 1              |
| Max Drawdown | Worst drop from peak    | Lower is better  |
| Win Rate     | % of profitable trades  | Higher is better |
| VaR (95%)    | Max expected daily loss | Lower is safer   |

---

## 🚀 Quick Commands

```bash
# Run dashboard
streamlit run dashboard.py

# Run all tests
python test_all.py

# Compare strategies
python main.py --compare

# Backtest specific symbol from Yahoo Finance
python main.py --yahoo --symbol RELIANCE.NS
```

---

## 📁 Project Structure

```
backtesting_engine/
├─ data/           # Historical data
├─ engine/         # Backtesting core logic
├─ strategies/     # Trading strategies
├─ visualization/  # Charts & plots
├─ ml/             # Predictive models
├─ dashboard.py    # Interactive dashboard
├─ main.py         # Run backtests
```

---

> Designed for developers and traders who want a **fast, accurate, and transparent way** to test strategies before risking real money.

---

Do you want me to do that?
