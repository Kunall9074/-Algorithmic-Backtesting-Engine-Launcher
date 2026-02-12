import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import DataLoader, generate_sample_data
from engine.backtest import Backtest
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy

st.set_page_config(
    page_title="AlgoTrade Pro | Backtester",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Light Theme CSS ---
st.markdown("""<style>
.stApp { background-color: #f0f2f6; }
section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
.metric-card { background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e8e8e8; margin-bottom: 15px; }
.metric-title { color: #6b7280; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.metric-val { color: #111827; font-size: 1.8rem; font-weight: 700; }
.metric-pos { color: #059669; font-size: 0.85rem; font-weight: 600; }
.metric-neg { color: #dc2626; font-size: 0.85rem; font-weight: 600; }
h1, h2, h3 { color: #1f2937; }
.stButton>button { background-color: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-weight: 600; width: 100%; }
.stButton>button:hover { background-color: #1d4ed8; }
.feature-card { background: white; border-radius: 12px; padding: 25px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e8e8e8; }
</style>""", unsafe_allow_html=True)

# --- Helpers ---
def render_metric(label, value, delta=None, suffix=""):
    dhtml = ""
    if delta is not None and isinstance(delta, (int, float)):
        cls = "metric-pos" if delta >= 0 else "metric-neg"
        sign = "+" if delta >= 0 else ""
        dhtml = f'<div class="{cls}">{sign}{delta}{suffix}</div>'
    st.markdown(f'<div class="metric-card"><div class="metric-title">{label}</div><div class="metric-val">{value}{suffix}</div>{dhtml}</div>', unsafe_allow_html=True)

def calc_monthly_returns(eq):
    if eq.empty:
        return pd.DataFrame()
    m = eq.resample('ME')['equity'].last()
    mr = m.pct_change() * 100
    mr.index = pd.to_datetime(mr.index)
    df = pd.DataFrame({'Year': mr.index.year, 'Month': mr.index.month, 'Return': mr.values})
    pt = df.pivot(index='Year', columns='Month', values='Return')
    pt.columns = [pd.to_datetime(f"2000-{c}-01").strftime('%b') for c in pt.columns]
    return pt

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚡ AlgoTrade Pro")
    st.caption("Professional Backtesting Suite")
    st.markdown("---")

    STOCKS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "TATAMOTORS.NS", "AXISBANK.NS", "MARUTI.NS", "TITAN.NS",
        "BAJFINANCE.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "M&M.NS", "ULTRACEMCO.NS",
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"
    ]

    st.markdown("**📊 Market Data**")
    data_source = st.radio("Source", ["Yahoo Finance", "Sample Data"], horizontal=True)
    if data_source == "Yahoo Finance":
        symbol = st.selectbox("Stock Symbol", STOCKS, index=0)
    else:
        symbol = "SAMPLE"
    initial_cash = st.number_input("Initial Capital (₹)", value=100000, step=10000)

    st.markdown("---")
    st.markdown("**🧠 Strategy**")
    strategy_name = st.selectbox("Select Strategy", ["SMA Crossover", "RSI Mean Reversion", "MACD"])

    if strategy_name == "SMA Crossover":
        c1, c2 = st.columns(2)
        fast = c1.number_input("Fast SMA", 5, 50, 10)
        slow = c2.number_input("Slow SMA", 20, 200, 30)
        strategy = SMACrossoverStrategy(fast_period=fast, slow_period=slow)
    elif strategy_name == "RSI Mean Reversion":
        period = st.slider("RSI Period", 5, 30, 14)
        c1, c2 = st.columns(2)
        oversold = c1.number_input("Oversold", 10, 40, 30)
        overbought = c2.number_input("Overbought", 60, 90, 70)
        strategy = RSIStrategy(rsi_period=period, oversold=oversold, overbought=overbought)
    elif strategy_name == "MACD":
        c1, c2 = st.columns(2)
        fast_ema = c1.number_input("Fast EMA", 5, 30, 12)
        slow_ema = c2.number_input("Slow EMA", 20, 100, 26)
        signal = st.slider("Signal Period", 5, 20, 9)
        strategy = MACDStrategy(fast_period=fast_ema, slow_period=slow_ema, signal_period=signal)

    st.markdown("---")
    run_btn = st.button("🚀 Run Backtest")

# --- Main ---
if run_btn:
    with st.spinner("Running backtest..."):
        loader = DataLoader()
        if data_source == "Sample Data":
            data = generate_sample_data(days=730)
        else:
            try:
                data = loader.load_yahoo(symbol, period="2y")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        bt = Backtest(data, strategy, initial_cash)
        results = bt.run()
        metrics = results.metrics.to_dict()

        st.markdown(f"## 📊 Analysis: {symbol}")
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric("Total Return", round(metrics['Total Return (%)'], 2), round(metrics['Total Return (%)'], 2), "%")
        with c2:
            render_metric("Win Rate", round(metrics['Win Rate (%)'], 1), suffix="%")
        with c3:
            render_metric("Sharpe Ratio", round(metrics['Sharpe Ratio'], 2))
        with c4:
            dd = round(metrics['Max Drawdown (%)'], 2)
            render_metric("Max Drawdown", dd, -dd, "%")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Trades", "📉 Drawdown", "🗓️ Monthly Returns", "📝 Trade Log"])

        with tab1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='OHLC'), secondary_y=True)
            fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve['equity'], name="Equity", line=dict(color="#2563eb", width=2), mode='lines'), secondary_y=False)
            tdf = results.trades
            if not tdf.empty:
                buys = tdf[tdf['side'] == 'buy']
                sells = tdf[tdf['side'] == 'sell']
                fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#059669'), name='Buy'), secondary_y=True)
                fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#dc2626'), name='Sell'), secondary_y=True)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0.0))
            fig.update_yaxes(title="Equity", secondary_y=False, showgrid=False)
            fig.update_yaxes(title="Price", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            eq = results.equity_curve['equity']
            dd_series = (eq - eq.cummax()) / eq.cummax() * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='#dc2626', width=1), name='Drawdown'))
            fig_dd.update_layout(height=400, template="plotly_white", yaxis_title="Drawdown %")
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab3:
            mr = calc_monthly_returns(results.equity_curve)
            if not mr.empty:
                fig_hm = px.imshow(mr, labels=dict(x="Month", y="Year", color="Return %"), color_continuous_scale="RdYlGn", text_auto=".2f")
                fig_hm.update_layout(height=400)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Not enough data for heatmap")

        with tab4:
            if hasattr(results, 'portfolio') and results.portfolio.trades_history:
                df_trades = pd.DataFrame(results.portfolio.trades_history)
            elif not results.trades.empty:
                trades = results.trades
                pnl_records = []
                buy_list = []
                for _, row in trades.iterrows():
                    if row['side'] == 'buy':
                        buy_list.append(row)
                    elif row['side'] == 'sell' and buy_list:
                        bt_row = buy_list.pop(0)
                        pnl = (row['price'] - bt_row['price']) * row['quantity'] - row['commission'] - bt_row['commission']
                        pct = ((row['price'] - bt_row['price']) / bt_row['price']) * 100
                        pnl_records.append({'entry_date': bt_row['timestamp'], 'exit_date': row['timestamp'], 'symbol': symbol, 'side': 'Long', 'entry_price': bt_row['price'], 'exit_price': row['price'], 'quantity': row['quantity'], 'pnl': pnl, 'return_pct': pct})
                df_trades = pd.DataFrame(pnl_records)
            else:
                df_trades = pd.DataFrame()

            if not df_trades.empty:
                if 'entry_date' in df_trades:
                    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date']).dt.strftime('%Y-%m-%d')
                if 'exit_date' in df_trades:
                    df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date']).dt.strftime('%Y-%m-%d')
                def color_pnl(val):
                    return 'background-color: #dcfce7; color: #166534' if val > 0 else 'background-color: #fee2e2; color: #991b1b'
                st.dataframe(df_trades.style.map(color_pnl, subset=['pnl', 'return_pct']).format({'pnl': '₹{:.2f}', 'return_pct': '{:.2f}%', 'entry_price': '{:.2f}', 'exit_price': '{:.2f}'}), use_container_width=True, height=500)

                wins = len(df_trades[df_trades['pnl'] > 0])
                losses = len(df_trades[df_trades['pnl'] <= 0])
                total_pnl = df_trades['pnl'].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total PnL", f"₹{total_pnl:.2f}")
                c2.metric("Winning Trades", wins)
                c3.metric("Losing Trades", losses)
            else:
                st.info("No completed trades.")
else:
    # Landing
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 📈 AlgoTrade Pro")
        st.markdown("#### Professional Algorithmic Backtesting Suite")
        st.markdown("")
        st.markdown("Analyze stocks, test strategies, and optimize returns.")
        st.markdown("")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown('<div class="feature-card">🧠<br><b>Smart Strategies</b></div>', unsafe_allow_html=True)
        with fc2:
            st.markdown('<div class="feature-card">⚡<br><b>Fast Execution</b></div>', unsafe_allow_html=True)
        with fc3:
            st.markdown('<div class="feature-card">📊<br><b>Deep Analytics</b></div>', unsafe_allow_html=True)
        st.markdown("")
        st.info("👈 Select your configuration from the sidebar and click **Run Backtest** to begin.")
