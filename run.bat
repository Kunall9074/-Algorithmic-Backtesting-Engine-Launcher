@echo off
echo ==========================================
echo    Algorithmic Backtesting Engine Launcher
echo ==========================================
echo.
echo 1. Run basic backtest (Sample Data)
echo 2. Run with Yahoo Finance (RELIANCE.NS)
echo 3. Run RSI Strategy
echo 4. Run Interactive Mode
echo 5. COMPARE STRATEGIES (New!)
echo 6. LAUNCH DASHBOARD (Streamlit)
echo.
set /p choice="Select an option (1-6): "

if "%choice%"=="1" (
    python main.py
) else if "%choice%"=="2" (
    python main.py --yahoo --symbol RELIANCE.NS
) else if "%choice%"=="3" (
    python main.py --strategy rsi
) else if "%choice%"=="4" (
    python main.py --interactive
) else if "%choice%"=="5" (
    python main.py --compare
) else if "%choice%"=="6" (
    streamlit run dashboard.py
) else (
    echo Invalid choice
    python main.py
)

pause
