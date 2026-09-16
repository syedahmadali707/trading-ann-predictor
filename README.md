# 📈 ANN Stock Trend Predictor

A deep learning project that uses an **Artificial Neural Network (ANN)** to
predict the next-day price direction (Up/Down) of a stock, wrapped in a
polished **Streamlit** dashboard.

## Features
- Live historical data fetch via `yfinance` (any ticker: AAPL, TSLA, BTC-USD, etc.)
- Automatic technical-indicator feature engineering (SMA, EMA, RSI, MACD, Bollinger Bands, Momentum, Volatility)
- Configurable ANN (Keras/TensorFlow) — layers, neurons, dropout, epochs, batch size — trained live in the browser
- Interactive candlestick charts, correlation heatmap, training curves, confusion matrix, ROC curve
- Live next-day Up/Down prediction with a confidence gauge
- Clean, dark, custom-styled UI

## Project structure
```
trading-ann-predictor/
├── app.py              # Streamlit app (UI)
├── utils.py             # Data fetching, feature engineering, ANN model
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to use
1. In the sidebar, enter a ticker (e.g. `AAPL`) and a date range, then click **Fetch Data**.
2. Explore the **Overview** and **Features** tabs to see the price chart and engineered indicators.
3. Adjust ANN hyperparameters in the sidebar and click **Train ANN Model**.
4. Check the **Training** tab for accuracy/loss curves, confusion matrix and ROC curve.
5. Go to the **Prediction** tab to see tomorrow's predicted direction and confidence.

## Notes
- TensorFlow install can take a few minutes and is large (~500MB+). If you only
  have a lightweight environment, consider `tensorflow-cpu` instead of `tensorflow`
  in `requirements.txt`.
- The target label is binary: `1` if next day's close > today's close, else `0`.
- Train/test split is **chronological** (not shuffled) to avoid lookahead bias.
- This project is for educational purposes only — **not financial advice**.
