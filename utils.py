"""
utils.py
Core utilities for the ANN Stock Trend Predictor:
- Data fetching (yfinance)
- Technical indicator / feature engineering
- Dataset preparation (scaling, train/test split)
- ANN model construction (Keras)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------
# DATA FETCHING
# --------------------------------------------------------------------------
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a ticker between start and end dates."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")
    # Flatten MultiIndex columns if present (happens with some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).capitalize() for c in df.columns]
    return df


# --------------------------------------------------------------------------
# TECHNICAL INDICATORS (implemented manually -> no extra heavy dependency)
# --------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a set of technical indicator features to the price dataframe."""
    data = df.copy()

    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

    data["RSI_14"] = _rsi(data["Close"], 14)

    macd_line, signal_line = _macd(data["Close"])
    data["MACD"] = macd_line
    data["MACD_signal"] = signal_line
    data["MACD_hist"] = macd_line - signal_line

    data["BB_mid"] = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_upper"] = data["BB_mid"] + 2 * bb_std
    data["BB_lower"] = data["BB_mid"] - 2 * bb_std
    data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / (data["BB_mid"] + 1e-9)

    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Volatility_10"] = data["Close"].rolling(window=10).std()
    data["Return_1d"] = data["Close"].pct_change()
    data["Volume_change"] = data["Volume"].pct_change()

    # Target: 1 if next day's close is higher than today's close, else 0
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    data = data.dropna().reset_index(drop=True)
    return data


FEATURE_COLUMNS = [
    "SMA_10", "SMA_50", "EMA_10", "EMA_50",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "Momentum_10", "Volatility_10",
    "Return_1d", "Volume_change",
]


# --------------------------------------------------------------------------
# DATASET PREPARATION
# --------------------------------------------------------------------------
def prepare_dataset(data: pd.DataFrame, test_size: float = 0.2):
    """Scale features and split into train/test sets (chronological split)."""
    X = data[FEATURE_COLUMNS].values
    y = data["Target"].values

    split_idx = int(len(X) * (1 - test_size))
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------------------------------
# ANN MODEL
# --------------------------------------------------------------------------
def build_ann_model(input_dim: int, hidden_layers=(128, 64, 32), dropout: float = 0.3, lr: float = 0.001):
    """Build and compile a feed-forward Artificial Neural Network (Keras Sequential)."""
    model = keras.Sequential(name="ANN_Trend_Predictor")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the ANN with early stopping and return the training history."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    return history
