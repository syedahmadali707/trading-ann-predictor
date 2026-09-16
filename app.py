"""
Stock Trend Predictor — ANN powered Streamlit App
Predicts next-day stock price direction (Up / Down) using an
Artificial Neural Network trained on technical indicators.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from utils import (
    fetch_data,
    add_technical_indicators,
    prepare_dataset,
    build_ann_model,
    train_ann,
    FEATURE_COLUMNS,
)

# --------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ANN Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #12151d 100%); }

    h1, h2, h3 { color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1c2333 0%, #232a3d 100%);
        border: 1px solid #2d3548;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-card h3 { font-size: 13px; color: #9aa4b2; margin-bottom: 6px; font-weight: 500; }
    .metric-card p { font-size: 26px; font-weight: 700; margin: 0; }

    .up { color: #00d68f; }
    .down { color: #ff4d6d; }
    .neutral { color: #f0f2f6; }

    .signal-box {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .signal-up { background: linear-gradient(135deg, #00382a 0%, #024a35 100%); border: 1px solid #00d68f; }
    .signal-down { background: linear-gradient(135deg, #3a0d18 0%, #4a0f1f 100%); border: 1px solid #ff4d6d; }

    .signal-box h1 { font-size: 42px; margin: 0; }
    .signal-box p { color: #c3c9d4; font-size: 15px; margin-top: 6px; }

    section[data-testid="stSidebar"] { background-color: #12151d; border-right: 1px solid #232a3d; }

    div.stButton > button {
        background: linear-gradient(135deg, #4f6df5 0%, #3752d6 100%);
        color: white; border: none; border-radius: 10px;
        padding: 10px 18px; font-weight: 600; width: 100%;
    }
    div.stButton > button:hover { background: linear-gradient(135deg, #5c78ff 0%, #4560e8 100%); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
for key in ["data", "featured_data", "model", "history", "scaler", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --------------------------------------------------------------------------
# SIDEBAR — CONTROLS
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker = st.text_input("Stock Ticker", value="AAPL", help="e.g. AAPL, MSFT, TSLA, BTC-USD").upper().strip()

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=5 * 365))
    with col_b:
        end_date = st.date_input("End date", value=dt.date.today())

    st.markdown("---")
    st.markdown("### 🧠 ANN Hyperparameters")
    n_layers = st.select_slider("Hidden layers", options=[1, 2, 3, 4], value=3)
    neurons = st.select_slider("Neurons per layer", options=[16, 32, 64, 128, 256], value=64)
    dropout = st.slider("Dropout rate", 0.0, 0.6, 0.3, 0.05)
    epochs = st.slider("Epochs", 10, 200, 60, 10)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    st.markdown("---")
    fetch_clicked = st.button("📥 Fetch Data")
    train_clicked = st.button("🚀 Train ANN Model")

    st.markdown("---")
    st.caption("⚠️ Educational project only. Not financial advice.")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("# 📈 ANN Stock Trend Predictor")
st.markdown(
    "Predict next-day stock price **direction** using an "
    "**Artificial Neural Network** trained on technical indicators "
    "(SMA, EMA, RSI, MACD, Bollinger Bands, Momentum, Volatility)."
)

# --------------------------------------------------------------------------
# FETCH DATA
# --------------------------------------------------------------------------
if fetch_clicked:
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            raw = fetch_data(ticker, str(start_date), str(end_date))
            featured = add_technical_indicators(raw)
        st.session_state.data = raw
        st.session_state.featured_data = featured
        st.session_state.model = None
        st.session_state.results = None
        st.success(f"Loaded {len(raw)} rows for {ticker} ({len(featured)} usable after feature engineering).")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_overview, tab_features, tab_train, tab_predict, tab_about = st.tabs(
    ["📊 Overview", "🧮 Features", "🏋️ Training", "🔮 Prediction", "ℹ️ About"]
)

# ---- OVERVIEW TAB ---------------------------------------------------------
with tab_overview:
    if st.session_state.data is None:
        st.info("👈 Enter a ticker and click **Fetch Data** in the sidebar to get started.")
    else:
        df = st.session_state.data
        c1, c2, c3, c4 = st.columns(4)
        last_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        change_pct = (last_close - prev_close) / prev_close * 100
        change_class = "up" if change_pct >= 0 else "down"

        c1.markdown(f"""<div class="metric-card"><h3>LAST CLOSE</h3><p class="neutral">${last_close:,.2f}</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><h3>DAILY CHANGE</h3><p class="{change_class}">{change_pct:+.2f}%</p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><h3>52W HIGH</h3><p class="neutral">${df['High'].tail(252).max():,.2f}</p></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card"><h3>52W LOW</h3><p class="neutral">${df['Low'].tail(252).min():,.2f}</p></div>""", unsafe_allow_html=True)

        st.markdown("### Price Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#00d68f", decreasing_line_color="#ff4d6d", name=ticker,
        )])
        fig.update_layout(
            template="plotly_dark", height=480, xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View raw data"):
            st.dataframe(df.tail(200), use_container_width=True)

# ---- FEATURES TAB -----------------------------------------------------------
with tab_features:
    if st.session_state.featured_data is None:
        st.info("Fetch data first to see engineered features.")
    else:
        fd = st.session_state.featured_data
        st.markdown("### Engineered Feature Set")
        st.dataframe(fd[["Date"] + FEATURE_COLUMNS + ["Target"]].tail(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RSI (14)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=fd["Date"], y=fd["RSI_14"], line=dict(color="#4f6df5")))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4d6d")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00d68f")
            fig_rsi.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            st.markdown("#### MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=fd["Date"], y=fd["MACD"], name="MACD", line=dict(color="#4f6df5")))
            fig_macd.add_trace(go.Scatter(x=fd["Date"], y=fd["MACD_signal"], name="Signal", line=dict(color="#ff9f43")))
            fig_macd.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_macd, use_container_width=True)

        st.markdown("#### Feature Correlation Heatmap")
        corr = fd[FEATURE_COLUMNS].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
        ))
        fig_corr.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_corr, use_container_width=True)

        target_balance = fd["Target"].value_counts(normalize=True) * 100
        st.markdown(
            f"**Class balance:** Up days = {target_balance.get(1, 0):.1f}% | "
            f"Down days = {target_balance.get(0, 0):.1f}%"
        )

# ---- TRAINING TAB -----------------------------------------------------------
with tab_train:
    if st.session_state.featured_data is None:
        st.info("Fetch data first, then click **Train ANN Model** in the sidebar.")
    elif train_clicked:
        with st.spinner("Training Artificial Neural Network..."):
            fd = st.session_state.featured_data
            X_train, X_test, y_train, y_test, scaler = prepare_dataset(fd, test_size=test_size)

            hidden_layers = tuple([neurons] * n_layers)
            model = build_ann_model(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout)
            history = train_ann(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

            y_pred_prob = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_pred_prob >= 0.5).astype(int)

            st.session_state.model = model
            st.session_state.history = history
            st.session_state.scaler = scaler
            st.session_state.results = dict(
                X_test=X_test, y_test=y_test, y_pred=y_pred, y_pred_prob=y_pred_prob,
            )
        st.success("Training complete! Explore results below and check the Prediction tab.")

    if st.session_state.history is not None:
        history = st.session_state.history
        results = st.session_state.results

        acc = accuracy_score(results["y_test"], results["y_pred"])
        prec = precision_score(results["y_test"], results["y_pred"], zero_division=0)
        rec = recall_score(results["y_test"], results["y_pred"], zero_division=0)
        f1 = f1_score(results["y_test"], results["y_pred"], zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card"><h3>ACCURACY</h3><p class="neutral">{acc*100:.1f}%</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><h3>PRECISION</h3><p class="neutral">{prec*100:.1f}%</p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><h3>RECALL</h3><p class="neutral">{rec*100:.1f}%</p></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card"><h3>F1 SCORE</h3><p class="neutral">{f1*100:.1f}%</p></div>""", unsafe_allow_html=True)

        st.markdown("### Training Curves")
        col1, col2 = st.columns(2)
        with col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss", line=dict(color="#4f6df5")))
            fig_loss.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss", line=dict(color="#ff4d6d")))
            fig_loss.update_layout(title="Loss", template="plotly_dark", height=320,
                                    margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_loss, use_container_width=True)
        with col2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=history.history["accuracy"], name="Train Acc", line=dict(color="#00d68f")))
            fig_acc.add_trace(go.Scatter(y=history.history["val_accuracy"], name="Val Acc", line=dict(color="#ff9f43")))
            fig_acc.update_layout(title="Accuracy", template="plotly_dark", height=320,
                                   margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_acc, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(results["y_test"], results["y_pred"])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred Down", "Pred Up"], y=["Actual Down", "Actual Up"],
                colorscale="Blues", text=cm, texttemplate="%{text}",
            ))
            fig_cm.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)
        with col4:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(results["y_test"], results["y_pred_prob"])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}", line=dict(color="#4f6df5")))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
            fig_roc.update_layout(template="plotly_dark", height=320, xaxis_title="FPR", yaxis_title="TPR",
                                   margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_roc, use_container_width=True)

        with st.expander("Model architecture summary"):
            lines = []
            st.session_state.model.summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines))

# ---- PREDICTION TAB -----------------------------------------------------------
with tab_predict:
    if st.session_state.model is None:
        st.info("Train the ANN model first (Training tab) to generate a live prediction.")
    else:
        fd = st.session_state.featured_data
        scaler = st.session_state.scaler
        model = st.session_state.model

        latest_row = fd[FEATURE_COLUMNS].iloc[[-1]].values
        latest_scaled = scaler.transform(latest_row)
        prob_up = float(model.predict(latest_scaled, verbose=0).ravel()[0])
        prob_down = 1 - prob_up
        signal = "UP" if prob_up >= 0.5 else "DOWN"
        signal_class = "signal-up" if signal == "UP" else "signal-down"
        arrow = "▲" if signal == "UP" else "▼"
        color_class = "up" if signal == "UP" else "down"

        st.markdown(f"### Next-Day Prediction for {ticker}")
        st.markdown(
            f"""
            <div class="signal-box {signal_class}">
                <h1 class="{color_class}">{arrow} {signal}</h1>
                <p>Model confidence: {max(prob_up, prob_down)*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_up * 100,
                title={"text": "P(Price Up Tomorrow)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00d68f" if prob_up >= 0.5 else "#ff4d6d"},
                    "steps": [
                        {"range": [0, 50], "color": "#3a0d18"},
                        {"range": [50, 100], "color": "#00382a"},
                    ],
                },
            ))
            fig_gauge.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=50, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("#### Latest indicator snapshot")
            snapshot = fd[FEATURE_COLUMNS].iloc[-1].to_frame(name="Value")
            st.dataframe(snapshot, use_container_width=True)

        st.warning(
            "This prediction reflects patterns in historical technical indicators only. "
            "It is **not** financial advice — markets are influenced by many factors a "
            "technical-indicator ANN cannot capture."
        )

# ---- ABOUT TAB -----------------------------------------------------------
with tab_about:
    st.markdown("""
### About this project

This app trains an **Artificial Neural Network (ANN)** to predict whether a stock's
closing price will go **up or down the next trading day**, using classic technical
indicators as input features:

- **Trend**: SMA(10/50), EMA(10/50)
- **Momentum**: RSI(14), Momentum(10)
- **Trend strength**: MACD, MACD Signal, MACD Histogram
- **Volatility**: Bollinger Band Width, Rolling Std Dev(10)
- **Price/Volume behavior**: Daily Return, Volume Change

**Architecture**: A configurable feed-forward network — Dense layers with
ReLU activation, Batch Normalization, and Dropout for regularization — ending
in a single sigmoid output (probability of an "Up" day). Trained with Adam
optimizer and binary cross-entropy loss, with early stopping on validation loss.

**Pipeline**: `yfinance` (data) → feature engineering → `StandardScaler` →
chronological train/test split → Keras ANN → evaluation (accuracy, precision,
recall, F1, ROC-AUC, confusion matrix) → live next-day prediction.

**Disclaimer**: This is an educational/demo project. Stock markets are noisy
and influenced by countless factors beyond technical indicators. Do not use
this tool for real trading decisions.
""")
