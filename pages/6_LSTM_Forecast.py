"""
pages/6_LSTM_Forecast.py
─────────────────────────────────────────────────────────────────────────────
LSTM Forecasting Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Visualise the LSTM-based time-series forecasting of total heavy-metal
concentration trends, compare actual vs predicted values, and show a
30-day future forecast.

WHAT THIS PAGE DOES
───────────────────
1.  Load lstm_model.keras + lstm_scaler.pkl from artifacts/.
2.  Rebuild the daily-aggregated pollution series from the dataset.
3.  Display the historical trend of Total_Heavy_Metal.
4.  Show actual vs LSTM-predicted values on the test portion.
5.  Evaluation metrics — MAE, RMSE, R².
6.  30-day recursive future forecast chart.
7.  Honest limitations about synthetic temporal validity.

LSTM ARCHITECTURE REMINDER (for the viva)
─────────────────────────────────────────
Sequential:
  LSTM(64, return_sequences=True) → Dropout(0.2) →
  LSTM(32) → Dropout(0.2) →
  Dense(16, relu) → Dense(1)
Optimizer: Adam | Loss: MSE
TIME_STEPS = 30 (use previous 30 daily aggregates to predict next)
Split: chronological 80/20 (no shuffle)

LIMITATIONS (must be flagged)
─────────────────────────────
- Dates in the synthetic dataset are randomly assigned. Sorting by date
  does not create a scientifically valid longitudinal monitoring series.
  The LSTM learns patterns from the data-generation logic, not genuine
  temporal environmental dynamics.
- R² may be misleadingly high because the synthetic data has constrained
  variance by construction.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import load_artifacts, load_dataset

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Forecast | Heavy Metal Dashboard",
    page_icon="📈",
    layout="wide",
)

# ─── Shared CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size:1.05rem; font-weight:600; color:#a0aec0;
    text-transform:uppercase; letter-spacing:0.07em;
    margin:1.4rem 0 0.6rem; padding-bottom:0.35rem;
    border-bottom:2px solid rgba(99,179,237,0.3);
}
.disclaimer {
    background:rgba(237,137,54,0.08); border-left:4px solid #ed8936;
    border-radius:8px; padding:10px 16px; font-size:0.83rem;
    color:#fbd38d; margin-bottom:1rem;
}
.metric-card {
    background:linear-gradient(135deg,#1e2738 0%,#252d3d 100%);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:14px 18px; text-align:center;
}
.metric-label { font-size:0.75rem; color:#718096; text-transform:uppercase; }
.metric-value { font-size:1.6rem; font-weight:700; color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
artifacts    = load_artifacts()
lstm_model   = artifacts.get("lstm_model")
lstm_scaler  = artifacts.get("lstm_scaler")
df           = load_dataset()

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        📈 LSTM Pollution Forecasting
    </h1>
    <p style="color:#718096;margin:0;">
        Time-series forecasting of total heavy-metal concentration trends
        using a deep LSTM neural network.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Important Temporal Limitation:</strong> Dates in this synthetic dataset
are randomly assigned. Sorting by date does not create a scientifically valid
longitudinal environmental monitoring series. The LSTM learns patterns from the
data-generation logic, not genuine temporal contamination dynamics.
Results are for educational ML demonstration only and should not be interpreted
as real pollution forecasts.
</div>
""", unsafe_allow_html=True)

# ─── TensorFlow Availability Check ───────────────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow as tf          # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    pass

if not TF_AVAILABLE:
    st.error(
        "❌ **TensorFlow is not installed** — the LSTM Forecasting page requires TensorFlow.\n\n"
        "**Your Python version:** 3.14 (no TensorFlow wheel available yet as of mid-2026)\n\n"
        "**Workaround options:**\n"
        "1. Use **Python 3.11 or 3.12** via a virtual environment: `py -3.12 -m venv venv`\n"
        "2. Use **Conda**: `conda create -n pollution python=3.12` then `conda activate pollution`\n"
        "   then `conda install tensorflow` or `pip install tensorflow`\n"
        "3. **Temporarily**: all other 6 pages work without TensorFlow — only Page 6 (LSTM) and "
        "   the LSTM portion of Page 7 (Report) are affected.\n\n"
        "The LSTM model was trained in Google Colab (Python 3.10/3.11) and saved as "
        "`lstm_model.keras`. It cannot be loaded without TensorFlow installed."
    )
    st.info(
        "💡 **For your B.Tech project viva:** You can show the LSTM architecture, "
        "training history, and results screenshots from Google Colab directly — "
        "the dashboard's other 6 pages (Pollution Analysis, Risk Assessment, "
        "Geographic Analysis, ML Prediction, Explainable AI, Report Generation) "
        "are fully functional."
    )
    st.stop()

# ─── Artifact Check ───────────────────────────────────────────────────────────
if lstm_model is None or lstm_scaler is None:
    st.error(
        "❌ LSTM artifacts not found. Required files:\n"
        "- `artifacts/lstm_model.keras`\n"
        "- `artifacts/lstm_scaler.pkl`\n\n"
        "Please run the training notebook to generate these."
    )
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 LSTM Settings")
    st.markdown("---")
    TIME_STEPS    = st.number_input("Time Steps (must match training)", min_value=5, max_value=90, value=30, step=5)
    forecast_days = st.slider("Future Forecast Days", min_value=7, max_value=90, value=30)
    st.markdown("---")
    st.markdown("""
    **LSTM Architecture:**
    - LSTM(64, return_sequences=True)
    - Dropout(0.2)
    - LSTM(32)
    - Dropout(0.2)
    - Dense(16, relu)
    - Dense(1)

    Optimizer: Adam | Loss: MSE
    """)
    st.caption("TIME_STEPS must match the value used during model training (default: 30).")

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DAILY POLLUTION SERIES (mirrors training notebook logic)
# ═══════════════════════════════════════════════════════════════════════════════
METALS = ["Lead_Pb_mgkg","Mercury_Hg_mgkg","Arsenic_As_mgkg","Cadmium_Cd_mgkg","Chromium_Cr_mgkg"]

df_lstm = df.copy()
df_lstm["Date"] = pd.to_datetime(df_lstm["Date"])
df_lstm = df_lstm.sort_values("Date")

df_lstm["Total_Heavy_Metal"] = df_lstm[METALS].sum(axis=1)

daily_pollution = (
    df_lstm.groupby("Date")["Total_Heavy_Metal"]
    .mean()
    .reset_index()
    .sort_values("Date")
)
daily_pollution.columns = ["Date","Total_Heavy_Metal"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HISTORICAL TREND
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Historical Total Heavy Metal Concentration</div>', unsafe_allow_html=True)
st.caption(
    "Daily mean of total heavy-metal concentration (sum of Pb+Hg+As+Cd+Cr) across all samples. "
    "Randomised date assignment means this is not a true time series."
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=daily_pollution["Date"], y=daily_pollution["Total_Heavy_Metal"],
    mode="lines", name="Daily Mean",
    line=dict(color="#63b3ed", width=1.5),
    fill="tozeroy", fillcolor="rgba(99,179,237,0.07)",
))
# 7-day rolling average
rolling = daily_pollution["Total_Heavy_Metal"].rolling(7, min_periods=1).mean()
fig_hist.add_trace(go.Scatter(
    x=daily_pollution["Date"], y=rolling,
    mode="lines", name="7-Day Rolling Avg",
    line=dict(color="#f6ad55", width=2, dash="dot"),
))
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", legend=dict(orientation="h", y=1.08),
    xaxis=dict(showgrid=False, title="Date"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Total Heavy Metal (mg/kg)"),
    margin=dict(t=10, b=30, l=10, r=10), height=320,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LSTM INFERENCE — rebuild sequences and predict on test set
# ═══════════════════════════════════════════════════════════════════════════════
scaled = lstm_scaler.transform(daily_pollution[["Total_Heavy_Metal"]])

# Build sliding-window sequences
def build_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X_all, y_all = build_sequences(scaled, TIME_STEPS)

if len(X_all) == 0:
    st.error(f"Not enough daily data points ({len(daily_pollution)}) for TIME_STEPS={TIME_STEPS}. Lower TIME_STEPS or check dataset.")
    st.stop()

# Chronological 80/20 split
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_test           = y_all[split:]
dates_test       = daily_pollution["Date"].iloc[TIME_STEPS + split : TIME_STEPS + split + len(y_test)].values

# Predict
with st.spinner("Running LSTM inference…"):
    y_pred_scaled = lstm_model.predict(X_test, verbose=0)

y_pred_actual = lstm_scaler.inverse_transform(y_pred_scaled).flatten()
y_test_actual = lstm_scaler.inverse_transform(y_test).flatten()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTUAL vs PREDICTED
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 Actual vs LSTM Predicted (Test Set)</div>', unsafe_allow_html=True)

fig_avp = go.Figure()
fig_avp.add_trace(go.Scatter(
    x=dates_test, y=y_test_actual,
    mode="lines", name="Actual",
    line=dict(color="#63b3ed", width=2),
))
fig_avp.add_trace(go.Scatter(
    x=dates_test, y=y_pred_actual,
    mode="lines", name="Predicted (LSTM)",
    line=dict(color="#fc8181", width=2, dash="dot"),
))
fig_avp.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", legend=dict(orientation="h", y=1.08),
    xaxis=dict(showgrid=False, title="Date"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Total Heavy Metal (mg/kg)"),
    margin=dict(t=10, b=30, l=10, r=10), height=340,
)
st.plotly_chart(fig_avp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📐 LSTM Evaluation Metrics</div>', unsafe_allow_html=True)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2   = r2_score(y_test_actual, y_pred_actual)

m1, m2, m3 = st.columns(3)
m1.markdown(
    f'<div class="metric-card"><div class="metric-label">MAE</div>'
    f'<div class="metric-value">{mae:.4f}</div>'
    f'<div class="metric-label">mg/kg</div></div>', unsafe_allow_html=True
)
m2.markdown(
    f'<div class="metric-card"><div class="metric-label">RMSE</div>'
    f'<div class="metric-value">{rmse:.4f}</div>'
    f'<div class="metric-label">mg/kg</div></div>', unsafe_allow_html=True
)
m3.markdown(
    f'<div class="metric-card"><div class="metric-label">R²</div>'
    f'<div class="metric-value">{r2:.4f}</div>'
    f'<div class="metric-label">coefficient of determination</div></div>',
    unsafe_allow_html=True
)

st.caption(
    "Note: High R² in synthetic data does not imply real-world forecasting accuracy. "
    "The model learns the statistical properties of the generated data, not genuine "
    "environmental dynamics."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — 30-DAY RECURSIVE FUTURE FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">🔮 {forecast_days}-Day Recursive Future Forecast</div>', unsafe_allow_html=True)
st.caption(
    f"Starting from the last {TIME_STEPS} known data points, the model autoregressively "
    f"predicts {forecast_days} days into the future. Each new prediction is fed back as "
    "the next input (recursive / multi-step forecast)."
)

# Seed with last TIME_STEPS scaled values
last_sequence = scaled[-TIME_STEPS:].copy()  # shape: (TIME_STEPS, 1)
future_preds  = []

for _ in range(forecast_days):
    input_seq  = last_sequence.reshape(1, TIME_STEPS, 1)
    next_pred  = lstm_model.predict(input_seq, verbose=0)[0, 0]
    future_preds.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

future_actual = lstm_scaler.inverse_transform(
    np.array(future_preds).reshape(-1,1)
).flatten()

last_date      = daily_pollution["Date"].max()
future_dates   = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
future_df      = pd.DataFrame({"Date": future_dates, "Predicted_Total_Heavy_Metal": future_actual})

# Plot: join historical tail + forecast
tail_n  = min(60, len(daily_pollution))
hist_tail = daily_pollution.tail(tail_n)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=hist_tail["Date"], y=hist_tail["Total_Heavy_Metal"],
    mode="lines", name="Historical (last 60 days)",
    line=dict(color="#63b3ed", width=2),
))
fig_fc.add_trace(go.Scatter(
    x=future_df["Date"], y=future_df["Predicted_Total_Heavy_Metal"],
    mode="lines+markers", name=f"Forecast ({forecast_days}d)",
    line=dict(color="#fc8181", width=2.5, dash="dash"),
    marker=dict(size=5),
))
# Shaded forecast region
fig_fc.add_vrect(
    x0=future_df["Date"].iloc[0], x1=future_df["Date"].iloc[-1],
    fillcolor="rgba(252,129,129,0.05)", layer="below", line_width=0,
)
fig_fc.add_vline(
    x=last_date, line_dash="dot", line_color="rgba(255,255,255,0.3)",
    annotation_text="Forecast Start", annotation_font_color="#a0aec0",
)
fig_fc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", legend=dict(orientation="h", y=1.08),
    xaxis=dict(showgrid=False, title="Date"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Total Heavy Metal (mg/kg)"),
    margin=dict(t=10, b=30, l=10, r=10), height=380,
)
st.plotly_chart(fig_fc, use_container_width=True)

# Forecast table
with st.expander(f"📋 View {forecast_days}-Day Forecast Table"):
    future_df["Predicted_Total_Heavy_Metal"] = future_df["Predicted_Total_Heavy_Metal"].round(4)
    st.dataframe(future_df, use_container_width=True)
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=future_df.to_csv(index=False),
        file_name="lstm_future_forecast.csv",
        mime="text/csv",
    )

# ─── Viva Explanation ─────────────────────────────────────────────────────────
with st.expander("🎓 LSTM Architecture — Viva Explanation"):
    st.markdown(f"""
**Model Architecture:**
```
LSTM(64 units, return_sequences=True)  ← captures long-range dependencies
Dropout(0.2)                           ← reduces overfitting
LSTM(32 units)                         ← extracts higher-level patterns
Dropout(0.2)
Dense(16, activation='relu')           ← non-linear transformation
Dense(1)                               ← scalar output: next step's concentration
```

**Why LSTM for pollution forecasting?**
LSTMs have a memory mechanism (cell state + gates) that allows them to model
long-range temporal dependencies — important for environmental signals that
may have seasonal or persistent patterns.

**TIME_STEPS = {int(TIME_STEPS)}:**
The model sees the previous {int(TIME_STEPS)} daily mean values to predict the next one.

**Recursive (autoregressive) forecasting:**
For multi-step future prediction, each predicted value is appended to the input
window and the oldest value is dropped (sliding window). This accumulates errors
over time, so long-horizon forecasts should be interpreted with caution.

**Known Limitation:**
The synthetic dataset's dates are randomly assigned. The LSTM learns statistical
patterns of the synthetic data, not genuine environmental temporal dynamics.
Temporal realism is limited and must be acknowledged in any assessment.
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 LSTM Forecasting Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
