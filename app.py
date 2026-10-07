"""
app.py — Dashboard Overview (Home Page)
Heavy Metal Pollution Prediction & Environmental Risk Assessment System
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import load_dataset, load_artifacts, RISK_COLORS, METAL_LABELS, METALS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heavy Metal Pollution Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2738 0%, #252d3d 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
    }

    /* Main background */
    .main .block-container {
        padding-top: 2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(99,179,237,0.3);
    }

    /* Info callout */
    .info-callout {
        background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%);
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #bee3f8;
    }

    /* Best model badge */
    .best-model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6b46c1, #4299e1);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df        = load_dataset()
artifacts = load_artifacts()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Filters")
    st.markdown("---")

    mediums = ["All"] + sorted(df["Medium"].dropna().unique().tolist())
    sel_medium = st.selectbox("🧪 Medium", mediums)

    states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    sel_state = st.selectbox("📍 State", states)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "A B.Tech AIML minor project exploring AI-driven heavy-metal "
        "pollution prediction across soil, water, and air mediums using "
        "synthetic environmental data."
    )

# Apply sidebar filters
filtered = df.copy()
if sel_medium != "All":
    filtered = filtered[filtered["Medium"] == sel_medium]
if sel_state != "All":
    filtered = filtered[filtered["State"] == sel_state]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <h1 style="font-size:2.4rem; font-weight:700; margin-bottom:0.2rem;">
        🌍 Heavy Metal Pollution
    </h1>
    <p style="font-size:1.1rem; color:#718096; margin:0;">
        AI-Driven Prediction &amp; Environmental Risk Assessment System
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="info-callout">⚠️ <strong>Synthetic Dataset Disclaimer:</strong> This dashboard uses a synthetically generated 10,000-record dataset designed to simulate realistic environmental relationships. It is <em>not</em> real-world monitoring data and results should not be interpreted as actual environmental measurements.</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── KPI Metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_samples = len(filtered)
avg_pli       = filtered["PLI"].mean() if "PLI" in filtered.columns else 0
avg_hpi       = filtered["HPI"].mean() if "HPI" in filtered.columns else 0
critical_cnt  = filtered[filtered["Risk_Level"] == "Critical"].shape[0] if "Risk_Level" in filtered.columns else 0
high_cnt      = filtered[filtered["Risk_Level"] == "High"].shape[0] if "Risk_Level" in filtered.columns else 0

col1.metric("📊 Total Samples",    f"{total_samples:,}")
col2.metric("📈 Avg PLI",          f"{avg_pli:.3f}")
col3.metric("📉 Avg HPI",          f"{avg_hpi:.2f}")
col4.metric("🚨 Critical Records", f"{critical_cnt:,}")
col5.metric("⚠️ High Risk Records", f"{high_cnt:,}")

st.markdown("---")

# ─── Charts Row 1 ─────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
    if "Risk_Level" in filtered.columns:
        risk_counts = filtered["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig_pie = px.pie(
            risk_counts, names="Risk_Level", values="Count",
            color="Risk_Level",
            color_discrete_map=RISK_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.markdown('<div class="section-header">Samples by Environmental Medium</div>', unsafe_allow_html=True)
    med_counts = filtered["Medium"].value_counts().reset_index()
    med_counts.columns = ["Medium", "Count"]
    fig_bar = px.bar(
        med_counts, x="Medium", y="Count",
        color="Medium",
        color_discrete_sequence=["#4299e1", "#48bb78", "#ed8936"],
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── Charts Row 2 ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Average Heavy Metal Concentrations (mg/kg or µg/m³)</div>', unsafe_allow_html=True)

metal_avgs = {METAL_LABELS[m]: filtered[m].mean() for m in METALS if m in filtered.columns}
fig_metals = px.bar(
    x=list(metal_avgs.keys()),
    y=list(metal_avgs.values()),
    labels={"x": "Heavy Metal", "y": "Avg Concentration"},
    color=list(metal_avgs.keys()),
    color_discrete_sequence=["#fc8181","#f6ad55","#fbd38d","#68d391","#63b3ed"],
)
fig_metals.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    showlegend=False,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    margin=dict(t=10, b=30, l=20, r=20),
)
st.plotly_chart(fig_metals, use_container_width=True)

# ─── Best Model Callout ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🏆 Best Performing ML Model</div>', unsafe_allow_html=True)

best_model_name = artifacts.get("best_model_name")
if best_model_name:
    st.markdown(
        f'<div style="text-align:center; padding:1rem;">'
        f'<span class="best-model-badge">🥇 {best_model_name}</span>'
        f'<p style="color:#718096; margin-top:0.6rem; font-size:0.85rem;">'
        f'Selected by highest F1-Score on the 20% hold-out test set</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Best model artifact not found — run model training first.")

# ─── Navigation Hint ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#718096; font-size:0.9rem; padding: 0.5rem 0 1.5rem;">
    Navigate using the <strong>sidebar pages</strong> to explore Pollution Analysis, Risk Assessment,
    Geographic Maps, ML Predictions, Explainable AI, LSTM Forecasts, and Report Generation.
</div>
""", unsafe_allow_html=True)
