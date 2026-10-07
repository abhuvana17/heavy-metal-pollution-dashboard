"""
pages/1_Pollution_Analysis.py
─────────────────────────────────────────────────────────────────────────────
Pollution Analysis Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Provide a comprehensive, interactive visual exploration of heavy-metal
concentrations (Pb, Hg, As, Cd, Cr) across different environmental mediums,
land-use categories, states, districts, and time periods.

WHAT THIS PAGE DOES
───────────────────
1.  Sidebar filters  — State, District, Medium, Land Use, Year range.
2.  Summary statistics table — count/mean/std/min/max per metal.
3.  Bar chart — average concentration per metal (grouped by selected grouping).
4.  Box plot — concentration distribution per metal to reveal outliers.
5.  Line/trend chart — annual mean concentration trend for each metal.
6.  Heatmap — correlation between metals and environmental parameters.
7.  Grouped bar — average metal concentrations by Land Use or Medium.
8.  Top-10 most polluted districts table.

DESIGN NOTES
────────────
- All charts use Plotly for interactivity.
- All data is read from the cached `load_dataset()` — no re-reads.
- Leakage-prone columns (HPI, HEI, PLI, Risk_Level) are displayed for
  information only, never used as ML inputs here.
- Synthetic-data caveat is shown as a collapsible disclaimer.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset, METALS, METAL_LABELS, RISK_COLORS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pollution Analysis | Heavy Metal Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─── Shared CSS (mirrors app.py style) ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.section-header {
    font-size: 1.05rem; font-weight: 600; color: #a0aec0;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(99,179,237,0.3);
}
.stat-card {
    background: linear-gradient(135deg,#1e2738 0%,#252d3d 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.stat-label { font-size:0.78rem; color:#718096; text-transform:uppercase; }
.stat-value { font-size:1.4rem; font-weight:700; color:#e2e8f0; }
.disclaimer {
    background: rgba(66,153,225,0.08);
    border-left: 4px solid #4299e1;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.83rem;
    color: #90cdf4;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_dataset()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Analysis Filters")
    st.markdown("---")

    # State
    states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    sel_state = st.selectbox("📍 State", states)

    # District (depends on state)
    if sel_state != "All":
        districts = ["All"] + sorted(
            df[df["State"] == sel_state]["District"].dropna().unique().tolist()
        )
    else:
        districts = ["All"] + sorted(df["District"].dropna().unique().tolist())
    sel_district = st.selectbox("🏘️ District", districts)

    # Medium
    mediums = ["All"] + sorted(df["Medium"].dropna().unique().tolist())
    sel_medium = st.selectbox("🧪 Medium", mediums)

    # Land Use
    land_uses = ["All"] + sorted(df["Land_Use"].dropna().unique().tolist())
    sel_land = st.selectbox("🌾 Land Use", land_uses)

    # Year range
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    sel_years = st.slider("📅 Year Range", yr_min, yr_max, (yr_min, yr_max))

    st.markdown("---")

    # Grouping axis for grouped bar chart
    group_by = st.radio(
        "📊 Group Bars By",
        ["Medium", "Land_Use", "Risk_Level"],
        horizontal=True,
    )

    st.markdown("---")
    st.caption("Filters apply to all charts on this page.")

# ─── Apply Filters ────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_state    != "All": fdf = fdf[fdf["State"]    == sel_state]
if sel_district != "All": fdf = fdf[fdf["District"] == sel_district]
if sel_medium   != "All": fdf = fdf[fdf["Medium"]   == sel_medium]
if sel_land     != "All": fdf = fdf[fdf["Land_Use"] == sel_land]
fdf = fdf[(fdf["Year"] >= sel_years[0]) & (fdf["Year"] <= sel_years[1])]

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        📊 Pollution Analysis
    </h1>
    <p style="color:#718096;margin:0;">
        Visual exploration of heavy-metal concentrations across environmental mediums,
        land-use categories, regions, and time.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚠️ <strong>Synthetic Data:</strong> '
    'Concentrations are simulated — not real environmental measurements. '
    'Trends reflect the data-generation logic, not actual monitoring observations.</div>',
    unsafe_allow_html=True,
)

# Guard against empty filter result
if fdf.empty:
    st.warning("⚠️ No records match the current filters. Please adjust the sidebar selections.")
    st.stop()

record_count = len(fdf)
st.caption(f"Showing **{record_count:,}** records after filtering.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Summary Statistics</div>', unsafe_allow_html=True)

stat_df = fdf[METALS].describe().T.round(4)
stat_df.index = [METAL_LABELS[m] for m in stat_df.index]
stat_df.columns = [c.capitalize() for c in stat_df.columns]
st.dataframe(
    stat_df.style.background_gradient(cmap="YlOrRd", subset=["Mean"]),
    use_container_width=True,
)

# Quick KPI row
k1, k2, k3, k4, k5 = st.columns(5)
for col, metal in zip([k1,k2,k3,k4,k5], METALS):
    avg = fdf[metal].mean()
    col.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{METAL_LABELS[metal]}</div>'
        f'<div class="stat-value">{avg:.3f}</div>'
        f'<div class="stat-label">Avg mg/kg</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BAR CHART: Average Concentration per Metal
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Average Concentration per Metal</div>', unsafe_allow_html=True)

avg_by_group = (
    fdf.groupby(group_by)[METALS]
    .mean()
    .reset_index()
    .rename(columns=METAL_LABELS)
)

fig_bar = px.bar(
    avg_by_group,
    x=group_by,
    y=list(METAL_LABELS.values()),
    barmode="group",
    labels={"value": "Avg Concentration (mg/kg)", "variable": "Metal"},
    color_discrete_sequence=["#fc8181","#f6ad55","#fbd38d","#68d391","#63b3ed"],
)
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", legend_title_text="Metal",
    xaxis=dict(showgrid=False, title=group_by),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    margin=dict(t=10, b=30, l=10, r=10),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BOX PLOT: Distribution per Metal
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📦 Concentration Distribution (Box Plot)</div>', unsafe_allow_html=True)
st.caption(
    "Box plots reveal the median, inter-quartile range, and outliers for each metal. "
    "Outliers may indicate industrial hotspots in the synthetic dataset."
)

# Melt for Plotly
melt_df = fdf[METALS].copy()
melt_df = melt_df.rename(columns=METAL_LABELS)
melt_long = melt_df.melt(var_name="Metal", value_name="Concentration")

fig_box = px.box(
    melt_long, x="Metal", y="Concentration",
    color="Metal",
    color_discrete_sequence=["#fc8181","#f6ad55","#fbd38d","#68d391","#63b3ed"],
    points="outliers",
)
fig_box.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", showlegend=False,
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        title="Concentration (mg/kg)",
    ),
    margin=dict(t=10, b=30, l=10, r=10),
)
st.plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LINE CHART: Annual Mean Trend per Metal
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Annual Mean Concentration Trend</div>', unsafe_allow_html=True)
st.caption(
    "Note: Because dates in the synthetic dataset are randomly assigned, "
    "this trend reflects sampling distribution rather than a genuine "
    "longitudinal monitoring series."
)

yearly = fdf.groupby("Year")[METALS].mean().reset_index()
yearly_long = yearly.melt(id_vars="Year", var_name="Metal", value_name="Mean Concentration")
yearly_long["Metal"] = yearly_long["Metal"].map(METAL_LABELS)

fig_line = px.line(
    yearly_long, x="Year", y="Mean Concentration",
    color="Metal",
    markers=True,
    color_discrete_sequence=["#fc8181","#f6ad55","#fbd38d","#68d391","#63b3ed"],
)
fig_line.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", legend_title_text="Metal",
    xaxis=dict(showgrid=False, title="Year", dtick=1),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Mean Concentration (mg/kg)"),
    margin=dict(t=10, b=30, l=10, r=10),
)
st.plotly_chart(fig_line, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HEATMAP: Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
st.caption(
    "Pearson correlation between heavy-metal concentrations and key environmental "
    "parameters. Strong positive correlation between metals is expected in synthetic data "
    "where all metals are generated from shared pollution-source factors."
)

corr_cols = METALS + ["Temperature_C","Rainfall_mm","Humidity_percent",
                       "Soil_pH","Soil_Moisture_percent","Distance_to_Industry_km"]
avail_cols = [c for c in corr_cols if c in fdf.columns]
corr_matrix = fdf[avail_cols].corr().round(3)

# Rename for display
rename_map = {**METAL_LABELS}
corr_matrix_disp = corr_matrix.rename(index=rename_map, columns=rename_map)

fig_heat = px.imshow(
    corr_matrix_disp,
    color_continuous_scale="RdYlGn",
    zmin=-1, zmax=1,
    text_auto=".2f",
    aspect="auto",
)
fig_heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    margin=dict(t=20, b=20, l=20, r=20),
    coloraxis_colorbar=dict(title="r"),
)
fig_heat.update_traces(textfont_size=10)
st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STACKED BAR: Metal Share by Pollution Source
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🏭 Metal Concentrations by Pollution Source</div>', unsafe_allow_html=True)

if "Pollution_Source" in fdf.columns:
    src_avg = (
        fdf.groupby("Pollution_Source")[METALS]
        .mean()
        .reset_index()
        .rename(columns=METAL_LABELS)
    )
    fig_src = px.bar(
        src_avg,
        x="Pollution_Source",
        y=list(METAL_LABELS.values()),
        barmode="stack",
        labels={"value": "Avg Concentration (mg/kg)", "variable": "Metal"},
        color_discrete_sequence=["#fc8181","#f6ad55","#fbd38d","#68d391","#63b3ed"],
    )
    fig_src.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", legend_title_text="Metal",
        xaxis=dict(showgrid=False, title="Pollution Source"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, b=30, l=10, r=10),
    )
    st.plotly_chart(fig_src, use_container_width=True)
else:
    st.info("Pollution_Source column not found in dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VIOLIN PLOT: Distribution by Medium
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎻 Metal Distribution by Medium (Violin)</div>', unsafe_allow_html=True)
st.caption("Violin plots show the full probability density of each metal's concentration across Soil, Water, and Air.")

sel_violin_metal = st.selectbox(
    "Select Metal for Violin Plot",
    options=METALS,
    format_func=lambda m: METAL_LABELS[m],
    key="violin_metal",
)

fig_violin = px.violin(
    fdf, x="Medium", y=sel_violin_metal,
    color="Medium",
    box=True, points="outliers",
    color_discrete_sequence=["#4299e1","#48bb78","#ed8936"],
    labels={sel_violin_metal: f"{METAL_LABELS[sel_violin_metal]} (mg/kg)"},
)
fig_violin.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", showlegend=False,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    margin=dict(t=10, b=30, l=10, r=10),
)
st.plotly_chart(fig_violin, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TOP 10 MOST POLLUTED DISTRICTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🏙️ Top 10 Most Polluted Districts (by Avg PLI)</div>', unsafe_allow_html=True)
st.caption(
    "Districts ranked by average Pollution Load Index (PLI). "
    "PLI = (Pb_CF × Hg_CF × As_CF × Cd_CF × Cr_CF)^(1/5) — "
    "a project-defined threshold, not a universal standard."
)

if "PLI" in fdf.columns:
    top_districts = (
        fdf.groupby(["State","District"])["PLI"]
        .mean()
        .reset_index()
        .sort_values("PLI", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_districts.index += 1
    top_districts["PLI"] = top_districts["PLI"].round(4)

    fig_top = px.bar(
        top_districts,
        x="PLI", y="District",
        orientation="h",
        color="PLI",
        color_continuous_scale="Reds",
        hover_data=["State"],
        labels={"PLI": "Average PLI"},
        text="PLI",
    )
    fig_top.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_top.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=False,
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, b=30, l=10, r=80),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.dataframe(
        top_districts.style.background_gradient(cmap="Reds", subset=["PLI"]),
        use_container_width=True,
    )
else:
    st.info("PLI column not found in dataset.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Pollution Analysis Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
