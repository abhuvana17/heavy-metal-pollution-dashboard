"""
pages/2_Risk_Assessment.py
─────────────────────────────────────────────────────────────────────────────
Risk Assessment Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Display HPI / HEI / PLI pollution indices and Risk_Level classifications with
rich visual indicators, providing an at-a-glance understanding of environmental
contamination severity across different filters.

WHAT THIS PAGE DOES
───────────────────
1.  Sidebar filters  — State, District, Medium, Land Use, Year range, Risk Level.
2.  KPI metrics      — total samples per risk tier, avg HPI / HEI / PLI.
3.  Risk gauge cards — visual coloured cards per risk category.
4.  Stacked bar      — risk distribution per Medium and per Land Use.
5.  Scatter plot     — PLI vs HEI coloured by Risk_Level.
6.  Index histogram  — distribution of HPI / HEI / PLI values.
7.  Recommendation summary table.
8.  Filterable raw data table (Risk_Level, Recommendation, Indices).

INDEX DEFINITIONS (project-defined, not official WHO/EPA standards)
───────────────────────────────────────────────────────────────────
CF   = Metal_Concentration / Reference_Value   (per metal)
HEI  = Pb_CF + Hg_CF + As_CF + Cd_CF + Cr_CF
PLI  = (Pb_CF × Hg_CF × As_CF × Cd_CF × Cr_CF)^(1/5)
HPI  = weighted indicator using reference values as weights (project-defined)

Risk thresholds (project-defined):
  PLI < 1  → Low       → Continue Monitoring
  PLI < 2  → Moderate  → Increase Monitoring Frequency
  PLI < 3  → High      → Soil and Water Remediation
  PLI ≥ 3  → Critical  → Immediate Government Intervention
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset, RISK_COLORS, RECOMMENDATIONS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Assessment | Heavy Metal Dashboard",
    page_icon="⚠️",
    layout="wide",
)

# ─── Shared CSS ───────────────────────────────────────────────────────────────
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
.risk-card {
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
}
.risk-card .risk-label  { font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.risk-card .risk-count  { font-size:2rem; font-weight:700; }
.risk-card .risk-action { font-size:0.72rem; color:rgba(255,255,255,0.6); margin-top:4px; }
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
    st.markdown("## ⚠️ Risk Filters")
    st.markdown("---")

    states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    sel_state = st.selectbox("📍 State", states)

    if sel_state != "All":
        districts = ["All"] + sorted(df[df["State"] == sel_state]["District"].dropna().unique().tolist())
    else:
        districts = ["All"] + sorted(df["District"].dropna().unique().tolist())
    sel_district = st.selectbox("🏘️ District", districts)

    mediums = ["All"] + sorted(df["Medium"].dropna().unique().tolist())
    sel_medium = st.selectbox("🧪 Medium", mediums)

    land_uses = ["All"] + sorted(df["Land_Use"].dropna().unique().tolist())
    sel_land = st.selectbox("🌾 Land Use", land_uses)

    risk_levels = ["All"] + sorted(df["Risk_Level"].dropna().unique().tolist())
    sel_risk = st.selectbox("🚦 Risk Level", risk_levels)

    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    sel_years = st.slider("📅 Year Range", yr_min, yr_max, (yr_min, yr_max))

# ─── Apply Filters ────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_state    != "All": fdf = fdf[fdf["State"]      == sel_state]
if sel_district != "All": fdf = fdf[fdf["District"]   == sel_district]
if sel_medium   != "All": fdf = fdf[fdf["Medium"]     == sel_medium]
if sel_land     != "All": fdf = fdf[fdf["Land_Use"]   == sel_land]
if sel_risk     != "All": fdf = fdf[fdf["Risk_Level"] == sel_risk]
fdf = fdf[(fdf["Year"] >= sel_years[0]) & (fdf["Year"] <= sel_years[1])]

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        ⚠️ Environmental Risk Assessment
    </h1>
    <p style="color:#718096;margin:0;">
        HPI, HEI, and PLI pollution indices with risk classification and remediation recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Project-Defined Thresholds:</strong> HPI formulation, reference values,
    and PLI risk thresholds are defined specifically for this project and are
    <em>not</em> official WHO or EPA standards. Synthetic data only.
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("⚠️ No records match the current filters.")
    st.stop()

st.caption(f"Showing **{len(fdf):,}** records after filtering.")

# ─── Index Definition Expander ────────────────────────────────────────────────
with st.expander("📖 Index & Threshold Definitions (expand to view)"):
    st.markdown("""
| Index | Formula | Meaning |
|-------|---------|---------|
| **CF (Contamination Factor)** | Metal_Concentration / Reference_Value | Per-metal contamination ratio |
| **HEI (Heavy-Element Index)** | Pb_CF + Hg_CF + As_CF + Cd_CF + Cr_CF | Sum of all CFs |
| **PLI (Pollution Load Index)** | (Pb_CF × Hg_CF × As_CF × Cd_CF × Cr_CF)^(1/5) | Geometric mean of CFs |
| **HPI (Heavy Pollution Index)** | Weighted indicator (project-defined weights) | Integrated weighted measure |

**Risk Classification (PLI-based, project-defined):**

| PLI Range | Risk Level | Recommendation |
|-----------|-----------|----------------|
| < 1       | 🟢 Low     | Continue Monitoring |
| 1 – 2     | 🟡 Moderate | Increase Monitoring Frequency |
| 2 – 3     | 🔴 High    | Soil and Water Remediation |
| ≥ 3       | 🟣 Critical | Immediate Government Intervention |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RISK LEVEL CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚦 Risk Level Overview</div>', unsafe_allow_html=True)

risk_order  = ["Low", "Moderate", "High", "Critical"]
risk_emojis = {"Low":"🟢","Moderate":"🟡","High":"🔴","Critical":"🟣"}

cols = st.columns(4)
for col, risk in zip(cols, risk_order):
    cnt   = fdf[fdf["Risk_Level"] == risk].shape[0]
    color = RISK_COLORS[risk]
    rec   = RECOMMENDATIONS[risk]
    pct   = f"{cnt/len(fdf)*100:.1f}%" if len(fdf) > 0 else "0%"
    col.markdown(
        f'<div class="risk-card" style="background:linear-gradient(135deg,{color}22,{color}11);border-color:{color}44;">'
        f'<div class="risk-label" style="color:{color};">{risk_emojis[risk]} {risk}</div>'
        f'<div class="risk-count" style="color:{color};">{cnt:,}</div>'
        f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.5);">{pct} of filtered</div>'
        f'<div class="risk-action">{rec}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INDEX KPIs
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Pollution Index Summary</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("📈 Avg PLI",  f"{fdf['PLI'].mean():.4f}",  help="Pollution Load Index — geometric mean of contamination factors")
m2.metric("📉 Avg HEI",  f"{fdf['HEI'].mean():.4f}",  help="Heavy Element Index — sum of contamination factors")
m3.metric("📊 Avg HPI",  f"{fdf['HPI'].mean():.4f}",  help="Heavy Pollution Index — weighted indicator (project-defined)")
m4.metric("🔢 Samples",  f"{len(fdf):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STACKED BAR: Risk Distribution by Medium & Land Use
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Risk Distribution by Medium & Land Use</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    med_risk = (
        fdf.groupby(["Medium","Risk_Level"]).size().reset_index(name="Count")
    )
    fig_med = px.bar(
        med_risk, x="Medium", y="Count", color="Risk_Level",
        barmode="stack",
        color_discrete_map=RISK_COLORS,
        category_orders={"Risk_Level": risk_order},
        title="By Medium",
    )
    fig_med.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", legend_title_text="Risk",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=30, b=20, l=10, r=10), title_font_size=14,
    )
    st.plotly_chart(fig_med, use_container_width=True)

with c2:
    lu_risk = (
        fdf.groupby(["Land_Use","Risk_Level"]).size().reset_index(name="Count")
    )
    fig_lu = px.bar(
        lu_risk, x="Land_Use", y="Count", color="Risk_Level",
        barmode="stack",
        color_discrete_map=RISK_COLORS,
        category_orders={"Risk_Level": risk_order},
        title="By Land Use",
    )
    fig_lu.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", legend_title_text="Risk",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=30, b=20, l=10, r=10), title_font_size=14,
    )
    st.plotly_chart(fig_lu, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCATTER: PLI vs HEI coloured by Risk Level
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔵 PLI vs HEI Scatter (coloured by Risk Level)</div>', unsafe_allow_html=True)
st.caption("Each point is one environmental sample. The scatter shows how PLI and HEI co-vary and how the risk thresholds separate the clusters.")

sample_df = fdf.sample(min(3000, len(fdf)), random_state=42)
fig_scat = px.scatter(
    sample_df, x="HEI", y="PLI",
    color="Risk_Level",
    color_discrete_map=RISK_COLORS,
    category_orders={"Risk_Level": risk_order},
    opacity=0.55,
    hover_data=["State","District","Medium","Land_Use"],
    labels={"HEI":"Heavy Element Index (HEI)","PLI":"Pollution Load Index (PLI)"},
)

# Add PLI threshold lines
for pli_val, label, color in [(1,"PLI=1 (Low/Mod)","#f6ad55"),(2,"PLI=2 (Mod/High)","#fc8181"),(3,"PLI=3 (High/Crit)","#9f7aea")]:
    fig_scat.add_hline(
        y=pli_val, line_dash="dash", line_color=color, line_width=1.5,
        annotation_text=label, annotation_position="right",
        annotation_font_color=color,
    )

fig_scat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    margin=dict(t=10, b=20, l=10, r=10),
)
st.plotly_chart(fig_scat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HISTOGRAMS: Index Distributions
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📉 Pollution Index Distributions</div>', unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)

for col, index_col, color, label in [
    (h1, "PLI",  "#63b3ed", "Pollution Load Index (PLI)"),
    (h2, "HEI",  "#68d391", "Heavy Element Index (HEI)"),
    (h3, "HPI",  "#f6ad55", "Heavy Pollution Index (HPI)"),
]:
    fig_h = px.histogram(
        fdf, x=index_col, nbins=50,
        color_discrete_sequence=[color],
        labels={index_col: label},
    )
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=False,
        xaxis=dict(showgrid=False, title=label),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
        margin=dict(t=10, b=20, l=10, r=10), bargap=0.05,
    )
    col.plotly_chart(fig_h, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RECOMMENDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💡 Recommendation Summary</div>', unsafe_allow_html=True)

rec_counts = fdf["Recommendation"].value_counts().reset_index()
rec_counts.columns = ["Recommendation","Count"]

fig_rec = px.bar(
    rec_counts, x="Count", y="Recommendation", orientation="h",
    color="Count", color_continuous_scale="Purples",
    text="Count",
)
fig_rec.update_traces(textposition="outside")
fig_rec.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", showlegend=False,
    coloraxis_showscale=False,
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(showgrid=False, autorange="reversed"),
    margin=dict(t=10, b=20, l=10, r=80),
)
st.plotly_chart(fig_rec, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FILTERABLE DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗂️ Sample-Level Risk Data</div>', unsafe_allow_html=True)

display_cols = ["Sample_ID","Date","State","District","Medium","Land_Use",
                "PLI","HEI","HPI","Risk_Level","Recommendation"]
avail_display = [c for c in display_cols if c in fdf.columns]

st.dataframe(
    fdf[avail_display]
      .sort_values("PLI", ascending=False)
      .reset_index(drop=True)
      .head(500)
      .style.map(
          lambda v: f"color:{RISK_COLORS.get(v,'#e2e8f0')}; font-weight:bold;"
          if v in RISK_COLORS else "",
          subset=["Risk_Level"],
      ),
    use_container_width=True,
)
st.caption("Table shows up to 500 records sorted by PLI (highest first).")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Risk Assessment Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
