"""
pages/3_Geographic_Analysis.py
─────────────────────────────────────────────────────────────────────────────
Geographic Analysis Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Visualise heavy-metal pollution spatially using Folium interactive maps
and Plotly bubble / choropleth-style charts based on Latitude/Longitude
and Risk_Level from the dataset.

WHAT THIS PAGE DOES
───────────────────
1.  Sidebar filters  — State, Medium, Land Use, Risk Level, Metal to visualise.
2.  Folium map       — colour-coded circle markers per Risk_Level; popups show
                        District, PLI, HEI, Risk, Recommendation.
3.  Plotly scatter-geo — bubble map coloured by PLI magnitude.
4.  State-level heatmap — avg PLI per State as a ranked bar chart
                          (we don't have shapefile data, so a bar chart is more
                           reliable and honest than a fake choropleth).
5.  District drilldown  — table + mini bar chart for a selected State.
6.  Hotspot summary     — top-20 highest-PLI sample locations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from utils import load_dataset, RISK_COLORS, METAL_LABELS, METALS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geographic Analysis | Heavy Metal Dashboard",
    page_icon="🗺️",
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
    background:rgba(66,153,225,0.08); border-left:4px solid #4299e1;
    border-radius:8px; padding:10px 16px; font-size:0.83rem;
    color:#90cdf4; margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_dataset()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ Map Filters")
    st.markdown("---")

    states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    sel_state = st.selectbox("📍 State", states)

    mediums = ["All"] + sorted(df["Medium"].dropna().unique().tolist())
    sel_medium = st.selectbox("🧪 Medium", mediums)

    land_uses = ["All"] + sorted(df["Land_Use"].dropna().unique().tolist())
    sel_land = st.selectbox("🌾 Land Use", land_uses)

    risk_levels = ["All", "Low", "Moderate", "High", "Critical"]
    sel_risk = st.selectbox("🚦 Risk Level", risk_levels)

    sel_metal = st.selectbox(
        "🔬 Metal for Bubble Map",
        options=METALS, format_func=lambda m: METAL_LABELS[m],
    )

    map_sample_n = st.slider(
        "📍 Max Map Markers", min_value=200, max_value=2000, value=800, step=100,
        help="Limit markers for performance. Higher = slower map render.",
    )

    drilldown_state = st.selectbox(
        "🔍 District Drilldown State",
        sorted(df["State"].dropna().unique().tolist()),
    )

    st.markdown("---")
    st.caption("Reduce markers if the map is slow to load.")

# ─── Apply Filters ────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_state  != "All": fdf = fdf[fdf["State"]      == sel_state]
if sel_medium != "All": fdf = fdf[fdf["Medium"]     == sel_medium]
if sel_land   != "All": fdf = fdf[fdf["Land_Use"]   == sel_land]
if sel_risk   != "All": fdf = fdf[fdf["Risk_Level"] == sel_risk]

# Drop rows with missing coordinates
fdf = fdf.dropna(subset=["Latitude","Longitude"])

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        🗺️ Geographic Analysis
    </h1>
    <p style="color:#718096;margin:0;">
        Interactive map of pollution hotspots, spatial distribution of risk levels,
        and state/district drilldown analysis.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Synthetic Data Notice:</strong> Latitude/Longitude values are synthetically
generated and may not correspond to actual administrative boundaries.
Geographic patterns reflect the data-generation logic, not real environmental monitoring.
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("⚠️ No records match the current filters.")
    st.stop()

st.caption(f"Showing **{len(fdf):,}** records with valid coordinates after filtering.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FOLIUM INTERACTIVE MAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗺️ Interactive Pollution Map</div>', unsafe_allow_html=True)
st.caption(
    "Circle markers are coloured by Risk Level. Click any marker for details. "
    f"Showing up to {map_sample_n} sampled points for performance."
)

# Sample for performance
map_df = fdf.sample(min(map_sample_n, len(fdf)), random_state=42)

# Centre map on data centroid
center_lat = map_df["Latitude"].mean()
center_lon = map_df["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="CartoDB dark_matter",
)

# Legend
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:rgba(15,17,23,0.92);border:1px solid rgba(255,255,255,0.15);
            border-radius:10px;padding:12px 16px;font-family:Inter,sans-serif;
            font-size:13px;color:#e2e8f0;">
  <b>Risk Level</b><br>
  <span style="color:#2ecc71">●</span> Low<br>
  <span style="color:#f39c12">●</span> Moderate<br>
  <span style="color:#e74c3c">●</span> High<br>
  <span style="color:#8e44ad">●</span> Critical
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

FOLIUM_COLORS = {
    "Low":      "#2ecc71",
    "Moderate": "#f39c12",
    "High":     "#e74c3c",
    "Critical": "#8e44ad",
}

for _, row in map_df.iterrows():
    risk   = row.get("Risk_Level","Low")
    color  = FOLIUM_COLORS.get(risk,"#63b3ed")
    pli    = row.get("PLI", 0)
    radius = max(4, min(14, pli * 3))   # scale radius with PLI

    popup_html = f"""
    <div style="font-family:sans-serif;min-width:180px;">
        <b>{row.get('District','N/A')}, {row.get('State','N/A')}</b><br>
        <hr style="margin:4px 0;">
        Medium  : <b>{row.get('Medium','N/A')}</b><br>
        Land Use: <b>{row.get('Land_Use','N/A')}</b><br>
        PLI     : <b>{pli:.3f}</b><br>
        HEI     : <b>{row.get('HEI',0):.3f}</b><br>
        Risk    : <b style="color:{color};">{risk}</b><br>
        Action  : {row.get('Recommendation','N/A')}
    </div>
    """
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=radius,
        color=color, fill=True, fill_color=color, fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{risk} | PLI={pli:.2f}",
    ).add_to(m)

st_folium(m, width="100%", height=520)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PLOTLY BUBBLE MAP (scatter_geo)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">🫧 Bubble Map — {METAL_LABELS[sel_metal]} Concentration</div>', unsafe_allow_html=True)
st.caption(
    "Bubble size ∝ metal concentration. Colour = PLI magnitude. "
    "Hover for location details. Uses Plotly's natural-earth projection."
)

bubble_df = fdf.sample(min(1500, len(fdf)), random_state=7)
fig_bubble = px.scatter_geo(
    bubble_df,
    lat="Latitude", lon="Longitude",
    size=sel_metal,
    color="PLI",
    color_continuous_scale="Reds",
    hover_name="District",
    hover_data={"State":True, "Medium":True, "Risk_Level":True,
                "PLI":":.3f", sel_metal:":.3f"},
    projection="natural earth",
    labels={"PLI":"PLI","size":METAL_LABELS[sel_metal]},
)
fig_bubble.update_layout(
    paper_bgcolor="rgba(15,17,23,1)",
    geo=dict(
        bgcolor="rgba(20,22,30,1)",
        showland=True,  landcolor="#1a1f2e",
        showocean=True, oceancolor="#0d1117",
        showcoastlines=True, coastlinecolor="#2d3748",
        showframe=False,
    ),
    font_color="#e2e8f0",
    coloraxis_colorbar=dict(title="PLI"),
    margin=dict(t=10, b=10, l=10, r=10),
)
st.plotly_chart(fig_bubble, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STATE-LEVEL AVERAGE PLI (ranked bar)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 State-Level Average PLI</div>', unsafe_allow_html=True)
st.caption(
    "Ranked bar chart showing average PLI per Indian state. "
    "Note: coordinate-based state aggregation may differ from official state boundaries."
)

state_pli = (
    df.groupby("State")["PLI"]
    .mean()
    .reset_index()
    .sort_values("PLI", ascending=False)
)

fig_state = px.bar(
    state_pli, x="PLI", y="State", orientation="h",
    color="PLI", color_continuous_scale="RdYlGn_r",
    text="PLI",
    labels={"PLI":"Average PLI","State":"State"},
)
fig_state.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_state.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    yaxis=dict(autorange="reversed", showgrid=False),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    coloraxis_showscale=False,
    margin=dict(t=10, b=10, l=10, r=80),
    height=max(350, len(state_pli) * 22),
)
st.plotly_chart(fig_state, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DISTRICT DRILLDOWN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">🔍 District Drilldown — {drilldown_state}</div>', unsafe_allow_html=True)

state_data = df[df["State"] == drilldown_state]
dist_pli = (
    state_data.groupby("District")[["PLI","HEI","HPI"] + METALS]
    .mean()
    .round(4)
    .reset_index()
    .sort_values("PLI", ascending=False)
)

if not dist_pli.empty:
    fig_dist = px.bar(
        dist_pli.head(15), x="PLI", y="District", orientation="h",
        color="PLI", color_continuous_scale="Oranges",
        text="PLI",
        title=f"Top Districts by PLI in {drilldown_state}",
    )
    fig_dist.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=40, b=20, l=10, r=80),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Table
    dist_pli_disp = dist_pli.rename(columns={
        "PLI":"Avg PLI","HEI":"Avg HEI","HPI":"Avg HPI",
    })
    st.dataframe(
        dist_pli_disp.style.background_gradient(cmap="Oranges", subset=["Avg PLI"]),
        use_container_width=True,
    )
else:
    st.info(f"No data found for {drilldown_state}.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HOTSPOT TABLE: Top 20 Highest PLI Samples
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔥 Top 20 Pollution Hotspots (Highest PLI)</div>', unsafe_allow_html=True)

hotspot_cols = ["Sample_ID","Date","State","District","Latitude","Longitude",
                "Medium","Land_Use","PLI","HEI","Risk_Level","Recommendation"]
avail_hs = [c for c in hotspot_cols if c in fdf.columns]
hotspots = fdf[avail_hs].sort_values("PLI", ascending=False).head(20).reset_index(drop=True)
hotspots.index += 1

st.dataframe(
    hotspots.style.background_gradient(cmap="Reds", subset=["PLI"]),
    use_container_width=True,
)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Geographic Analysis Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
