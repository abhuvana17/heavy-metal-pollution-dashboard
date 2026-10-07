"""
pages/5_Explainable_AI.py
─────────────────────────────────────────────────────────────────────────────
Explainable AI (SHAP) Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Explain the ML model's predictions using SHAP (SHapley Additive exPlanations),
showing which features drive risk classification decisions.

WHAT THIS PAGE DOES
───────────────────
1.  Load shap_importance_df.csv from artifacts/ (pre-computed SHAP values).
2.  Bar chart — mean absolute SHAP importance per feature (global).
3.  Feature importance ranking table with clear metal/environmental labels.
4.  Single-prediction SHAP waterfall explanation (using the best/selected model
    and a randomly sampled or user-selected test point from the dataset).
5.  Known workaround note: GradientBoostingClassifier only supports binary SHAP
    via TreeExplainer; LightGBM is used for SHAP when GB is the best model.
6.  Viva explanation of SHAP methodology.

SHAP BACKGROUND (for the viva)
───────────────────────────────
SHAP assigns each feature a value representing its contribution to the prediction
for a specific sample, grounded in cooperative game theory (Shapley values).
TreeExplainer is used for tree-based models (RF, XGB, LGBM, CatBoost, GB).
Positive SHAP → pushes prediction toward the class; negative → away from it.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_artifacts, load_dataset, METAL_LABELS, METALS, RISK_COLORS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable AI | Heavy Metal Dashboard",
    page_icon="🔍",
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
.shap-note {
    background:rgba(159,122,234,0.08); border-left:4px solid #9f7aea;
    border-radius:8px; padding:10px 16px; font-size:0.83rem;
    color:#d6bcfa; margin:0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts & Data ────────────────────────────────────────────────────
artifacts       = load_artifacts()
df              = load_dataset()
best_model_name = artifacts.get("best_model_name") or "Unknown"
feature_columns = artifacts.get("feature_columns") or []

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
shap_csv_path = os.path.join(ARTIFACTS_DIR, "shap_importance_df.csv")

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        🔍 Explainable AI — SHAP Analysis
    </h1>
    <p style="color:#718096;margin:0;">
        Understanding which features drive the ML model's risk-level predictions
        using SHapley Additive exPlanations (SHAP).
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Synthetic Data:</strong> SHAP values reflect patterns in synthetically
generated data. Because Risk_Level is derived from metal concentrations via PLI,
SHAP will naturally rank metal features as highest-importance — this is an expected
artifact of the data generation, not a generalisable real-world finding.
</div>
""", unsafe_allow_html=True)

# ─── SHAP Workaround Note ─────────────────────────────────────────────────────
if best_model_name == "Gradient Boosting":
    st.markdown("""
    <div class="shap-note">
    ⚠️ <strong>Known SHAP Workaround:</strong> GradientBoostingClassifier only supports
    binary classification in SHAP's TreeExplainer. Since this is a 4-class problem,
    <strong>LightGBM</strong> was used as the SHAP model instead of Gradient Boosting.
    This substitution is documented here and does not affect the main classification results.
    </div>
    """, unsafe_allow_html=True)
    shap_model_note = "LightGBM (used for SHAP — see workaround note above)"
else:
    shap_model_note = best_model_name

st.markdown(f"**SHAP Model:** {shap_model_note} &nbsp;|&nbsp; **Best Classifier (F1):** {best_model_name}")

# ─── SHAP Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 SHAP Settings")
    st.markdown("---")
    top_n = st.slider("Top N Features", min_value=5, max_value=20, value=13)
    st.markdown("---")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)**

    SHAP values quantify how much each feature
    contributes to moving the model output from
    the expected value toward the actual prediction.

    - Positive SHAP → increases predicted class probability
    - Negative SHAP → decreases predicted class probability
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GLOBAL SHAP IMPORTANCE (from pre-computed CSV)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🌐 Global Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)

if os.path.exists(shap_csv_path):
    shap_df = pd.read_csv(shap_csv_path)

    # Ensure expected columns
    if "Feature" not in shap_df.columns or "Mean_Absolute_SHAP" not in shap_df.columns:
        st.error("shap_importance_df.csv must have columns: Feature, Mean_Absolute_SHAP")
        st.stop()

    # Map technical column names to readable labels
    label_map = {**METAL_LABELS}
    shap_df["Feature_Label"] = shap_df["Feature"].apply(
        lambda f: label_map.get(f, f.replace("_", " ").replace("Medium ", "Medium: ").replace("Land Use ", "Land Use: "))
    )

    shap_top = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False).head(top_n)

    # Colour gradient: highest SHAP = most intense
    max_shap = shap_top["Mean_Absolute_SHAP"].max()
    colors   = [
        f"rgba(159,122,234,{0.4 + 0.6 * v/max_shap})"
        for v in shap_top["Mean_Absolute_SHAP"]
    ]

    fig_shap = go.Figure(go.Bar(
        x=shap_top["Mean_Absolute_SHAP"].values,
        y=shap_top["Feature_Label"].values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in shap_top["Mean_Absolute_SHAP"].values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Mean |SHAP Value|"),
        yaxis=dict(autorange="reversed", showgrid=False, title="Feature"),
        margin=dict(t=10, b=20, l=10, r=80),
        height=max(300, top_n * 32),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Table
    st.markdown('<div class="section-header">📋 SHAP Importance Table</div>', unsafe_allow_html=True)
    shap_table = shap_top[["Feature_Label","Mean_Absolute_SHAP"]].copy()
    shap_table.columns = ["Feature","Mean |SHAP|"]
    shap_table = shap_table.reset_index(drop=True)
    shap_table.index += 1
    shap_table["Mean |SHAP|"] = shap_table["Mean |SHAP|"].round(6)
    st.dataframe(
        shap_table.style.background_gradient(cmap="Purples", subset=["Mean |SHAP|"]),
        use_container_width=True,
    )

    # ─── Insight callout ──────────────────────────────────────────────────────
    top_feat = shap_top.iloc[0]["Feature_Label"]
    st.markdown(f"""
    <div class="shap-note">
    💡 <strong>Key Insight:</strong> <em>{top_feat}</em> has the highest mean absolute
    SHAP value, meaning it has the largest average impact on the model's risk-level
    prediction across the test set. Because Risk_Level is derived from PLI (which is
    derived from metal concentrations), metal features dominate — this is expected in
    the synthetic dataset.
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning(
        "⚠️ `shap_importance_df.csv` not found in `artifacts/`. "
        "Please run the training/SHAP notebook to generate this file."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SINGLE-SAMPLE SHAP WATERFALL (live computation)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔬 Single-Sample SHAP Explanation (Waterfall)</div>', unsafe_allow_html=True)
st.caption(
    "Select a sample from the dataset to see how each feature pushed the "
    "model's prediction away from or toward a specific risk class."
)

# Check which model to use for SHAP
MODEL_MAP = {
    "Random Forest":     artifacts.get("rf_model"),
    "XGBoost":           artifacts.get("xgb_model"),
    "LightGBM":          artifacts.get("lgbm_model"),
    "CatBoost":          artifacts.get("cat_model"),
    "Gradient Boosting": artifacts.get("gb_model"),
}

shap_model_key = best_model_name
# Workaround: GB → use LightGBM for SHAP
if shap_model_key == "Gradient Boosting" and MODEL_MAP.get("LightGBM") is not None:
    shap_model_key = "LightGBM"

shap_model = MODEL_MAP.get(shap_model_key)
encoder    = artifacts.get("encoder")

if shap_model is None or encoder is None or not feature_columns:
    st.info("Live SHAP explanations require: model artifact, encoder.pkl, and feature_columns.pkl.")
else:
    from utils import align_input_row

    # Build encoded test set (sample 500 for UI speed)
    NUMERIC_FEATURES = [
        "Distance_to_Industry_km","Temperature_C","Rainfall_mm","Humidity_percent",
        "Soil_pH","Soil_Moisture_percent","Lead_Pb_mgkg","Mercury_Hg_mgkg",
        "Arsenic_As_mgkg","Cadmium_Cd_mgkg","Chromium_Cr_mgkg",
    ]
    CAT_FEATURES = ["Medium","Land_Use"]

    sample_pool = df.dropna(subset=NUMERIC_FEATURES + CAT_FEATURES).sample(
        min(500, len(df)), random_state=99
    )

    # Encode categorical
    try:
        cat_encoded = encoder.transform(sample_pool[CAT_FEATURES])
        cat_names   = encoder.get_feature_names_out(CAT_FEATURES)
        cat_part    = pd.DataFrame(cat_encoded, columns=cat_names, index=sample_pool.index)
        num_part    = sample_pool[NUMERIC_FEATURES].reset_index(drop=True)
        cat_part    = cat_part.reset_index(drop=True)
        X_pool      = pd.concat([num_part, cat_part], axis=1)
        X_pool      = X_pool.reindex(columns=feature_columns, fill_value=0)
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    sample_idx = st.slider(
        "Select sample index (from 500-sample pool)",
        min_value=0, max_value=len(X_pool)-1, value=0,
    )

    sample_row      = X_pool.iloc[[sample_idx]]
    actual_risk     = sample_pool.iloc[sample_idx].get("Risk_Level", "N/A")
    actual_district = sample_pool.iloc[sample_idx].get("District", "N/A")
    actual_state    = sample_pool.iloc[sample_idx].get("State", "N/A")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("State",     actual_state)
    col_info2.metric("District",  actual_district)
    col_info3.metric("Actual Risk", actual_risk,
                     delta_color="off",
                     help="Risk level from the dataset (derived from PLI)")

    if st.button("🔬 Compute SHAP for this Sample"):
        with st.spinner("Computing SHAP values (this may take a moment)…"):
            try:
                import shap
                explainer   = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(sample_row)

                # Multiclass → shap_values is (samples, features, classes) or list of arrays
                # Determine predicted class index for waterfall
                pred_encoded = shap_model.predict(sample_row)[0]
                label_encoder = artifacts.get("label_encoder")
                if isinstance(pred_encoded, (np.integer, int)) and label_encoder is not None:
                    pred_class = label_encoder.inverse_transform([int(pred_encoded)])[0]
                else:
                    pred_class = str(pred_encoded)

                risk_order = ["Low","Moderate","High","Critical"]

                # Determine class index for SHAP slice
                if hasattr(shap_model, "classes_"):
                    classes = list(shap_model.classes_)
                    if label_encoder is not None and isinstance(classes[0], (np.integer, int)):
                        classes = list(label_encoder.inverse_transform(classes))
                else:
                    classes = risk_order

                # Handle both list-of-arrays and 3D array formats
                if isinstance(shap_values, list):
                    # list of arrays, one per class
                    cls_idx = classes.index(pred_class) if pred_class in classes else 0
                    sv = shap_values[cls_idx][0]          # shape: (features,)
                elif shap_values.ndim == 3:
                    cls_idx = classes.index(pred_class) if pred_class in classes else 0
                    sv = shap_values[0, :, cls_idx]       # shape: (features,)
                else:
                    sv = shap_values[0]

                # Build waterfall data
                feat_names = list(sample_row.columns)
                label_map  = {**METAL_LABELS}
                feat_labels = [
                    label_map.get(f, f.replace("_"," ")) for f in feat_names
                ]

                sv_df = pd.DataFrame({
                    "Feature": feat_labels,
                    "SHAP_Value": sv,
                }).sort_values("SHAP_Value", key=abs, ascending=False).head(15)

                colors_wf = [
                    "#fc8181" if v > 0 else "#68d391"
                    for v in sv_df["SHAP_Value"]
                ]

                fig_wf = go.Figure(go.Bar(
                    x=sv_df["SHAP_Value"],
                    y=sv_df["Feature"],
                    orientation="h",
                    marker_color=colors_wf,
                    text=[f"{v:+.4f}" for v in sv_df["SHAP_Value"]],
                    textposition="outside",
                ))
                fig_wf.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                fig_wf.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title=f"SHAP Waterfall — Predicted: {pred_class} | Actual: {actual_risk}",
                    title_font_size=14,
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="SHAP Value"),
                    yaxis=dict(autorange="reversed", showgrid=False),
                    margin=dict(t=40, b=20, l=10, r=80),
                    height=450,
                )
                st.plotly_chart(fig_wf, use_container_width=True)

                st.caption(
                    "🔴 Red bars push prediction toward the predicted class. "
                    "🟢 Green bars push prediction away from the predicted class. "
                    "Values are for the predicted class only."
                )

            except Exception as e:
                st.error(f"❌ SHAP computation failed: {e}")
                st.info(
                    "If SHAP fails with a 'binary only' error, this is because "
                    "GradientBoostingClassifier does not support multiclass SHAP via TreeExplainer. "
                    "Set the SHAP model to LightGBM in the training notebook."
                )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VIVA EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🎓 SHAP Methodology — Viva Explanation"):
    st.markdown("""
**What is SHAP?**

SHAP (SHapley Additive exPlanations) is a framework for explaining individual
machine-learning predictions. It is grounded in cooperative game theory:
each feature's "Shapley value" represents its fair contribution to the prediction,
computed by averaging over all possible subsets of features.

**Why TreeExplainer?**

`shap.TreeExplainer` is optimised for tree-based models (Random Forest, XGBoost,
LightGBM, CatBoost, Gradient Boosting). It computes exact Shapley values in
polynomial time, unlike the model-agnostic KernelExplainer which is much slower.

**Multiclass Output:**

For a 4-class problem (Low/Moderate/High/Critical), SHAP produces values for
each class. We visualise SHAP for the predicted class, showing which features
pushed the model to choose that class over others.

**Why do metal features dominate?**

Risk_Level is derived from PLI, which is derived from the five metal concentrations
used as features. The model essentially learns the PLI formula — so SHAP correctly
identifies metal concentrations as the dominant drivers. This is an expected
synthetic-data artifact; in real-world data, environmental and geographical features
may play a larger role.

**Known Limitation:**

GradientBoostingClassifier in scikit-learn supports only binary SHAP via
TreeExplainer for multiclass problems. When GB is the best classifier, LightGBM
is substituted for SHAP computation. This is explicitly documented here.
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Explainable AI Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
