"""
pages/4_ML_Prediction.py
─────────────────────────────────────────────────────────────────────────────
ML Prediction Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Allow a user to enter environmental parameters through a form and get a
real-time pollution risk classification from any of the five trained models.

WHAT THIS PAGE DOES
───────────────────
1.  Load artifacts via load_artifacts() — models, encoder, feature_columns.
2.  User input form — Medium, Land_Use, Distance_to_Industry_km,
    Temperature_C, Rainfall_mm, Humidity_percent, Soil_pH,
    Soil_Moisture_percent, Pb, Hg, As, Cd, Cr.
3.  align_input_row() from utils.py aligns the form input to training columns.
4.  User selects which model to predict with (or best model by default).
5.  Model predicts Risk_Level; probabilities shown as a bar chart.
6.  Recommendation mapped from prediction using RECOMMENDATIONS dict.
7.  Index estimation (HEI from CF values using dataset reference medians).
8.  Viva-friendly explanation of the prediction pipeline.

DATA LEAKAGE NOTE
─────────────────
HPI, HEI, PLI, Risk_Level, Recommendation are NEVER used as input features.
Only the 13 ML features from training are used (medium/land-use + 11 numeric).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import load_artifacts, load_dataset, align_input_row, RISK_COLORS, RECOMMENDATIONS, METAL_LABELS, METALS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Prediction | Heavy Metal Dashboard",
    page_icon="🤖",
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
.pred-badge {
    border-radius:12px; padding:20px 24px; text-align:center;
    border:2px solid; margin:8px 0;
}
.pred-label { font-size:0.85rem; font-weight:600; text-transform:uppercase; color:rgba(255,255,255,0.6); }
.pred-value { font-size:2.4rem; font-weight:800; margin:4px 0; }
.pred-action { font-size:0.9rem; margin-top:6px; color:rgba(255,255,255,0.75); }
.disclaimer {
    background:rgba(66,153,225,0.08); border-left:4px solid #4299e1;
    border-radius:8px; padding:10px 16px; font-size:0.83rem;
    color:#90cdf4; margin-bottom:1rem;
}
.info-box {
    background:rgba(72,187,120,0.08); border-left:4px solid #48bb78;
    border-radius:8px; padding:10px 16px; font-size:0.83rem;
    color:#9ae6b4; margin:0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
artifacts = load_artifacts()
df        = load_dataset()

encoder         = artifacts.get("encoder")
feature_columns = artifacts.get("feature_columns")
best_model_name = artifacts.get("best_model_name") or "Random Forest"
label_encoder   = artifacts.get("label_encoder")

MODEL_MAP = {
    "Random Forest":      artifacts.get("rf_model"),
    "XGBoost":            artifacts.get("xgb_model"),
    "LightGBM":           artifacts.get("lgbm_model"),
    "CatBoost":           artifacts.get("cat_model"),
    "Gradient Boosting":  artifacts.get("gb_model"),
}
# Keep only models that actually loaded
available_models = {k: v for k, v in MODEL_MAP.items() if v is not None}

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        🤖 ML Risk Prediction
    </h1>
    <p style="color:#718096;margin:0;">
        Enter environmental parameters to predict pollution risk using trained ML models.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Leakage-Free Prediction:</strong> HPI, HEI, PLI, and Risk_Level are
<em>not</em> used as model inputs — only the 13 environmental/pollution features
used during training are supplied here. Predictions are from models trained on
synthetic data and are for educational demonstration only.
</div>
""", unsafe_allow_html=True)

# ─── Artifact Check ───────────────────────────────────────────────────────────
if encoder is None or feature_columns is None:
    st.error(
        "❌ Required artifacts (encoder.pkl / feature_columns.pkl) not found. "
        "Please ensure all model artifacts are in the `artifacts/` folder and "
        "run the training notebook first."
    )
    st.stop()

if not available_models:
    st.error("❌ No trained model artifacts found in `artifacts/`. Run the training notebook first.")
    st.stop()

# ─── Sidebar — Model Selector ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Model Settings")
    st.markdown("---")

    default_idx = (
        list(available_models.keys()).index(best_model_name)
        if best_model_name in available_models
        else 0
    )
    sel_model_name = st.selectbox(
        "🧠 Select Model",
        list(available_models.keys()),
        index=default_idx,
        help=f"Best model by F1-Score: {best_model_name}",
    )
    st.markdown(f"**Best Model (F1):** {best_model_name}")
    st.markdown("---")

    # Dataset statistics for default hints
    st.markdown("### 📊 Dataset Ranges")
    for col, label in [
        ("Distance_to_Industry_km", "Distance to Industry"),
        ("Temperature_C", "Temperature"),
        ("Rainfall_mm", "Rainfall"),
    ]:
        if col in df.columns:
            st.caption(f"{label}: {df[col].min():.1f} – {df[col].max():.1f}")

# ─── INPUT FORM ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Environmental Input Parameters</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    # ── Categorical
    with c1:
        st.markdown("**🏷️ Categorical Parameters**")
        medium = st.selectbox(
            "Environmental Medium", ["Soil", "Water", "Air"],
            help="The medium from which the sample was collected."
        )
        land_use = st.selectbox(
            "Land Use Type",
            ["Industrial", "Agricultural", "Residential", "Mining", "Forest"],
            help="The dominant land use at the sampling location."
        )

    # ── Environmental Numeric
    with c2:
        st.markdown("**🌡️ Environmental Parameters**")
        dist_ind = st.number_input(
            "Distance to Industry (km)", min_value=0.0, max_value=200.0, value=10.0, step=0.5,
        )
        temp_c = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5,
        )
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0,
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**🌱 Soil Parameters**")
        soil_ph = st.number_input(
            "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1,
        )
        soil_moist = st.number_input(
            "Soil Moisture (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0,
        )

    with c4:
        st.markdown("**⚗️ Heavy Metal Concentrations (mg/kg or µg/m³)**")
        pb  = st.number_input("Lead (Pb)",     min_value=0.0, max_value=10000.0, value=45.0,  step=0.5)
        hg  = st.number_input("Mercury (Hg)",  min_value=0.0, max_value=500.0,   value=0.5,   step=0.01)
        ars = st.number_input("Arsenic (As)",  min_value=0.0, max_value=2000.0,  value=20.0,  step=0.5)
        cd  = st.number_input("Cadmium (Cd)",  min_value=0.0, max_value=200.0,   value=1.5,   step=0.1)
        cr  = st.number_input("Chromium (Cr)", min_value=0.0, max_value=5000.0,  value=80.0,  step=1.0)

    submitted = st.form_submit_button("🔍 Predict Risk Level", use_container_width=True)

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if submitted:
    raw_input = {
        "Medium":                   medium,
        "Land_Use":                 land_use,
        "Distance_to_Industry_km":  dist_ind,
        "Temperature_C":            temp_c,
        "Rainfall_mm":              rainfall,
        "Humidity_percent":         humidity,
        "Soil_pH":                  soil_ph,
        "Soil_Moisture_percent":    soil_moist,
        "Lead_Pb_mgkg":             pb,
        "Mercury_Hg_mgkg":          hg,
        "Arsenic_As_mgkg":          ars,
        "Cadmium_Cd_mgkg":          cd,
        "Chromium_Cr_mgkg":         cr,
    }

    # Align to training feature order
    X_input = align_input_row(raw_input, encoder, feature_columns)

    # Select model
    model = available_models[sel_model_name]

    # Predict
    try:
        # For XGBoost — target may be label-encoded integers
        y_pred_raw = model.predict(X_input)[0]

        # Decode if label encoder was applied (XGBoost path)
        if isinstance(y_pred_raw, (np.integer, int)) and label_encoder is not None:
            predicted_risk = label_encoder.inverse_transform([int(y_pred_raw)])[0]
        else:
            predicted_risk = str(y_pred_raw)

        # Probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            # Get class labels
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if label_encoder is not None and isinstance(classes[0], (np.integer, int)):
                    classes = list(label_encoder.inverse_transform(classes))
            else:
                classes = ["Low","Moderate","High","Critical"]
        else:
            proba   = None
            classes = None

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    recommendation = RECOMMENDATIONS.get(predicted_risk, "Consult Environmental Authority")
    risk_color     = RISK_COLORS.get(predicted_risk, "#63b3ed")

    # ─── Result Display ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown(
            f'<div class="pred-badge" style="border-color:{risk_color};'
            f'background:linear-gradient(135deg,{risk_color}22,{risk_color}11);">'
            f'<div class="pred-label">Predicted Risk Level</div>'
            f'<div class="pred-value" style="color:{risk_color};">{predicted_risk}</div>'
            f'<div class="pred-action">📋 {recommendation}</div>'
            f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:6px;">'
            f'Model: {sel_model_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r2:
        if proba is not None and classes is not None:
            risk_order = ["Low", "Moderate", "High", "Critical"]
            # Align proba to risk_order
            prob_dict = dict(zip(classes, proba))
            ordered_classes = [c for c in risk_order if c in prob_dict]
            ordered_proba   = [prob_dict[c] for c in ordered_classes]
            ordered_colors  = [RISK_COLORS.get(c, "#63b3ed") for c in ordered_classes]

            fig_prob = go.Figure(go.Bar(
                x=ordered_classes,
                y=ordered_proba,
                marker_color=ordered_colors,
                text=[f"{p:.1%}" for p in ordered_proba],
                textposition="outside",
            ))
            fig_prob.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title="Class Probabilities",
                title_font_size=13,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           range=[0, 1.1], title="Probability"),
                margin=dict(t=40, b=20, l=10, r=10), showlegend=False,
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("This model does not support probability estimates.")

    # ─── Estimated Index Values ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Estimated Contamination Factors (Indicative)</div>', unsafe_allow_html=True)
    st.caption(
        "CF = Metal_Concentration / Reference_Value. Reference values used here are "
        "dataset-level medians from the training data — not official WHO/EPA standards. "
        "These are indicative estimates only."
    )

    # Use dataset medians as reference values
    ref_vals = {
        "Lead_Pb_mgkg":     df["Lead_Pb_mgkg"].median(),
        "Mercury_Hg_mgkg":  df["Mercury_Hg_mgkg"].median(),
        "Arsenic_As_mgkg":  df["Arsenic_As_mgkg"].median(),
        "Cadmium_Cd_mgkg":  df["Cadmium_Cd_mgkg"].median(),
        "Chromium_Cr_mgkg": df["Chromium_Cr_mgkg"].median(),
    }
    concs = {
        "Lead_Pb_mgkg": pb, "Mercury_Hg_mgkg": hg,
        "Arsenic_As_mgkg": ars, "Cadmium_Cd_mgkg": cd, "Chromium_Cr_mgkg": cr,
    }
    cf_vals = {k: (concs[k] / ref_vals[k] if ref_vals[k] > 0 else 0) for k in concs}

    est_hei = sum(cf_vals.values())
    est_pli = np.prod(list(cf_vals.values())) ** (1/5) if all(v >= 0 for v in cf_vals.values()) else 0

    cf_data = pd.DataFrame({
        "Metal":                [METAL_LABELS.get(k,k) for k in cf_vals],
        "Input Concentration":  [concs[k] for k in cf_vals],
        "Reference (Median)":   [round(ref_vals[k], 4) for k in cf_vals],
        "CF (estimated)":       [round(cf_vals[k], 4) for k in cf_vals],
    })

    ix1, ix2, ix3 = st.columns(3)
    ix1.metric("Σ CF (est. HEI)", f"{est_hei:.4f}")
    ix2.metric("∏CF^(1/5) (est. PLI)", f"{est_pli:.4f}")
    ix3.metric("Predicted Risk", predicted_risk)

    st.dataframe(
        cf_data.style.background_gradient(cmap="YlOrRd", subset=["CF (estimated)"]),
        use_container_width=True,
    )

    # ─── Viva Pipeline Explanation ────────────────────────────────────────────
    with st.expander("🎓 How the prediction works (Viva Explanation)"):
        st.markdown(f"""
**Pipeline Summary:**

1. **Input** — You entered environmental and metal concentration values (13 features total).
2. **Encoding** — `Medium` and `Land_Use` are one-hot encoded using the saved `encoder.pkl`.
3. **Alignment** — `align_input_row()` reindexes the encoded row to match the exact column
   order in `feature_columns.pkl` (the same order used during training). Missing dummy
   columns are filled with 0.
4. **Prediction** — `{sel_model_name}` predicts the `Risk_Level` class.
5. **Probability** — `predict_proba()` gives confidence across all four risk classes.
6. **Recommendation** — mapped from the predicted class using the `RECOMMENDATIONS` dict.

**Why is accuracy so high on training data?**
Because `Risk_Level` is derived from `PLI`, which is derived from the same metal
concentrations used as features. The model essentially re-learns the PLI formula.
This is a known synthetic-data artifact and would not generalise to real-world data.

**Leakage prevention:** HPI, HEI, PLI, and Risk_Level are *never* fed as inputs.
        """)

# ─── Model Comparison Reference ───────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Comparison (Training Results)</div>', unsafe_allow_html=True)

import os, joblib
results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "results_df.csv")
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    st.dataframe(
        results_df.style.highlight_max(subset=[c for c in results_df.columns if c != "Model"], color="#2d4a2d"),
        use_container_width=True,
    )
    st.caption("Green cells indicate the best value per metric column.")
else:
    st.info("results_df.csv not found in artifacts/. Run the training notebook to generate it.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 ML Prediction Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
