"""
pages/7_Report_Generation.py
─────────────────────────────────────────────────────────────────────────────
Report Generation Page — Heavy Metal Pollution Dashboard
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Generate a downloadable PDF environmental risk report combining:
- Location metadata (State, District, Date, Medium, Land Use)
- Environmental parameters (Temperature, Rainfall, Humidity, Soil pH, etc.)
- Heavy metal concentrations (Pb, Hg, As, Cd, Cr)
- Pollution indices (HPI, HEI, PLI, Risk Level)
- ML model prediction and class probabilities
- SHAP top-3 feature explanation
- LSTM forecast summary

LIBRARY
───────
Uses fpdf2 (FPDF2) — lightweight, no external system dependencies, works
on Windows, Linux, macOS without LaTeX/wkhtmltopdf. Pure Python.

HOW TO USE
──────────
1. Fill in the report form on this page.
2. Click "Generate Report".
3. Click "Download PDF Report" to save the generated PDF.

NOTE ON SYNTHETIC DATA
──────────────────────
The report is clearly marked as generated from a synthetic dataset.
It is for academic demonstration purposes only.
"""

import os
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np

from utils import load_artifacts, load_dataset, align_input_row, RISK_COLORS, RECOMMENDATIONS, METAL_LABELS, METALS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Report Generation | Heavy Metal Dashboard",
    page_icon="📄",
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
.success-box {
    background:rgba(72,187,120,0.08); border-left:4px solid #48bb78;
    border-radius:8px; padding:12px 16px; font-size:0.9rem; color:#9ae6b4;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
artifacts       = load_artifacts()
df              = load_dataset()
encoder         = artifacts.get("encoder")
feature_columns = artifacts.get("feature_columns")
best_model_name = artifacts.get("best_model_name") or "Random Forest"
label_encoder   = artifacts.get("label_encoder")

MODEL_MAP = {
    "Random Forest":     artifacts.get("rf_model"),
    "XGBoost":           artifacts.get("xgb_model"),
    "LightGBM":          artifacts.get("lgbm_model"),
    "CatBoost":          artifacts.get("cat_model"),
    "Gradient Boosting": artifacts.get("gb_model"),
}
available_models = {k: v for k, v in MODEL_MAP.items() if v is not None}

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
shap_csv_path = os.path.join(ARTIFACTS_DIR, "shap_importance_df.csv")

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.2rem;">
    <h1 style="font-size:2rem;font-weight:700;margin-bottom:0.1rem;">
        📄 Environmental Risk Report Generation
    </h1>
    <p style="color:#718096;margin:0;">
        Generate a downloadable PDF report combining pollution assessment,
        ML prediction, SHAP explanation, and LSTM forecast.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Synthetic Dataset:</strong> All reports generated from this dashboard
use synthetically simulated data. Reports are for academic demonstration only —
not for actual environmental decision-making.
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Report Settings")
    st.markdown("---")
    if available_models:
        default_idx = (
            list(available_models.keys()).index(best_model_name)
            if best_model_name in available_models else 0
        )
        sel_model_name = st.selectbox("🧠 ML Model", list(available_models.keys()), index=default_idx)
    else:
        sel_model_name = None
        st.warning("No model artifacts found.")

    include_shap  = st.checkbox("Include SHAP explanation", value=True)
    include_lstm  = st.checkbox("Include LSTM forecast summary", value=True)
    st.markdown("---")
    st.caption("Fill in the form below and click 'Generate Report'.")

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT FORM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📝 Report Input Form</div>', unsafe_allow_html=True)

with st.form("report_form"):
    # ── Location & Date
    st.markdown("**📍 Location & Date**")
    l1, l2, l3 = st.columns(3)
    r_state    = l1.text_input("State",    value="Maharashtra")
    r_district = l2.text_input("District", value="Pune")
    r_date     = l3.date_input("Sample Date", value=datetime.date.today())

    # ── Sample Type
    st.markdown("**🧪 Sample Classification**")
    c1, c2 = st.columns(2)
    r_medium   = c1.selectbox("Medium",   ["Soil","Water","Air"])
    r_land_use = c2.selectbox("Land Use", ["Industrial","Agricultural","Residential","Mining","Forest"])

    # ── Environmental Parameters
    st.markdown("**🌡️ Environmental Parameters**")
    e1, e2, e3, e4 = st.columns(4)
    r_dist   = e1.number_input("Distance to Industry (km)", 0.0, 200.0, 10.0, 0.5)
    r_temp   = e2.number_input("Temperature (°C)",          -10.0, 55.0, 28.0, 0.5)
    r_rain   = e3.number_input("Rainfall (mm)",             0.0, 5000.0, 800.0, 10.0)
    r_humid  = e4.number_input("Humidity (%)",              0.0, 100.0, 60.0, 1.0)

    s1, s2 = st.columns(2)
    r_ph    = s1.number_input("Soil pH",          0.0, 14.0, 6.5, 0.1)
    r_moist = s2.number_input("Soil Moisture (%)", 0.0, 100.0, 35.0, 1.0)

    # ── Heavy Metal Concentrations
    st.markdown("**⚗️ Heavy Metal Concentrations (mg/kg or µg/m³)**")
    m1c, m2c, m3c, m4c, m5c = st.columns(5)
    r_pb  = m1c.number_input("Lead (Pb)",     0.0, 10000.0, 45.0,  0.5)
    r_hg  = m2c.number_input("Mercury (Hg)",  0.0, 500.0,   0.5,   0.01)
    r_as  = m3c.number_input("Arsenic (As)",  0.0, 2000.0,  20.0,  0.5)
    r_cd  = m4c.number_input("Cadmium (Cd)",  0.0, 200.0,   1.5,   0.1)
    r_cr  = m5c.number_input("Chromium (Cr)", 0.0, 5000.0,  80.0,  1.0)

    # ── Report Metadata
    st.markdown("**📋 Report Metadata**")
    rm1, rm2 = st.columns(2)
    r_prepared_by = rm1.text_input("Prepared By", value="B.Tech AIML Student")
    r_institution = rm2.text_input("Institution",  value="Engineering College")

    r_notes = st.text_area(
        "Additional Notes (optional)",
        value="This report is generated from a synthetic dataset for academic purposes only.",
        height=80,
    )

    submitted = st.form_submit_button("🔍 Generate Report", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
if submitted:
    raw_input = {
        "Medium": r_medium, "Land_Use": r_land_use,
        "Distance_to_Industry_km": r_dist,  "Temperature_C": r_temp,
        "Rainfall_mm": r_rain,               "Humidity_percent": r_humid,
        "Soil_pH": r_ph,                     "Soil_Moisture_percent": r_moist,
        "Lead_Pb_mgkg": r_pb,               "Mercury_Hg_mgkg": r_hg,
        "Arsenic_As_mgkg": r_as,            "Cadmium_Cd_mgkg": r_cd,
        "Chromium_Cr_mgkg": r_cr,
    }

    # ── ML Prediction ────────────────────────────────────────────────────────
    predicted_risk    = "N/A"
    recommendation    = "N/A"
    proba_str         = "N/A"
    model_used        = sel_model_name or "None"

    if encoder is not None and feature_columns and sel_model_name in available_models:
        try:
            X_input   = align_input_row(raw_input, encoder, feature_columns)
            model     = available_models[sel_model_name]
            pred_raw  = model.predict(X_input)[0]

            if isinstance(pred_raw, (np.integer, int)) and label_encoder is not None:
                predicted_risk = label_encoder.inverse_transform([int(pred_raw)])[0]
            else:
                predicted_risk = str(pred_raw)

            recommendation = RECOMMENDATIONS.get(predicted_risk, "Consult Environmental Authority")

            if hasattr(model, "predict_proba"):
                proba  = model.predict_proba(X_input)[0]
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    if label_encoder is not None and isinstance(classes[0], (np.integer, int)):
                        classes = list(label_encoder.inverse_transform(classes))
                else:
                    classes = ["Low","Moderate","High","Critical"]
                proba_str = " | ".join([f"{c}: {p:.1%}" for c, p in zip(classes, proba)])
        except Exception as e:
            predicted_risk = f"Error: {e}"

    # ── Estimated Indices ────────────────────────────────────────────────────
    ref_vals = {
        "Lead_Pb_mgkg": df["Lead_Pb_mgkg"].median(),
        "Mercury_Hg_mgkg": df["Mercury_Hg_mgkg"].median(),
        "Arsenic_As_mgkg": df["Arsenic_As_mgkg"].median(),
        "Cadmium_Cd_mgkg": df["Cadmium_Cd_mgkg"].median(),
        "Chromium_Cr_mgkg": df["Chromium_Cr_mgkg"].median(),
    }
    concs = {
        "Lead_Pb_mgkg": r_pb, "Mercury_Hg_mgkg": r_hg,
        "Arsenic_As_mgkg": r_as, "Cadmium_Cd_mgkg": r_cd, "Chromium_Cr_mgkg": r_cr,
    }
    cf_vals = {k: (concs[k] / ref_vals[k] if ref_vals[k] > 0 else 0) for k in concs}
    est_hei = sum(cf_vals.values())
    est_pli = np.prod(list(cf_vals.values())) ** (1/5) if all(v >= 0 for v in cf_vals.values()) else 0

    # ── SHAP Top Features ────────────────────────────────────────────────────
    shap_top_str = "SHAP data not available."
    if include_shap and os.path.exists(shap_csv_path):
        shap_df = pd.read_csv(shap_csv_path)
        if "Feature" in shap_df.columns and "Mean_Absolute_SHAP" in shap_df.columns:
            top3    = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False).head(3)
            rows    = []
            for _, row in top3.iterrows():
                label = METAL_LABELS.get(row["Feature"], row["Feature"].replace("_"," "))
                rows.append(f"{label}: {row['Mean_Absolute_SHAP']:.4f}")
            shap_top_str = " | ".join(rows)

    # ── LSTM Forecast Summary ─────────────────────────────────────────────────
    lstm_summary = "LSTM forecast not included."
    if include_lstm:
        lstm_model  = artifacts.get("lstm_model")
        lstm_scaler = artifacts.get("lstm_scaler")
        if lstm_model is not None and lstm_scaler is not None:
            try:
                TIME_STEPS = 30
                df_lstm    = df.copy()
                df_lstm["Date"] = pd.to_datetime(df_lstm["Date"])
                df_lstm["Total_Heavy_Metal"] = df_lstm[METALS].sum(axis=1)
                daily = (
                    df_lstm.groupby("Date")["Total_Heavy_Metal"]
                    .mean().reset_index().sort_values("Date")
                )
                scaled = lstm_scaler.transform(daily[["Total_Heavy_Metal"]])
                last_seq = scaled[-TIME_STEPS:].copy()
                future_preds = []
                for _ in range(7):
                    inp       = last_seq.reshape(1, TIME_STEPS, 1)
                    nxt       = lstm_model.predict(inp, verbose=0)[0,0]
                    future_preds.append(nxt)
                    last_seq  = np.append(last_seq[1:], [[nxt]], axis=0)
                future_actual = lstm_scaler.inverse_transform(
                    np.array(future_preds).reshape(-1,1)
                ).flatten()
                avg_f = future_actual.mean()
                min_f = future_actual.min()
                max_f = future_actual.max()
                lstm_summary = (
                    f"7-day recursive forecast — "
                    f"Avg: {avg_f:.3f} mg/kg | Min: {min_f:.3f} | Max: {max_f:.3f}"
                )
            except Exception as e:
                lstm_summary = f"LSTM forecast error: {e}"
        else:
            lstm_summary = "LSTM artifacts not found."

    # ════════════════════════════════════════════════════════════════════════════
    # BUILD PDF WITH FPDF2
    # ════════════════════════════════════════════════════════════════════════════
    try:
        from fpdf import FPDF

        def sanitize(text):
            """
            Replace non-latin-1 characters with ASCII equivalents.
            Prevents UnicodeEncodeError when using Helvetica (latin-1 only) in fpdf2.
            """
            replacements = {
                '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
                '\u201c': '"', '\u201d': '"', '\u2022': '*',
                '\u00b0': 'deg', '\u00b5': 'u', '\u03bc': 'u',
                '\u2212': '-', '\u00e9': 'e',
            }
            for char, rep in replacements.items():
                text = text.replace(char, rep)
            return text.encode('latin-1', errors='replace').decode('latin-1')

        class ReportPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 11)
                self.set_fill_color(15, 17, 23)
                self.set_text_color(99, 179, 237)
                self.cell(0, 10, "HEAVY METAL POLLUTION - ENVIRONMENTAL RISK REPORT",
                          align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(113, 128, 150)
                self.set_font("Helvetica", "", 8)
                self.cell(0, 6, "B.Tech AIML Minor Project | Synthetic Dataset - Not real environmental data",
                          align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(2)
                self.set_draw_color(99, 179, 237)
                self.set_line_width(0.5)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(113, 128, 150)
                self.cell(0, 10,
                          f"Page {self.page_no()} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                          align="C")

            def section_title(self, title):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(99, 179, 237)
                self.set_fill_color(30, 39, 56)
                self.cell(0, 8, sanitize(f"  {title}"), fill=True, new_x="LMARGIN", new_y="NEXT")
                self.ln(1)

            def two_col_row(self, label, value, label_w=55):
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(160, 174, 192)
                self.cell(label_w, 7, sanitize(label + ":"))
                self.set_font("Helvetica", "", 9)
                self.set_text_color(226, 232, 240)
                # In fpdf2 v2.8+, multi_cell(0,...) means full usable width, NOT remaining.
                # Must explicitly pass remaining width = usable_width - label_w.
                usable_w = self.w - self.l_margin - self.r_margin
                value_w  = usable_w - label_w
                self.multi_cell(value_w, 7, sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")

            def risk_badge(self, risk_level):
                colors = {
                    "Low": (46,204,113), "Moderate": (243,156,18),
                    "High": (231,76,60), "Critical": (142,68,173),
                }
                c = colors.get(risk_level, (99,179,237))
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(*c)
                self.cell(0, 12, sanitize(f"  Predicted Risk: {risk_level}"),
                          new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 9)
                self.set_text_color(160, 174, 192)
                self.cell(0, 7,
                          sanitize(f"  Recommendation: {RECOMMENDATIONS.get(risk_level,'Consult Authority')}"),
                          new_x="LMARGIN", new_y="NEXT")
                self.ln(2)

        pdf = ReportPDF()
        pdf.set_margins(12, 18, 12)
        pdf.add_page()

        # ── Report Metadata
        pdf.section_title("1. REPORT METADATA")
        pdf.two_col_row("Report Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf.two_col_row("Prepared By", r_prepared_by)
        pdf.two_col_row("Institution", r_institution)
        pdf.two_col_row("Dashboard", "Heavy Metal Pollution Prediction & Risk Assessment System")
        pdf.two_col_row("Dataset Type", "Synthetic - 10,000 records | NOT real environmental data")
        pdf.ln(3)

        # ── Location & Sample Info
        pdf.section_title("2. LOCATION & SAMPLE INFORMATION")
        pdf.two_col_row("State",          r_state)
        pdf.two_col_row("District",       r_district)
        pdf.two_col_row("Sample Date",    str(r_date))
        pdf.two_col_row("Medium",         r_medium)
        pdf.two_col_row("Land Use",       r_land_use)
        pdf.ln(3)

        # ── Environmental Parameters
        pdf.section_title("3. ENVIRONMENTAL PARAMETERS")
        pdf.two_col_row("Distance to Industry (km)", f"{r_dist:.2f}")
        pdf.two_col_row("Temperature (degC)",        f"{r_temp:.1f}")
        pdf.two_col_row("Rainfall (mm)",             f"{r_rain:.1f}")
        pdf.two_col_row("Humidity (%)",              f"{r_humid:.1f}")
        pdf.two_col_row("Soil pH",                  f"{r_ph:.2f}")
        pdf.two_col_row("Soil Moisture (%)",         f"{r_moist:.1f}")
        pdf.ln(3)

        # ── Heavy Metal Concentrations
        pdf.section_title("4. HEAVY METAL CONCENTRATIONS (mg/kg or ug/m3)")
        for metal_key, label, val in [
            ("Lead_Pb_mgkg",     "Lead (Pb)",     r_pb),
            ("Mercury_Hg_mgkg",  "Mercury (Hg)",  r_hg),
            ("Arsenic_As_mgkg",  "Arsenic (As)",  r_as),
            ("Cadmium_Cd_mgkg",  "Cadmium (Cd)",  r_cd),
            ("Chromium_Cr_mgkg", "Chromium (Cr)", r_cr),
        ]:
            cf = cf_vals.get(metal_key, 0)
            pdf.two_col_row(label, f"{val:.4f} (CF = {cf:.4f})")
        pdf.ln(3)

        # ── Pollution Indices
        pdf.section_title("5. ESTIMATED POLLUTION INDICES (Project-Defined)")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(113,128,150)
        pdf.multi_cell(0, 6, "Reference values = dataset medians. Thresholds are project-defined, not official WHO/EPA standards.")
        pdf.ln(1)
        pdf.two_col_row("Estimated HEI (sum of CFs)",         f"{est_hei:.4f}")
        pdf.two_col_row("Estimated PLI (geometric mean CFs)", f"{est_pli:.4f}")
        pdf.two_col_row("PLI Risk Mapping",
                        "< 1: Low | 1-2: Moderate | 2-3: High | >= 3: Critical")
        pdf.ln(3)

        # ── ML Prediction
        pdf.section_title("6. MACHINE LEARNING RISK PREDICTION")
        pdf.two_col_row("Model Used",          model_used)
        pdf.two_col_row("Best Model (F1)",     best_model_name)
        pdf.risk_badge(predicted_risk)
        pdf.two_col_row("Class Probabilities", proba_str)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(113,128,150)
        pdf.multi_cell(0, 6,
            "Note: High accuracy is expected because Risk_Level is derived from PLI, "
            "which is derived from the same metal concentrations used as features. "
            "This is a known synthetic-data artifact."
        )
        pdf.ln(3)

        # ── SHAP Explanation
        if include_shap:
            pdf.section_title("7. EXPLAINABLE AI — TOP SHAP FEATURES")
            pdf.two_col_row("SHAP Model",   shap_model_note if 'shap_model_note' in dir() else model_used)
            pdf.two_col_row("Top Features", shap_top_str)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(113,128,150)
            pdf.multi_cell(0, 6,
                "SHAP (SHapley Additive exPlanations) assigns each feature a value "
                "representing its contribution to the prediction. Metal features dominate "
                "because Risk_Level is derived from metal concentrations (synthetic data artifact)."
            )
            pdf.ln(3)

        # ── LSTM Forecast
        if include_lstm:
            pdf.section_title("8. LSTM POLLUTION FORECAST (7-Day Summary)")
            pdf.two_col_row("Forecast", lstm_summary)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(113,128,150)
            pdf.multi_cell(0, 6,
                "Note: LSTM forecasts are based on synthetic data with randomly assigned dates. "
                "This does not represent a genuine longitudinal monitoring series."
            )
            pdf.ln(3)

        # ── Additional Notes
        pdf.section_title("9. ADDITIONAL NOTES")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(226,232,240)
        pdf.multi_cell(0, 7, r_notes if r_notes else "None.")
        pdf.ln(3)

        # ── Disclaimer
        pdf.section_title("DISCLAIMER")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(113,128,150)
        pdf.multi_cell(0, 6,
                sanitize(
                    "This report is generated by an AI/ML system trained on a synthetically generated "
                    "dataset. It is produced as part of a B.Tech AIML Minor Project for academic and "
                    "educational purposes only. Results, indices, risk classifications, and forecasts "
                    "must NOT be used for actual environmental policy, remediation decisions, or public "
                    "health guidance. Pollution thresholds, reference values, and index formulations are "
                    "project-defined and do not represent official WHO, EPA, or Government of India standards."
                )
        )

        # Output PDF bytes
        pdf_bytes = bytes(pdf.output())

        # ── Preview in Streamlit ─────────────────────────────────────────────
        st.markdown('<div class="section-header">✅ Report Generated</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="success-box">
        ✅ PDF report generated successfully!<br>
        📍 <strong>{r_district}, {r_state}</strong> &nbsp;|&nbsp;
        🧪 {r_medium} &nbsp;|&nbsp; 🌾 {r_land_use}<br>
        🚦 Predicted Risk: <strong>{predicted_risk}</strong> &nbsp;|&nbsp;
        📋 {recommendation}
        </div>
        """, unsafe_allow_html=True)

        # ── Preview Table ─────────────────────────────────────────────────────
        preview_data = {
            "Field": [
                "State", "District", "Date", "Medium", "Land Use",
                "Temperature (°C)", "Rainfall (mm)", "Humidity (%)",
                "Soil pH", "Soil Moisture (%)", "Distance to Industry (km)",
                "Lead Pb (mg/kg)", "Mercury Hg (mg/kg)", "Arsenic As (mg/kg)",
                "Cadmium Cd (mg/kg)", "Chromium Cr (mg/kg)",
                "Est. HEI", "Est. PLI",
                "Predicted Risk", "Recommendation", "ML Model",
            ],
            "Value": [
                r_state, r_district, str(r_date), r_medium, r_land_use,
                r_temp, r_rain, r_humid,
                r_ph, r_moist, r_dist,
                r_pb, r_hg, r_as, r_cd, r_cr,
                round(est_hei, 4), round(est_pli, 4),
                predicted_risk, recommendation, model_used,
            ],
        }
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        # ── Download Button ────────────────────────────────────────────────────
        fname = f"pollution_risk_report_{r_district.replace(' ','_')}_{r_date}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )

    except ImportError:
        st.error(
            "❌ `fpdf2` is not installed. Run: `pip install fpdf2` and restart the app."
        )
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
        import traceback
        st.code(traceback.format_exc())

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Report Generation Page | Heavy Metal Pollution Prediction & Risk Assessment System | "
    "B.Tech AIML Minor Project | Synthetic Dataset — Not real environmental data."
)
