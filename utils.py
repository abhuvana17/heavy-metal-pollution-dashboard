"""
utils.py — Shared utilities for the Heavy Metal Pollution Dashboard
Centralises artifact loading, dataset loading, input alignment, and constants.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(BASE_DIR, "artifacts")
DATASET_PATH = os.path.join(BASE_DIR, "heavy_metal_pollution_dataset.csv")

# ─── Shared Constants ─────────────────────────────────────────────────────────

RISK_COLORS = {
    "Low":      "#2ecc71",   # green
    "Moderate": "#f39c12",   # amber
    "High":     "#e74c3c",   # red
    "Critical": "#8e44ad",   # purple
}

RECOMMENDATIONS = {
    "Low":      "Continue Monitoring",
    "Moderate": "Increase Monitoring Frequency",
    "High":     "Soil and Water Remediation",
    "Critical": "Immediate Government Intervention",
}

METALS = ["Lead_Pb_mgkg", "Mercury_Hg_mgkg", "Arsenic_As_mgkg",
          "Cadmium_Cd_mgkg", "Chromium_Cr_mgkg"]

METAL_LABELS = {
    "Lead_Pb_mgkg":     "Lead (Pb)",
    "Mercury_Hg_mgkg":  "Mercury (Hg)",
    "Arsenic_As_mgkg":  "Arsenic (As)",
    "Cadmium_Cd_mgkg":  "Cadmium (Cd)",
    "Chromium_Cr_mgkg": "Chromium (Cr)",
}

# ─── Artifact Loader ──────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """
    Load every model, encoder, and metadata file from the artifacts/ folder.
    Returns a dict keyed by logical name.
    Cached as a resource so models are loaded only once per session.
    """
    def _load(filename):
        path = os.path.join(ARTIFACTS, filename)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def _load_keras(filename):
        path = os.path.join(ARTIFACTS, filename)
        if not os.path.exists(path):
            return None
        try:
            import tensorflow as tf          # noqa: F401
            return tf.keras.models.load_model(path)
        except ImportError:
            # TensorFlow not installed (e.g. Python 3.14 — no TF wheel yet).
            # The LSTM page will show an informative warning instead of crashing.
            return None
        except Exception:
            return None

    artifacts = {
        "encoder":          _load("encoder.pkl"),
        "label_encoder":    _load("label_encoder.pkl"),
        "rf_model":         _load("rf_model.pkl"),
        "xgb_model":        _load("xgb_model.pkl"),
        "lgbm_model":       _load("lgbm_model.pkl"),
        "cat_model":        _load("cat_model.pkl"),
        "gb_model":         _load("gb_model.pkl"),
        "best_model_name":  _load("best_model_name.pkl"),
        "feature_columns":  _load("feature_columns.pkl"),
        "lstm_model":       _load_keras("lstm_model.keras"),
        "lstm_scaler":      _load("lstm_scaler.pkl"),
    }
    return artifacts

# ─── Dataset Loader ───────────────────────────────────────────────────────────

@st.cache_data
def load_dataset():
    """
    Load the 10,000-record synthetic dataset.
    Returns a cleaned DataFrame with Date parsed and Year extracted.
    """
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    return df

# ─── Input Alignment ──────────────────────────────────────────────────────────

def align_input_row(raw_input: dict, encoder, feature_columns: list) -> pd.DataFrame:
    """
    Convert a user-supplied dict of raw field values into a single-row
    DataFrame whose columns exactly match feature_columns (training order).

    Steps
    -----
    1. Extract categorical fields (Medium, Land_Use) and one-hot-encode them
       using the saved encoder (same transform used during training).
    2. Combine the encoded categorical part with the numeric fields.
    3. Reindex to feature_columns — fills any unseen dummies with 0.

    Parameters
    ----------
    raw_input      : dict  — keys match the form field names
    encoder        : fitted OneHotEncoder (for Medium & Land_Use)
    feature_columns: list  — column order from feature_columns.pkl

    Returns
    -------
    pd.DataFrame with shape (1, len(feature_columns))
    """
    import numpy as np

    # Numeric fields (order must match training X)
    numeric_fields = [
        "Distance_to_Industry_km", "Temperature_C", "Rainfall_mm",
        "Humidity_percent", "Soil_pH", "Soil_Moisture_percent",
        "Lead_Pb_mgkg", "Mercury_Hg_mgkg", "Arsenic_As_mgkg",
        "Cadmium_Cd_mgkg", "Chromium_Cr_mgkg",
    ]

    # One-hot encode categorical columns
    cat_df = pd.DataFrame([[raw_input["Medium"], raw_input["Land_Use"]]],
                          columns=["Medium", "Land_Use"])
    encoded_cats = encoder.transform(cat_df)
    cat_col_names = encoder.get_feature_names_out(["Medium", "Land_Use"])
    cat_part = pd.DataFrame(encoded_cats, columns=cat_col_names)

    # Numeric part
    num_part = pd.DataFrame(
        [[raw_input[f] for f in numeric_fields]],
        columns=numeric_fields
    )

    # Combine and reindex to exact training column order
    combined = pd.concat([num_part, cat_part], axis=1)
    combined = combined.reindex(columns=feature_columns, fill_value=0)
    return combined
