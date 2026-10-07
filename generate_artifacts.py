"""
generate_artifacts.py
─────────────────────────────────────────────────────────────────────────────
Trains all 5 ML classifiers + computes SHAP values + saves all artifacts
needed by the Streamlit dashboard.

Run from inside pollution_dashboard/:
    python generate_artifacts.py

Outputs to artifacts/:
    encoder.pkl, label_encoder.pkl, feature_columns.pkl
    rf_model.pkl, xgb_model.pkl, lgbm_model.pkl, cat_model.pkl, gb_model.pkl
    best_model_name.pkl, results_df.csv, shap_importance_df.csv

NOTE: lstm_model.keras and lstm_scaler.pkl are NOT generated here because
TensorFlow is not available on Python 3.14. They must be generated in
Google Colab with Python 3.10/3.11 and copied here.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save(obj, name):
    path = os.path.join(ARTIFACTS_DIR, name)
    joblib.dump(obj, path)
    print(f"  Saved: {name}")

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv("heavy_metal_pollution_dataset.csv", parse_dates=["Date"])
print(f"      {len(df):,} records, {len(df.columns)} columns")

# ─── 2. Feature Engineering ───────────────────────────────────────────────────
print("\n[2/7] Preparing features...")

NUMERIC_FEATURES = [
    "Distance_to_Industry_km", "Temperature_C", "Rainfall_mm", "Humidity_percent",
    "Soil_pH", "Soil_Moisture_percent",
    "Lead_Pb_mgkg", "Mercury_Hg_mgkg", "Arsenic_As_mgkg", "Cadmium_Cd_mgkg", "Chromium_Cr_mgkg",
]
CAT_FEATURES = ["Medium", "Land_Use"]
TARGET       = "Risk_Level"

# NOTE: HPI, HEI, PLI, Risk_Level, Recommendation intentionally excluded
#       to avoid target leakage.

X_num = df[NUMERIC_FEATURES]
X_cat = df[CAT_FEATURES]
y     = df[TARGET]

# One-hot encode categorical
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_enc = encoder.fit_transform(X_cat)
cat_col_names = encoder.get_feature_names_out(CAT_FEATURES)
X_cat_df  = pd.DataFrame(X_cat_enc, columns=cat_col_names, index=X_num.index)

X = pd.concat([X_num, X_cat_df], axis=1)
feature_columns = list(X.columns)
print(f"      Features: {len(feature_columns)} columns")
print(f"      Classes : {sorted(y.unique())}")

save(encoder, "encoder.pkl")
save(feature_columns, "feature_columns.pkl")

# ─── 3. Train/Test Split ──────────────────────────────────────────────────────
print("\n[3/7] Splitting dataset (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Label-encode target for XGBoost
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
save(le, "label_encoder.pkl")

# ─── 4. Train Models ──────────────────────────────────────────────────────────
print("\n[4/7] Training models...")

results = []

def evaluate(name, model, X_tr, y_tr, X_te, y_te, encoded_target=False):
    """Fit, evaluate, and save a model."""
    print(f"  Training {name}...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    if encoded_target:
        y_pred = le.inverse_transform(y_pred)
        y_te   = le.inverse_transform(y_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    print(f"    Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    results.append({"Model": name, "Accuracy": round(acc,4), "Precision": round(prec,4),
                    "Recall": round(rec,4), "F1_Score": round(f1,4)})
    return model

# Random Forest
rf = evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test,
)
save(rf, "rf_model.pkl")

# Gradient Boosting
gb = evaluate(
    "Gradient Boosting",
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test,
)
save(gb, "gb_model.pkl")

# XGBoost
try:
    from xgboost import XGBClassifier
    xgb = evaluate(
        "XGBoost",
        XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                      eval_metric="mlogloss", verbosity=0),
        X_train, y_train_enc, X_test, y_test_enc,
        encoded_target=True,
    )
    save(xgb, "xgb_model.pkl")
except ImportError:
    print("  XGBoost not available — skipping.")
    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))  # fallback

# LightGBM
try:
    from lightgbm import LGBMClassifier
    lgbm = evaluate(
        "LightGBM",
        LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
        X_train, y_train, X_test, y_test,
    )
    save(lgbm, "lgbm_model.pkl")
except ImportError:
    print("  LightGBM not available — skipping.")
    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))  # fallback

# CatBoost
try:
    from catboost import CatBoostClassifier
    cat = evaluate(
        "CatBoost",
        CatBoostClassifier(iterations=200, random_seed=42, verbose=0),
        X_train, y_train, X_test, y_test,
    )
    save(cat, "cat_model.pkl")
except ImportError:
    print("  CatBoost not available — skipping.")
    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "cat_model.pkl"))  # fallback

# ─── 5. Select Best Model ─────────────────────────────────────────────────────
print("\n[5/7] Selecting best model by F1 Score...")
results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\n  Best model: {best_model_name}")

results_df.to_csv(os.path.join(ARTIFACTS_DIR, "results_df.csv"), index=False)
save(best_model_name, "best_model_name.pkl")

# ─── 6. SHAP Feature Importance ───────────────────────────────────────────────
print("\n[6/7] Computing SHAP feature importance...")
try:
    import shap

    # Choose SHAP model — GradientBoosting → use LightGBM (multiclass workaround)
    # documented in 5_Explainable_AI.py
    model_map = {}
    for row in results:
        n = row["Model"]
        pkl = {"Random Forest":"rf_model.pkl","XGBoost":"xgb_model.pkl",
               "LightGBM":"lgbm_model.pkl","CatBoost":"cat_model.pkl",
               "Gradient Boosting":"gb_model.pkl"}.get(n)
        if pkl:
            model_map[n] = joblib.load(os.path.join(ARTIFACTS_DIR, pkl))

    shap_model_name = best_model_name
    if shap_model_name == "Gradient Boosting" and "LightGBM" in model_map:
        shap_model_name = "LightGBM"
        print(f"  (GB workaround) Using LightGBM for SHAP instead of Gradient Boosting")

    shap_model = model_map.get(shap_model_name, rf)

    # Use up to 1000 test samples for SHAP
    n_shap = min(1000, len(X_test))
    X_shap = X_test.iloc[:n_shap]

    print(f"  Running TreeExplainer on {n_shap} samples...")
    explainer   = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap)

    # Handle list-of-arrays (multiclass) or 3D array
    if isinstance(shap_values, list):
        # List: one array per class — mean abs across all classes
        sv_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        sv_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        sv_abs = np.abs(shap_values).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "Feature":             feature_columns,
        "Mean_Absolute_SHAP":  sv_abs.tolist(),
    }).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    shap_importance_df.to_csv(
        os.path.join(ARTIFACTS_DIR, "shap_importance_df.csv"), index=False
    )
    print(f"  SHAP top-5 features:")
    print(shap_importance_df.head(5).to_string(index=False))

except Exception as e:
    print(f"  SHAP failed: {e}")
    # Create dummy SHAP file so dashboard doesn't crash
    dummy_shap = pd.DataFrame({"Feature": feature_columns,
                               "Mean_Absolute_SHAP": np.zeros(len(feature_columns))})
    dummy_shap.to_csv(os.path.join(ARTIFACTS_DIR, "shap_importance_df.csv"), index=False)
    print("  Saved dummy shap_importance_df.csv")

# ─── 7. Done ──────────────────────────────────────────────────────────────────
print("\n[7/7] All artifacts saved to artifacts/")
print("\nFiles in artifacts/:")
for f in sorted(os.listdir(ARTIFACTS_DIR)):
    size = os.path.getsize(os.path.join(ARTIFACTS_DIR, f))
    print(f"  {f:40s}  {size/1024:.1f} KB")

print("\nDone. Run: streamlit run app.py")
