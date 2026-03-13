"""
XGBoost training script.

Usage:
    python -m src.models.train_xgb          # from sentinel-trade/ directory
    python src/models/train_xgb.py

Steps:
  1. Load features_long.parquet
  2. Temporal train/val/test split (train 2016-2019, val 2020, test 2021)
  3. Train XGBoost with early stopping on val RMSE
  4. Evaluate on test set; print all metrics + baseline comparison
  5. Save model (JSON) + SHAP explainer (pickle)
  6. Predict all missing freight rates → graph_edges_full.parquet
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd  # noqa: F401 — needed inside PermutationExplainer lambda
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    ML_FEATURES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    MODEL_PATH, EXPLAINER_PATH, EDGES_PATH, ARTIFACTS_DIR,
    PROCESSED_DIR, FEATURES_PATH
)
from src.data.feature_pipeline import load_features


# ─── XGBoost hyperparameters ──────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
)


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 1e-9
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def within_pct(y_true, y_pred, pct=0.10):
    mask = y_true > 1e-9
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask] <= pct) * 100)


def train_and_evaluate():
    print("=" * 60)
    print("SONAR — XGBoost Training")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    print("\nLoading feature table...")
    df = load_features()
    observed = df[df["freight_rate"].notna()].copy()
    print(f"  Observed rows: {len(observed):,}")

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df = observed[observed["year"].isin(TRAIN_YEARS)]
    val_df   = observed[observed["year"].isin(VAL_YEARS)]
    test_df  = observed[observed["year"].isin(TEST_YEARS)]
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Target: log1p transform
    y_train = np.log1p(train_df["freight_rate"].values)
    y_val   = np.log1p(val_df["freight_rate"].values)
    y_test  = np.log1p(test_df["freight_rate"].values)

    X_train = train_df[ML_FEATURES].astype(float)
    X_val   = val_df[ML_FEATURES].astype(float)
    X_test  = test_df[ML_FEATURES].astype(float)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_pred_log  = model.predict(X_test)
    y_pred      = np.expm1(y_pred_log)
    y_true      = np.expm1(y_test)
    y_pred      = np.clip(y_pred, 0, None)

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred)
    pct10 = within_pct(y_true, y_pred, 0.10)

    # Baseline: median per product
    median_preds = (
        train_df.groupby("product_code")["freight_rate"].median()
        .rename("median_pred")
    )
    test_with_baseline = test_df.merge(median_preds, on="product_code")
    baseline_rmse = np.sqrt(mean_squared_error(
        test_with_baseline["freight_rate"].values,
        test_with_baseline["median_pred"].values,
    ))
    improvement = (baseline_rmse - rmse) / baseline_rmse * 100

    print("\n" + "─" * 40)
    print("TEST SET RESULTS (year 2021)")
    print("─" * 40)
    print(f"  RMSE            : {rmse:.4f}   (target < 0.05)")
    print(f"  MAE             : {mae:.4f}")
    print(f"  R²              : {r2:.4f}   (target > 0.85)")
    print(f"  MAPE            : {mape:.1f}%   (target < 15%)")
    print(f"  Within ±10%     : {pct10:.1f}%   (target > 70%)")
    print(f"  Baseline RMSE   : {baseline_rmse:.4f}")
    print(f"  Improvement     : {improvement:.1f}%  (target > 20%)")
    print("─" * 40)

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── SHAP / feature importance ─────────────────────────────────────────────
    print("Computing feature importance...")

    # XGBoost native gain importance (always available, no version quirks)
    importance = model.get_booster().get_score(importance_type="gain")
    # Map f0,f1,... back to feature names
    feat_map = {f"f{i}": name for i, name in enumerate(ML_FEATURES)}
    importance_named = {feat_map.get(k, k): v for k, v in importance.items()}
    importance_sorted = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    print("\nTop-10 features by XGBoost gain importance:")
    for feat, gain in importance_sorted[:10]:
        print(f"  {feat:<35s} {gain:.2f}")

    # Save importance dict as pickle (used by Streamlit explainability page)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump({"type": "gain", "importance": dict(importance_sorted)}, f)
    print(f"Feature importance saved → {EXPLAINER_PATH}")

    # ── Impute missing freight rates ──────────────────────────────────────────
    print("\nPredicting missing freight rates...")
    missing = df[df["freight_rate"].isna()].copy()
    print(f"  Missing rows: {len(missing):,}")

    if len(missing) > 0:
        X_missing = missing[ML_FEATURES].astype(float)
        preds_log = model.predict(X_missing)
        preds     = np.clip(np.expm1(preds_log), 0, None)
        missing["freight_rate"] = preds
        missing["is_predicted"] = True
    else:
        missing["is_predicted"] = True

    observed["is_predicted"] = False
    edges_df = pd.concat([observed, missing], ignore_index=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    edges_df.to_parquet(EDGES_PATH, index=False)
    print(f"Graph edges saved → {EDGES_PATH}")
    print(f"  Total edges: {len(edges_df):,}")

    return model, importance_named


if __name__ == "__main__":
    train_and_evaluate()
