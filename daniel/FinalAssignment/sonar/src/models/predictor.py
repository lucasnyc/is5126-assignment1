"""
Model predictor — load trained XGBoost model and run inference.
Used by the Streamlit app and notebooks for on-demand predictions.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODEL_PATH, EXPLAINER_PATH, ML_FEATURES, EDGES_PATH


class FreightPredictor:
    """Wraps the trained XGBoost model for batch and single-row inference."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run src/models/train_xgb.py first."
            )
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_PATH)
        self._importance_data = None

    @property
    def importance_data(self) -> dict:
        """Load feature importance dict (gain-based) from disk."""
        if self._importance_data is None:
            if os.path.exists(EXPLAINER_PATH):
                with open(EXPLAINER_PATH, "rb") as f:
                    self._importance_data = pickle.load(f)
            else:
                # Fallback: compute from model directly
                raw = self.model.get_booster().get_score(importance_type="gain")
                feat_map = {f"f{i}": name for i, name in enumerate(ML_FEATURES)}
                named = {feat_map.get(k, k): v for k, v in raw.items()}
                self._importance_data = {"type": "gain", "importance": named}
        return self._importance_data

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict freight rates for a feature DataFrame.
        Input must have all columns in ML_FEATURES.
        Returns array of non-negative ad-valorem freight rates.
        """
        X_f = X[ML_FEATURES].astype(float)
        preds_log = self.model.predict(X_f)
        return np.clip(np.expm1(preds_log), 0, None)

    def feature_importance(self) -> pd.Series:
        """Return feature importance as a sorted pandas Series."""
        imp = self.importance_data.get("importance", {})
        # Fill any missing features with 0
        all_imp = {f: imp.get(f, 0.0) for f in ML_FEATURES}
        return pd.Series(all_imp).sort_values(ascending=False)

    def explain_edge(
        self, origin: str, destination: str, product_code: int, year: int
    ) -> dict:
        """
        Return feature importance explanation for a single edge.
        Uses XGBoost gain importance (global, not per-instance) as a proxy.

        Returns dict with keys: feature_names, importance_vals, prediction, is_predicted
        """
        edges = load_edges()
        mask = (
            (edges["origin"] == origin)
            & (edges["destination"] == destination)
            & (edges["product_code"] == product_code)
            & (edges["year"] == year)
        )
        row = edges[mask]
        if len(row) == 0:
            raise ValueError(
                f"Edge {origin}→{destination} ({product_code}, {year}) not found."
            )

        X_row   = row[ML_FEATURES].astype(float)
        pred    = float(self.predict(X_row)[0])
        imp_ser = self.feature_importance()

        return {
            "feature_names":    ML_FEATURES,
            "importance_vals":  imp_ser[ML_FEATURES].tolist(),
            "feature_values":   X_row.iloc[0].to_dict(),
            "prediction":       pred,
            "is_predicted":     bool(row["is_predicted"].iloc[0]),
        }


def load_edges() -> pd.DataFrame:
    """Load the complete edge matrix (observed + ML-predicted freight rates)."""
    if not os.path.exists(EDGES_PATH):
        raise FileNotFoundError(
            f"Edge matrix not found at {EDGES_PATH}. "
            "Run src/models/train_xgb.py first."
        )
    return pd.read_parquet(EDGES_PATH)
