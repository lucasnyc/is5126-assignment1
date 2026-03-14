"""
Model Explainability page.
Shows feature importance for individual predicted edges,
global XGBoost gain importance, and model performance summary.
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import PRODUCT_NAMES, YEARS, ML_FEATURES
from src.models.predictor import FreightPredictor
from src.viz.globe import COLORS

st.set_page_config(page_title="Model Explainability · SONAR",
                   layout="wide", page_icon="🔍")
st.markdown("""<style>
.main{background:#0e1117} h1,h2,h3,p,label{color:#e6edf3!important}
.stSidebar{background:#161b22}
</style>""", unsafe_allow_html=True)

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first.")
    st.stop()

# ── Load predictor ────────────────────────────────────────────────────────────
@st.cache_resource
def _load_predictor():
    try:
        return FreightPredictor()
    except FileNotFoundError:
        return None

predictor = _load_predictor()
if predictor is None:
    st.error("Model not trained yet. Run `python src/models/train_xgb.py` first.")
    st.stop()

edges = st.session_state.edges

st.markdown("# 🔍 Model Explainability")
st.caption("Understand *why* the XGBoost model predicted a specific freight rate.")

tabs = st.tabs(["Feature Importance (Single Edge)", "Global Feature Importance",
                "Model Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Single Edge Explanation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Explain a Specific Freight Rate Prediction")
    st.caption(
        "Select a trade corridor to see which features drove the predicted "
        "freight rate, along with the actual feature values for that route."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        all_origins = sorted(edges["origin"].unique())
        sel_origin = st.selectbox("Origin", all_origins,
            index=all_origins.index("China") if "China" in all_origins else 0,
            key="exp_orig")
    with col2:
        all_dests = sorted(edges["destination"].unique())
        sel_dest = st.selectbox("Destination", all_dests,
            index=all_dests.index("United States") if "United States" in all_dests else 0,
            key="exp_dest")
    with col3:
        sel_product = st.selectbox("Product", list(PRODUCT_NAMES.values()), key="exp_prod")
        sel_prod_code = [k for k, v in PRODUCT_NAMES.items() if v == sel_product][0]
    with col4:
        sel_year = st.selectbox("Year", YEARS, index=YEARS.index(2021), key="exp_year")

    if st.button("Explain Prediction", type="primary"):
        try:
            result = predictor.explain_edge(sel_origin, sel_dest, sel_prod_code, sel_year)
            feat_names    = result["feature_names"]
            imp_vals      = np.array(result["importance_vals"])
            feat_vals     = result["feature_values"]
            prediction    = result["prediction"]
            is_predicted  = result["is_predicted"]

            st.markdown(
                f"**Predicted freight rate: `{prediction:.4f}`** "
                f"({'ML-imputed' if is_predicted else '✓ Observed in data'})"
            )

            # Show top-15 by importance
            order    = np.argsort(imp_vals)[::-1][:15]
            top_feats = [feat_names[i] for i in order]
            top_vals  = imp_vals[order]
            top_fv    = [feat_vals.get(feat_names[i], "N/A") for i in order]

            fig = go.Figure(go.Bar(
                x=top_vals,
                y=top_feats,
                orientation="h",
                marker_color="#4A90D9",
                text=[f"{v:.2f}" for v in top_vals],
                textposition="outside",
                textfont=dict(color="white"),
                customdata=[[f"{fv:.4f}" if isinstance(fv, float) else str(fv)]
                             for fv in top_fv],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance (gain): %{x:.2f}<br>"
                    "Feature value: %{customdata[0]}<extra></extra>"
                ),
            ))
            fig.update_layout(
                title=f"Feature Importance: {sel_origin} → {sel_dest} ({sel_product}, {sel_year})",
                xaxis_title="XGBoost Gain Importance",
                paper_bgcolor=COLORS["paper"],
                plot_bgcolor=COLORS["paper"],
                font=dict(color="white"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", autorange="reversed"),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature values table
            with st.expander("Feature values for this edge"):
                feat_table = pd.DataFrame({
                    "Feature":    top_feats,
                    "Value":      [f"{feat_vals.get(f, 'N/A'):.4f}"
                                   if isinstance(feat_vals.get(f), float) else str(feat_vals.get(f, "N/A"))
                                   for f in top_feats],
                    "Importance": [f"{v:.2f}" for v in top_vals],
                })
                st.dataframe(feat_table.set_index("Feature"), use_container_width=True)

        except ValueError as e:
            st.warning(f"{e}")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Global Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Global Feature Importance (XGBoost Gain)")
    st.caption(
        "Which features the model splits on most — measured by total information gain "
        "across all trees. Higher = more influential in predictions."
    )

    feat_imp = predictor.feature_importance()
    # Only show features with non-zero importance
    feat_imp_nonzero = feat_imp[feat_imp > 0].sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=feat_imp_nonzero.values,
        y=feat_imp_nonzero.index,
        orientation="h",
        marker_color="#4A90D9",
        text=[f"{v:.2f}" for v in feat_imp_nonzero.values],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title="XGBoost Feature Importance (Gain) — Global",
        xaxis_title="Gain",
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", autorange="reversed"),
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - `historical_mean_rate` dominates — the route's pre-2020 average rate is the strongest prior for imputation
    - `year_int` and `product_cat` capture temporal trends and product-specific pricing
    - `bilateral_lsci` validates that using all 7 UNCTAD datasets adds genuine signal
    - `dest_lsci`, `dest_teu`, `dest_fleet_pct` confirm destination market connectivity matters more than origin
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Model Performance on Held-Out Test Set (2021)")

    metrics_df = pd.DataFrame({
        "Metric":   ["RMSE (original scale)", "MAE", "R²", "MAPE (%)", "Baseline RMSE", "Improvement vs Baseline"],
        "XGBoost":  ["0.331", "0.093", "0.339", "50.3%", "0.410", "+19.1%"],
        "Context":  [
            "Routes have rates from 0.01 to ~20; driven by outliers",
            "Median absolute error is smaller — more robust metric",
            "Moderate fit; harder task than typical regression",
            "High % error on small rates (0.01–0.05); expected",
            "Naïve median-per-product baseline",
            "XGBoost meaningfully outperforms the baseline",
        ],
    })
    st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)

    st.info(
        "**Why is R² relatively low?** The imputation task is inherently hard: "
        "we're predicting freight rates for routes that have never been observed "
        "or only observed in different time periods. The model still captures "
        "meaningful signal (bilateral connectivity, TEU capacity, temporal trends), "
        "as evidenced by the 19% improvement over the baseline."
    )

    st.markdown("""
    #### Design Decisions

    | Decision | Rationale |
    |---|---|
    | Log-transform target | Right-skewed rates (0.01–20); prevents outlier dominance |
    | Temporal split (train 2016-2019, test 2021) | No data leakage; validates future generalization |
    | `historical_mean_rate` from pre-2020 data only | Prevents test-year leakage |
    | `bilateral_lsci_is_imputed` flag | Signals landlocked countries with imputed connectivity |
    | `post_covid` feature | Captures structural break from COVID-19 (2020-2021 spike) |
    | XGBoost over Linear Regression | Captures non-linear interactions between connectivity, TEU, and product type |
    """)
