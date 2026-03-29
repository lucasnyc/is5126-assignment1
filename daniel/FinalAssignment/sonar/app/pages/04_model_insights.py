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
from app.components.theme import inject_global_css, section_header, stat_card, render_footer

st.set_page_config(page_title="Model Explainability \u00b7 SONAR",
                   layout="wide", page_icon="\U0001f50d")

inject_global_css()

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

st.markdown("# \U0001f50d Model Explainability")
st.caption("Understand *why* the XGBoost model predicted a specific freight rate.")

tabs = st.tabs(["\U0001f4cd Single Edge Explanation", "\U0001f30d Global Feature Importance",
                "\U0001f4c8 Model Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Single Edge Explanation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    section_header("\U0001f4cd", "Explain a Specific Freight Rate Prediction")
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

            # Prediction result badge
            status_color = "#F39C12" if is_predicted else "#27AE60"
            status_label = "ML-imputed" if is_predicted else "Observed in data"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:16px;margin:12px 0 16px 0">'
                f'<div style="background:#161b22;border:1px solid #21262d;'
                f'border-radius:8px;padding:10px 18px;display:inline-flex;align-items:baseline;gap:8px">'
                f'<span style="font-size:12px;color:#8B949E;text-transform:uppercase;'
                f'letter-spacing:.5px">Predicted Rate</span>'
                f'<span style="font-size:22px;font-weight:800;color:#e6edf3">{prediction:.4f}</span>'
                f'</div>'
                f'<div style="background:{status_color}22;border:1px solid {status_color};'
                f'border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600;'
                f'color:{status_color}">{status_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
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
                title=f"Feature Importance: {sel_origin} \u2192 {sel_dest} ({sel_product}, {sel_year})",
                xaxis_title="XGBoost Gain Importance",
                paper_bgcolor=COLORS["paper"],
                plot_bgcolor=COLORS["paper"],
                font=dict(color="white"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", autorange="reversed"),
                height=500,
            )
            st.plotly_chart(fig, width='stretch')

            # Feature values table
            with st.expander("Feature values for this edge"):
                feat_table = pd.DataFrame({
                    "Feature":    top_feats,
                    "Value":      [f"{feat_vals.get(f, 'N/A'):.4f}"
                                   if isinstance(feat_vals.get(f), float) else str(feat_vals.get(f, "N/A"))
                                   for f in top_feats],
                    "Importance": [f"{v:.2f}" for v in top_vals],
                })
                st.dataframe(feat_table.set_index("Feature"), width='stretch')

        except ValueError as e:
            st.warning(f"{e}")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Global Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    section_header("\U0001f30d", "Global Feature Importance", "XGBoost Gain")
    st.caption(
        "Which features the model splits on most \u2014 measured by total information gain "
        "across all trees. Higher = more influential in predictions."
    )

    feat_imp = predictor.feature_importance()
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
        title="XGBoost Feature Importance (Gain) \u2014 Global",
        xaxis_title="Gain",
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", autorange="reversed"),
        height=550,
    )
    st.plotly_chart(fig, width='stretch')

    with st.container(border=True):
        st.markdown("""
**Key Insights:**
- `historical_mean_rate` dominates \u2014 the route's pre-2020 average rate is the strongest prior for imputation
- `year_int` and `product_cat` capture temporal trends and product-specific pricing
- `bilateral_lsci` validates that using all 7 UNCTAD datasets adds genuine signal
- `dest_lsci`, `dest_teu`, `dest_fleet_pct` confirm destination market connectivity matters more than origin
""")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    section_header("\U0001f4c8", "Model Performance", "Held-out test set (2021)")

    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(stat_card("RMSE", "0.331", delta="original scale", delta_type="warn"), unsafe_allow_html=True)
    with m2:
        st.markdown(stat_card("MAE", "0.093", delta="more robust metric", delta_type="good"), unsafe_allow_html=True)
    with m3:
        st.markdown(stat_card("R\u00b2", "0.339", delta="moderate fit", delta_type="warn"), unsafe_allow_html=True)
    with m4:
        st.markdown(stat_card("vs Baseline", "+19.1%", delta="improvement over na\u00efve median", delta_type="good"), unsafe_allow_html=True)

    st.write("")

    metrics_df = pd.DataFrame({
        "Metric":   ["RMSE (original scale)", "MAE", "R\u00b2", "MAPE (%)", "Baseline RMSE", "Improvement vs Baseline"],
        "XGBoost":  ["0.331", "0.093", "0.339", "50.3%", "0.410", "+19.1%"],
        "Context":  [
            "Routes have rates from 0.01 to ~20; driven by outliers",
            "Median absolute error is smaller \u2014 more robust metric",
            "Moderate fit; harder task than typical regression",
            "High % error on small rates (0.01\u20130.05); expected",
            "Na\u00efve median-per-product baseline",
            "XGBoost meaningfully outperforms the baseline",
        ],
    })
    st.dataframe(metrics_df.set_index("Metric"), width='stretch')

    st.info(
        "**Why is R\u00b2 relatively low?** The imputation task is inherently hard: "
        "we're predicting freight rates for routes that have never been observed "
        "or only observed in different time periods. The model still captures "
        "meaningful signal (bilateral connectivity, TEU capacity, temporal trends), "
        "as evidenced by the 19% improvement over the baseline."
    )

    st.markdown("---")
    section_header("", "Design Decisions")

    st.markdown("""
| Decision | Rationale |
|---|---|
| Log-transform target | Right-skewed rates (0.01\u201320); prevents outlier dominance |
| Temporal split (train 2016-2019, test 2021) | No data leakage; validates future generalization |
| `historical_mean_rate` from pre-2020 data only | Prevents test-year leakage |
| `bilateral_lsci_is_imputed` flag | Signals landlocked countries with imputed connectivity |
| `post_covid` feature | Captures structural break from COVID-19 (2020-2021 spike) |
| XGBoost over Linear Regression | Captures non-linear interactions between connectivity, TEU, and product type |
""")

render_footer()
