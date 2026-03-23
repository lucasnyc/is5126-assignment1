"""
SONAR — Supply-chain Optimization and Network Analysis for Resilience.
Streamlit entry point.

Run with:
    cd sonar
    streamlit run app/main.py
"""

import os
import sys

import networkx as nx
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from config import PRODUCT_CODES, PRODUCT_NAMES, LATEST_YEAR, TOP_CORRIDORS
from src.graph.builder import load_latest_graphs_cache, build_latest_graphs
from src.graph.routing import find_k_routes
from src.graph.chokepoints import get_tariff_multipliers
from src.models.predictor import load_edges
from src.scoring.resilience import ResilienceScorer
from app.components.theme import inject_global_css, section_header, stat_card, render_footer

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SONAR",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading shipping network...")
def _load_graphs():
    """Load 5 pre-built graphs (one per product, latest year). Build if missing."""
    try:
        return load_latest_graphs_cache()
    except FileNotFoundError:
        st.warning("Graph cache not found. Building now — this takes about 30s...")
        edges = _load_edges()
        return build_latest_graphs(edges_df=edges, save_cache=True)


@st.cache_resource(show_spinner="Loading edge matrix...")
def _load_edges():
    try:
        return load_edges()
    except FileNotFoundError:
        st.error(
            "Edge matrix not found. Please run the training pipeline first:\n"
            "```\npython src/models/train_xgb.py\n```"
        )
        st.stop()


@st.cache_resource
def _load_scorer():
    return ResilienceScorer()


@st.cache_resource(show_spinner="Pre-computing resilience baseline...")
def _compute_heatmap_baseline():
    """
    Compute baseline resilience scores for all TOP_CORRIDORS × PRODUCT_CODES.
    Cached at process level — survives page refreshes and new sessions.
    Only re-runs if the Streamlit server restarts.
    """
    graphs = _load_graphs()
    scorer = _load_scorer()
    rows   = []
    for orig, dest in TOP_CORRIDORS:
        for prod in PRODUCT_CODES:
            gkey = (LATEST_YEAR, prod)
            if gkey not in graphs:
                continue
            G = graphs[gkey]
            try:
                routes = find_k_routes(G, orig, dest, k=3)
                rs     = scorer.score_from_routes(routes, G)
                rows.append({
                    "origin":       orig,
                    "destination":  dest,
                    "product_name": PRODUCT_NAMES[prod],
                    "score":        rs["score"],
                    "label":        rs["label"],
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                rows.append({
                    "origin":       orig,
                    "destination":  dest,
                    "product_name": PRODUCT_NAMES[prod],
                    "score":        0.0,
                    "label":        "No Route",
                })
    return rows


# ── Bootstrap session state ───────────────────────────────────────────────────
if "graphs" not in st.session_state:
    st.session_state.graphs = _load_graphs()
if "edges" not in st.session_state:
    st.session_state.edges  = _load_edges()
if "scorer" not in st.session_state:
    st.session_state.scorer = _load_scorer()

if "heatmap_baseline" not in st.session_state:
    st.session_state.heatmap_baseline = _compute_heatmap_baseline()

# ── Data refs ─────────────────────────────────────────────────────────────────
edges  = st.session_state.edges
graphs = st.session_state.graphs
baseline = st.session_state.heatmap_baseline

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1929 0%, #162447 50%, #1a3a5c 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 32px;
">
    <div style="font-size: 36px; font-weight: 800; color: #e6edf3;
                letter-spacing: -1px; line-height: 1.15; margin-bottom: 8px;">
        <span style="color: #4A90D9;">SONAR</span>
    </div>
    <div style="font-size: 18px; font-weight: 500; color: #c9d1d9;
                margin-bottom: 16px; letter-spacing: -0.2px;">
        Supply-chain Optimization &amp; Network Analysis for Resilience
    </div>
    <div style="font-size: 13px; color: #8B949E; line-height: 1.7; max-width: 720px;">
        Strategic supply chain planning tool for profit-driven companies.
        Compare trade corridors, quantify disruption risk in dollars, and identify
        your best sourcing strategy — powered by UNCTAD bilateral trade data,
        ML-imputed freight rates, and AHP-weighted resilience scoring across 375K+ corridors.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI stats row ─────────────────────────────────────────────────────────────
n_corridors = len(edges[["origin", "destination"]].drop_duplicates())
n_countries = max(G.number_of_nodes() for G in graphs.values())

# Compute aggregate resilience stats from baseline
baseline_df = pd.DataFrame(baseline)
avg_rs = baseline_df["score"].mean()
high_rs_pct = int((baseline_df["score"] >= 75).sum() / len(baseline_df) * 100)
critical_count = int((baseline_df["score"] < 25).sum())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(stat_card("Trade Corridors", f"{n_corridors:,}"), unsafe_allow_html=True)
with c2:
    st.markdown(stat_card("Products Tracked", "5 HS Codes"), unsafe_allow_html=True)
with c3:
    st.markdown(stat_card("Network Countries", f"~{n_countries:,}"), unsafe_allow_html=True)
with c4:
    st.markdown(
        stat_card("Avg Resilience", f"{avg_rs:.1f}",
                  delta=f"{high_rs_pct}% high resilience",
                  delta_type="good" if avg_rs >= 50 else "warn"),
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        stat_card("Critical Routes", str(critical_count),
                  delta="below RS 25" if critical_count > 0 else "none",
                  delta_type="bad" if critical_count > 5 else "warn" if critical_count > 0 else "good"),
        unsafe_allow_html=True,
    )

# ── Navigation cards ──────────────────────────────────────────────────────────
st.write("")
section_header("", "Explore", "Select a module from the sidebar to begin")

st.markdown("""
<div style="display:flex;gap:16px;margin:8px 0 28px 0">
    <div class="nav-card" style="border-top:3px solid #4A90D9">
        <div class="nav-icon">🗺</div>
        <div class="nav-title">Route Explorer</div>
        <div class="nav-desc">
            Compare the most resilient, cheapest, and fastest routes between
            any two countries. Simulate chokepoint closures and tariff shocks in real-time,
            or let the guided wizard recommend the best route for your business profile.
        </div>
    </div>
    <div class="nav-card" style="border-top:3px solid #27AE60">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Resilience Analysis</div>
        <div class="nav-desc">
            Heatmap view of resilience scores across the top 20 global trade corridors
            and 5 product categories. Drill down into all 5 score components —
            Delivery Confidence, Backup Options, Weather Safety, Port Health, and Security Level.
        </div>
    </div>
    <div class="nav-card" style="border-top:3px solid #9B59B6">
        <div class="nav-icon">🛡</div>
        <div class="nav-title">Resilience Score</div>
        <div class="nav-desc">
            Full explanation of the 5-factor AHP-weighted resilience model.
            Understand how each factor is calculated, why the weights were chosen,
            and how to interpret scores for any corridor.
        </div>
    </div>
    <div class="nav-card" style="border-top:3px solid #F5A623">
        <div class="nav-icon">🔍</div>
        <div class="nav-title">Model Explainability</div>
        <div class="nav-desc">
            Understand why the XGBoost model predicted a specific freight rate.
            Inspect per-edge SHAP-style feature importance, global gain rankings,
            and held-out test performance with design decision rationale.
        </div>
    </div>
    <div class="nav-card" style="border-top:3px solid #58a6ff">
        <div class="nav-icon">⚖</div>
        <div class="nav-title">Corridor Comparison</div>
        <div class="nav-desc">
            Compare 2–4 trade lanes side by side across all planning metrics:
            Resilience Score, freight cost, lead time, rate volatility, and
            chokepoint exposure. Rank by your own priority weights to identify
            the best sourcing strategy.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
section_header("", "How It Works", "5-stage analytical pipeline")

# Pipeline steps as styled cards
_steps = [
    ("1", "ML Imputation", "#4A90D9",
     "XGBoost imputes missing bilateral freight rates using 25 UNCTAD maritime indicators — "
     "bilateral LSCI, TEU throughput, fleet ownership, and historical averages."),
    ("2", "Graph Engine", "#27AE60",
     "NetworkX + Yen's K-shortest paths finds up to 20 candidate routes per corridor, "
     "constrained to 2 hops (direct or single transshipment hub)."),
    ("3", "Multi-Criteria Routing", "#F5A623",
     "Selects the best route by three independent criteria: Most Resilient (highest RS), "
     "Cheapest (lowest freight cost), Fastest (lowest lead time at 15 knots)."),
    ("4", "Scenario Simulator", "#E74C3C",
     "Applies chokepoint closures and tariff multipliers in real-time — reroutes around "
     "blocked nodes and reprices affected edges instantly."),
    ("5", "Resilience Score", "#9B59B6",
     "A 0\u2013100 composite index via AHP-TOPSIS (Saaty 1980): "
     "RS = 100 \u00d7 (0.37\u00b7Rel + 0.21\u00b7Flex + 0.21\u00b7Env + 0.11\u00b7Port + 0.10\u00b7Sec), CR = 0.003."),
]

cols = st.columns(5)
for col, (num, title, color, desc) in zip(cols, _steps):
    with col:
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;'
            f'padding:20px 16px;height:220px;box-sizing:border-box;border-top:3px solid {color}">'
            f'<div style="font-size:28px;font-weight:800;color:{color};margin-bottom:6px">{num}</div>'
            f'<div style="font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:10px">{title}</div>'
            f'<div style="font-size:12px;color:#8B949E;line-height:1.65">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
render_footer()
