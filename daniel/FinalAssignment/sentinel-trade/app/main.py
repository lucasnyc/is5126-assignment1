"""
SONAR — Supply-chain Optimization and Network Analysis for Resilience.
Streamlit entry point.

Run with:
    cd sentinel-trade
    streamlit run app/main.py
"""

import os
import sys

import networkx as nx
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

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SONAR",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme override ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22; }
    h1, h2, h3, h4, p { color: #e6edf3 !important; }
    .sentinel-logo {
        font-size: 26px; font-weight: 700;
        color: #4A90D9; letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)


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


# ── Bootstrap session state ───────────────────────────────────────────────────
if "graphs" not in st.session_state:
    st.session_state.graphs = _load_graphs()
if "edges" not in st.session_state:
    st.session_state.edges  = _load_edges()
if "scorer" not in st.session_state:
    st.session_state.scorer = _load_scorer()

# ── Pre-compute heatmap baseline (no scenario, all products, all 20 corridors) ─
# This runs once at startup so page 02 loads instantly for the default view.
if "heatmap_baseline" not in st.session_state:
    _graphs  = st.session_state.graphs
    _scorer  = st.session_state.scorer
    _rows    = []
    with st.spinner("Pre-computing resilience baseline..."):
        for orig, dest in TOP_CORRIDORS:
            for prod in PRODUCT_CODES:
                gkey = (LATEST_YEAR, prod)
                if gkey not in _graphs:
                    continue
                G = _graphs[gkey]
                try:
                    routes = find_k_routes(G, orig, dest, k=3)
                    rs     = _scorer.score_from_routes(routes, G)
                    _rows.append({
                        "origin":       orig,
                        "destination":  dest,
                        "product_name": PRODUCT_NAMES[prod],
                        "score":        rs["score"],
                        "label":        rs["label"],
                    })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    _rows.append({
                        "origin":       orig,
                        "destination":  dest,
                        "product_name": PRODUCT_NAMES[prod],
                        "score":        0.0,
                        "label":        "No Route",
                    })
    st.session_state.heatmap_baseline = _rows

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_tagline = st.columns([1, 3])
with col_logo:
    st.markdown('<div class="sentinel-logo">🌊 SONAR</div>', unsafe_allow_html=True)
with col_tagline:
    st.markdown(
        "<small style='color:#8B949E;'>Supply-chain Optimization &amp; Network Analysis for Resilience · "
        "UNCTAD Maritime Data 2016–2021</small>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.info(
    "**Navigate using the sidebar pages** → "
    "🗺 Route Explorer · 📊 Resilience Analysis · 🔍 Model Explainability"
)

# ── Summary stats ─────────────────────────────────────────────────────────────
edges  = st.session_state.edges
graphs = st.session_state.graphs

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trade Corridors", f"{len(edges[['origin','destination']].drop_duplicates()):,}")
c2.metric("Products Tracked", "5 HS Codes")
c3.metric("Data Year", str(LATEST_YEAR))
c4.metric("Graph Countries", f"~{max(G.number_of_nodes() for G in graphs.values()):,}")

st.markdown("---")
st.markdown("""
### How It Works
1. **ML Model** (XGBoost) imputes missing bilateral freight rates using UNCTAD maritime indicators — bilateral LSCI, TEU throughput, fleet ownership, and historical averages
2. **Graph Engine** (NetworkX + Yen's K-shortest) finds up to 20 candidate routes per corridor, constrained to ≤ 2 hops (direct or single transshipment hub)
3. **Multi-Criteria Routing** selects the best route by three independent criteria: **Most Resilient** (highest RS score), **Cheapest** (lowest freight cost), **Fastest** (lowest haversine lead time at 15 knots)
4. **Scenario Simulator** applies chokepoint closures and tariff multipliers in real-time — reroutes around blocked nodes and reprices affected edges
5. **Resilience Score (RS)** — a 0–100 composite index derived via AHP (Saaty 1980): `RS = 100 × (0.47·Alt + 0.28·Chk + 0.17·Bil + 0.07·Fleet)`, where weights were validated with Consistency Ratio CR = 0.019 < 0.10

> Navigate to **Route Explorer** in the sidebar to run your first simulation.
""")
