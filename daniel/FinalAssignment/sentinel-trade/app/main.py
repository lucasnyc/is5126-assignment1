"""
SONAR — Supply-chain Optimization and Network Analysis for Resilience.
Streamlit entry point.

Run with:
    cd sentinel-trade
    streamlit run app/main.py
"""

import os
import sys

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.graph.builder import load_graphs_cache, build_all_graphs
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

@st.cache_resource(show_spinner="Loading shipping network (first load ~30s)...")
def _load_graphs():
    """Load all 30 pre-built graphs from cache. Build if missing."""
    try:
        return load_graphs_cache()
    except FileNotFoundError:
        st.warning("Graph cache not found. Building now — this may take a minute...")
        edges = _load_edges()
        return build_all_graphs(edges_df=edges, save_cache=True)


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
c3.metric("Years Covered", "2016 – 2021")
c4.metric("Graph Countries", f"~{max(G.number_of_nodes() for G in graphs.values()):,}")

st.markdown("---")
st.markdown("""
### How It Works
1. **ML Model** (XGBoost + SHAP) fills in missing freight rates for unobserved trade corridors
2. **Graph Engine** (NetworkX + Dijkstra / Yen's K-shortest) finds optimal shipping paths
3. **Scenario Simulator** removes chokepoint nodes and applies tariff multipliers in real-time
4. **Resilience Score** quantifies route stability across redundancy, connectivity, chokepoint exposure, and fleet availability

> Navigate to **Route Explorer** in the sidebar to run your first simulation.
""")
