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
from src.graph.routing import find_k_routes, find_multi_criteria_routes, find_pareto_frontier, apply_scenario
from src.viz.globe import make_multi_criteria_globe
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
    """Load 5 pre-built graphs (one per product, latest year). Build if missing or stale."""
    try:
        graphs = load_latest_graphs_cache()
        # Guard: cache may be from a previous LATEST_YEAR — rebuild if stale
        if (LATEST_YEAR, PRODUCT_CODES[0]) not in graphs:
            st.warning(f"Graph cache is for an older year. Rebuilding for {LATEST_YEAR}...")
            edges = _load_edges()
            return build_latest_graphs(edges_df=edges, save_cache=True)
        return graphs
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


_PRELOAD_ORIGIN  = "China"
_PRELOAD_DEST    = "Germany"
_PRELOAD_PRODUCT = 8517          # Telephones & Electronics
_PRELOAD_KEY     = (_PRELOAD_ORIGIN, _PRELOAD_DEST, _PRELOAD_PRODUCT, LATEST_YEAR, (), 0, 0, 0, 0)
_CRITERIA_KEYS   = {"cheapest", "fastest", "most_resilient"}


@st.cache_resource(show_spinner="Pre-computing default route...")
def _preload_default_routes() -> tuple[dict, dict]:
    """
    Compute the China → Germany frontier AND globe figure once per server
    process for both the zero-tariff case (mode selector / guided wizard)
    and the us_tariff=10 case (expert sidebar default).
    Returns (routes_cache, globe_cache) dicts ready to copy into session_state.
    """
    graphs = _load_graphs()
    scorer = _load_scorer()
    gkey   = (LATEST_YEAR, _PRELOAD_PRODUCT)
    if gkey not in graphs:
        return {}, {}
    G           = graphs[gkey]
    all_lsci    = [G.nodes[n].get("lsci", 0) for n in G.nodes()]
    median_lsci = float(pd.Series(all_lsci).replace(0, pd.NA).median() or 50.0)

    routes_cache: dict = {}
    globe_cache:  dict = {}

    # Scenarios to precompute: (us_tariff, eu_tariff, china_tariff, asean_tariff)
    scenarios = [
        (0,  0, 0, 0),   # zero-tariff — used while mode selector is shown
        (10, 0, 0, 0),   # expert sidebar default (US Tariff slider starts at 10)
    ]

    for us_t, eu_t, cn_t, as_t in scenarios:
        cache_key = (_PRELOAD_ORIGIN, _PRELOAD_DEST, _PRELOAD_PRODUCT,
                     LATEST_YEAR, (), us_t, eu_t, cn_t, as_t)
        tariff_mults = get_tariff_multipliers(
            us_pct=float(us_t), eu_pct=float(eu_t),
            china_pct=float(cn_t), asean_pct=float(as_t),
        )
        G_active = apply_scenario(G, tariff_multipliers=tariff_mults) if any([us_t, eu_t, cn_t, as_t]) else G
        try:
            routes = find_pareto_frontier(
                G_active, _PRELOAD_ORIGIN, _PRELOAD_DEST, scorer,
                k_per_pass=10, median_lsci=median_lsci,
                blocked_wps=frozenset(),
            )
            criteria_dicts = {k: r.to_dict() for k, r in routes.items() if k in _CRITERIA_KEYS}
            globe_fig = make_multi_criteria_globe(
                criteria_routes=criteria_dicts,
                blocked_chokepoints=[],
            )
            routes_cache[cache_key] = routes
            globe_cache[cache_key]  = (criteria_dicts, globe_fig)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return routes_cache, globe_cache


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
                    "lead_time":    routes[0].lead_time_days,
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                rows.append({
                    "origin":       orig,
                    "destination":  dest,
                    "product_name": PRODUCT_NAMES[prod],
                    "score":        0.0,
                    "label":        "No Route",
                    "lead_time":    None,
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

# ── Preload default route: China → Germany, Telephones & Electronics ──────────
# Seed both caches from the process-level precomputed result so the first render
# of the Route Explorer is instant — routes AND globe are ready on arrival.
if "_re_routes_cache" not in st.session_state or "_re_globe_cache" not in st.session_state:
    _pre_routes_cache, _pre_globe_cache = _preload_default_routes()
    st.session_state.setdefault("_re_routes_cache", dict(_pre_routes_cache))
    st.session_state.setdefault("_re_globe_cache",  dict(_pre_globe_cache))

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
        TFT-forecasted 2022 freight rates, and resilience scoring across 438K+ corridors.
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
section_header("", "Explore", "Click any card to open that module")

_nav_tiles = [
    ("pages/01_route_explorer.py",      "#4A90D9", "🗺",  "Route Explorer",
     "Compare the most resilient, cheapest, and fastest routes between any two countries. "
     "Simulate chokepoint closures and tariff shocks in real-time, or let the guided wizard "
     "recommend the best route for your business profile."),
    ("pages/02_resilience_analysis.py", "#27AE60", "📊", "Resilience Analysis",
     "Heatmap view of resilience scores across the top 20 global trade corridors and "
     "5 product categories. Drill down into all 5 score components — Reliability, "
     "Redundancy, Weather, Ports, and Security."),
    ("pages/03_compare_corridors.py",   "#58a6ff", "🌐",  "Corridor Comparison",
     "Fix your manufacturing origin and target market, then compare shipping direct vs. "
     "routing through an intermediate hub country. Stress-test each strategy under tariff "
     "shocks and chokepoint closures to find your most resilient expansion path."),
    ("pages/04_model_insights.py",      "#F5A623", "🔍", "Model Insights",
     "Explore 2022 out-of-sample freight forecasts from the Temporal Fusion Transformer. "
     "View quantile uncertainty bands per corridor, TFT architecture overview, and a "
     "model comparison with the XGBoost baseline."),
    ("pages/05_score_methodology.py",   "#9B59B6", "🛡",  "Score Methodology",
     "Full explanation of the 5-factor resilience model with equal weights. Understand how "
     "each factor is calculated and how to interpret scores for any corridor."),
]

# ── Tile CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Wrapper chain: make all 5 columns the same height ───────────────────── */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    gap: 14px !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stButton"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 !important;
}
/* ── Button = the card ───────────────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: linear-gradient(160deg, #1a2030 0%, #161b22 60%) !important;
    border: 1px solid #2a3140 !important;
    border-radius: 14px !important;
    width: 100% !important;
    flex: 1 !important;
    text-align: left !important;
    padding: 26px 22px 24px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: flex-start !important;
    white-space: normal !important;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s, background 0.2s !important;
    position: relative !important;
    overflow: hidden !important;
}
/* subtle shimmer line at top-right corner */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button::after {
    content: "→" !important;
    position: absolute !important;
    bottom: 18px !important;
    right: 20px !important;
    font-size: 16px !important;
    opacity: 0 !important;
    transition: opacity 0.2s, transform 0.2s !important;
    transform: translateX(-4px) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
    transform: translateY(-3px) !important;
    background: linear-gradient(160deg, #1e2a3a 0%, #1a2030 60%) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover::after {
    opacity: 0.5 !important;
    transform: translateX(0) !important;
}
/* Inner markdown container */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button > div {
    width: 100% !important;
}
/* Icon */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p:first-child {
    font-size: 30px !important;
    line-height: 1 !important;
    margin: 0 0 16px 0 !important;
    display: block !important;
}
/* Title */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p:nth-child(2) {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    margin: 0 0 10px 0 !important;
    line-height: 1.25 !important;
    letter-spacing: -0.1px !important;
}
/* Description */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p:last-child {
    font-size: 12px !important;
    color: #7d8590 !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
/* Per-tile top accent + hover glow */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) button {
    border-top: 3px solid #4A90D9 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) button:hover {
    border-color: #4A90D9 !important;
    box-shadow: 0 8px 32px rgba(74,144,217,0.18) !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button {
    border-top: 3px solid #27AE60 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover {
    border-color: #27AE60 !important;
    box-shadow: 0 8px 32px rgba(39,174,96,0.18) !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button {
    border-top: 3px solid #58a6ff !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button:hover {
    border-color: #58a6ff !important;
    box-shadow: 0 8px 32px rgba(88,166,255,0.18) !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) button {
    border-top: 3px solid #F5A623 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) button:hover {
    border-color: #F5A623 !important;
    box-shadow: 0 8px 32px rgba(245,166,35,0.18) !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(5) button {
    border-top: 3px solid #9B59B6 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(5) button:hover {
    border-color: #9B59B6 !important;
    box-shadow: 0 8px 32px rgba(155,89,182,0.18) !important;
}
</style>
""", unsafe_allow_html=True)

_tile_cols = st.columns(5)
for _col, (_page, _color, _icon, _title, _desc) in zip(_tile_cols, _nav_tiles):
    with _col:
        _label = f"{_icon}\n\n**{_title}**\n\n{_desc}"
        if st.button(_label, key=f"nav_{_title}", width='stretch'):
            st.switch_page(_page)

# ── How It Works ──────────────────────────────────────────────────────────────
section_header("", "How It Works", "5-stage analytical pipeline")

# Pipeline steps as styled cards
_steps = [
    ("1", "TFT Forecasting", "#4A90D9",
     "A Temporal Fusion Transformer (TFT) forecasts 2022 freight rates across 62K+ corridors "
     "with quantile uncertainty bands, replacing the XGBoost imputer used for 2016–2021."),
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
     "A 0\u2013100 composite index: RS\u00a0=\u00a0100\u00a0\u00d7\u00a0(Reliability\u00a0\u00d7\u00a0Redundancy\u00a0\u00d7\u00a0Weather\u00a0\u00d7\u00a0Ports\u00a0\u00d7\u00a0Security)^0.20. "
     "Geometric mean — one near-zero factor collapses the entire score."),
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
