"""
Route Explorer — multi-criteria route comparison.
Finds the best route from A → B optimised for three distinct criteria:
  • Most Resilient  (highest Resilience Score)
  • Cheapest        (lowest freight cost)
  • Fastest         (lowest estimated lead time)
"""

import os
import sys

import streamlit as st
import pandas as pd
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import CHOKEPOINTS, CHOKEPOINT_WAYPOINTS, PRODUCT_NAMES, LATEST_YEAR
from src.graph.routing import find_multi_criteria_routes, apply_scenario
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import make_multi_criteria_globe, make_resilience_gauge, CRITERIA_COLORS

st.set_page_config(page_title="Route Explorer · SONAR", layout="wide", page_icon="🗺")

st.markdown("""
<style>
.main{background:#0e1117}
h1,h2,h3,p,label{color:#e6edf3!important}
.stSidebar{background:#161b22}
.route-card{
    background:#161b22;border:1px solid #21262d;
    border-radius:10px;padding:18px 16px;margin:4px 0;
}
.route-card h4{margin:0 0 10px 0;font-size:15px}
.metric-big{font-size:26px;font-weight:700;margin:4px 0}
.metric-label{font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:.5px}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ── Guard: session state ───────────────────────────────────────────────────────
if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first to initialise the app.")
    st.stop()

graphs = st.session_state.graphs
scorer = st.session_state.scorer

sample_graph = graphs[(2021, 8517)]
ALL_COUNTRIES = sorted(sample_graph.nodes())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Route Configuration")
    origin = st.selectbox(
        "Origin Country", ALL_COUNTRIES,
        index=ALL_COUNTRIES.index("China") if "China" in ALL_COUNTRIES else 0,
    )
    destination = st.selectbox(
        "Destination Country", ALL_COUNTRIES,
        index=ALL_COUNTRIES.index("Germany") if "Germany" in ALL_COUNTRIES else 1,
    )
    product_label = st.selectbox("Product", list(PRODUCT_NAMES.values()))
    product_code = [k for k, v in PRODUCT_NAMES.items() if v == product_label][0]
    year = LATEST_YEAR

    st.markdown("---")
    st.markdown("## 🚨 Chokepoint Scenarios")
    blocked = [cp for cp in CHOKEPOINTS if st.checkbox(cp, key=f"cp_{cp}")]

    st.markdown("---")
    st.markdown("## 💹 Tariff Scenarios")
    us_tariff    = st.slider("US Tariff (%)",    0, 50, 10, step=5)
    eu_tariff    = st.slider("EU Tariff (%)",    0, 50,  0, step=5)
    china_tariff = st.slider("China Tariff (%)", 0, 50,  0, step=5)
    asean_tariff = st.slider("ASEAN Tariff (%)", 0, 50,  0, step=5)

    st.markdown("---")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🗺 Route Explorer")
st.caption(f"**{origin}** → **{destination}** | {product_label} | Latest data ({year})")

if origin == destination:
    st.warning("Please select different origin and destination countries.")
    st.stop()

# ── Routing ────────────────────────────────────────────────────────────────────
key = (year, product_code)
if key not in graphs:
    st.error(f"No graph for year={year}, product={product_code}.")
    st.stop()

G_base = graphs[key]
has_scenario = bool(blocked) or any([us_tariff, eu_tariff, china_tariff, asean_tariff])

tariff_multipliers = get_tariff_multipliers(
    us_pct=float(us_tariff),
    eu_pct=float(eu_tariff),
    china_pct=float(china_tariff),
    asean_pct=float(asean_tariff),
)

# ── Compute multi-criteria routes (session-state memoised) ────────────────────
_cache_key = (
    origin, destination, product_code, year,
    tuple(sorted(blocked)),
    us_tariff, eu_tariff, china_tariff, asean_tariff,
)
_routes_cache = st.session_state.setdefault("_re_routes_cache", {})

if _cache_key not in _routes_cache:
    G_active    = apply_scenario(G_base, blocked, tariff_multipliers) if has_scenario else G_base
    all_lsci    = [G_base.nodes[n].get("lsci", 0) for n in G_base.nodes()]
    median_lsci = float(pd.Series(all_lsci).replace(0, pd.NA).median() or 50.0)
    blocked_wps = frozenset(wp for cp in blocked for wp in CHOKEPOINT_WAYPOINTS.get(cp, []))
    try:
        _routes_cache[_cache_key] = find_multi_criteria_routes(
            G_active, origin, destination, scorer,
            k_candidates=20, median_lsci=median_lsci,
            blocked_wps=blocked_wps,
        )
    except nx.NodeNotFound as e:
        st.error(f"Node error: {e}")
        st.stop()
    except nx.NetworkXNoPath as e:
        st.error(f"No path found: {e}")
        st.stop()

routes = _routes_cache[_cache_key]

# ── Globe (session-state memoised) ────────────────────────────────────────────
_globe_cache = st.session_state.setdefault("_re_globe_cache", {})
if _cache_key not in _globe_cache:
    criteria_dicts = {k: r.to_dict() for k, r in routes.items()}
    _globe_cache[_cache_key] = (
        criteria_dicts,
        make_multi_criteria_globe(criteria_routes=criteria_dicts, blocked_chokepoints=blocked),
    )
criteria_dicts, globe_fig = _globe_cache[_cache_key]

st.plotly_chart(globe_fig, use_container_width=True, config={"displayModeBar": False})

if has_scenario:
    st.info(
        f"🚨 Scenario active: blocked=[{', '.join(blocked)}]  "
        f"US tariff={us_tariff}%  EU={eu_tariff}%  China={china_tariff}%  ASEAN={asean_tariff}%"
    )

# ── 3-column route cards ───────────────────────────────────────────────────────
st.markdown("### Route Comparison by Criterion")

CARD_CONFIG = [
    ("most_resilient", "Most Resilient",  "🛡",  CRITERIA_COLORS["most_resilient"], "Resilience Score"),
    ("cheapest",       "Cheapest",        "💰",  CRITERIA_COLORS["cheapest"],       "Freight Cost"),
    ("fastest",        "Fastest",         "⚡",  CRITERIA_COLORS["fastest"],        "Lead Time"),
]

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for col, (crit_key, crit_label, icon, color, highlight_label) in zip(cols, CARD_CONFIG):
    r = routes[crit_key]
    rd = criteria_dicts[crit_key]

    # Highlighted metric varies by criterion
    if crit_key == "most_resilient":
        highlight_val = f"{r.rs:.1f} / 100"
    elif crit_key == "cheapest":
        highlight_val = f"{r.cost:.4f}"
    else:
        highlight_val = f"{r.lead_time_days:.0f} days"

    # RS label colours
    rs_score = r.rs
    rs_color = (
        "#27AE60" if rs_score >= 75 else
        "#F39C12" if rs_score >= 50 else
        "#E74C3C" if rs_score >= 25 else "#8E44AD"
    )

    with col:
        st.markdown(
            f"""<div class="route-card" style="border-top:3px solid {color}">
            <h4>{icon} {crit_label}</h4>
            <div class="metric-label">{highlight_label}</div>
            <div class="metric-big" style="color:{color}">{highlight_val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Route path
        st.markdown(f"**Path** ({r.hops} hop{'s' if r.hops != 1 else ''})")
        st.markdown(" → ".join(f"`{c}`" for c in r.path))

        # Key metrics table
        st.dataframe(
            pd.DataFrame([
                {"Metric": "Freight Cost",    "Value": f"{r.cost:.4f}"},
                {"Metric": "Lead Time",       "Value": f"{r.lead_time_days:.0f} d"},
                {"Metric": "Hops",            "Value": str(r.hops)},
                {"Metric": "Chokepoint Exp.", "Value": f"{rd['chk_exposure']:.0%}"},
                {"Metric": "RS Score",        "Value": f"{rs_score:.1f} / 100"},
                {"Metric": "ML Predicted",    "Value": "Yes ⚠" if rd["has_predicted"] else "No ✓"},
            ]).set_index("Metric"),
            use_container_width=True,
        )

        # RS gauge
        st.plotly_chart(
            make_resilience_gauge(rs_score, crit_label),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # RS component breakdown
        if hasattr(r, "rs_detail") and r.rs_detail:
            with st.expander("Resilience breakdown"):
                comp = r.rs_detail.get("components_pct", {})
                st.dataframe(
                    pd.DataFrame([
                        {"Component": k, "Contribution (pts)": f"{v:.1f}"}
                        for k, v in comp.items()
                    ]).set_index("Component"),
                    use_container_width=True,
                )

# ── Summary trade-off table ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Trade-off Summary")
summary_rows = []
for crit_key, crit_label, icon, color, _ in CARD_CONFIG:
    r  = routes[crit_key]
    rd = criteria_dicts[crit_key]
    summary_rows.append({
        "Criterion":      f"{icon} {crit_label}",
        "Route":          " → ".join(r.path),
        "Freight Cost":   round(r.cost, 4),
        "Lead Time (d)":  r.lead_time_days,
        "Hops":           r.hops,
        "Chk Exposure":   f"{rd['chk_exposure']:.0%}",
        "RS Score":       round(r.rs, 1),
    })
st.dataframe(
    pd.DataFrame(summary_rows).set_index("Criterion"),
    use_container_width=True,
)
