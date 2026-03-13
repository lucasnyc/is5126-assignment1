"""
Route Explorer — primary page.
Allows users to select O/D/product/year, toggle chokepoints, set tariffs,
and see real-time Dijkstra rerouting on an interactive globe.
"""

import os
import sys

import streamlit as st
import pandas as pd
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import CHOKEPOINTS, PRODUCT_NAMES, YEARS, PRODUCT_CODES
from src.graph.routing import find_k_routes, apply_scenario, compare_scenarios
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import make_route_globe, make_resilience_gauge

st.set_page_config(page_title="Route Explorer · SONAR", layout="wide", page_icon="🗺")

st.markdown("""
<style>
.main{background:#0e1117} h1,h2,h3,p,label{color:#e6edf3!important}
.stSidebar{background:#161b22}
.route-card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;margin:6px 0}
</style>""", unsafe_allow_html=True)

# ── Ensure session state is populated (may be accessed via direct page nav) ──
if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first to initialize the app.")
    st.stop()

graphs  = st.session_state.graphs
scorer  = st.session_state.scorer

# ─── Get list of countries from the graph ────────────────────────────────────
sample_graph = graphs[(2021, 8517)]
ALL_COUNTRIES = sorted(sample_graph.nodes())

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Route Configuration")
    origin = st.selectbox("Origin Country", ALL_COUNTRIES,
                          index=ALL_COUNTRIES.index("China") if "China" in ALL_COUNTRIES else 0)
    destination = st.selectbox("Destination Country", ALL_COUNTRIES,
                                index=ALL_COUNTRIES.index("Germany") if "Germany" in ALL_COUNTRIES else 1)
    product_label = st.selectbox(
        "Product",
        options=list(PRODUCT_NAMES.values()),
    )
    product_code = [k for k, v in PRODUCT_NAMES.items() if v == product_label][0]
    year = st.selectbox("Year", YEARS, index=YEARS.index(2021))

    st.markdown("---")
    st.markdown("## 🚨 Chokepoint Scenarios")
    blocked = []
    for cp_name in CHOKEPOINTS.keys():
        if st.checkbox(f"{cp_name}", key=f"cp_{cp_name}"):
            blocked.append(cp_name)

    st.markdown("---")
    st.markdown("## 💹 Tariff Scenarios")
    us_tariff    = st.slider("US Tariff (%)",    0, 50, 10, step=5)
    eu_tariff    = st.slider("EU Tariff (%)",    0, 50, 0,  step=5)
    china_tariff = st.slider("China Tariff (%)", 0, 50, 0,  step=5)
    asean_tariff = st.slider("ASEAN Tariff (%)", 0, 50, 0,  step=5)

    st.markdown("---")
    show_top_k = st.radio("Routes to display", [1, 3], index=0,
                          help="Show top 1 or top 3 alternative routes")
    run_btn = st.button("🔄 Run Simulation", type="primary", use_container_width=True)

# ─── Main panel ───────────────────────────────────────────────────────────────
st.markdown("# 🗺 Route Explorer")
st.caption(f"**{origin}** → **{destination}** | {product_label} | {year}")

# ─── Routing ─────────────────────────────────────────────────────────────────
key = (year, product_code)
if key not in graphs:
    st.error(f"No graph available for year={year}, product={product_code}.")
    st.stop()

G_base = graphs[key]

tariff_multipliers = get_tariff_multipliers(
    us_pct=float(us_tariff),
    eu_pct=float(eu_tariff),
    china_pct=float(china_tariff),
    asean_pct=float(asean_tariff),
)
G_scenario = apply_scenario(G_base, blocked, tariff_multipliers)

# Compute median LSCI for lead time estimation
all_lsci = [G_base.nodes[n].get("lsci", 0) for n in G_base.nodes()]
median_lsci = float(pd.Series(all_lsci).replace(0, pd.NA).median() or 50.0)

# Baseline routing
baseline_error = None
baseline_routes_obj = []
try:
    baseline_routes_obj = find_k_routes(G_base, origin, destination,
                                        k=show_top_k, median_lsci=median_lsci)
except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
    baseline_error = str(e)

# Scenario routing
scenario_error = None
scenario_routes_obj = []
try:
    scenario_routes_obj = find_k_routes(G_scenario, origin, destination,
                                        k=show_top_k, median_lsci=median_lsci)
except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
    scenario_error = str(e)

baseline_dicts = [r.to_dict() for r in baseline_routes_obj]
scenario_dicts = [r.to_dict() for r in scenario_routes_obj]

# ─── Globe ────────────────────────────────────────────────────────────────────
globe_fig = make_route_globe(
    baseline_routes=baseline_dicts,
    scenario_routes=scenario_dicts if (blocked or any([us_tariff, eu_tariff, china_tariff, asean_tariff])) else [],
    blocked_chokepoints=blocked,
    show_top_k=show_top_k,
)
st.plotly_chart(globe_fig, use_container_width=True, config={"displayModeBar": False})

# ─── Results panel ────────────────────────────────────────────────────────────
if baseline_error:
    st.error(f"Baseline routing failed: {baseline_error}")
elif baseline_routes_obj:
    b_best = baseline_routes_obj[0]
    rs_base = scorer.score_from_routes(baseline_routes_obj, G_base)

    has_scenario = bool(blocked) or any([us_tariff, eu_tariff, china_tariff, asean_tariff])
    s_best = scenario_routes_obj[0] if scenario_routes_obj else None
    rs_scen = scorer.score_from_routes(scenario_routes_obj, G_scenario) if s_best else None

    # Layout: results table + gauge side by side
    col_table, col_gauge = st.columns([3, 1])

    with col_table:
        st.markdown("### Route Comparison")
        rows = []

        # Baseline row
        rows.append({
            "Scenario":        "Baseline",
            "Route":           " → ".join(b_best.path),
            "Freight Rate":    f"{b_best.cost:.4f}",
            "Lead Time":       f"{b_best.lead_time_days:.0f} days",
            "Hops":            b_best.hops,
            "RS Score":        f"{rs_base['score']:.1f} / 100",
            "RS Label":        rs_base["label"],
            "ML Predicted":    "⚠ Yes" if b_best.has_predicted else "✓ Observed",
        })

        if has_scenario:
            if scenario_error:
                st.warning(f"No viable route under scenario: {scenario_error}")
            elif s_best:
                premium_pct = (s_best.cost - b_best.cost) / (b_best.cost + 1e-9) * 100
                lead_delta  = s_best.lead_time_days - b_best.lead_time_days
                rows.append({
                    "Scenario":     "After Scenario",
                    "Route":        " → ".join(s_best.path),
                    "Freight Rate": f"{s_best.cost:.4f}",
                    "Lead Time":    f"{s_best.lead_time_days:.0f} days",
                    "Hops":         s_best.hops,
                    "RS Score":     f"{rs_scen['score']:.1f} / 100",
                    "RS Label":     rs_scen["label"],
                    "ML Predicted": "⚠ Yes" if s_best.has_predicted else "✓ Observed",
                })
                # Cost delta callout
                delta_color = "🔴" if premium_pct > 0 else "🟢"
                st.markdown(
                    f"{delta_color} **Cost change: {premium_pct:+.1f}%** · "
                    f"Lead time change: **{lead_delta:+.0f} days**"
                )

        st.dataframe(pd.DataFrame(rows).set_index("Scenario"),
                     use_container_width=True)

        # Component breakdown
        st.markdown("#### Resilience Score Breakdown (Baseline)")
        comp_df = pd.DataFrame([{
            "Component":   k,
            "Contribution (pts)": v,
        } for k, v in rs_base["components_pct"].items()])
        st.dataframe(comp_df.set_index("Component"), use_container_width=True)

    with col_gauge:
        gauge_score  = rs_base["score"]
        gauge_label  = rs_base["label"]
        if has_scenario and rs_scen:
            gauge_score  = rs_scen["score"]
            gauge_label  = f"Scenario: {rs_scen['label']}"
        st.plotly_chart(make_resilience_gauge(gauge_score, gauge_label),
                        use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown(f"""
        <div class="route-card">
        <b>Score Components</b><br>
        🔄 Redundancy: {rs_base['alt']:.2f}<br>
        📡 Connectivity: {rs_base['bil']:.2f}<br>
        ⚠ Chokepoint: {rs_base['chk']:.2f}<br>
        🚢 Fleet: {rs_base['fleet']:.2f}
        </div>""", unsafe_allow_html=True)

# ─── All k routes table ───────────────────────────────────────────────────────
if len(baseline_routes_obj) > 1:
    with st.expander(f"All {len(baseline_routes_obj)} Baseline Routes"):
        all_rows = [r.to_dict() for r in baseline_routes_obj]
        st.dataframe(pd.DataFrame(all_rows)[
            ["path_str", "cost", "hops", "lead_time_days", "chk_exposure", "has_predicted"]
        ], use_container_width=True)
