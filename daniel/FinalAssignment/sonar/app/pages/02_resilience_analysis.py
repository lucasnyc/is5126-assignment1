"""
Resilience Analysis page.
Shows resilience score heatmap across top trade corridors and component breakdown.

Default view (no scenario) uses the baseline pre-computed at startup — instant load.
Scenario changes trigger a focused recompute via @st.cache_data.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

import plotly.express as px

from config import PRODUCT_NAMES, PRODUCT_CODES, LATEST_YEAR, CHOKEPOINTS, TOP_CORRIDORS, RS_TIER_HIGH, COST_TIER_MED
from src.graph.routing import find_k_routes, apply_scenario
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import make_corridor_heatmap, COLORS
from app.components.theme import inject_global_css, section_header, stat_card, render_footer

st.set_page_config(page_title="Resilience Analysis \u00b7 SONAR",
                   layout="wide", page_icon="\U0001f4ca")

inject_global_css()

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first.")
    st.stop()

graphs  = st.session_state.graphs
scorer  = st.session_state.scorer
year    = LATEST_YEAR

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## \U0001f4ca Resilience Analysis")
    selected_products = st.multiselect(
        "Products",
        options=list(PRODUCT_NAMES.values()),
        default=list(PRODUCT_NAMES.values()),
    )
    n_corridors = st.slider("Top N corridors", 10, len(TOP_CORRIDORS), 20)
    st.markdown("---")
    st.markdown("### Scenario (optional)")
    blocked = []
    for cp_name in CHOKEPOINTS.keys():
        if st.checkbox(cp_name, key=f"rs_cp_{cp_name}"):
            blocked.append(cp_name)
    us_t = st.slider("US Tariff (%)", 0, 50, 0, step=5, key="rs_us")
    eu_t = st.slider("EU Tariff (%)", 0, 50, 0, step=5, key="rs_eu")
    cn_t = st.slider("China Tariff (%)", 0, 50, 0, step=5, key="rs_cn")
    st.caption("Scores update automatically when inputs change.")

st.markdown("# \U0001f4ca Resilience Analysis")
st.caption("Compare route resilience across the top global trade corridors.")

# ─── Resolve active corridor / product slice ──────────────────────────────────
active_corridors = TOP_CORRIDORS[:n_corridors]
selected_codes   = [k for k, v in PRODUCT_NAMES.items() if v in selected_products]
has_scenario     = bool(blocked) or any([us_t, eu_t, cn_t])

# ─── Fetch heatmap data ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing scenario scores...")
def compute_scenario_data(
    corridors: list, product_codes: list, year: int,
    blocked_cps: tuple, us_t: int, eu_t: int, cn_t: int
) -> list[dict]:
    tariff_mult = get_tariff_multipliers(float(us_t), float(eu_t), float(cn_t), 0.0)
    rows = []
    for orig, dest in corridors:
        for prod in product_codes:
            gkey = (year, prod)
            if gkey not in graphs:
                continue
            G = apply_scenario(graphs[gkey], list(blocked_cps), tariff_mult)
            try:
                routes = find_k_routes(G, orig, dest, k=3)
                rs = scorer.score_from_routes(routes, G)
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


if not has_scenario and "heatmap_baseline" in st.session_state:
    top_set = {(o, d) for o, d in active_corridors}
    data = [
        r for r in st.session_state.heatmap_baseline
        if (r["origin"], r["destination"]) in top_set
        and r["product_name"] in selected_products
    ]
else:
    data = compute_scenario_data(
        active_corridors, selected_codes, year,
        tuple(sorted(blocked)), us_t, eu_t, cn_t,
    )

if not data:
    st.warning("No data available for the selected configuration.")
    st.stop()

# ─── Tabs: Heatmap | Planning Tiers ──────────────────────────────────────────
tab_heatmap, tab_tiers = st.tabs(["🗺 Resilience Heatmap", "📐 Planning Tiers"])

with tab_heatmap:
    section_header("\U0001f5fa", "Resilience Heatmap", f"Top {n_corridors} corridors \u00b7 {year}")
    heatmap_fig = make_corridor_heatmap(
        data,
        title=f"Resilience Scores \u2014 Top {n_corridors} Corridors ({year})"
    )
    st.plotly_chart(heatmap_fig, width='stretch')

with tab_tiers:
    section_header("\U0001f4d0", "Planning Tier View",
                   "Strategy quadrant: RS Score vs. Freight Cost — where should you commit?")
    st.caption(
        "Each bubble is a corridor × product pair. "
        "Use this to classify your lanes into strategy tiers before committing resources."
    )

    # Build tier dataframe: join RS scores with median freight cost from edges
    _edges_df = st.session_state.edges
    _tier_rows = []
    for row in data:
        o, d, pname = row["origin"], row["destination"], row["product_name"]
        rs_score    = row["score"]
        # Get median freight rate for this corridor × product (across all available years)
        pcode = [k for k, v in PRODUCT_NAMES.items() if v == pname]
        if not pcode:
            continue
        _mask = (
            (_edges_df["origin"]       == o) &
            (_edges_df["destination"]  == d) &
            (_edges_df["product_code"] == pcode[0]) &
            (_edges_df["year"]         == year)
        )
        _sub = _edges_df[_mask]
        median_cost = float(_sub["freight_rate"].median()) if not _sub.empty else None
        if median_cost is None or median_cost <= 0:
            continue

        # Classify tier
        high_rs   = rs_score >= RS_TIER_HIGH
        high_cost = median_cost >= COST_TIER_MED
        if high_rs and not high_cost:
            tier = "Tier 1 — Commit"
            tier_color = "#27AE60"
        elif high_rs and high_cost:
            tier = "Tier 2 — Critical Only"
            tier_color = "#4A90D9"
        elif not high_rs and not high_cost:
            tier = "Tier 3 — Buffer Stock"
            tier_color = "#F39C12"
        else:
            tier = "Tier 4 — Diversify"
            tier_color = "#E74C3C"

        _tier_rows.append({
            "Corridor":     f"{o} → {d}",
            "Product":      pname,
            "RS Score":     rs_score,
            "Freight Rate (%)": median_cost * 100,
            "Tier":         tier,
            "_color":       tier_color,
        })

    if _tier_rows:
        _tier_df = pd.DataFrame(_tier_rows)

        # Scatter plot
        _tier_color_map = {
            "Tier 1 — Commit":        "#27AE60",
            "Tier 2 — Critical Only": "#4A90D9",
            "Tier 3 — Buffer Stock":  "#F39C12",
            "Tier 4 — Diversify":     "#E74C3C",
        }
        _scatter = px.scatter(
            _tier_df,
            x="Freight Rate (%)",
            y="RS Score",
            color="Tier",
            color_discrete_map=_tier_color_map,
            symbol="Product",
            hover_data=["Corridor", "Product", "RS Score", "Freight Rate (%)"],
            title="Planning Tiers: RS Score vs Freight Cost",
            labels={"Freight Rate (%)": "Freight Cost (% of cargo value)", "RS Score": "Resilience Score (0–100)"},
        )

        # Quadrant dividers
        _scatter.add_hline(y=RS_TIER_HIGH,    line_dash="dash", line_color="#555", line_width=1)
        _scatter.add_vline(x=COST_TIER_MED * 100, line_dash="dash", line_color="#555", line_width=1)

        # Quadrant labels
        x_max = _tier_df["Freight Rate (%)"].max() * 1.1
        _scatter.add_annotation(x=COST_TIER_MED * 100 / 2,       y=RS_TIER_HIGH + 5,
                                 text="Tier 1: Commit", showarrow=False,
                                 font=dict(color="#27AE60", size=10))
        _scatter.add_annotation(x=(COST_TIER_MED * 100 + x_max) / 2, y=RS_TIER_HIGH + 5,
                                 text="Tier 2: Critical Only", showarrow=False,
                                 font=dict(color="#4A90D9", size=10))
        _scatter.add_annotation(x=COST_TIER_MED * 100 / 2,       y=RS_TIER_HIGH - 8,
                                 text="Tier 3: Buffer Stock", showarrow=False,
                                 font=dict(color="#F39C12", size=10))
        _scatter.add_annotation(x=(COST_TIER_MED * 100 + x_max) / 2, y=RS_TIER_HIGH - 8,
                                 text="Tier 4: Diversify", showarrow=False,
                                 font=dict(color="#E74C3C", size=10))

        _scatter.update_layout(
            paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["paper"],
            font=dict(color="white"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
            height=480,
        )
        st.plotly_chart(_scatter, width='stretch', config={"displayModeBar": False})

        # Tier legend cards
        st.markdown(
            '<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 16px 0">',
            unsafe_allow_html=True,
        )
        _tier_descs = {
            "Tier 1 — Commit":        ("High RS, Low Cost",  "Best of both worlds — commit confidently to long-term contracts."),
            "Tier 2 — Critical Only": ("High RS, High Cost", "Resilient but expensive — reserve for mission-critical products."),
            "Tier 3 — Buffer Stock":  ("Low RS, Low Cost",   "Cheap but fragile — acceptable with safety stock buffer."),
            "Tier 4 — Diversify":     ("Low RS, High Cost",  "Worst position — diversify sourcing or renegotiate immediately."),
        }
        tier_cards_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 16px 0">'
        for tier_name, (subtitle, desc) in _tier_descs.items():
            color = _tier_color_map[tier_name]
            tier_cards_html += (
                f'<div style="background:{color}11;border:1px solid {color}44;border-left:3px solid {color};'
                f'border-radius:8px;padding:10px 14px;min-width:200px;flex:1">'
                f'<div style="font-size:12px;font-weight:700;color:{color};margin-bottom:3px">{tier_name}</div>'
                f'<div style="font-size:11px;color:#8B949E;margin-bottom:4px">{subtitle}</div>'
                f'<div style="font-size:11px;color:#c9d1d9">{desc}</div>'
                f'</div>'
            )
        tier_cards_html += '</div>'
        st.markdown(tier_cards_html, unsafe_allow_html=True)

        # Sortable tier table
        _display_tier_df = (
            _tier_df[["Tier", "Corridor", "Product", "RS Score", "Freight Rate (%)"]]
            .sort_values(["Tier", "RS Score"], ascending=[True, False])
            .reset_index(drop=True)
        )
        _display_tier_df["Freight Rate (%)"] = _display_tier_df["Freight Rate (%)"].round(2)
        _display_tier_df["RS Score"]         = _display_tier_df["RS Score"].round(1)
        st.dataframe(_display_tier_df, width='stretch')
    else:
        st.info("Not enough freight rate data to build planning tiers for the selected corridors.")

# ─── Summary statistics ───────────────────────────────────────────────────────
df = pd.DataFrame(data)
avg_score     = df["score"].mean()
high_count    = int((df["score"] >= 75).sum())
critical_count = int((df["score"] < 25).sum())
noroute_count = int((df["label"] == "No Route").sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        stat_card("Avg Resilience Score", f"{avg_score:.1f}",
                  delta="of 100", delta_type="good" if avg_score >= 50 else "warn"),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        stat_card("High Resilience", str(high_count),
                  delta="corridors \u2265 75 RS", delta_type="good"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        stat_card("Critical Risk", str(critical_count),
                  delta="corridors < 25 RS",
                  delta_type="bad" if critical_count > 0 else "good"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        stat_card("No Route", str(noroute_count),
                  delta="unreachable pairs",
                  delta_type="bad" if noroute_count > 0 else "good"),
        unsafe_allow_html=True,
    )

# ─── Table view ──────────────────────────────────────────────────────────────
st.markdown("---")
section_header("\U0001f4cb", "Detailed Scores")
display_df = df.sort_values("score", ascending=False).copy()
display_df["corridor"] = display_df["origin"] + " \u2192 " + display_df["destination"]
table_df = (
    display_df[["corridor", "product_name", "score", "label"]]
    .rename(columns={
        "corridor":     "Corridor",
        "product_name": "Product",
        "score":        "RS Score",
        "label":        "Rating",
    }).reset_index(drop=True)
)
st.dataframe(table_df, width='stretch')

st.download_button(
    "Download as CSV",
    data=table_df.to_csv(index=False),
    file_name=f"sonar_resilience_{n_corridors}_corridors.csv",
    mime="text/csv",
)

# ─── Component analysis for selected corridor ─────────────────────────────────
st.markdown("---")
section_header("\U0001f50d", "Drill Down: Score Components")
chosen_corridor = st.selectbox(
    "Select corridor to analyze",
    [f"{o} \u2192 {d}" for o, d in active_corridors]
)
chosen_product   = st.selectbox("Product", list(PRODUCT_NAMES.values()), key="dd_prod")
chosen_prod_code = [k for k, v in PRODUCT_NAMES.items() if v == chosen_product][0]

orig_c, dest_c = chosen_corridor.split(" \u2192 ", 1)
gkey = (year, chosen_prod_code)
if gkey in graphs:
    tariff_mult  = get_tariff_multipliers(float(us_t), float(eu_t), float(cn_t), 0.0)
    _drill_key   = (orig_c, dest_c, chosen_prod_code, year, tuple(sorted(blocked)), us_t, eu_t, cn_t)
    _drill_cache = st.session_state.setdefault("_ra_drill_cache", {})
    if _drill_key not in _drill_cache:
        G_s = apply_scenario(graphs[gkey], blocked, tariff_mult)
        try:
            _dr = find_k_routes(G_s, orig_c, dest_c, k=3)
            _rs = scorer.score_from_routes(_dr, G_s)
            _drill_cache[_drill_key] = ("ok", _dr, _rs)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            _drill_cache[_drill_key] = ("err", str(e))

    _result = _drill_cache[_drill_key]
    if _result[0] == "err":
        st.warning(f"No route found: {_result[1]}")
    else:
        _, routes, rs = _result
        comp = rs["components_pct"]

        # Score badge
        score_val = rs["score"]
        badge_color = (
            "#27AE60" if score_val >= 75 else
            "#F39C12" if score_val >= 50 else
            "#E74C3C" if score_val >= 25 else "#8E44AD"
        )
        best_path_str = " \u2192 ".join(routes[0].path)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;margin:12px 0 16px 0">'
            f'<div style="background:{badge_color}22;border:1px solid {badge_color};'
            f'border-radius:8px;padding:8px 16px;display:inline-flex;align-items:baseline;gap:6px">'
            f'<span style="font-size:24px;font-weight:800;color:{badge_color}">{score_val:.1f}</span>'
            f'<span style="font-size:12px;color:#8B949E">/ 100</span>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:14px;font-weight:600;color:#e6edf3">{rs["label"]}</div>'
            f'<div style="font-size:12px;color:#8B949E">Best route: {best_path_str}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        _comp_colors = {
            "Reliability": "#4A90D9",
            "Redundancy":  "#27AE60",
            "Weather":     "#F39C12",
            "Ports":       "#9B59B6",
            "Security":    "#E74C3C",
        }
        fig_bar = go.Figure(go.Bar(
            x=list(comp.keys()),
            y=list(comp.values()),
            marker_color=[_comp_colors.get(k, "#8B949E") for k in comp.keys()],
            text=[f"{v:.1f}" for v in comp.values()],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_bar.update_layout(
            title=f"RS Components: {chosen_corridor} ({chosen_product})",
            yaxis_title="Component Score (0–100 each)",
            paper_bgcolor=COLORS["paper"],
            plot_bgcolor=COLORS["paper"],
            font=dict(color="white"),
            yaxis=dict(range=[0, 105], gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            height=350,
        )
        st.plotly_chart(fig_bar, width='stretch')

render_footer()
