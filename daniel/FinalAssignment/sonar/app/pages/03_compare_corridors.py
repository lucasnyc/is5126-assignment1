"""
Corridor Comparison

Helps supply chain planners compare shipping strategies for a specific
origin-destination pair.

The planner fixes an origin and a destination, then compares supply chain
configurations:
  - Direct shipping (baseline)
  - Via Hub Country A  (e.g. routing through Mexico)
  - Via Hub Country B  (e.g. routing through Brazil)
  ...

Each configuration is stress-tested under tariff and chokepoint scenarios so
the planner can see which strategy holds up under disruption.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import (
    PRODUCT_NAMES, PRODUCT_CODES, LATEST_YEAR,
    CHOKEPOINTS, CHOKEPOINT_WAYPOINTS,
)
from src.graph.routing import (
    find_k_routes, Route, apply_scenario, maritime_chokepoint_exposure,
    pareto_filter, score_frontier,
)
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import COLORS, make_route_radar
from app.components.theme import inject_global_css, section_header, render_footer

st.set_page_config(
    page_title="Compare Corridors · SONAR",
    layout="wide",
    page_icon="🌐",
)
inject_global_css()

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first to initialise the app.")
    st.stop()

graphs = st.session_state.graphs
scorer = st.session_state.scorer
edges  = st.session_state.edges

sample_graph  = graphs[(LATEST_YEAR, PRODUCT_CODES[0])]
ALL_COUNTRIES = sorted(sample_graph.nodes())

# ─── Hub suggestions per target market ────────────────────────────────────────
# Curated nearshoring candidates: geographically sensible hubs with strong
# port infrastructure between common origin-destination pairs.
_HUB_SUGGESTIONS: dict[str, list[str]] = {
    "United States":     ["Mexico", "Brazil", "Vietnam", "India"],
    "Germany":           ["Turkey", "Morocco", "United Arab Emirates"],
    "United Kingdom":    ["Morocco", "Turkey", "Netherlands"],
    "Japan":             ["Republic of Korea", "Vietnam", "Singapore"],
    "Republic of Korea": ["Japan", "Vietnam", "Singapore"],
    "Australia":         ["Singapore", "Malaysia", "India"],
    "Brazil":            ["Mexico", "Colombia", "South Africa"],
    "India":             ["United Arab Emirates", "Singapore", "Sri Lanka"],
    "China":             ["Vietnam", "Malaysia", "Bangladesh"],
    "_default":          ["Singapore", "United Arab Emirates", "Netherlands"],
}


def _hub_suggestions(destination: str) -> list[str]:
    return _HUB_SUGGESTIONS.get(destination, _HUB_SUGGESTIONS["_default"])


# ─── Colours for up to 4 strategies ──────────────────────────────────────────
_STRATEGY_COLORS = ["#4A90D9", "#27AE60", "#F5A623", "#E74C3C"]
MEDAL = ["🥇", "🥈", "🥉", "4️⃣"]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🌐 Compare Corridors")
st.caption(
    "Choose your origin and destination, then compare shipping strategies "
    "head-to-head — direct routes, via-hub configurations, and custom options — "
    "stress-tested under tariff shocks and chokepoint closures."
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Compare Corridors")
    st.markdown("### Route Configuration")

    origin = st.selectbox(
        "Origin",
        ALL_COUNTRIES,
        index=ALL_COUNTRIES.index("China") if "China" in ALL_COUNTRIES else 0,
        key="ns_origin",
    )
    destination = st.selectbox(
        "Destination",
        ALL_COUNTRIES,
        index=ALL_COUNTRIES.index("United States") if "United States" in ALL_COUNTRIES else 1,
        key="ns_dest",
    )
    product_label = st.selectbox(
        "Product", list(PRODUCT_NAMES.values()), key="ns_product"
    )
    product_code = [k for k, v in PRODUCT_NAMES.items() if v == product_label][0]

    st.markdown("---")
    st.markdown("### Scenario Simulation")
    st.caption("Stress-test strategies under disruption.")

    with st.expander("Tariff Scenarios", expanded=False):
        us_tariff    = st.slider("US Import Tariff (%)",    0, 100, 0, key="ns_us_t")
        eu_tariff    = st.slider("EU Import Tariff (%)",    0, 100, 0, key="ns_eu_t")
        cn_tariff    = st.slider("China Export Tariff (%)", 0, 100, 0, key="ns_cn_t")
        asean_tariff = st.slider("ASEAN Tariff (%)",        0, 100, 0, key="ns_asean_t")

    with st.expander("Chokepoint Closures", expanded=False):
        blocked = [
            cp for cp in CHOKEPOINTS
            if st.checkbox(cp, key=f"ns_cp_{cp}")
        ]

    st.markdown("---")
    st.markdown("### Priority Weights")
    st.caption("What matters most to your routing decision?")
    w_rs   = st.slider("Resilience Score",    0, 10, 5, key="ns_w_rs")
    w_cost = st.slider("Freight Cost",        0, 10, 4, key="ns_w_cost")
    w_lt   = st.slider("Speed",               0, 10, 3, key="ns_w_lt")
    w_chk  = st.slider("Redundancy",           0, 10, 3, key="ns_w_chk")
    w_vol  = st.slider("Rate Stability",      0, 10, 2, key="ns_w_vol")

if origin == destination:
    st.warning("Origin and destination must be different.")
    st.stop()

# ─── Build scenario graph ─────────────────────────────────────────────────────
gkey = (LATEST_YEAR, product_code)
if gkey not in graphs:
    st.error(f"No graph for {product_label}.")
    st.stop()

G_base = graphs[gkey]
tariff_mults = get_tariff_multipliers(
    us_pct=us_tariff, eu_pct=eu_tariff,
    china_pct=cn_tariff, asean_pct=asean_tariff,
)
G = apply_scenario(
    G_base,
    blocked_chokepoints=blocked or None,
    tariff_multipliers=tariff_mults or None,
)
median_lsci = float(
    pd.Series([G.nodes[n].get("lsci", 0) for n in G.nodes()]).median()
)
blocked_wps = frozenset(
    wp for cp in blocked for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])
)

# ─── Hub selector ─────────────────────────────────────────────────────────────
section_header(
    "🏭", "Hub Countries (Optional)",
    "Add up to 3 intermediate hubs to compare against the direct route baseline",
)
st.caption(
    f"The direct route ({origin} → {destination}) is always included as the baseline. "
    f"Add hub countries to evaluate routing via an intermediate stop."
)

_suggested = [
    c for c in _hub_suggestions(destination)
    if c in ALL_COUNTRIES and c not in (origin, destination)
]
_other = [c for c in ALL_COUNTRIES if c not in (origin, destination)]

selected_hubs = st.multiselect(
    "Hub countries to evaluate (up to 3)",
    options=_other,
    default=[c for c in _suggested[:2] if c in _other],
    max_selections=3,
    key="ns_hubs",
    help="Suggestions are based on your destination. Add or swap freely.",
)

if not selected_hubs:
    st.info(
        "No hub countries selected — showing the **Direct Route** only. "
        "Add hub countries above to compare alternative routing strategies.",
        icon="ℹ️",
    )

# ─── Compute strategy metrics ─────────────────────────────────────────────────
def _compute_strategies(
    origin: str,
    destination: str,
    hubs: list[str],
) -> list[dict]:
    """
    Compute supply chain metrics for:
      - Direct Route (baseline)
      - Via each hub country (two-stage routing: origin→hub→destination)
    """
    rows = []
    strategies = [(None, "Direct Route")] + [(h, f"Via {h}") for h in hubs]

    for hub, label in strategies:
        try:
            if hub is None:
                routes = find_k_routes(
                    G, origin, destination, k=5,
                    median_lsci=median_lsci, blocked_wps=blocked_wps,
                )
                best = routes[0]
                path = best.path
                cost = best.cost
                lt   = best.lead_time_days
            else:
                # Two-stage: find best leg to hub, then best leg from hub to destination
                r_to   = find_k_routes(G, origin, hub, k=1,
                                       median_lsci=median_lsci, blocked_wps=blocked_wps)
                r_from = find_k_routes(G, hub, destination, k=1,
                                       median_lsci=median_lsci, blocked_wps=blocked_wps)
                # Merge: remove duplicate hub node at the join
                combined = Route(
                    path=r_to[0].path + r_from[0].path[1:],
                    cost=r_to[0].cost + r_from[0].cost,
                    graph=G,
                    median_lsci=median_lsci,
                    blocked_wps=blocked_wps,
                )
                path = combined.path
                cost = combined.cost
                lt   = combined.lead_time_days

            rs      = scorer.score(path_k1=path, cost_k1=cost, G=G)
            chk_exp = maritime_chokepoint_exposure(path) * 100

            # Historical rate volatility on the direct origin→destination leg
            _mask = (
                (edges["origin"]       == origin) &
                (edges["destination"]  == destination) &
                (edges["product_code"] == product_code)
            )
            vol = float(edges[_mask]["freight_rate"].std() * 100) if len(edges[_mask]) > 1 else 0.0

            rows.append({
                "label":           label,
                "hub":             hub,
                "path_str":        " → ".join(path),
                "rs_score":        rs["score"],
                "rs_label":        rs["label"],
                "rs_detail":       rs,
                "freight_rate":    cost * 100,
                "lead_time":       lt,
                "chk_exposure":    chk_exp,
                "rate_volatility": vol,
            })

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            rows.append({
                "label":           label,
                "hub":             hub,
                "path_str":        "No viable route found",
                "rs_score":        0.0,
                "rs_label":        "No Route",
                "rs_detail":       {},
                "freight_rate":    None,
                "lead_time":       None,
                "chk_exposure":    100.0,
                "rate_volatility": 0.0,
            })

    return rows


# Manual session-state cache (G is not hashable, so can't use @st.cache_data)
_cache_key = (
    origin, destination, tuple(sorted(selected_hubs)),
    product_code, LATEST_YEAR,
    tuple(sorted(blocked)),
    tuple(sorted(tariff_mults.items())),
)
_cache = st.session_state.setdefault("_ns_cache", {})
if _cache_key not in _cache:
    with st.spinner("Evaluating supply chain strategies..."):
        _cache[_cache_key] = _compute_strategies(origin, destination, selected_hubs)

data     = _cache[_cache_key]
df       = pd.DataFrame(data)
_valid   = df[df["freight_rate"].notna()].copy()

# ─── Planning score ────────────────────────────────────────────────────────────
total_w = max(w_rs + w_cost + w_lt + w_chk + w_vol, 1)

def _norm_hi(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(s), index=s.index)

def _norm_lo(s: pd.Series) -> pd.Series:
    return 1.0 - _norm_hi(s)

if not _valid.empty:
    _valid["_n_rs"]   = _norm_hi(_valid["rs_score"])
    _valid["_n_cost"] = _norm_lo(_valid["freight_rate"])
    _valid["_n_lt"]   = _norm_lo(_valid["lead_time"])
    _valid["_n_chk"]  = _norm_lo(_valid["chk_exposure"])
    _valid["_n_vol"]  = _norm_lo(_valid["rate_volatility"])
    _valid["planning_score"] = (
        _valid["_n_rs"]   * (w_rs   / total_w) +
        _valid["_n_cost"] * (w_cost / total_w) +
        _valid["_n_lt"]   * (w_lt   / total_w) +
        _valid["_n_chk"]  * (w_chk  / total_w) +
        _valid["_n_vol"]  * (w_vol  / total_w)
    ) * 100
    df = df.merge(_valid[["label", "planning_score"]], on="label", how="left")
    df["planning_score"] = df["planning_score"].fillna(0)
else:
    df["planning_score"] = 0.0

# ─── Pareto filtering & frontier-based ranking ────────────────────────────────
# Build lightweight proxy objects so pareto_filter and score_frontier can be
# applied to the strategy pool (they expect .cost, .lead_time_days, .rs, .path).
class _StrategyProxy:
    """Thin wrapper so strategy dicts work with pareto_filter / score_frontier."""
    def __init__(self, row):
        self.cost           = float(row["freight_rate"] or 0) / 100.0
        self.lead_time_days = float(row["lead_time"]    or 0)
        self.rs             = float(row["rs_score"]     or 0)
        self.path           = row["path_str"].split(" → ")
        self.score          = 0.0
        self._label         = row["label"]

_valid_proxies = [_StrategyProxy(r) for _, r in df[df["freight_rate"].notna()].iterrows()]

# Map sliders to (w_c, w_t, w_r)
_w_r_raw = w_rs   + w_chk * 0.5
_w_c_raw = w_cost + w_vol * 0.5
_w_t_raw = w_lt
_proxy_total = _w_r_raw + _w_c_raw + _w_t_raw or 1.0
_w_c = _w_c_raw / _proxy_total
_w_t = _w_t_raw / _proxy_total
_w_r = _w_r_raw / _proxy_total

dominated_labels   = set()
recommended_label  = None

if len(_valid_proxies) > 1:
    dominated_proxies = [p for p in _valid_proxies if p not in pareto_filter(_valid_proxies)]
    dominated_labels  = {p._label for p in dominated_proxies}

if _valid_proxies:
    _scored = score_frontier(_valid_proxies, _w_c, _w_t, _w_r)
    recommended_label = _scored[0]._label if _scored else None

df_sorted = df.sort_values("planning_score", ascending=False).reset_index(drop=True)

# ─── Ranked strategy cards ─────────────────────────────────────────────────────
section_header(
    "🏆", "Strategy Ranking",
    f"{origin} → [hub] → {destination}  ·  ranked by your weighted priorities",
)

cols = st.columns(len(df_sorted))
for i, (col, (_, row)) in enumerate(zip(cols, df_sorted.iterrows())):
    color    = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
    medal    = MEDAL[i] if i < len(MEDAL) else str(i + 1)
    rs_color = "#27AE60" if row["rs_score"] >= 75 else "#F39C12" if row["rs_score"] >= 50 else "#E74C3C"
    fr_str   = f"{row['freight_rate']:.2f}%" if pd.notna(row["freight_rate"]) else "N/A"
    lt_str   = f"{row['lead_time']:.0f} d"   if pd.notna(row["lead_time"])    else "N/A"
    chk_str  = f"{row['chk_exposure']:.0f}%"

    hub_line = (
        f'<div style="font-size:10px;color:#58a6ff;margin-bottom:6px">'
        f'Hub: {row["hub"]}</div>'
        if row["hub"] else
        '<div style="font-size:10px;color:#8B949E;margin-bottom:6px">No intermediate hub</div>'
    )

    _is_recommended = row["label"] == recommended_label
    _is_dominated   = row["label"] in dominated_labels
    _badge_html = ""
    if _is_recommended:
        _badge_html = (
            '<div style="display:inline-block;background:#27AE6033;border:1px solid #27AE60;'
            'border-radius:4px;padding:2px 8px;font-size:10px;color:#27AE60;margin-bottom:6px">'
            '★ Recommended</div>'
        )
    elif _is_dominated:
        _badge_html = (
            '<div style="display:inline-block;background:#F5A62333;border:1px solid #F5A623;'
            'border-radius:4px;padding:2px 8px;font-size:10px;color:#F5A623;margin-bottom:6px">'
            '⚠ Dominated</div>'
        )

    with col:
        st.markdown(
            "".join([
                f'<div style="background:#161b22;border:1px solid #21262d;'
                f'border-top:3px solid {color};border-radius:10px;padding:16px;height:100%">',
                f'<div style="font-size:22px;margin-bottom:4px">{medal}</div>',
                f'<div style="font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:4px">'
                f'{row["label"]}</div>',
                _badge_html,
                hub_line,
                '<div style="font-size:10px;color:#8B949E;text-transform:uppercase;'
                'letter-spacing:.04em">Planning Score</div>',
                f'<div style="font-size:24px;font-weight:800;color:{color};margin-bottom:10px">'
                f'{row["planning_score"]:.1f}</div>',
                '<div style="display:flex;gap:10px;flex-wrap:wrap">',
                f'<div><div style="font-size:9px;color:#aaa">RS Score</div>'
                f'<div style="font-weight:600;color:{rs_color}">{row["rs_score"]:.1f}</div></div>',
                f'<div><div style="font-size:9px;color:#aaa">Freight</div>'
                f'<div style="font-weight:600;color:#ccc">{fr_str}</div></div>',
                f'<div><div style="font-size:9px;color:#aaa">Lead Time</div>'
                f'<div style="font-weight:600;color:#ccc">{lt_str}</div></div>',
                f'<div><div style="font-size:9px;color:#aaa">Chk. Exposure</div>'
                f'<div style="font-weight:600;color:#ccc">{chk_str}</div></div>',
                '</div>',
                f'<div style="margin-top:8px;font-size:10px;color:#6e7681">'
                f'{row["path_str"]}</div>',
                '</div>',
            ]),
            unsafe_allow_html=True,
        )

# ─── Metric bar chart ─────────────────────────────────────────────────────────
st.markdown("---")
section_header("📊", "Strategy Comparison", "All metrics side by side")

_bar_metrics = {
    "Resilience Score (0–100)": "rs_score",
    "Freight Cost (%)":         "freight_rate",
    "Speed (days)":             "lead_time",
    "Redundancy (% exposure)":  "chk_exposure",
    "Rate Stability (σ pp)":    "rate_volatility",
}
fig_bar = go.Figure()
for i, (_, row) in enumerate(df_sorted.iterrows()):
    color  = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
    y_vals = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in _bar_metrics.values()]
    fig_bar.add_trace(go.Bar(
        name=row["label"],
        x=list(_bar_metrics.keys()),
        y=y_vals,
        marker_color=color,
        text=[f"{v:.1f}" for v in y_vals],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
fig_bar.update_layout(
    barmode="group",
    paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["paper"],
    font=dict(color="white"),
    xaxis=dict(gridcolor="#21262d"),
    yaxis=dict(gridcolor="#21262d"),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
    height=400, margin=dict(t=20),
)
st.plotly_chart(fig_bar, width='stretch', config={"displayModeBar": False})

# ─── Resilience radar ─────────────────────────────────────────────────────────
st.markdown("---")
section_header("🕸", "Resilience Profile", "RS sub-components for each strategy")

# Build lightweight Route-like objects that make_route_radar can consume.
# The function reads r.rs_detail for the 5 sub-components.
class _StrategyRoute:
    def __init__(self, row: pd.Series, color: str):
        self.rs             = row["rs_score"]
        self.cost           = (row["freight_rate"] / 100) if pd.notna(row["freight_rate"]) else 0.0
        self.lead_time_days = row["lead_time"] if pd.notna(row["lead_time"]) else 0.0
        self.chk_exposure   = row["chk_exposure"] / 100
        self.rs_detail      = row["rs_detail"] if isinstance(row["rs_detail"], dict) else {}
        self._color         = color

# Temporarily extend the module-level lookup dicts so make_route_radar picks up
# our strategy labels and colours (it falls back gracefully to key as label / #ccc).
from src.viz import globe as _globe
_saved_labels = dict(_globe.CRITERIA_LABELS)
_saved_colors = dict(_globe.CRITERIA_COLORS)

radar_routes: dict = {}
for i, (_, row) in enumerate(df_sorted.iterrows()):
    key   = row["label"]
    color = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
    _globe.CRITERIA_LABELS[key] = key
    _globe.CRITERIA_COLORS[key] = color
    radar_routes[key] = _StrategyRoute(row, color)

fig_radar = make_route_radar(radar_routes)
st.plotly_chart(fig_radar, width='stretch', config={"displayModeBar": False})

# Restore original dicts
_globe.CRITERIA_LABELS = _saved_labels
_globe.CRITERIA_COLORS = _saved_colors

# ─── RS Component Breakdown ────────────────────────────────────────────────────
st.markdown("---")
section_header(
    "🔬", "Resilience Component Breakdown",
    "Which of the 5 factors drives each strategy's RS score?",
)
st.caption(
    "Each component scored 0–100. Compare across strategies to see how routing via a hub "
    "shifts individual risk factors."
)

_comp_fig = go.Figure()
for i, (_, row) in enumerate(df_sorted.iterrows()):
    _detail = row.get("rs_detail", {})
    _comps  = _detail.get("components_pct", {}) if isinstance(_detail, dict) else {}
    if not _comps:
        continue
    _comp_fig.add_trace(go.Bar(
        name=row["label"],
        x=list(_comps.keys()),
        y=list(_comps.values()),
        marker_color=_STRATEGY_COLORS[i % len(_STRATEGY_COLORS)],
        text=[f"{v:.1f}" for v in _comps.values()],
        textposition="outside",
        textfont=dict(color="white", size=9),
    ))

if _comp_fig.data:
    _comp_fig.update_layout(
        barmode="group",
        paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        yaxis=dict(title="Component Score (0–100)", range=[0, 115], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
        height=380, margin=dict(t=20),
    )
    st.plotly_chart(_comp_fig, width='stretch', config={"displayModeBar": False})

# ─── Full metrics table ───────────────────────────────────────────────────────
st.markdown("---")
section_header("📋", "Full Metrics Table")

_display_cols = {
    "label":           "Strategy",
    "path_str":        "Supply Chain Path",
    "rs_score":        "RS Score",
    "rs_label":        "Resilience Rating",
    "freight_rate":    "Freight Rate (%)",
    "lead_time":       "Lead Time (d)",
    "chk_exposure":    "Chokepoint Exp. (%)",
    "rate_volatility": "Rate Volatility (σ pp)",
    "planning_score":  "Planning Score",
}
_out = df_sorted[[c for c in _display_cols if c in df_sorted.columns]].rename(
    columns=_display_cols
).copy()
for num_col in ["RS Score", "Freight Rate (%)", "Lead Time (d)",
                "Chokepoint Exp. (%)", "Rate Volatility (σ pp)", "Planning Score"]:
    if num_col in _out.columns:
        _out[num_col] = pd.to_numeric(_out[num_col], errors="coerce").round(1)
_out = _out.set_index("Strategy")
st.dataframe(_out, width='stretch')

st.download_button(
    "Download Strategy Comparison as CSV",
    data=_out.to_csv(),
    file_name=(
        f"sonar_corridors_{origin.replace(' ', '_')}"
        f"_to_{destination.replace(' ', '_')}.csv"
    ),
    mime="text/csv",
)

# ─── Historical rate trend (direct corridor) ──────────────────────────────────
st.markdown("---")
section_header(
    "📈", "Historical Freight Rate — Direct Corridor (2016–2021)",
    "Context for rate stability on the baseline direct route",
)
_mask = (
    (edges["origin"]       == origin) &
    (edges["destination"]  == destination) &
    (edges["product_code"] == product_code)
)
_trend = edges[_mask].sort_values("year")
if not _trend.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=_trend["year"].tolist(),
        y=(_trend["freight_rate"] * 100).tolist(),
        mode="lines+markers",
        name=f"{origin} → {destination}",
        line=dict(color="#4A90D9", width=2),
        marker=dict(size=7),
    ))
    fig_trend.add_vrect(
        x0=2019.5, x1=2021.5,
        fillcolor="#F39C12", opacity=0.06,
        line_width=0,
        annotation_text="COVID era",
        annotation_position="top left",
        annotation_font_color="#F39C12",
        annotation_font_size=10,
    )
    fig_trend.update_layout(
        paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        xaxis=dict(title="Year", gridcolor="#21262d", tickvals=list(range(2016, 2022))),
        yaxis=dict(title="Freight Rate (% of cargo value)", gridcolor="#21262d"),
        height=280, margin=dict(t=10),
    )
    st.plotly_chart(fig_trend, width='stretch', config={"displayModeBar": False})
else:
    st.info("No historical rate data available for this corridor.")

render_footer()
