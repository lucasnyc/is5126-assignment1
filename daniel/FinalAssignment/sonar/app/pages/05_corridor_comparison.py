"""
Corridor Comparison — side-by-side strategic comparison of 2–4 trade lanes.

Lets planners compare any combination of origin/destination pairs across all
key metrics (RS Score, Freight Rate, Lead Time, Chokepoint Exposure, Rate
Volatility) in a single view, ranked by a user-weighted composite score.

Designed for sourcing decisions: "Should I source electronics from China,
Korea, or Vietnam for my US distribution?"
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import PRODUCT_NAMES, PRODUCT_CODES, LATEST_YEAR, TOP_CORRIDORS
from src.graph.routing import find_k_routes
from src.viz.globe import COLORS
from app.components.theme import inject_global_css, section_header, render_footer

st.set_page_config(
    page_title="Corridor Comparison · SONAR",
    layout="wide",
    page_icon="⚖",
)
inject_global_css()

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first to initialise the app.")
    st.stop()

graphs  = st.session_state.graphs
scorer  = st.session_state.scorer
edges   = st.session_state.edges

sample_graph  = graphs[(LATEST_YEAR, PRODUCT_CODES[0])]
ALL_COUNTRIES = sorted(sample_graph.nodes())

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# ⚖ Corridor Comparison")
st.caption(
    "Compare 2–4 trade lanes side by side. "
    "Rank by your priorities to identify the best sourcing strategy."
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖ Corridor Comparison")
    st.markdown("### Product")
    product_label = st.selectbox("Product", list(PRODUCT_NAMES.values()), key="cc_product")
    product_code  = [k for k, v in PRODUCT_NAMES.items() if v == product_label][0]

    st.markdown("### Priority Weights")
    st.caption("Set your planning priorities. Weights are normalised automatically.")
    w_rs   = st.slider("Resilience Score",  0, 10, 5, key="w_rs")
    w_cost = st.slider("Freight Cost",      0, 10, 4, key="w_cost")
    w_lt   = st.slider("Lead Time",         0, 10, 3, key="w_lt")
    w_vol  = st.slider("Rate Volatility",   0, 10, 2, key="w_vol")
    w_chk  = st.slider("Chokepoint Safety", 0, 10, 2, key="w_chk")

# ─── Corridor selector ───────────────────────────────────────────────────────
st.markdown("### Select corridors to compare (2–4)")

# Preset from top corridors
preset_options = ["Custom"] + [f"{o} → {d}" for o, d in TOP_CORRIDORS[:10]]
preset = st.selectbox(
    "Quick-load a preset corridor set",
    preset_options,
    key="cc_preset",
)

N_CORRIDORS = st.number_input(
    "Number of corridors", min_value=2, max_value=4, value=2, step=1, key="cc_n"
)

corridors = []
default_pairs = [
    ("China",             "United States"),
    ("Republic of Korea", "United States"),
    ("Japan",             "United States"),
    ("India",             "United States"),
]

for i in range(int(N_CORRIDORS)):
    c1, c2 = st.columns(2)
    if preset != "Custom" and i == 0:
        default_o, default_d = preset.split(" → ", 1)
    else:
        default_o = default_pairs[i][0] if i < len(default_pairs) else ALL_COUNTRIES[0]
        default_d = default_pairs[i][1] if i < len(default_pairs) else ALL_COUNTRIES[1]
    with c1:
        orig = st.selectbox(
            f"Corridor {i+1} — Origin",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index(default_o) if default_o in ALL_COUNTRIES else 0,
            key=f"cc_orig_{i}",
        )
    with c2:
        dest = st.selectbox(
            f"Corridor {i+1} — Destination",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index(default_d) if default_d in ALL_COUNTRIES else 0,
            key=f"cc_dest_{i}",
        )
    if orig != dest:
        corridors.append((orig, dest))

if len(corridors) < 2:
    st.warning("Please select at least 2 distinct corridors (origin ≠ destination).")
    st.stop()

# ─── Compute metrics for each corridor ───────────────────────────────────────
gkey = (LATEST_YEAR, product_code)
if gkey not in graphs:
    st.error(f"No graph for product {product_label}.")
    st.stop()

G = graphs[gkey]

@st.cache_data(show_spinner="Computing corridor metrics...")
def _compute_corridor_metrics(
    corridors: list, product_code: int, year: int
) -> list[dict]:
    rows = []
    for orig, dest in corridors:
        try:
            routes = find_k_routes(G, orig, dest, k=3)
            rs     = scorer.score_from_routes(routes, G)
            best   = routes[0]
            # Chokepoint exposure from best route
            from config import ALL_CHOKEPOINT_COUNTRIES
            chk_count = sum(1 for c in best.path if c in ALL_CHOKEPOINT_COUNTRIES)
            chk_exp   = chk_count / len(ALL_CHOKEPOINT_COUNTRIES)

            # Rate volatility across 2016–2021
            _mask = (
                (edges["origin"]       == orig) &
                (edges["destination"]  == dest) &
                (edges["product_code"] == product_code)
            )
            _sub = edges[_mask]
            vol = float(_sub["freight_rate"].std()) * 100 if len(_sub) > 1 else 0.0

            rows.append({
                "corridor":    f"{orig} → {dest}",
                "origin":      orig,
                "destination": dest,
                "rs_score":    rs["score"],
                "rs_label":    rs["label"],
                "freight_rate": best.cost * 100,
                "lead_time":   best.lead_time_days,
                "chk_exposure": chk_exp,
                "rate_volatility": vol,
                "best_path":   " → ".join(best.path),
            })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            rows.append({
                "corridor":    f"{orig} → {dest}",
                "origin":      orig,
                "destination": dest,
                "rs_score":    0.0,
                "rs_label":    "No Route",
                "freight_rate": None,
                "lead_time":   None,
                "chk_exposure": 1.0,
                "rate_volatility": 0.0,
                "best_path":   "N/A",
            })
    return rows


_cache_key = (tuple(sorted(corridors)), product_code, LATEST_YEAR)
_corridor_cache = st.session_state.setdefault("_cc_cache", {})
if _cache_key not in _corridor_cache:
    _corridor_cache[_cache_key] = _compute_corridor_metrics(corridors, product_code, LATEST_YEAR)

metrics = _corridor_cache[_cache_key]
df = pd.DataFrame(metrics)

if df.empty:
    st.warning("No data available for the selected corridors.")
    st.stop()

# ─── Composite planning score ─────────────────────────────────────────────────
# Normalise each metric to [0, 1] range (higher = better for all dimensions)
total_w = w_rs + w_cost + w_lt + w_vol + w_chk
if total_w == 0:
    total_w = 1

_valid = df[df["freight_rate"].notna()].copy()

def _norm_higher_better(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(s), index=s.index)

def _norm_lower_better(s: pd.Series) -> pd.Series:
    return 1 - _norm_higher_better(s)

if not _valid.empty:
    _valid["_n_rs"]   = _norm_higher_better(_valid["rs_score"])
    _valid["_n_cost"] = _norm_lower_better(_valid["freight_rate"])
    _valid["_n_lt"]   = _norm_lower_better(_valid["lead_time"])
    _valid["_n_vol"]  = _norm_lower_better(_valid["rate_volatility"])
    _valid["_n_chk"]  = _norm_lower_better(_valid["chk_exposure"])

    _valid["planning_score"] = (
        _valid["_n_rs"]   * (w_rs   / total_w) +
        _valid["_n_cost"] * (w_cost / total_w) +
        _valid["_n_lt"]   * (w_lt   / total_w) +
        _valid["_n_vol"]  * (w_vol  / total_w) +
        _valid["_n_chk"]  * (w_chk  / total_w)
    ) * 100

    df = df.merge(
        _valid[["corridor", "planning_score"]],
        on="corridor", how="left",
    )
    df["planning_score"] = df["planning_score"].fillna(0)
else:
    df["planning_score"] = 0

df_sorted = df.sort_values("planning_score", ascending=False).reset_index(drop=True)

# ─── Ranked summary cards ─────────────────────────────────────────────────────
section_header("🏆", "Ranking", "Sorted by your weighted planning score")
MEDAL = ["🥇", "🥈", "🥉", "4️⃣"]
CARD_COLORS = ["#F5A623", "#8B949E", "#C0A070", "#555"]

cols = st.columns(len(df_sorted))
for i, (col, (_, row)) in enumerate(zip(cols, df_sorted.iterrows())):
    color = CARD_COLORS[i] if i < len(CARD_COLORS) else "#4A90D9"
    medal = MEDAL[i] if i < len(MEDAL) else str(i + 1)
    rs_color = "#27AE60" if row["rs_score"] >= 75 else "#F39C12" if row["rs_score"] >= 50 else "#E74C3C"
    fr_str   = f"{row['freight_rate']:.2f}%" if pd.notna(row["freight_rate"]) else "N/A"
    lt_str   = f"{row['lead_time']:.0f} d"  if pd.notna(row["lead_time"])   else "N/A"
    with col:
        st.markdown(
            "".join([
                f'<div style="background:#161b22;border:1px solid #21262d;border-top:3px solid {color};'
                f'border-radius:10px;padding:16px;height:100%">',
                f'<div style="font-size:22px;margin-bottom:4px">{medal}</div>',
                f'<div style="font-size:13px;font-weight:700;color:#e6edf3;margin-bottom:10px">{row["corridor"]}</div>',
                f'<div style="font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.04em">Planning Score</div>',
                f'<div style="font-size:24px;font-weight:800;color:{color};margin-bottom:10px">{row["planning_score"]:.1f}</div>',
                '<div style="display:flex;gap:10px;flex-wrap:wrap">',
                f'<div><div style="font-size:9px;color:#aaa">RS Score</div>',
                f'<div style="font-weight:600;color:{rs_color}">{row["rs_score"]:.1f}</div></div>',
                f'<div><div style="font-size:9px;color:#aaa">Freight</div>',
                f'<div style="font-weight:600;color:#ccc">{fr_str}</div></div>',
                f'<div><div style="font-size:9px;color:#aaa">Lead Time</div>',
                f'<div style="font-weight:600;color:#ccc">{lt_str}</div></div>',
                '</div>',
                f'<div style="margin-top:8px;font-size:10px;color:#8B949E">{row["best_path"]}</div>',
                '</div>',
            ]),
            unsafe_allow_html=True,
        )

# ─── Grouped bar chart ────────────────────────────────────────────────────────
st.markdown("---")
section_header("📊", "Metric Comparison", "All corridors side by side")

_metrics_to_plot = {
    "RS Score (0–100)":        ("rs_score",       False),
    "Freight Rate (%)":        ("freight_rate",    False),
    "Lead Time (days)":        ("lead_time",       False),
    "Rate Volatility (σ pp)":  ("rate_volatility", False),
    "Chokepoint Exposure (%)": ("chk_exposure",    False),
}

_bar_colors = ["#4A90D9", "#27AE60", "#F5A623", "#E74C3C"]
fig_bar = go.Figure()

for i, row in df_sorted.iterrows():
    color = _bar_colors[i % len(_bar_colors)]
    y_vals = []
    x_labs = []
    for label, (col, _) in _metrics_to_plot.items():
        v = row[col]
        if col == "chk_exposure":
            v = v * 100 if v is not None else 0
        y_vals.append(float(v) if v is not None else 0)
        x_labs.append(label)
    fig_bar.add_trace(go.Bar(
        name=row["corridor"],
        x=x_labs,
        y=y_vals,
        marker_color=color,
        text=[f"{v:.1f}" for v in y_vals],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))

fig_bar.update_layout(
    barmode="group",
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["paper"],
    font=dict(color="white"),
    xaxis=dict(gridcolor="#21262d"),
    yaxis=dict(gridcolor="#21262d"),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
    height=400,
    margin=dict(t=20),
)
st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ─── Detailed comparison table ────────────────────────────────────────────────
st.markdown("---")
section_header("📋", "Full Metrics Table")

display_cols = {
    "corridor":         "Corridor",
    "rs_score":         "RS Score",
    "rs_label":         "Rating",
    "freight_rate":     "Freight Rate (%)",
    "lead_time":        "Lead Time (d)",
    "rate_volatility":  "Rate Volatility (σ pp)",
    "chk_exposure":     "Chokepoint Exp. (%)",
    "planning_score":   "Planning Score",
    "best_path":        "Best Route",
}

_out_df = df_sorted[list(display_cols.keys())].rename(columns=display_cols).copy()
_out_df["Freight Rate (%)"]      = _out_df["Freight Rate (%)"].round(2)
_out_df["RS Score"]              = _out_df["RS Score"].round(1)
_out_df["Lead Time (d)"]         = _out_df["Lead Time (d)"].round(0)
_out_df["Rate Volatility (σ pp)"]= _out_df["Rate Volatility (σ pp)"].round(2)
_out_df["Chokepoint Exp. (%)"]   = (_out_df["Chokepoint Exp. (%)"] * 100).round(1)
_out_df["Planning Score"]        = _out_df["Planning Score"].round(1)
_out_df = _out_df.set_index("Corridor")

st.dataframe(_out_df, use_container_width=True)

st.download_button(
    "Download Comparison as CSV",
    data=_out_df.to_csv(),
    file_name=f"sonar_comparison_{product_label.replace(' ', '_')}.csv",
    mime="text/csv",
)

# ─── Volatility trend sparklines ─────────────────────────────────────────────
st.markdown("---")
section_header("📈", "Rate Trend Overlay (2016–2021)",
               "Compare how freight costs evolved across selected corridors")

_trend_fig = go.Figure()
_trend_colors = ["#4A90D9", "#27AE60", "#F5A623", "#E74C3C"]
_has_trend = False
for i, (_, row) in enumerate(df_sorted.iterrows()):
    _mask = (
        (edges["origin"]       == row["origin"]) &
        (edges["destination"]  == row["destination"]) &
        (edges["product_code"] == product_code)
    )
    _sub = edges[_mask].sort_values("year")
    if _sub.empty:
        continue
    _has_trend = True
    _trend_fig.add_trace(go.Scatter(
        x=_sub["year"].tolist(),
        y=(_sub["freight_rate"] * 100).tolist(),
        mode="lines+markers",
        name=row["corridor"],
        line=dict(color=_trend_colors[i % len(_trend_colors)], width=2),
        marker=dict(size=7),
    ))

if _has_trend:
    _trend_fig.add_vrect(
        x0=2019.5, x1=2021.5, fillcolor="#F39C12", opacity=0.06,
        line_width=0, annotation_text="COVID era",
        annotation_position="top left",
        annotation_font_color="#F39C12", annotation_font_size=10,
    )
    _trend_fig.update_layout(
        paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        xaxis=dict(title="Year", gridcolor="#21262d",
                   tickvals=list(range(2016, 2022))),
        yaxis=dict(title="Freight Rate (% of cargo value)", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
        height=320,
        margin=dict(t=20),
    )
    st.plotly_chart(_trend_fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No historical rate data found for the selected corridors and product.")

render_footer()
