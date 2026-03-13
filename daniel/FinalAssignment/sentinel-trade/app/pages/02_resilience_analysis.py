"""
Resilience Analysis page.
Shows resilience score heatmap across top trade corridors and component breakdown.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import PRODUCT_NAMES, PRODUCT_CODES, YEARS, CHOKEPOINTS
from src.graph.routing import find_k_routes, apply_scenario
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import make_corridor_heatmap, COLORS

st.set_page_config(page_title="Resilience Analysis · SONAR",
                   layout="wide", page_icon="📊")
st.markdown("""<style>
.main{background:#0e1117} h1,h2,h3,p,label{color:#e6edf3!important}
.stSidebar{background:#161b22}
</style>""", unsafe_allow_html=True)

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first.")
    st.stop()

graphs  = st.session_state.graphs
scorer  = st.session_state.scorer

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Resilience Analysis")
    year = st.selectbox("Year", YEARS, index=YEARS.index(2021))
    selected_products = st.multiselect(
        "Products",
        options=list(PRODUCT_NAMES.values()),
        default=list(PRODUCT_NAMES.values()),
    )
    n_corridors = st.slider("Top N corridors", 10, 50, 20)
    st.markdown("---")
    st.markdown("### Scenario (optional)")
    blocked = []
    for cp_name in CHOKEPOINTS.keys():
        if st.checkbox(cp_name, key=f"rs_cp_{cp_name}"):
            blocked.append(cp_name)
    us_t = st.slider("US Tariff (%)", 0, 50, 0, step=5, key="rs_us")
    eu_t = st.slider("EU Tariff (%)", 0, 50, 0, step=5, key="rs_eu")
    cn_t = st.slider("China Tariff (%)", 0, 50, 0, step=5, key="rs_cn")
    compute_btn = st.button("Compute Heatmap", type="primary", use_container_width=True)

st.markdown("# 📊 Resilience Analysis")
st.caption("Compare route resilience across the top global trade corridors.")

# ─── Compute heatmap ─────────────────────────────────────────────────────────
# Pre-defined top corridors (by seaborne trade importance)
TOP_CORRIDORS = [
    ("China", "United States"),
    ("China", "Germany"),
    ("China", "United Kingdom"),
    ("China", "Japan"),
    ("China", "Republic of Korea"),
    ("China", "Australia"),
    ("China", "India"),
    ("Japan", "United States"),
    ("Republic of Korea", "United States"),
    ("Germany", "United States"),
    ("India", "United States"),
    ("India", "Germany"),
    ("United States", "Germany"),
    ("United States", "Japan"),
    ("Brazil", "China"),
    ("Brazil", "United States"),
    ("Australia", "China"),
    ("Singapore", "United States"),
    ("United Arab Emirates", "India"),
    ("Saudi Arabia", "China"),
][:n_corridors]

selected_codes = [k for k, v in PRODUCT_NAMES.items() if v in selected_products]

@st.cache_data(show_spinner="Computing resilience scores...")
def compute_heatmap_data(
    corridors: list, product_codes: list, year: int,
    blocked_cps: tuple, us_t: int, eu_t: int, cn_t: int
) -> list[dict]:
    tariff_mult = get_tariff_multipliers(float(us_t), float(eu_t), float(cn_t), 0.0)
    rows = []
    for orig, dest in corridors:
        for prod in product_codes:
            key = (year, prod)
            if key not in graphs:
                continue
            G_base = graphs[key]
            G = apply_scenario(G_base, list(blocked_cps), tariff_mult)
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

with st.spinner("Computing resilience scores for selected corridors..."):
    data = compute_heatmap_data(
        TOP_CORRIDORS, selected_codes, year,
        tuple(sorted(blocked)), us_t, eu_t, cn_t
    )

if not data:
    st.warning("No data available for the selected configuration.")
    st.stop()

# ─── Heatmap ─────────────────────────────────────────────────────────────────
heatmap_fig = make_corridor_heatmap(
    data,
    title=f"Resilience Scores — Top {n_corridors} Corridors ({year})"
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# ─── Summary statistics ───────────────────────────────────────────────────────
df = pd.DataFrame(data)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Resilience Score", f"{df['score'].mean():.1f}")
col2.metric("High Resilience Corridors",
            int((df['score'] >= 75).sum()))
col3.metric("Critical Risk Corridors",
            int((df['score'] < 25).sum()))
col4.metric("No-Route Corridors",
            int((df['label'] == 'No Route').sum()))

# ─── Table view ──────────────────────────────────────────────────────────────
st.markdown("### Detailed Scores")
display_df = df.sort_values("score", ascending=False).copy()
display_df["corridor"] = display_df["origin"] + " → " + display_df["destination"]
st.dataframe(
    display_df[["corridor", "product_name", "score", "label"]]
    .rename(columns={
        "corridor":     "Corridor",
        "product_name": "Product",
        "score":        "RS Score",
        "label":        "Rating",
    }).reset_index(drop=True),
    use_container_width=True,
)

# ─── Component analysis for selected corridor ─────────────────────────────────
st.markdown("### Drill Down: Score Components")
chosen_corridor = st.selectbox(
    "Select corridor to analyze",
    [f"{o} → {d}" for o, d in TOP_CORRIDORS]
)
chosen_product = st.selectbox("Product", list(PRODUCT_NAMES.values()), key="dd_prod")
chosen_prod_code = [k for k, v in PRODUCT_NAMES.items() if v == chosen_product][0]

orig_c, dest_c = chosen_corridor.split(" → ", 1)
key = (year, chosen_prod_code)
if key in graphs:
    G_b = graphs[key]
    tariff_mult = get_tariff_multipliers(float(us_t), float(eu_t), float(cn_t), 0.0)
    G_s = apply_scenario(G_b, blocked, tariff_mult)
    try:
        routes = find_k_routes(G_s, orig_c, dest_c, k=3)
        rs = scorer.score_from_routes(routes, G_s)
        comp = rs["components_pct"]

        fig_bar = go.Figure(go.Bar(
            x=list(comp.keys()),
            y=list(comp.values()),
            marker_color=["#4A90D9", "#27AE60", "#F39C12", "#9B59B6"],
            text=[f"{v:.1f}" for v in comp.values()],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_bar.update_layout(
            title=f"RS Components: {chosen_corridor} ({chosen_product})",
            yaxis_title="Contribution (pts, max=35/25/25/15)",
            paper_bgcolor=COLORS["paper"],
            plot_bgcolor=COLORS["paper"],
            font=dict(color="white"),
            yaxis=dict(range=[0, 40], gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(f"**Overall Score: {rs['score']:.1f} / 100** — {rs['label']}")
        st.markdown(f"Best route: `{routes[0].to_dict()['path_str']}`")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        st.warning(f"No route found: {e}")
