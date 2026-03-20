"""
Resilience Score Explainer page.
Explains the 5-factor AHP-weighted resilience model to the user,
with live interactive scoring for any corridor.
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import (
    PRODUCT_NAMES, PRODUCT_CODES, LATEST_YEAR,
    RS_WEIGHT_REL, RS_WEIGHT_FLEX, RS_WEIGHT_ENV,
    RS_WEIGHT_PORT, RS_WEIGHT_SEC,
)
from src.graph.routing import find_k_routes
from src.scoring.resilience import ResilienceScorer, sensitivity_analysis
from src.viz.globe import COLORS
from app.components.theme import inject_global_css, section_header, render_footer

st.set_page_config(
    page_title="Resilience Scoring Explainability · SONAR",
    layout="wide",
    page_icon="🛡",
)
inject_global_css()

if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first.")
    st.stop()

graphs = st.session_state.graphs
scorer = st.session_state.scorer

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px">
    <h1 style="font-size:28px;font-weight:800;color:#e6edf3;margin:0">
        🛡 Resilience Scoring Explainability
    </h1>
    <p style="color:#8B949E;font-size:14px;margin:4px 0 0 0">
        How SONAR measures supply chain resilience — 5 explainable factors,
        grounded in maritime logistics research.
    </p>
</div>
""", unsafe_allow_html=True)

# ── 1. Formula overview ───────────────────────────────────────────────────────
section_header("📐", "The Formula")

st.markdown("""
<div style="background:#161b22;border:1px solid #21262d;border-left:4px solid #9B59B6;
            border-radius:8px;padding:20px 24px;margin:8px 0 16px 0;font-family:monospace">
    <div style="font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:6px">
        RS = 100 × (0.37·Rel + 0.21·Flex + 0.21·Env + 0.11·Port + 0.10·Sec)
    </div>
    <div style="font-size:12px;color:#8B949E">
        Each component ∈ [0, 1] &nbsp;|&nbsp; RS ∈ [0, 100] &nbsp;|&nbsp;
        Weights via Analytic Hierarchy Process (Saaty 1980), CR = 0.003 &lt; 0.10
    </div>
</div>
""", unsafe_allow_html=True)

# Weight bar chart
_factors = ["Delivery Confidence", "Backup Options", "Weather Safety", "Port Health", "Security Level"]
_weights = [RS_WEIGHT_REL, RS_WEIGHT_FLEX, RS_WEIGHT_ENV, RS_WEIGHT_PORT, RS_WEIGHT_SEC]
_colors  = ["#4A90D9", "#27AE60", "#F39C12", "#9B59B6", "#E74C3C"]
_keys    = ["Rel", "Flex", "Env", "Port", "Sec"]

fig_weights = go.Figure(go.Bar(
    x=_factors,
    y=[w * 100 for w in _weights],
    marker_color=_colors,
    text=[f"{w*100:.0f}%" for w in _weights],
    textposition="outside",
    textfont=dict(color="white", size=13),
))
fig_weights.update_layout(
    height=280,
    yaxis=dict(title="Weight (%)", range=[0, 45], gridcolor="#21262d", tickformat=".0f"),
    xaxis=dict(gridcolor="#21262d"),
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["paper"],
    font=dict(color="white"),
    margin=dict(t=20, b=10),
    showlegend=False,
)
st.plotly_chart(fig_weights, use_container_width=True)

# ── 2. Factor cards ───────────────────────────────────────────────────────────
section_header("🔍", "Factor Breakdown", "What each component measures and why it matters")

_factor_details = [
    {
        "key":    "Rel",
        "label":  "Delivery Confidence",
        "weight": RS_WEIGHT_REL,
        "color":  "#4A90D9",
        "icon":   "📦",
        "formula": "0.60 × OTD Rate  +  0.40 × (1 − Mean Delay / 10 days)",
        "source":  "global_supply_chain_disruption_v1.csv — 10K shipments, 6 trade lanes",
        "why": (
            "On-time delivery rate is the gold standard of carrier performance, "
            "typically benchmarked at ≥95% (PDF §4). High delay variability makes it "
            "impossible to give accurate ETAs, damaging customer trust. "
            "Together these measure both the frequency and severity of lateness."
        ),
        "interpretation": "Countries with Suez-route exposure (India, UK) score lower due to geopolitical-conflict delays (83–85% OTD). Atlantic and commodity routes score highest (89–91% OTD).",
    },
    {
        "key":    "Flex",
        "label":  "Backup Options",
        "weight": RS_WEIGHT_FLEX,
        "color":  "#27AE60",
        "icon":   "🔀",
        "formula": "0.60 × (1 − k2 cost premium)  +  0.40 × (1 − chokepoint exposure)",
        "source":  "Graph-derived at query time — Yen's K-shortest paths",
        "why": (
            "Routes that pass through a single chokepoint cannot recover from disruption "
            "(PDF §3). A route with no viable alternative is catastrophically fragile — "
            "the 2021 Ever Given blockage halted $9B/day in trade with zero alternatives. "
            "The cost premium of the 2nd-best route quantifies how 'trapped' a shipper is."
        ),
        "interpretation": "Isolated island routes (Japan, Australia) score low on Flex due to few alternative hub connections. Major hub-to-hub corridors (China–US, China–Germany) score high.",
    },
    {
        "key":    "Env",
        "label":  "Weather Safety",
        "weight": RS_WEIGHT_ENV,
        "color":  "#F39C12",
        "icon":   "🌦",
        "formula": "1 − mean(weather severity) over all countries in path",
        "source":  "country_date_conditions.csv — 129K observations, 211 countries, 669 dates",
        "why": (
            "Meteorological conditions are the most persistent external factor in maritime "
            "resilience (PDF §1, professor's recommendation). Wind, wave height, and "
            "visibility directly affect vessel speed, cargo safety, and berthing delays. "
            "48 weather condition strings are mapped to a [0–1] severity scale per IMO/WMO "
            "hazard classifications."
        ),
        "interpretation": "Tropical routes through Southeast Asia and the Indian Ocean score lower. Northern European and trans-Pacific routes benefit from more moderate seasonal weather profiles.",
    },
    {
        "key":    "Port",
        "label":  "Port Health",
        "weight": RS_WEIGHT_PORT,
        "color":  "#9B59B6",
        "icon":   "🏗",
        "formula": "0.60 × (TEU / 95th-pct TEU)  +  0.40 × (1 − port congestion rate)",
        "source":  "UNCTAD container_port_throughput.csv + disruption dataset congestion rates",
        "why": (
            "Seaports are primary failure nodes — congestion cascades into the entire "
            "land-side network (PDF §2). TEU throughput proxies for infrastructure "
            "capacity and crane efficiency; high-throughput ports absorb demand shocks. "
            "When yard density exceeds 80%, terminal efficiency collapses in a 'vicious cycle' "
            "of congestion (PDF §2)."
        ),
        "interpretation": "China, South Korea, and the US score high (massive TEU throughput). Brazil and the Netherlands score lower due to observed port congestion in the disruption dataset.",
    },
    {
        "key":    "Sec",
        "label":  "Security Level",
        "weight": RS_WEIGHT_SEC,
        "color":  "#E74C3C",
        "icon":   "🔐",
        "formula": "0.50 × (1 − geo conflict rate)  +  0.50 × (1 − Geopolitical Risk Index)",
        "source":  "global_supply_chain_disruption_v1.csv — Geopolitical_Risk_Index + conflict events",
        "why": (
            "Geopolitical risks including conflict, piracy, and terrorism can make routes "
            "financially unviable — insurers spike premiums 300–500% in high-tension areas "
            "like the Red Sea or Strait of Hormuz (PDF §5). A 1% increase in geopolitical "
            "distance between nations leads to a 10% decrease in trade efficiency."
        ),
        "interpretation": "Routes transiting the Suez Canal (India–UK, Shenzhen–Rotterdam) show the highest geopolitical conflict rates (14–16%). Intra-Pacific routes are significantly safer.",
    },
]

for det in _factor_details:
    col_badge, col_body = st.columns([1, 5])
    with col_badge:
        st.markdown(
            f'<div style="background:{det["color"]}22;border:1px solid {det["color"]}55;'
            f'border-radius:10px;padding:16px 12px;text-align:center;margin-top:8px">'
            f'<div style="font-size:28px">{det["icon"]}</div>'
            f'<div style="font-size:18px;font-weight:800;color:{det["color"]};margin:4px 0 2px 0">{det["key"]}</div>'
            f'<div style="font-size:11px;color:#8B949E;font-weight:600">{det["weight"]*100:.0f}% weight</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_body:
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;'
            f'padding:16px 20px;margin-top:8px">'
            f'<div style="font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:8px">'
            f'{det["label"]}</div>'
            f'<div style="background:#0d1117;border-radius:6px;padding:8px 12px;margin-bottom:10px;'
            f'font-family:monospace;font-size:12px;color:{det["color"]}">'
            f'{det["formula"]}</div>'
            f'<div style="font-size:12px;color:#c9d1d9;line-height:1.7;margin-bottom:8px">'
            f'{det["why"]}</div>'
            f'<div style="font-size:11px;color:#8B949E;border-top:1px solid #21262d;padding-top:8px">'
            f'<b style="color:#6e7681">Source:</b> {det["source"]}<br>'
            f'<b style="color:#6e7681">In practice:</b> {det["interpretation"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

# ── 3. AHP weighting rationale ────────────────────────────────────────────────
section_header("⚖️", "AHP Pairwise Comparison Matrix", "Consistency Ratio = 0.003 (threshold: < 0.10)")

ahp_labels = ["Rel", "Flex", "Env", "Port", "Sec"]
ahp_matrix = [
    [1,     2,    2,    3,    3],
    [1/2,   1,    1,    2,    2],
    [1/2,   1,    1,    2,    2],
    [1/3,   1/2,  1/2,  1,    1],
    [1/3,   1/2,  1/2,  1,    1],
]
ahp_df = pd.DataFrame(ahp_matrix, index=ahp_labels, columns=ahp_labels)

col_ahp, col_rationale = st.columns([2, 3])

with col_ahp:
    # Format fractions nicely
    def _fmt(v):
        if v == 1/2: return "1/2"
        if v == 1/3: return "1/3"
        return str(int(v))

    display_matrix = [[_fmt(v) for v in row] for row in ahp_matrix]
    display_df = pd.DataFrame(display_matrix, index=ahp_labels, columns=ahp_labels)

    st.dataframe(
        display_df.style.set_properties(**{
            "text-align": "center",
            "font-weight": "600",
        }),
        use_container_width=True,
    )
    st.caption("Values > 1 mean the row factor dominates the column factor.")

with col_rationale:
    st.markdown("""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:16px 20px;font-size:13px;line-height:1.75;color:#c9d1d9">
        <b style="color:#e6edf3">Weighting rationale</b><br><br>
        Multi-criteria decision-making research in transportation logistics consistently
        identifies <b style="color:#4A90D9">delivery reliability</b> as the primary
        resilience driver, often accounting for 35–45% of composite scores in AHP-based
        freight models. <b>Rel</b> is therefore rated dominant (2×) over all other factors.<br><br>
        <b style="color:#27AE60">Route flexibility</b> and
        <b style="color:#F39C12">environmental stability</b> receive equal weight as
        co-primary structural and environmental resilience dimensions. Route flexibility
        captures the network's ability to absorb disruptions through alternative paths,
        while weather stability directly affects vessel kinematics, port berthing, and
        cargo integrity — both are operationally irreplaceable.<br><br>
        <b style="color:#9B59B6">Port capacity</b> and
        <b style="color:#E74C3C">geopolitical security</b> are treated as supporting
        factors: significant but more amenable to mitigation through advance planning,
        carrier selection, and inventory buffering than structural path limitations.
    </div>
    """, unsafe_allow_html=True)

# ── 4. Scoring labels ─────────────────────────────────────────────────────────
section_header("🏷", "Score Interpretation")

_labels = [
    ("75 – 100", "High Resilience",     "#27AE60", "Route is well-protected against disruptions. Multiple alternatives exist, low chokepoint exposure, strong carrier reliability, and stable conditions."),
    ("50 – 74",  "Moderate Resilience", "#F39C12", "Acceptable resilience but with identifiable vulnerabilities. Monitor actively and maintain contingency plans for disruption scenarios."),
    ("25 – 49",  "Low Resilience",      "#E74C3C", "Significant exposure to disruptions. Route likely passes through high-risk chokepoints or has few backup options. Consider diversification."),
    ("0 – 24",   "Critical Risk",       "#8E44AD", "Route is highly fragile. Any major disruption event will cause severe delays. Immediate action required — alternate routing strongly recommended."),
]

cols = st.columns(4)
for col, (rng, lbl, color, desc) in zip(cols, _labels):
    with col:
        st.markdown(
            f'<div style="background:{color}15;border:1px solid {color}44;border-top:3px solid {color};'
            f'border-radius:8px;padding:16px;height:160px;box-sizing:border-box">'
            f'<div style="font-size:13px;font-weight:800;color:{color};margin-bottom:2px">{lbl}</div>'
            f'<div style="font-size:20px;font-weight:800;color:{color};margin-bottom:10px">{rng}</div>'
            f'<div style="font-size:11px;color:#8B949E;line-height:1.55">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── 5. Live corridor scorer ───────────────────────────────────────────────────
st.markdown("---")
section_header("⚡", "Live Score Explorer", "See how any corridor scores on each factor")

all_countries = sorted(set(
    n for G in graphs.values() for n in G.nodes()
))

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    origin = st.selectbox("Origin country", all_countries,
                          index=all_countries.index("China") if "China" in all_countries else 0)
with c2:
    dest_list = [c for c in all_countries if c != origin]
    destination = st.selectbox("Destination country", dest_list,
                               index=dest_list.index("Germany") if "Germany" in dest_list else 0)
with c3:
    product_name = st.selectbox("Product", list(PRODUCT_NAMES.values()))
    prod_code    = [k for k, v in PRODUCT_NAMES.items() if v == product_name][0]

gkey = (LATEST_YEAR, prod_code)

if gkey in graphs:
    G = graphs[gkey]
    try:
        routes = find_k_routes(G, origin, destination, k=3)
        if not routes:
            st.warning(f"No shipping route found between **{origin}** and **{destination}**.")
        else:
            rs   = scorer.score_from_routes(routes, G)
            comp = rs.get("components_pct", {})

            score_val   = rs["score"]
            badge_color = (
                "#27AE60" if score_val >= 75 else
                "#F39C12" if score_val >= 50 else
                "#E74C3C" if score_val >= 25 else "#8E44AD"
            )

            col_gauge, col_bars = st.columns([1, 2])

            with col_gauge:
                st.markdown(
                    f'<div style="background:{badge_color}15;border:1px solid {badge_color}44;'
                    f'border-radius:12px;padding:28px 20px;text-align:center;margin-top:8px">'
                    f'<div style="font-size:56px;font-weight:900;color:{badge_color};line-height:1">'
                    f'{score_val:.1f}</div>'
                    f'<div style="font-size:13px;color:#8B949E;margin:4px 0">/ 100</div>'
                    f'<div style="font-size:15px;font-weight:700;color:{badge_color};margin:8px 0 12px">'
                    f'{rs["label"]}</div>'
                    f'<div style="font-size:11px;color:#8B949E">'
                    f'Best path: {" → ".join(routes[0].path)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_bars:
                fig_comp = go.Figure(go.Bar(
                    x=list(comp.keys()),
                    y=list(comp.values()),
                    marker_color=_colors,
                    text=[f"{v:.1f} pts" for v in comp.values()],
                    textposition="outside",
                    textfont=dict(color="white", size=12),
                ))
                _max_weights = [RS_WEIGHT_REL*100, RS_WEIGHT_FLEX*100,
                                RS_WEIGHT_ENV*100,  RS_WEIGHT_PORT*100, RS_WEIGHT_SEC*100]
                fig_comp.update_layout(
                    height=300,
                    yaxis=dict(
                        title="Contribution (pts)",
                        range=[0, max(_max_weights) + 6],
                        gridcolor="#21262d",
                    ),
                    xaxis=dict(gridcolor="#21262d"),
                    paper_bgcolor=COLORS["paper"],
                    plot_bgcolor=COLORS["paper"],
                    font=dict(color="white"),
                    margin=dict(t=20, b=10),
                    showlegend=False,
                )
                # Add max-possible markers
                for i, max_w in enumerate(_max_weights):
                    fig_comp.add_shape(
                        type="line",
                        x0=i - 0.4, x1=i + 0.4, y0=max_w, y1=max_w,
                        line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1.5),
                    )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.caption("Dashed lines show the maximum possible contribution for each factor.")

            # Per-factor breakdown table
            raw_vals = {
                "Delivery Confidence": (rs["rel"],  f"OTD: {rs['rel']:.0%}", "#4A90D9"),
                "Backup Options":      (rs["flex"], f"Alt+Chk combined", "#27AE60"),
                "Weather Safety":      (rs["env"],  f"1 − avg severity", "#F39C12"),
                "Port Health":         (rs["port"], f"TEU + congestion", "#9B59B6"),
                "Security Level":      (rs["sec"],  f"1 − conflict/GRI", "#E74C3C"),
            }
            weights_map = dict(zip(
                ["Delivery Confidence", "Backup Options", "Weather Safety", "Port Health", "Security Level"],
                _weights
            ))

            rows_detail = []
            for name, (raw, desc, color) in raw_vals.items():
                rows_detail.append({
                    "Factor":      name,
                    "Raw [0–1]":   f"{raw:.3f}",
                    "Max Weight":  f"{weights_map[name]*100:.0f}%",
                    "Contribution": f"{raw * weights_map[name] * 100:.1f} pts",
                    "Status":      "✅ Good" if raw >= 0.80 else "⚠️ Fair" if raw >= 0.60 else "🔴 Weak",
                })
            st.dataframe(pd.DataFrame(rows_detail).set_index("Factor"), use_container_width=True)

            # ── Sensitivity analysis ──────────────────────────────────────────────
            with st.expander("Sensitivity Analysis — how stable is this score?"):
                sa = sensitivity_analysis(
                    scorer,
                    routes[0].path, routes[0].cost, G,
                    path_k2=routes[1].path if len(routes) >= 2 else None,
                    cost_k2=routes[1].cost if len(routes) >= 2 else None,
                    delta=0.10,
                )
                sa_rows = []
                for comp_key, perturbs in sa["perturbations"].items():
                    plus_score  = perturbs.get("plus", score_val)
                    minus_score = perturbs.get("minus", score_val)
                    sa_rows.append({
                        "Factor":     comp_key.upper(),
                        "+10% weight": f"{plus_score:.1f}",
                        "−10% weight": f"{minus_score:.1f}",
                        "Δ range":    f"±{(abs(plus_score - score_val) + abs(minus_score - score_val)) / 2:.1f}",
                    })
                st.dataframe(pd.DataFrame(sa_rows).set_index("Factor"), use_container_width=True)
                st.caption(
                    f"Base score: **{score_val:.1f}**. Perturbing each weight ±10% while "
                    "rescaling the others to maintain a sum of 1.0."
                )

    except (nx.NetworkXNoPath, nx.NodeNotFound, IndexError, KeyError, ValueError) as e:
        st.warning(f"Could not compute score for {origin} → {destination}: {e}")
else:
    st.warning("Graph not available for this product.")

# ── 6. Data sources ───────────────────────────────────────────────────────────
st.markdown("---")
section_header("📂", "Data Sources")

_sources = [
    ("global_supply_chain_disruption_v1.csv",
     "10,000 shipment records across 6 major trade lanes. Provides per-country OTD rates, "
     "mean delay, port congestion rates, and geopolitical conflict rates used in Rel, Port, and Sec.",
     "Rel · Port · Sec"),
    ("country_date_conditions.csv",
     "129,000+ daily weather observations for 211 countries over 669 dates. Each condition "
     "string (e.g. 'Heavy rain', 'Blizzard') is mapped to a [0–1] severity score to compute "
     "per-country mean weather risk.",
     "Env"),
    ("container_port_throughput.csv",
     "UNCTAD container port TEU throughput by country, 2016–2021. Used as a proxy for port "
     "infrastructure capacity — normalised to the 95th percentile globally.",
     "Port"),
    ("Graph structure (Yen's K-shortest paths)",
     "Route alternatives and chokepoint exposure are computed live from the trade graph at "
     "query time. The cost premium of the 2nd-best route quantifies how 'trapped' a shipper "
     "is on a given corridor.",
     "Flex"),
]

for fname, desc, factors in _sources:
    st.markdown(
        f'<div style="display:flex;gap:16px;background:#161b22;border:1px solid #21262d;'
        f'border-radius:8px;padding:14px 18px;margin-bottom:8px;align-items:flex-start">'
        f'<div style="min-width:90px">'
        f'<span style="background:#9B59B622;border:1px solid #9B59B655;color:#9B59B6;'
        f'font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px">{factors}</span>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:12px;font-weight:700;color:#e6edf3;font-family:monospace;'
        f'margin-bottom:4px">{fname}</div>'
        f'<div style="font-size:12px;color:#8B949E;line-height:1.6">{desc}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

render_footer()
