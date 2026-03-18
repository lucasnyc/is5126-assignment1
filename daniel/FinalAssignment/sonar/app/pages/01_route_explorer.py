"""
Route Explorer — multi-criteria route comparison with stakeholder persona wizard.
Finds the best route from A → B optimised for three distinct criteria:
  • Most Resilient  (highest Resilience Score)
  • Cheapest        (lowest freight cost)
  • Fastest         (lowest estimated lead time)

The "Help Me Choose" wizard asks 4 business-context questions and recommends
the route that best fits the stakeholder's profile, including a profit-margin
viability check (freight cost % vs declared margin %).
"""

import os
import re
import sys

import streamlit as st
import pandas as pd
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from config import CHOKEPOINTS, CHOKEPOINT_WAYPOINTS, PRODUCT_NAMES, LATEST_YEAR
from src.graph.routing import find_multi_criteria_routes, apply_scenario
from src.graph.chokepoints import get_tariff_multipliers
from src.viz.globe import make_multi_criteria_globe, make_resilience_gauge, make_route_radar, CRITERIA_COLORS
from app.components.theme import inject_global_css, section_header, render_footer, wizard_step_indicator

st.set_page_config(page_title="Route Explorer · SONAR", layout="wide", page_icon="🗺")

inject_global_css()

# ── Guard: session state ───────────────────────────────────────────────────────
if "graphs" not in st.session_state:
    st.warning("Please visit the Home page first to initialise the app.")
    st.stop()

graphs = st.session_state.graphs
scorer = st.session_state.scorer

sample_graph  = graphs[(2021, 8517)]
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
    product_code  = [k for k, v in PRODUCT_NAMES.items() if v == product_label][0]
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
if st.session_state.get("explorer_mode") is not None:
    if st.button("← Change mode", key="back_to_mode_top"):
        st.session_state.explorer_mode  = None
        st.session_state.wiz_step       = -1
        st.session_state.wiz_done       = False
        st.session_state.turnstile_idx  = 0
        st.rerun()

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

G_base       = graphs[key]
has_scenario = bool(blocked) or any([us_tariff, eu_tariff, china_tariff, asean_tariff])

tariff_multipliers = get_tariff_multipliers(
    us_pct=float(us_tariff),
    eu_pct=float(eu_tariff),
    china_pct=float(china_tariff),
    asean_pct=float(asean_tariff),
)

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

# ── Globe ─────────────────────────────────────────────────────────────────────
_globe_cache = st.session_state.setdefault("_re_globe_cache", {})
if _cache_key not in _globe_cache:
    criteria_dicts = {k: r.to_dict() for k, r in routes.items()}
    _globe_cache[_cache_key] = (
        criteria_dicts,
        make_multi_criteria_globe(criteria_routes=criteria_dicts, blocked_chokepoints=blocked),
    )
criteria_dicts, globe_fig = _globe_cache[_cache_key]

# (Globe is rendered inside each mode section so it can be filtered per-mode)

# ══════════════════════════════════════════════════════════════════════════════
# STAKEHOLDER PERSONA WIZARD
# ══════════════════════════════════════════════════════════════════════════════

# ── Wizard session state ───────────────────────────────────────────────────────
st.session_state.setdefault("wiz_step",      -1)   # -1 = not started
st.session_state.setdefault("wiz_answers",   {})
st.session_state.setdefault("wiz_done",      False)
st.session_state.setdefault("turnstile_idx", 0)    # 0 = recommended route

# ── Wizard question definitions ────────────────────────────────────────────────
WIZARD_QUESTIONS = [
    {
        "id":      "margin",
        "title":   "What is your expected gross profit margin on this shipment?",
        "caption": (
            "Freight cost is expressed as a % of cargo value. "
            "Any route whose freight cost % exceeds your margin makes this shipment unprofitable."
        ),
        "type":    "slider",
        "min": 1, "max": 100, "default": 20,
    },
    {
        "id":      "deadline",
        "title":   "Do you have a delivery deadline?",
        "caption": "Routes that cannot meet your deadline will be flagged regardless of cost or resilience.",
        "type":    "deadline",
    },
    {
        "id":      "continuity",
        "title":   "How critical is uninterrupted supply for this product?",
        "caption": (
            "High = just-in-time: a disruption halts your operations. "
            "Low = you carry buffer stock and can absorb delays."
        ),
        "type":    "select",
        "options": ["Low — I hold buffer stock", "Medium", "High — just-in-time"],
        "default": "Medium",
    },
    {
        "id":      "cost_sensitivity",
        "title":   "How sensitive is your business to unexpected freight cost increases?",
        "caption": "This helps us balance cost efficiency against supply chain resilience in our recommendation.",
        "type":    "select",
        "options": ["Can absorb increases", "Moderate sensitivity", "Very sensitive"],
        "default": "Moderate sensitivity",
    },
]

N_QUESTIONS = len(WIZARD_QUESTIONS)


# ── Persona computation ────────────────────────────────────────────────────────
def _compute_persona(answers: dict, routes: dict) -> dict:
    margin        = answers.get("margin", 20)
    deadline_days = answers.get("deadline_days")          # int or None
    cont_raw      = answers.get("continuity",       "Medium")
    cost_raw      = answers.get("cost_sensitivity", "Moderate sensitivity")

    cont_key = (
        "High"   if "High"   in cont_raw else
        "Low"    if "Low"    in cont_raw else "Medium"
    )
    cost_key = (
        "very"   if "Very"   in cost_raw else
        "absorb" if "absorb" in cost_raw else "moderate"
    )

    # Economic viability: freight_cost% vs declared margin%
    viability = {
        rk: {
            "viable":      (r.cost * 100) <= margin,
            "freight_pct": r.cost * 100,
        }
        for rk, r in routes.items()
    }

    # Deadline compliance
    deadline_ok = {
        rk: (deadline_days is None) or (r.lead_time_days <= deadline_days)
        for rk, r in routes.items()
    }

    # Persona + initial recommendation from priority matrix
    _persona_map = {
        ("High",   "absorb"):   ("Risk-Averse Supply Manager",  "most_resilient",
                                 "Supply continuity is critical — resilience is paramount."),
        ("High",   "moderate"): ("Risk-Averse Supply Manager",  "most_resilient",
                                 "Supply continuity is critical — resilience is paramount."),
        ("High",   "very"):     ("Cautious Cost-Watcher",       "most_resilient",
                                 "Resilience is non-negotiable, but you monitor costs closely."),
        ("Medium", "absorb"):   ("Balanced Supply Manager",     "most_resilient",
                                 "You want reliability without overpaying — resilience wins."),
        ("Medium", "moderate"): ("Pragmatic Trader",            "cheapest",
                                 "A balanced profile — cost efficiency with reasonable risk tolerance."),
        ("Medium", "very"):     ("Lean Trader",                 "cheapest",
                                 "Cost control is key; you'll manage moderate disruptions."),
        ("Low",    "absorb"):   ("Speed-Focused Distributor",   "fastest",
                                 "Buffer stock reduces your risk — speed drives your competitive edge."),
        ("Low",    "moderate"): ("Cost-Conscious Shipper",      "cheapest",
                                 "Low supply risk and cost pressure make the cheapest route ideal."),
        ("Low",    "very"):     ("Cost-Cutter",                 "cheapest",
                                 "Minimal disruption risk and strong cost pressure — cheapest route fits."),
    }

    persona_name, rec_key, story = _persona_map.get(
        (cont_key, cost_key),
        ("Balanced Supply Manager", "most_resilient",
         "A balanced profile suggests the most resilient route."),
    )

    warnings = []

    # Override: recommended route is economically unviable
    if not viability[rec_key]["viable"]:
        fp = viability[rec_key]["freight_pct"]
        warnings.append(
            f"The **{rec_key.replace('_', ' ')}** route's freight cost ({fp:.1f}%) "
            f"exceeds your profit margin ({margin}%) — this shipment would not be profitable."
        )
        fallback_order = [k for k in ["most_resilient", "cheapest", "fastest"] if k != rec_key]
        for alt in fallback_order:
            if viability[alt]["viable"] and deadline_ok[alt]:
                rec_key = alt
                warnings.append(
                    f"Switching recommendation to **{alt.replace('_', ' ')}** "
                    f"— the best route within your margin."
                )
                break
        else:
            warnings.append(
                "All available routes exceed your profit margin. "
                "Consider renegotiating freight rates or adjusting your product pricing."
            )

    # Override: recommended route misses deadline
    elif deadline_days and not deadline_ok[rec_key]:
        lt = routes[rec_key].lead_time_days
        warnings.append(
            f"The recommended route takes {lt:.0f} days, "
            f"exceeding your {deadline_days}-day deadline."
        )
        for alt in ["fastest", "cheapest", "most_resilient"]:
            if alt != rec_key and deadline_ok[alt] and viability[alt]["viable"]:
                rec_key = alt
                warnings.append(
                    f"Switching recommendation to **{alt.replace('_', ' ')}** "
                    f"— within your deadline and margin."
                )
                break

    # Informational warnings for the non-recommended routes
    for rk in routes:
        if rk == rec_key:
            continue
        if not viability[rk]["viable"]:
            fp = viability[rk]["freight_pct"]
            warnings.append(
                f"Note: the **{rk.replace('_', ' ')}** route ({fp:.1f}%) "
                f"also exceeds your {margin}% margin."
            )

    return {
        "persona_name": persona_name,
        "story":        story,
        "rec_key":      rec_key,
        "viability":    viability,
        "deadline_ok":  deadline_ok,
        "warnings":     warnings,
        "margin":       margin,
        "deadline_days": deadline_days,
    }


# ── Wizard question renderer ───────────────────────────────────────────────────
def _render_wizard_question(step: int, answers: dict) -> None:
    q = WIZARD_QUESTIONS[step]

    # Step indicator (numbered circles)
    wizard_step_indicator(step, N_QUESTIONS)

    # Animated question header
    st.markdown(
        f"""<div class="wiz-wrap">
          <div style="font-size:17px;font-weight:600;color:#e6edf3;margin-bottom:5px">
            {q["title"]}
          </div>
          <div style="font-size:12px;color:#8B949E;margin-bottom:18px">
            {q["caption"]}
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Input widget
    if q["type"] == "slider":
        val = st.slider(
            "margin_slider",
            q["min"], q["max"],
            answers.get(q["id"], q["default"]),
            key=f"wiz_input_{step}",
            label_visibility="collapsed",
            format="%d%%",
        )
        st.caption(f"Selected: **{val}%** gross margin")

    elif q["type"] == "deadline":
        deadline_options = ["No deadline", "Flexible (±2 weeks)", "Firm deadline"]
        saved_opt  = answers.get("deadline_option", "No deadline")
        opt = st.radio(
            "deadline_radio",
            deadline_options,
            index=deadline_options.index(saved_opt),
            key=f"wiz_input_{step}",
            horizontal=True,
            label_visibility="collapsed",
        )
        days_val = None
        if opt == "Firm deadline":
            days_val = st.number_input(
                "Maximum lead time (days)",
                min_value=1, max_value=365,
                value=int(answers.get("deadline_days") or 45),
                key=f"wiz_days_{step}",
            )
        val = {"option": opt, "days": days_val}

    elif q["type"] == "select":
        default_idx = q["options"].index(answers.get(q["id"], q["default"]))
        val = st.radio(
            q["id"],
            q["options"],
            index=default_idx,
            key=f"wiz_input_{step}",
            label_visibility="collapsed",
        )

    # Navigation buttons
    st.write("")
    col_back, col_space, col_next = st.columns([1, 4, 1])
    with col_back:
        if step > 0 and st.button("← Back", key=f"wiz_back_{step}", use_container_width=True):
            st.session_state.wiz_step -= 1
            st.rerun()
    with col_next:
        btn_label = "Finish ✓" if step == N_QUESTIONS - 1 else "Next →"
        if st.button(btn_label, key=f"wiz_next_{step}", type="primary", use_container_width=True):
            # Persist answer
            if q["type"] == "deadline":
                st.session_state.wiz_answers["deadline_option"] = val["option"]
                st.session_state.wiz_answers["deadline_days"]   = val["days"]
            else:
                st.session_state.wiz_answers[q["id"]] = val
            # Advance or finish
            if step < N_QUESTIONS - 1:
                st.session_state.wiz_step += 1
            else:
                st.session_state.wiz_done = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_LABELS = {"most_resilient": "Most Resilient", "cheapest": "Cheapest", "fastest": "Fastest"}
_ICONS  = {"most_resilient": "🛡", "cheapest": "💰", "fastest": "⚡"}
ALL_CARD_CONFIG = [
    ("most_resilient", "Most Resilient", "🛡", CRITERIA_COLORS["most_resilient"], "Resilience Score"),
    ("cheapest",       "Cheapest",       "💰", CRITERIA_COLORS["cheapest"],       "Freight Cost"),
    ("fastest",        "Fastest",        "⚡", CRITERIA_COLORS["fastest"],        "Lead Time"),
]


def _rs_color(rs: float) -> str:
    return "#27AE60" if rs >= 75 else "#F39C12" if rs >= 50 else "#E74C3C" if rs >= 25 else "#8E44AD"


def _route_card_html(crit_key: str, crit_label: str, icon: str, color: str,
                     highlight_label: str, r, is_rec: bool = False) -> str:
    """Build route card HTML as a flat joined list — no embedded newlines that could
    terminate Streamlit's markdown HTML block parser prematurely."""
    rs = r.rs
    rsc = _rs_color(rs)
    if crit_key == "most_resilient":
        hval = f"{rs:.1f} / 100"
    elif crit_key == "cheapest":
        hval = f"{r.cost * 100:.2f}%"
    else:
        hval = f"{r.lead_time_days:.0f} days"
    border = f"border-top:3px solid {color}" + (f";box-shadow:0 0 16px {color}55" if is_rec else "")
    rec_badge = (
        f'<div style="background:{color}22;border:1px solid {color};border-radius:20px;'
        f'padding:2px 10px;display:inline-block;font-size:11px;font-weight:700;'
        f'color:{color};margin-bottom:8px">&#9733; Recommended</div>'
        if is_rec else ""
    )
    return "".join([
        f'<div class="route-card" style="{border}">',
        f'<h4>{icon} {crit_label}</h4>',
        rec_badge,
        f'<div class="metric-label">{highlight_label}</div>',
        f'<div class="metric-big" style="color:{color}">{hval}</div>',
        '<div style="margin-top:0.75rem;display:flex;gap:1.2rem;flex-wrap:wrap">',
        '<div>',
        '<div style="font-size:0.65rem;color:#aaa;text-transform:uppercase;letter-spacing:.05em">RS Score</div>',
        f'<div style="font-size:1rem;font-weight:600;color:{rsc}">{rs:.1f}'
        f'<span style="font-size:0.7rem;color:#aaa"> / 100</span></div>',
        '</div><div>',
        '<div style="font-size:0.65rem;color:#aaa;text-transform:uppercase;letter-spacing:.05em">Freight Cost</div>',
        f'<div style="font-size:1rem;font-weight:600;color:#ccc">{r.cost * 100:.2f}%</div>',
        '</div><div>',
        '<div style="font-size:0.65rem;color:#aaa;text-transform:uppercase;letter-spacing:.05em">Lead Time</div>',
        f'<div style="font-size:1rem;font-weight:600;color:#ccc">{r.lead_time_days:.0f}'
        f'<span style="font-size:0.7rem;color:#aaa"> d</span></div>',
        '</div></div></div>',
    ])


def _render_route_details(crit_key: str, crit_label: str, r, rd: dict,
                          persona_result: dict | None = None) -> None:
    """Render path + metrics table + gauge + RS breakdown (call inside the target column/container)."""
    st.markdown(f"**Path** ({r.hops} hop{'s' if r.hops != 1 else ''})")
    st.markdown(" → ".join(f"`{c}`" for c in r.path))
    rows = [
        {"Metric": "Freight Cost",    "Value": f"{r.cost * 100:.2f}%"},
        {"Metric": "Lead Time",       "Value": f"{r.lead_time_days:.0f} d"},
        {"Metric": "Hops",            "Value": str(r.hops)},
        {"Metric": "Chokepoint Exp.", "Value": f"{rd['chk_exposure']:.0%}"},
        {"Metric": "RS Score",        "Value": f"{r.rs:.1f} / 100"},
        {"Metric": "ML Predicted",    "Value": "Yes ⚠" if rd["has_predicted"] else "No ✓"},
    ]
    if persona_result:
        fp = persona_result["viability"][crit_key]["freight_pct"]
        viable = persona_result["viability"][crit_key]["viable"]
        rows.append({
            "Metric": "vs Your Margin",
            "Value":  f"{fp:.1f}% vs {persona_result['margin']}% → "
                      + ("✓ Viable" if viable else "✗ Unviable"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
    st.plotly_chart(
        make_resilience_gauge(r.rs, crit_label),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    if hasattr(r, "rs_detail") and r.rs_detail:
        with st.expander("Resilience breakdown"):
            comp = r.rs_detail.get("components_pct", {})
            st.dataframe(
                pd.DataFrame([{"Component": k, "Contribution (pts)": f"{v:.1f}"}
                              for k, v in comp.items()]).set_index("Component"),
                use_container_width=True,
                )


def _render_cost_of_certainty(routes: dict, base_key: str) -> None:
    """
    Answer the key question: 'How much more do I pay for more certainty?'
    Shows the cost/benefit of switching from the base route to each alternative.
    """
    base = routes[base_key]
    others = [k for k in ["most_resilient", "cheapest", "fastest"] if k != base_key]

    cols = st.columns(len(others))
    for col, alt_key in zip(cols, others):
        alt   = routes[alt_key]
        d_cost = (alt.cost - base.cost) * 100          # freight % delta
        d_lt   = alt.lead_time_days - base.lead_time_days
        d_rs   = alt.rs - base.rs

        cost_color = "#E74C3C" if d_cost > 0.1 else "#27AE60" if d_cost < -0.1 else "#8B949E"
        lt_color   = "#E74C3C" if d_lt   > 0   else "#27AE60" if d_lt   < 0   else "#8B949E"
        rs_color   = "#27AE60" if d_rs   > 0   else "#E74C3C" if d_rs   < 0   else "#8B949E"

        cost_str = (f"+{d_cost:.1f}%" if d_cost > 0.05 else f"{d_cost:.1f}%") if abs(d_cost) > 0.05 else "same cost"
        lt_str   = (f"+{d_lt:.0f}d"   if d_lt   > 0    else f"{d_lt:.0f}d")   if abs(d_lt)   > 0    else "same time"
        rs_str   = (f"+{d_rs:.1f}"    if d_rs   > 0    else f"{d_rs:.1f}")    if abs(d_rs)   > 0.05 else "same RS"

        # Plain-English verdict
        if d_cost > 0.1 and d_rs > 0:
            verdict = (f"Pay **{cost_str}** more in freight and wait **{lt_str}** longer "
                       f"to gain **{rs_str} resilience points**.")
        elif d_cost < -0.1 and d_rs < 0:
            verdict = (f"Save **{abs(d_cost):.1f}%** in freight but accept "
                       f"**{abs(d_rs):.1f} fewer resilience points**.")
        elif abs(d_lt) > 0 and abs(d_cost) <= 0.1:
            verdict = f"Arrive **{abs(d_lt):.0f} days {'faster' if d_lt < 0 else 'later'}** at roughly the same cost."
        elif d_cost < -0.1:
            verdict = f"Save **{abs(d_cost):.1f}%** in freight — but check lead time and resilience."
        else:
            verdict = f"Freight: **{cost_str}** · Lead time: **{lt_str}** · Resilience: **{rs_str}**"

        with col:
            with st.container(border=True):
                color = CRITERIA_COLORS[alt_key]
                st.markdown(
                    f'<div style="font-size:13px;font-weight:700;color:{color};margin-bottom:6px">'
                    f'{_ICONS[alt_key]} Switch to {_LABELS[alt_key]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(verdict)
                delta_html = "".join([
                    '<div style="display:flex;gap:20px;margin-top:10px;flex-wrap:wrap">',
                    f'<div><div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.05em">Freight Δ</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{cost_color}">{cost_str}</div></div>',
                    f'<div><div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.05em">Lead Time Δ</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{lt_color}">{lt_str}</div></div>',
                    f'<div><div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.05em">Resilience Δ</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{rs_color}">{rs_str}</div></div>',
                    '</div>',
                ])
                st.markdown(delta_html, unsafe_allow_html=True)


# ── Chatbot recommendation text ────────────────────────────────────────────────
def _chatbot_message(answers: dict, persona_result: dict, routes: dict) -> str:
    rec_key   = persona_result["rec_key"]
    rec_route = routes[rec_key]
    rec_label = _LABELS[rec_key]
    margin        = persona_result["margin"]
    deadline_days = persona_result["deadline_days"]
    cont_raw  = answers.get("continuity",       "Medium")
    cost_raw  = answers.get("cost_sensitivity", "Moderate sensitivity")

    ctx = [f"a **{margin}%** profit margin"]
    if deadline_days:
        ctx.append(f"a **{deadline_days}-day** delivery deadline")
    if "High" in cont_raw:
        ctx.append("just-in-time supply requirements")
    elif "Low" in cont_raw:
        ctx.append("buffer stock on hand")
    else:
        ctx.append("moderate supply continuity needs")
    if "Very" in cost_raw:
        ctx.append("high sensitivity to cost increases")
    elif "absorb" in cost_raw:
        ctx.append("flexibility on cost")
    else:
        ctx.append("moderate cost sensitivity")

    path_str    = " → ".join(rec_route.path)
    freight     = rec_route.cost * 100
    lt          = int(rec_route.lead_time_days)
    rs          = rec_route.rs
    margin_note = "well within" if freight < margin * 0.75 else "within"

    return (
        f"Based on your requirements — {', '.join(ctx)} — "
        f"I recommend the **{rec_label}** route: **{path_str}**. "
        f"At **{freight:.1f}%** freight cost it's {margin_note} your {margin}% margin, "
        f"with a resilience score of **{rs:.0f}/100** and an estimated **{lt}-day** lead time."
    )


# ── Turnstile explanation ──────────────────────────────────────────────────────
def _why_this_route(t_key: str, t_idx: int, rec_key: str,
                    persona_result: dict, routes: dict) -> str:
    """Return a plain-English explanation of why this turnstile position is worth considering."""
    if t_idx == 0:
        return f"**Recommended for you:** {persona_result['story']}"
    rec_r  = routes[rec_key]
    curr_r = routes[t_key]
    d_cost = (curr_r.cost - rec_r.cost) * 100
    d_lt   = curr_r.lead_time_days - rec_r.lead_time_days
    d_rs   = curr_r.rs - rec_r.rs
    why_label = {
        "most_resilient": "highest resilience",
        "cheapest":       "lowest freight cost",
        "fastest":        "fastest delivery",
    }[t_key]
    parts = []
    if d_rs > 0.5:
        parts.append(f"**+{d_rs:.1f} resilience points** stronger")
    elif d_rs < -0.5:
        parts.append(f"**{abs(d_rs):.1f} fewer resilience points**")
    if d_cost < -0.05:
        parts.append(f"saves **{abs(d_cost):.1f}%** in freight")
    elif d_cost > 0.05:
        parts.append(f"costs **+{d_cost:.1f}%** more in freight")
    if d_lt < 0:
        parts.append(f"arrives **{abs(d_lt):.0f} days faster**")
    elif d_lt > 0:
        parts.append(f"takes **{d_lt:.0f} more days**")
    vs_str = " and ".join(parts) if parts else "offers similar overall performance"
    return (
        f"**Next best — optimised for {why_label}.** "
        f"Compared to your recommended {_LABELS[rec_key]} route: {vs_str}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODE STATE + UI
# ══════════════════════════════════════════════════════════════════════════════

st.session_state.setdefault("explorer_mode", None)   # None | "expert" | "guided"

# ── Mode selector ──────────────────────────────────────────────────────────────
if st.session_state.explorer_mode is None:
    st.markdown("#### How would you like to explore routes?")
    st.markdown("".join([
        '<div style="display:flex;gap:16px;margin-bottom:16px">',
        '<div class="mode-card" style="border-top:3px solid #27AE60">',
        '<div style="font-size:18px;font-weight:700;margin-bottom:10px">🧭 Guided Mode</div>',
        '<div style="color:#8B949E;font-size:14px;line-height:1.6">',
        'Help me choose. Answer four quick questions about your shipment — ',
        'profit margin, deadline, supply criticality, and cost sensitivity — ',
        "and I'll recommend the right route and show you exactly what trade-offs exist.",
        '</div></div>',
        '<div class="mode-card" style="border-top:3px solid #4A90D9">',
        '<div style="font-size:18px;font-weight:700;margin-bottom:10px">🔍 Expert View</div>',
        '<div style="color:#8B949E;font-size:14px;line-height:1.6">',
        'I know what I need. Show me all three optimised routes — ',
        'cheapest, fastest, and most resilient — side by side with full metrics ',
        'and a cost-of-certainty breakdown.',
        '</div></div>',
        '</div>',
    ]), unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        if st.button("Get My Recommendation →", key="mode_guided", type="primary", use_container_width=True):
            st.session_state.explorer_mode = "guided"
            st.rerun()
    with m2:
        if st.button("Show All Routes →", key="mode_expert", use_container_width=True):
            st.session_state.explorer_mode = "expert"
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT MODE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.explorer_mode == "expert":
    section_header("🛡", "Route Comparison by Criterion", f"{origin} → {destination}")

    cols = st.columns(3)
    for col, (crit_key, crit_label, icon, color, highlight_label) in zip(cols, ALL_CARD_CONFIG):
        r  = routes[crit_key]
        rd = criteria_dicts[crit_key]
        with col:
            st.markdown(
                _route_card_html(crit_key, crit_label, icon, color, highlight_label, r),
                unsafe_allow_html=True,
            )
            _render_route_details(crit_key, crit_label, r, rd)

    # ── Globe — all three routes (below comparisons) ──────────────────────────
    st.markdown("---")
    section_header("🌐", "Route Map")
    st.plotly_chart(globe_fig, use_container_width=True, config={"displayModeBar": False})
    if has_scenario:
        st.info(
            f"🚨 Scenario active: blocked=[{', '.join(blocked)}]  "
            f"US tariff={us_tariff}%  EU={eu_tariff}%  China={china_tariff}%  ASEAN={asean_tariff}%"
        )

    # ── Cost of certainty ─────────────────────────────────────────────────────
    st.markdown("---")
    section_header("💡", "What does more certainty cost?")
    st.caption(
        "Starting from the **Cheapest** route as your baseline, "
        "here is what you gain — and pay — by switching."
    )
    _render_cost_of_certainty(routes, base_key="cheapest")

    # ── Radar chart ────────────────────────────────────────────────────────────
    st.markdown("---")
    section_header("🕸", "Route Comparison Radar")
    st.caption(
        "Each axis is normalised to 0\u2013100 (higher = better). "
        "A larger polygon means a stronger route on that dimension."
    )
    st.plotly_chart(
        make_route_radar(routes),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Summary table + CSV export ────────────────────────────────────────────
    st.markdown("---")
    section_header("📋", "Trade-off Summary")
    summary_rows = []
    for crit_key, crit_label, icon, _, _ in ALL_CARD_CONFIG:
        r  = routes[crit_key]
        rd = criteria_dicts[crit_key]
        summary_rows.append({
            "Criterion":     f"{icon} {crit_label}",
            "Route":         " → ".join(r.path),
            "Freight Cost":  f"{r.cost * 100:.2f}%",
            "Lead Time (d)": r.lead_time_days,
            "Hops":          r.hops,
            "Chk Exposure":  f"{rd['chk_exposure']:.0%}",
            "RS Score":      round(r.rs, 1),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Criterion")
    st.dataframe(summary_df, use_container_width=True)

    st.download_button(
        "Download as CSV",
        data=summary_df.to_csv(),
        file_name=f"sonar_routes_{origin}_{destination}.csv",
        mime="text/csv",
    )

    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# GUIDED MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Wizard (not yet complete) ──────────────────────────────────────────────
    if not st.session_state.wiz_done:
        # Start questions immediately — no "Start" button needed
        if st.session_state.wiz_step == -1:
            st.session_state.wiz_step = 0
            st.rerun()
        with st.container(border=True):
            _render_wizard_question(st.session_state.wiz_step, st.session_state.wiz_answers)

    # ── Wizard complete — show personalised recommendation + turnstile ─────────
    else:
        persona_result = _compute_persona(st.session_state.wiz_answers, routes)
        rec_key        = persona_result["rec_key"]
        rec_color      = CRITERIA_COLORS[rec_key]
        alt_keys       = [k for k in ["most_resilient", "cheapest", "fastest"] if k != rec_key]

        # Turnstile order: recommended first, then alternatives
        turnstile_order = [rec_key] + alt_keys

        # Clamp turnstile_idx in case routes changed
        t_idx = min(st.session_state.turnstile_idx, len(turnstile_order) - 1)
        st.session_state.turnstile_idx = t_idx
        t_key   = turnstile_order[t_idx]
        t_r     = routes[t_key]
        t_rd    = criteria_dicts[t_key]
        t_color = CRITERIA_COLORS[t_key]
        t_label = _LABELS[t_key]
        t_icon  = _ICONS[t_key]
        t_hl    = {"most_resilient": "Resilience Score", "cheapest": "Freight Cost", "fastest": "Lead Time"}[t_key]

        # ── Chatbot message bar ────────────────────────────────────────────────
        with st.container(border=True):
            msg_col, edit_col = st.columns([6, 1])
            with msg_col:
                st.markdown(
                    '<div style="font-size:11px;color:#8B949E;text-transform:uppercase;'
                    'letter-spacing:.06em;margin-bottom:6px">Route Advisor</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_chatbot_message(st.session_state.wiz_answers, persona_result, routes))
            with edit_col:
                st.write("")
                if st.button("✏️ Edit", key="wiz_reset", use_container_width=True):
                    st.session_state.wiz_step      = 0
                    st.session_state.wiz_done      = False
                    st.session_state.turnstile_idx = 0
                    st.rerun()

        for w in persona_result["warnings"]:
            st.warning(w)

        # ── Turnstile carousel ─────────────────────────────────────────────────
        st.markdown("---")

        # Dots + route label centred above card
        dots = "  ".join("●" if i == t_idx else "○" for i in range(len(turnstile_order)))
        is_rec_pos = t_idx == 0
        badge_html = (
            f'<span style="background:{rec_color}22;border:1px solid {rec_color};'
            f'border-radius:12px;padding:1px 8px;font-size:11px;font-weight:700;color:{rec_color}">'
            f'★ Recommended</span> ' if is_rec_pos else ""
        )
        st.markdown(
            "".join([
                '<div style="text-align:center;padding:4px 0 10px 0">',
                badge_html,
                f'<span style="font-size:15px;font-weight:700;color:#e6edf3">{t_icon} {t_label}</span>',
                f'<div style="color:#58a6ff;font-size:13px;margin-top:4px">{dots}</div>',
                f'<div style="color:#8B949E;font-size:11px">Route {t_idx+1} of {len(turnstile_order)}</div>',
                '</div>',
            ]),
            unsafe_allow_html=True,
        )

        # Arrows sit beside the card — vertically padded to align with card mid-point
        _ts_l, _ts_c, _ts_r = st.columns([2, 6, 2])

        # Pre-compute why text so both columns can reference it
        why_text = _why_this_route(t_key, t_idx, rec_key, persona_result, routes)
        why_html_body = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e6edf3">\1</strong>', why_text)

        with _ts_l:
            # Push arrow down to mid-card
            st.markdown('<div style="height:130px"></div>', unsafe_allow_html=True)
            prev_disabled = t_idx == 0
            if not prev_disabled:
                prev_label = _LABELS[turnstile_order[t_idx - 1]]
                st.markdown(
                    f'<div style="text-align:center;font-size:13px;color:#8B949E;margin-bottom:4px">'
                    f'← {prev_label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:13px;margin-bottom:4px;visibility:hidden">&nbsp;</div>',
                    unsafe_allow_html=True,
                )
            if st.button("◀", key="ts_prev", use_container_width=True, disabled=prev_disabled):
                st.session_state.turnstile_idx -= 1
                st.rerun()

        with _ts_c:
            # Route card
            st.markdown(
                f'<div class="turnstile-slide">'
                + _route_card_html(t_key, t_label, t_icon, t_color, t_hl, t_r, is_rec=is_rec_pos)
                + '</div>',
                unsafe_allow_html=True,
            )
            # Why this route box with text inside
            st.markdown(
                "".join([
                    f'<div class="turnstile-slide" style="background:#161b22;border:1px solid #21262d;',
                    f'border-left:3px solid {t_color};border-radius:8px;padding:14px 16px;margin:8px 0">',
                    f'<div style="font-size:11px;color:#8B949E;text-transform:uppercase;',
                    f'letter-spacing:.06em;margin-bottom:8px">Why this route?</div>',
                    f'<div style="font-size:13px;color:#c9d1d9;line-height:1.6">{why_html_body}</div>',
                    '</div>',
                ]),
                unsafe_allow_html=True,
            )
            # Full route details
            _render_route_details(t_key, t_label, t_r, t_rd, persona_result)

        with _ts_r:
            st.markdown('<div style="height:130px"></div>', unsafe_allow_html=True)
            next_disabled = t_idx == len(turnstile_order) - 1
            if not next_disabled:
                next_label = _LABELS[turnstile_order[t_idx + 1]]
                st.markdown(
                    f'<div style="text-align:center;font-size:13px;color:{CRITERIA_COLORS[turnstile_order[t_idx+1]]};'
                    f'font-weight:600;margin-bottom:4px">{next_label} →</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:13px;margin-bottom:4px;visibility:hidden">&nbsp;</div>',
                    unsafe_allow_html=True,
                )
            if st.button("▶", key="ts_next", use_container_width=True, disabled=next_disabled):
                st.session_state.turnstile_idx += 1
                st.rerun()

        # ── Globe — single route for current turnstile position ────────────────
        st.markdown("---")
        section_header("🌐", "Route Map")
        _guided_globe_cache = st.session_state.setdefault("_re_guided_globe_cache", {})
        _gk = (_cache_key, t_key)
        if _gk not in _guided_globe_cache:
            _guided_globe_cache[_gk] = make_multi_criteria_globe(
                criteria_routes={t_key: criteria_dicts[t_key]},
                blocked_chokepoints=blocked,
            )
        st.plotly_chart(_guided_globe_cache[_gk], use_container_width=True, config={"displayModeBar": False})
        if has_scenario:
            st.info(
                f"🚨 Scenario active: blocked=[{', '.join(blocked)}]  "
                f"US tariff={us_tariff}%  EU={eu_tariff}%  China={china_tariff}%  ASEAN={asean_tariff}%"
            )

        # ── Cost of certainty — alternatives ──────────────────────────────────
        st.markdown("---")
        section_header("💡", "What does switching cost?")
        st.caption(
            f"Starting from your recommended **{_LABELS[rec_key]}** route, "
            "here is exactly what you gain — and pay — by choosing a different option."
        )
        _render_cost_of_certainty(routes, base_key=rec_key)

        # ── Radar chart ────────────────────────────────────────────────────────
        st.markdown("---")
        section_header("🕸", "Route Comparison Radar")
        st.caption(
            "Each axis is normalised to 0\u2013100 (higher = better). "
            "A larger polygon means a stronger route on that dimension."
        )
        st.plotly_chart(
            make_route_radar(routes),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Summary table + CSV export ────────────────────────────────────────
        st.markdown("---")
        section_header("📋", "Trade-off Summary")
        summary_rows = []
        for crit_key, crit_label, icon, _, _ in ALL_CARD_CONFIG:
            r  = routes[crit_key]
            rd = criteria_dicts[crit_key]
            v  = persona_result["viability"][crit_key]
            row = {
                "Criterion":     f"{icon} {crit_label}",
                "Route":         " → ".join(r.path),
                "Freight Cost":  f"{r.cost * 100:.2f}%",
                "Lead Time (d)": r.lead_time_days,
                "RS Score":      round(r.rs, 1),
                "Within Margin": "\u2713" if v["viable"] else f"\u2717 ({v['freight_pct']:.1f}% > {persona_result['margin']}%)",
            }
            if persona_result["deadline_days"]:
                row["Meets Deadline"] = (
                    "\u2713" if persona_result["deadline_ok"][crit_key]
                    else f"\u2717 ({r.lead_time_days:.0f}d > {persona_result['deadline_days']}d)"
                )
            summary_rows.append(row)
        guided_df = pd.DataFrame(summary_rows).set_index("Criterion")
        st.dataframe(guided_df, use_container_width=True)

        st.download_button(
            "Download as CSV",
            data=guided_df.to_csv(),
            file_name=f"sonar_routes_{origin}_{destination}.csv",
            mime="text/csv",
        )

        render_footer()

