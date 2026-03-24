"""
Plotly globe / map visualization factory.
Builds Scattergeo figures showing shipping routes, blocked chokepoints,
and country risk overlays.

Routes are drawn along realistic maritime waypoints (straits, canals, ocean
crossing points) rather than straight point-to-point lines.
"""

import math
import sys
import os
import functools
from typing import Optional

import networkx as nx
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    COUNTRY_COORDS, CHOKEPOINTS, CHOKEPOINT_WAYPOINTS,
    MARITIME_WAYPOINTS, MARITIME_EDGES, COUNTRY_PORT_WAYPOINT,
)


# ─── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "baseline":       "#4A90D9",
    "scenario":       "#F5A623",
    "blocked":        "#D0021B",
    "node_normal":    "#7ED321",
    "node_risk":      "#F8E71C",
    "node_blocked":   "#D0021B",
    "bg":             "#0e1117",
    "paper":          "#0e1117",
    "geo":            "#1a2035",
    "coastline":      "#2a3f5f",
    "land":           "#1e2a3a",
    "ocean":          "#0d1929",
}

ROUTE_COLORS = [COLORS["baseline"], COLORS["scenario"], "#9B59B6"]

CRITERIA_COLORS = {
    "most_resilient": "#27AE60",
    "cheapest":       "#4A90D9",
    "fastest":        "#F5A623",
}

CRITERIA_LABELS = {
    "most_resilient": "Most Resilient",
    "cheapest":       "Cheapest",
    "fastest":        "Fastest",
}


# ─── Maritime routing helpers ─────────────────────────────────────────────────

def _hav(c1: tuple, c2: tuple) -> float:
    """Haversine distance in km between (lat, lon) tuples."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = c1[0], c1[1], c2[0], c2[1]
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


@functools.lru_cache(maxsize=1)
def _maritime_graph() -> nx.Graph:
    """Build and cache the maritime waypoint graph (built once per process)."""
    G = nx.Graph()
    for node in MARITIME_WAYPOINTS:
        G.add_node(node)
    for u, v in MARITIME_EDGES:
        cu, cv = MARITIME_WAYPOINTS[u], MARITIME_WAYPOINTS[v]
        G.add_edge(u, v, weight=_hav(cu, cv))
    return G


def _country_port_wp(country: str) -> str | None:
    """
    Return the maritime waypoint key for a country.
    Falls back to nearest waypoint by haversine if not in the explicit mapping.
    """
    if country in COUNTRY_PORT_WAYPOINT:
        return COUNTRY_PORT_WAYPOINT[country]
    coord = COUNTRY_COORDS.get(country)
    if not coord:
        return None
    return min(MARITIME_WAYPOINTS.items(),
               key=lambda kv: _hav(coord, kv[1]))[0]


def _maritime_lats_lons(
    path: list[str],
    blocked_wps: set[str] | None = None,
) -> tuple[list, list]:
    """
    Build a lat/lon trace that follows actual maritime sea lanes.

    For each consecutive pair in `path`, routes through maritime waypoints
    (Dijkstra on the waypoint graph) instead of drawing a straight line.
    Starts and ends at each country's geographic centroid.

    Parameters
    ----------
    path        : list of country names (trade-graph route)
    blocked_wps : waypoint node keys to remove before path-finding, so the
                  visual route honours the same chokepoint closures as the
                  trade routing engine (e.g. removing SUEZ_S + SUEZ_N forces
                  the drawn line around Africa when Suez Canal is blocked).
    """
    if not path:
        return [], []

    blocked_wps = blocked_wps or set()
    G = _maritime_graph()
    if blocked_wps:
        G = G.copy()          # don't mutate the lru_cache singleton
        for wp in blocked_wps:
            if G.has_node(wp):
                G.remove_node(wp)

    lats: list = []
    lons: list = []

    def _add(coord):
        if coord:
            lats.append(coord[0])
            lons.append(coord[1])

    _add(COUNTRY_COORDS.get(path[0]))

    for i in range(len(path) - 1):
        wp_start = _country_port_wp(path[i])
        wp_end   = _country_port_wp(path[i + 1])

        # If a country's own port waypoint was blocked, skip waypoint routing
        # for that leg and fall through to the straight-segment fallback.
        if wp_start in blocked_wps:
            wp_start = None
        if wp_end in blocked_wps:
            wp_end = None

        if wp_start and wp_end and wp_start != wp_end:
            try:
                for wp in nx.shortest_path(G, wp_start, wp_end, weight="weight"):
                    _add(MARITIME_WAYPOINTS[wp])
            except nx.NetworkXNoPath:
                pass  # fall back to straight segment

        _add(COUNTRY_COORDS.get(path[i + 1]))

    return lats, lons


def _coords(country: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a country centroid, or None if unknown."""
    return COUNTRY_COORDS.get(country)


def _path_lats_lons(path: list[str]) -> tuple[list, list]:
    """Straight-line centroid path — used only for node label markers."""
    lats, lons = [], []
    for c in path:
        coord = _coords(c)
        if coord:
            lats.append(coord[0])
            lons.append(coord[1])
        else:
            lats.append(None)
            lons.append(None)
    return lats, lons


# ─── Geo layout template ──────────────────────────────────────────────────────

def _geo_layout() -> dict:
    return dict(
        projection_type="natural earth",
        showcoastlines=True,  coastlinecolor=COLORS["coastline"],
        showland=True,        landcolor=COLORS["land"],
        showocean=True,       oceancolor=COLORS["ocean"],
        showlakes=False, showrivers=False, showframe=False,
        bgcolor=COLORS["geo"],
    )


def _legend_style() -> dict:
    return dict(
        font=dict(color="white", size=11),
        bgcolor="rgba(30,42,58,0.8)",
        bordercolor="#2a3f5f", borderwidth=1,
        x=0.01, y=0.99, xanchor="left", yanchor="top",
    )


# ─── Figure builders ──────────────────────────────────────────────────────────

def make_route_globe(
    baseline_routes: list[dict],
    scenario_routes: list[dict],
    blocked_chokepoints: list[str],
    projection: str = "natural earth",
    show_top_k: int = 1,
) -> go.Figure:
    """Baseline vs. scenario route comparison globe."""
    fig = go.Figure()

    # Blocked markers sit at maritime waypoint coords (the actual strait/canal),
    # not at country centroids.
    blocked_countries = [c for cp in blocked_chokepoints
                         for c in CHOKEPOINTS.get(cp, [])]
    if blocked_chokepoints:
        b_lats, b_lons, b_names = [], [], []
        for cp in blocked_chokepoints:
            coords = [MARITIME_WAYPOINTS[wp]
                      for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])
                      if wp in MARITIME_WAYPOINTS]
            if coords:
                b_lats.append(sum(c[0] for c in coords) / len(coords))
                b_lons.append(sum(c[1] for c in coords) / len(coords))
                b_names.append(cp)
        if b_lats:
            fig.add_trace(go.Scattergeo(
                lat=b_lats, lon=b_lons,
                mode="markers+text",
                marker=dict(size=14, symbol="x", color=COLORS["blocked"],
                            line=dict(width=2, color="white")),
                text=b_names, textposition="top center",
                textfont=dict(color="white", size=10),
                name="Blocked",
                hovertemplate="%{text}<br><b>BLOCKED</b><extra></extra>",
            ))

    blocked_wps = {wp for cp in blocked_chokepoints
                   for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])}

    for i, route_dict in enumerate(baseline_routes[:show_top_k]):
        path = route_dict["path"]
        lats, lons = _maritime_lats_lons(path, blocked_wps)
        is_best = (i == 0)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=4 if is_best else 2, color=ROUTE_COLORS[i]),
            name="Baseline (Optimal)" if is_best else f"Baseline Route {i+1}",
            hoverinfo="skip",
            legendgroup="baseline",
        ))

    if scenario_routes:
        for i, route_dict in enumerate(scenario_routes[:show_top_k]):
            path = route_dict["path"]
            lats, lons = _maritime_lats_lons(path, blocked_wps)
            is_best = (i == 0)
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(width=4 if is_best else 2,
                          color=COLORS["scenario"] if is_best else "#E67E22",
                          dash="dash"),
                name="Scenario (Rerouted)" if is_best else f"Scenario Route {i+1}",
                hoverinfo="skip",
                legendgroup="scenario",
            ))

    # Country node markers — dots + labels only at actual country centroids
    all_path_countries: set[str] = set()
    for r in baseline_routes[:1] + scenario_routes[:1]:
        all_path_countries.update(r.get("path", []))
    all_path_countries -= set(blocked_countries)

    node_lats = [_coords(c)[0] for c in all_path_countries if _coords(c)]
    node_lons = [_coords(c)[1] for c in all_path_countries if _coords(c)]
    node_names = [c for c in all_path_countries if _coords(c)]
    if node_names:
        fig.add_trace(go.Scattergeo(
            lat=node_lats, lon=node_lons,
            mode="markers+text",
            marker=dict(size=8, color=COLORS["node_normal"],
                        line=dict(width=1, color="white")),
            text=node_names, textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.7)", size=9),
            name="Route Nodes", showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_geos(**_geo_layout())
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        legend=_legend_style(),
    )
    return fig


def make_multi_criteria_globe(
    criteria_routes: dict,
    blocked_chokepoints: list[str] | None = None,
    projection: str = "natural earth",
) -> go.Figure:
    """
    Render up to 3 routes on the same globe, each coloured by criterion.
    Routes follow maritime sea lanes via waypoint routing.
    """
    fig = go.Figure()
    blocked_chokepoints = blocked_chokepoints or []

    blocked_countries = [c for cp in blocked_chokepoints
                         for c in CHOKEPOINTS.get(cp, [])]
    if blocked_chokepoints:
        b_lats, b_lons, b_names = [], [], []
        for cp in blocked_chokepoints:
            coords = [MARITIME_WAYPOINTS[wp]
                      for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])
                      if wp in MARITIME_WAYPOINTS]
            if coords:
                b_lats.append(sum(c[0] for c in coords) / len(coords))
                b_lons.append(sum(c[1] for c in coords) / len(coords))
                b_names.append(cp)
        if b_lats:
            fig.add_trace(go.Scattergeo(
                lat=b_lats, lon=b_lons,
                mode="markers+text",
                marker=dict(size=14, symbol="x", color=COLORS["blocked"],
                            line=dict(width=2, color="white")),
                text=b_names, textposition="top center",
                textfont=dict(color="white", size=10),
                name="Blocked",
                hovertemplate="%{text}<br><b>BLOCKED</b><extra></extra>",
            ))

    blocked_wps = {wp for cp in blocked_chokepoints
                   for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])}

    all_path_countries: set[str] = set()
    drawn_paths: list[tuple] = []

    for key, route_dict in criteria_routes.items():
        if not route_dict:
            continue
        path       = route_dict["path"]
        color      = CRITERIA_COLORS.get(key, "#FFFFFF")
        label      = CRITERIA_LABELS.get(key, key)
        path_tuple = tuple(path)

        # Slight lat offset for overlapping paths so all 3 remain visible
        offset = sum(0.8 for prev in drawn_paths if prev == path_tuple)
        drawn_paths.append(path_tuple)

        lats, lons = _maritime_lats_lons(path, blocked_wps)
        lats_off = [lat + offset if lat is not None else None for lat in lats]

        fig.add_trace(go.Scattergeo(
            lat=lats_off, lon=lons,
            mode="lines",
            line=dict(width=4, color=color),
            name=label,
            hoverinfo="skip",
            legendgroup=key,
        ))

        all_path_countries.update(path)

    # Country centroid node labels — dots only at actual country nodes
    all_path_countries -= set(blocked_countries)
    node_lats  = [_coords(c)[0] for c in all_path_countries if _coords(c)]
    node_lons  = [_coords(c)[1] for c in all_path_countries if _coords(c)]
    node_names = [c for c in all_path_countries if _coords(c)]
    if node_names:
        fig.add_trace(go.Scattergeo(
            lat=node_lats, lon=node_lons,
            mode="markers+text",
            marker=dict(size=9, color="white",
                        line=dict(width=1, color="#4A90D9")),
            text=node_names, textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.85)", size=9),
            name="Nodes", showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_geos(**_geo_layout())
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(30,42,58,0.85)",
            bordercolor="#2a3f5f", borderwidth=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            itemsizing="constant",
        ),
    )
    return fig


def make_resilience_gauge(score: float, label: str) -> go.Figure:
    """Create a Plotly gauge chart for the Resilience Score."""
    color = (
        "#27AE60" if score >= 75 else
        "#F39C12" if score >= 50 else
        "#E74C3C" if score >= 25 else
        "#8E44AD"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(font=dict(color="white", size=40)),
        title=dict(text=f"Resilience Score<br><sub>{label}</sub>",
                   font=dict(color="white", size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="white",
                      tickfont=dict(color="white")),
            bar=dict(color=color),
            bgcolor="rgba(30,42,58,0.8)",
            steps=[
                dict(range=[0,  25], color="#3D0000"),
                dict(range=[25, 50], color="#3D2000"),
                dict(range=[50, 75], color="#2C3E00"),
                dict(range=[75, 100], color="#0D3B00"),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.75,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor=COLORS["paper"],
        font=dict(color="white"),
    )
    return fig


def make_corridor_heatmap(
    corridor_data: list[dict],
    title: str = "Resilience Score — Top Trade Corridors",
) -> go.Figure:
    """Build a heatmap of resilience scores for top corridors × products."""
    import pandas as pd
    df = pd.DataFrame(corridor_data)
    pivot = df.pivot_table(
        index=["origin", "destination"],
        columns="product_name",
        values="score",
        aggfunc="mean",
    )
    z = pivot.values
    y = [f"{o} → {d}" for o, d in pivot.index]
    x = pivot.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#8E44AD"], [0.25, "#E74C3C"],
                    [0.5, "#F39C12"], [0.75, "#27AE60"], [1, "#1E8449"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}" if v == v else "" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
        colorbar=dict(
            title=dict(text="RS", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        height=600,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font=dict(color="white"),
        xaxis=dict(tickfont=dict(color="white")),
        yaxis=dict(tickfont=dict(color="white"), autorange="reversed"),
    )
    return fig


# ─── Radar chart for route comparison ────────────────────────────────────────

def make_route_radar(routes: dict) -> go.Figure:
    """
    Build a radar (spider) chart comparing up to 3 routes across 5 dimensions.

    Axes are chosen to guarantee visual differentiation between the three route
    criteria (cheapest / fastest / most resilient):
      1. Resilience Score   — composite RS (0–100), highest for most_resilient
      2. Affordability      — relative inverted cost, highest for cheapest
      3. Speed              — relative inverted lead time, highest for fastest
      4. Chokepoint Safety  — 1 − chokepoint exposure (0–100)
      5. Political Safety   — geopolitical risk component (0–100)

    Axes 2 and 3 are normalised relative to the compared routes so differences
    are always visible (scaled to [20, 100]).

    Parameters
    ----------
    routes : dict[str, Route]
        Keys are criteria names (e.g. "most_resilient"), values are Route objects.
    """
    categories = [
        "Resilience Score",
        "Affordability",
        "Speed",
        "Chokepoint Safety",
        "Political Safety",
    ]

    # Collect range data for relative normalisation of cost and lead time.
    costs = [r.cost            for r in routes.values()]
    lts   = [r.lead_time_days  for r in routes.values()]
    cost_lo, cost_hi = min(costs), max(costs)
    lt_lo,   lt_hi   = min(lts),   max(lts)

    def _norm_inv(val, lo, hi):
        """Lower raw value → higher score. Maps to [20, 100]."""
        if hi - lo < 1e-9:
            return 60.0          # all routes equal on this dimension
        return 20.0 + (1.0 - (val - lo) / (hi - lo)) * 80.0

    fig = go.Figure()

    for crit_key, r in routes.items():
        label  = CRITERIA_LABELS.get(crit_key, crit_key)
        color  = CRITERIA_COLORS.get(crit_key, "#ccc")
        detail = getattr(r, "rs_detail", {})

        values = [
            float(r.rs),                                       # Resilience Score
            _norm_inv(r.cost,           cost_lo, cost_hi),    # Affordability
            _norm_inv(r.lead_time_days, lt_lo,   lt_hi),      # Speed
            (1.0 - float(r.chk_exposure)) * 100.0,            # Chokepoint Safety
            detail.get("sec", 0.5) * 100.0,                   # Political Safety
        ]

        # Do NOT manually close the polygon — fill="toself" closes the fill
        # automatically, and avoiding the repeated first point prevents a
        # spurious coloured line overlaid on the first axis spoke.
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
            line=dict(color=color, width=2),
            name=label,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Score: %{r:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#21262d", linecolor="#21262d",
                tickfont=dict(color="#8B949E", size=10),
            ),
            angularaxis=dict(
                gridcolor="#21262d", linecolor="#21262d",
                tickfont=dict(color="#e6edf3", size=12),
            ),
        ),
        paper_bgcolor=COLORS["paper"],
        font=dict(color="#e6edf3"),
        legend=dict(
            font=dict(color="#e6edf3", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=40, b=40),
        height=420,
    )
    return fig
