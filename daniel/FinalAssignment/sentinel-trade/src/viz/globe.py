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
    COUNTRY_COORDS, CHOKEPOINTS,
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


def _maritime_lats_lons(path: list[str]) -> tuple[list, list]:
    """
    Build a lat/lon trace that follows actual maritime sea lanes.

    For each consecutive pair in `path`, routes through maritime waypoints
    (Dijkstra on the waypoint graph) instead of drawing a straight line.
    Starts and ends at each country's geographic centroid.
    """
    if not path:
        return [], []

    G = _maritime_graph()
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

    blocked_countries = [c for cp in blocked_chokepoints
                         for c in CHOKEPOINTS.get(cp, [])]
    if blocked_countries:
        b_lats = [_coords(c)[0] for c in blocked_countries if _coords(c)]
        b_lons = [_coords(c)[1] for c in blocked_countries if _coords(c)]
        b_names = [c for c in blocked_countries if _coords(c)]
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

    for i, route_dict in enumerate(baseline_routes[:show_top_k]):
        path = route_dict["path"]
        lats, lons = _maritime_lats_lons(path)
        is_best = (i == 0)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(width=4 if is_best else 2, color=ROUTE_COLORS[i]),
            marker=dict(size=6 if is_best else 4, color=ROUTE_COLORS[i]),
            name="Baseline (Optimal)" if is_best else f"Baseline Route {i+1}",
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=[f"{c}<br>Cost: {route_dict['cost']:.4f}<br>"
                  f"Lead time: {route_dict['lead_time_days']:.0f}d" if j == 0 else c
                  for j, c in enumerate(path)],
            legendgroup="baseline",
        ))

    if scenario_routes:
        for i, route_dict in enumerate(scenario_routes[:show_top_k]):
            path = route_dict["path"]
            lats, lons = _maritime_lats_lons(path)
            is_best = (i == 0)
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode="lines+markers",
                line=dict(width=4 if is_best else 2,
                          color=COLORS["scenario"] if is_best else "#E67E22",
                          dash="dash"),
                marker=dict(size=6 if is_best else 4, color=COLORS["scenario"]),
                name="Scenario (Rerouted)" if is_best else f"Scenario Route {i+1}",
                legendgroup="scenario",
                hovertemplate="<b>%{text}</b><extra></extra>",
                text=path,
            ))

    # Node labels — use centroids (not maritime path)
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
    if blocked_countries:
        b_lats = [_coords(c)[0] for c in blocked_countries if _coords(c)]
        b_lons = [_coords(c)[1] for c in blocked_countries if _coords(c)]
        b_names = [c for c in blocked_countries if _coords(c)]
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

        lats, lons = _maritime_lats_lons(path)
        lats_off = [lat + offset if lat is not None else None for lat in lats]

        # Hover text: detailed info on origin node only
        hover_texts = []
        for j, c in enumerate(path):
            if j == 0:
                hover_texts.append(
                    f"<b>{c}</b><br>"
                    f"Cost: {route_dict['cost']:.4f}<br>"
                    f"Lead time: {route_dict['lead_time_days']:.0f} d<br>"
                    f"Hops: {route_dict['hops']}"
                )
            else:
                hover_texts.append(f"<b>{c}</b>")

        # Waypoints between countries have no hover text — pad with empty strings
        # We only have country-level hover_texts but lats/lons is longer (includes waypoints).
        # Build hover text aligned to lats: country nodes get their text, waypoints get "".
        # Reconstruct which indices correspond to country nodes.
        country_lats = [(_coords(c) or (None, None))[0] for c in path]
        full_hover = []
        ci = 0  # country index
        tol = 0.01
        for lat in lats_off:
            if ci < len(path) and lat is not None and country_lats[ci] is not None:
                if abs(lat - country_lats[ci] - offset) < tol or abs(lat - offset - (country_lats[ci] or 0)) < tol:
                    full_hover.append(hover_texts[ci] if ci < len(hover_texts) else "")
                    ci += 1
                    continue
            full_hover.append("")

        fig.add_trace(go.Scattergeo(
            lat=lats_off, lon=lons,
            mode="lines+markers",
            line=dict(width=4, color=color),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            name=label,
            text=full_hover,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=key,
        ))

        all_path_countries.update(path)

    # Country centroid node labels
    all_path_countries -= set(blocked_countries)
    node_lats  = [_coords(c)[0] for c in all_path_countries if _coords(c)]
    node_lons  = [_coords(c)[1] for c in all_path_countries if _coords(c)]
    node_names = [c for c in all_path_countries if _coords(c)]
    if node_names:
        fig.add_trace(go.Scattergeo(
            lat=node_lats, lon=node_lons,
            mode="markers+text",
            marker=dict(size=6, color="white", opacity=0.5),
            text=node_names, textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.6)", size=9),
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
            title="RS",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
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
