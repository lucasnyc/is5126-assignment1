"""
Plotly globe / map visualization factory.
Builds Scattergeo figures showing shipping routes, blocked chokepoints,
and country risk overlays.
"""

import sys
import os
from typing import Optional

import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import COUNTRY_COORDS, CHOKEPOINTS


# ─── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "baseline":       "#4A90D9",   # blue
    "scenario":       "#F5A623",   # amber
    "blocked":        "#D0021B",   # red
    "node_normal":    "#7ED321",   # green
    "node_risk":      "#F8E71C",   # yellow
    "node_blocked":   "#D0021B",   # red
    "bg":             "#0e1117",
    "paper":          "#0e1117",
    "geo":            "#1a2035",
    "coastline":      "#2a3f5f",
    "land":           "#1e2a3a",
    "ocean":          "#0d1929",
}

ROUTE_COLORS = [COLORS["baseline"], COLORS["scenario"], "#9B59B6"]  # k1, k2, k3


def _coords(country: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a country, or None if unknown."""
    return COUNTRY_COORDS.get(country)


def _path_lats_lons(path: list[str]) -> tuple[list, list]:
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


def make_route_globe(
    baseline_routes: list[dict],
    scenario_routes: list[dict],
    blocked_chokepoints: list[str],
    projection: str = "natural earth",
    show_top_k: int = 1,
) -> go.Figure:
    """
    Build a Plotly Scattergeo figure showing baseline and scenario routes.

    Parameters
    ----------
    baseline_routes     : list of route dicts (from Route.to_dict())
    scenario_routes     : list of route dicts under scenario
    blocked_chokepoints : list of chokepoint display names to mark as blocked
    projection          : Plotly geo projection type
    show_top_k          : how many routes to draw (1 = best only, 3 = all top-3)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Blocked chokepoint country markers ────────────────────────────────────
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
            text=b_names,
            textposition="top center",
            textfont=dict(color="white", size=10),
            name="Blocked",
            hovertemplate="%{text}<br><b>BLOCKED</b><extra></extra>",
        ))

    # ── Baseline routes ───────────────────────────────────────────────────────
    routes_to_draw = baseline_routes[:show_top_k]
    for i, route_dict in enumerate(routes_to_draw):
        path = route_dict["path"]
        lats, lons = _path_lats_lons(path)
        is_best = (i == 0)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(
                width=4 if is_best else 2,
                color=ROUTE_COLORS[i],
            ),
            marker=dict(size=6 if is_best else 4, color=ROUTE_COLORS[i]),
            name=f"Baseline Route {i+1}" if not is_best else "Baseline (Optimal)",
            hovertemplate=(
                "<b>%{text}</b><extra></extra>"
            ),
            text=[f"{c}<br>Cost: {route_dict['cost']:.4f}<br>"
                  f"Lead time: {route_dict['lead_time_days']:.0f}d" if j == 0 else c
                  for j, c in enumerate(path)],
            legendgroup="baseline",
        ))

    # ── Scenario routes ───────────────────────────────────────────────────────
    if scenario_routes:
        scen_routes_to_draw = scenario_routes[:show_top_k]
        for i, route_dict in enumerate(scen_routes_to_draw):
            path = route_dict["path"]
            lats, lons = _path_lats_lons(path)
            is_best = (i == 0)
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode="lines+markers",
                line=dict(
                    width=4 if is_best else 2,
                    color=COLORS["scenario"] if is_best else "#E67E22",
                    dash="dash",
                ),
                marker=dict(size=6 if is_best else 4, color=COLORS["scenario"]),
                name=f"Scenario Route {i+1}" if not is_best else "Scenario (Rerouted)",
                legendgroup="scenario",
                hovertemplate="<b>%{text}</b><extra></extra>",
                text=path,
            ))

    # ── Node markers for path countries ───────────────────────────────────────
    all_path_countries = set()
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
            text=node_names,
            textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.7)", size=9),
            name="Route Nodes",
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_geos(
        projection_type=projection,
        showcoastlines=True,
        coastlinecolor=COLORS["coastline"],
        showland=True,
        landcolor=COLORS["land"],
        showocean=True,
        oceancolor=COLORS["ocean"],
        showlakes=False,
        showrivers=False,
        showframe=False,
        bgcolor=COLORS["geo"],
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(30,42,58,0.8)",
            bordercolor="#2a3f5f",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
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
            axis=dict(
                range=[0, 100],
                tickcolor="white",
                tickfont=dict(color="white"),
            ),
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
    """
    Build a heatmap table of resilience scores for top corridors.
    corridor_data: list of dicts with keys: origin, destination, product_name, score
    """
    import pandas as pd
    df = pd.DataFrame(corridor_data)
    pivot = df.pivot_table(
        index=["origin", "destination"],
        columns="product_name",
        values="score",
        aggfunc="mean",
    )
    z     = pivot.values
    y     = [f"{o} → {d}" for o, d in pivot.index]
    x     = pivot.columns.tolist()

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
