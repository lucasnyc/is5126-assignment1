"""
Routing engine — Dijkstra + Yen's K-shortest paths with scenario simulation.
"""

import itertools
import math
import sys
import os
from typing import Optional

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import K_ROUTES, MAX_HOPS, MAX_HOPS_FALLBACK, COUNTRY_COORDS
from src.graph.chokepoints import get_countries_to_remove, chokepoint_exposure

# Average container ship speed: 15 knots = 27.78 km/h = 666.7 km/day
_SHIP_SPEED_KM_PER_DAY = 666.7
# Port handling time per intermediate stop (loading, customs, berthing)
_PORT_DAYS_PER_STOP = 0.75
# Maximum allowed ratio of (total path distance) / (direct great-circle distance).
# Real transshipment adds 10–80 % extra distance (e.g. China→Singapore→Europe ≈ 1.9×).
# A ratio > 3.0 indicates a geographically absurd phantom route.
_MAX_DETOUR_RATIO = 3.0

# Minimum LSCI score for a country to serve as an intermediate transshipment hub.
# LSCI reflects a country's integration into global liner shipping networks.
# Setting the bar at 100 retains established hubs (Singapore=557, Korea=536,
# Malaysia=478, Netherlands=397, Panama=204, Morocco=202, Philippines=160)
# while excluding minor island states (Bahamas=74, Jamaica=95) and landlocked
# countries (Bolivia=0, Serbia=0) that are not real transshipment centres.
_MIN_HUB_LSCI = 100.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _path_distance_km(path: list[str]) -> float:
    """Sum of haversine legs along a path. Returns 0 if coords missing."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        cu, cv = COUNTRY_COORDS.get(u), COUNTRY_COORDS.get(v)
        if cu and cv:
            total += _haversine_km(cu[0], cu[1], cv[0], cv[1])
    return total


def _is_geographically_valid(path: list[str]) -> bool:
    """
    Return False if the path takes a geographically absurd detour.

    Computes the ratio of total path distance to direct source→target distance.
    Ratios > _MAX_DETOUR_RATIO (3.0) indicate phantom routes created by the ML
    model assigning low freight rates to non-existent corridors (e.g. China →
    Paraguay → Singapore). Real transshipment routes (e.g. via Singapore, UAE,
    Macao) produce ratios of 1.0–2.0.
    """
    if len(path) < 2:
        return True
    src, dst = path[0], path[-1]
    cs, cd = COUNTRY_COORDS.get(src), COUNTRY_COORDS.get(dst)
    if not cs or not cd:
        return True  # can't validate, allow through
    direct_km = _haversine_km(cs[0], cs[1], cd[0], cd[1])
    if direct_km < 500:          # very short corridors — no filtering needed
        return True
    total_km = _path_distance_km(path)
    if total_km == 0:
        return True              # coords missing along path — allow through
    return (total_km / direct_km) <= _MAX_DETOUR_RATIO


# ─── Route result dataclass ───────────────────────────────────────────────────

class Route:
    """Represents a single route result."""

    def __init__(self, path: list[str], cost: float, graph: nx.DiGraph,
                 median_lsci: float = 50.0):
        self.path          = path
        self.cost          = cost
        self.hops          = len(path) - 1
        self.chk_exposure  = chokepoint_exposure(path)
        self.lead_time_days = self._estimate_lead_time(graph, median_lsci)
        self.has_predicted = self._check_predicted(graph)

    def _estimate_lead_time(self, G: nx.DiGraph, median_lsci: float) -> float:
        """
        Lead time = (total haversine distance / avg ship speed) + port handling.

        Methodology
        -----------
        - Great-circle distance between each consecutive pair of countries
          using their geographic centroids (COUNTRY_COORDS).
        - Average container ship speed: 15 knots ≈ 667 km/day.
        - Port handling: 0.75 days per intermediate stop.
        - Falls back to 5 days/hop when coordinates are missing.
        """
        days = 0.0
        for u, v in zip(self.path[:-1], self.path[1:]):
            c_u = COUNTRY_COORDS.get(u)
            c_v = COUNTRY_COORDS.get(v)
            if c_u and c_v:
                dist_km = _haversine_km(c_u[0], c_u[1], c_v[0], c_v[1])
                days += dist_km / _SHIP_SPEED_KM_PER_DAY
            else:
                days += 5.0  # fallback if coords unknown
        # Add port handling for every stop except the final destination
        intermediate_stops = max(0, len(self.path) - 2)
        days += intermediate_stops * _PORT_DAYS_PER_STOP
        return round(days, 1)

    def _check_predicted(self, G: nx.DiGraph) -> bool:
        """True if any edge on this path is ML-predicted (not observed)."""
        for u, v in zip(self.path[:-1], self.path[1:]):
            if G.has_edge(u, v) and G[u][v].get("is_predicted", False):
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "path":           self.path,
            "path_str":       " → ".join(self.path),
            "cost":           round(self.cost, 4),
            "hops":           self.hops,
            "lead_time_days": self.lead_time_days,
            "chk_exposure":   round(self.chk_exposure, 3),
            "has_predicted":  self.has_predicted,
        }


# ─── Scenario application ─────────────────────────────────────────────────────

def apply_scenario(
    G: nx.DiGraph,
    blocked_chokepoints: list[str] | None = None,
    tariff_multipliers: dict[str, float] | None = None,
) -> nx.DiGraph:
    """
    Return a modified copy of G with:
      - Chokepoint country nodes removed
      - Edge weights multiplied by tariff multipliers

    IMPORTANT: Never mutates the original cached graph.
    """
    H = G.copy()  # shallow copy of graph structure; deep copy of dicts

    # Remove blocked chokepoint nodes
    if blocked_chokepoints:
        to_remove = get_countries_to_remove(blocked_chokepoints)
        for country in to_remove:
            if country in H:
                H.remove_node(country)

    # Apply tariff multipliers to edge weights
    if tariff_multipliers:
        for u, v, data in H.edges(data=True):
            mult_u = tariff_multipliers.get(u, 1.0)
            mult_v = tariff_multipliers.get(v, 1.0)
            multiplier = max(mult_u, mult_v)
            if multiplier > 1.0:
                H[u][v]["weight"] = data["weight"] * multiplier

    return H


# ─── K-shortest paths ─────────────────────────────────────────────────────────

def find_k_routes(
    G: nx.DiGraph,
    source: str,
    target: str,
    k: int = K_ROUTES,
    cutoff: int = MAX_HOPS,
    median_lsci: float = 50.0,
) -> list[Route]:
    """
    Find top-k shortest paths (by freight cost) using Yen's algorithm.

    Hop strategy (real-world shipping realism)
    ------------------------------------------
    Tries ``cutoff`` (default MAX_HOPS = 2) first — direct service or a single
    transshipment hub.  If fewer than k routes are found, automatically expands
    to MAX_HOPS_FALLBACK (= 3) to handle niche/remote corridors.  Paths longer
    than the fallback are never returned.

    Parameters
    ----------
    G           : NetworkX DiGraph (after any scenario modifications)
    source      : origin country name
    target      : destination country name
    k           : number of paths to return
    cutoff      : preferred hop limit (falls back to MAX_HOPS_FALLBACK)
    median_lsci : used for lead time calculation

    Returns
    -------
    List of Route objects in non-decreasing cost order.

    Raises
    ------
    nx.NodeNotFound   if source or target not in graph
    nx.NetworkXNoPath if no path exists within the fallback hop limit
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source node '{source}' not in graph.")
    if target not in G:
        raise nx.NodeNotFound(f"Target node '{target}' not in graph.")

    # ── Build hub-qualified subgraph ──────────────────────────────────────────
    # Only LSCI ≥ _MIN_HUB_LSCI countries (plus the endpoints) can participate
    # as transshipment nodes.  This excludes landlocked and non-maritime
    # countries from ever appearing as intermediate stops.
    hub_nodes = {
        n for n in G.nodes()
        if (G.nodes[n].get("lsci", 0.0) or 0.0) >= _MIN_HUB_LSCI
    }
    hub_nodes.update([source, target])
    G_hubs = G.subgraph(hub_nodes)

    # ── Phase-by-phase route collection ───────────────────────────────────────
    # We search explicitly by hop count rather than relying on Yen's cost
    # ordering.  This prevents cheap ML-predicted phantom routes (which rank
    # high in cost order) from burying valid direct/short-hop routes.

    def _routes_for_hops(graph: nx.DiGraph, max_hops: int,
                         geo_filter: bool = True) -> list[Route]:
        """Collect all distinct paths of ≤ max_hops hops from the graph."""
        if source not in graph or target not in graph:
            return []
        found: list[Route] = []
        seen: set[tuple] = set()
        try:
            path_gen = nx.shortest_simple_paths(graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        # Scan up to a generous window; stop early once we have k routes
        for path in itertools.islice(path_gen, k * 50):
            hops = len(path) - 1
            if hops > max_hops:
                # Yen's is cost-sorted; once hop count is this long,
                # shorter alternatives are still possible — keep scanning
                continue
            if geo_filter and not _is_geographically_valid(path):
                continue
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            cost = sum(
                G[u][v]["weight"] for u, v in zip(path[:-1], path[1:])
                if G.has_edge(u, v)
            )
            found.append(Route(path, cost, G, median_lsci))
            if len(found) == k:
                break
        return found

    # Phase 1 – Direct (0 intermediate hubs)
    routes: list[Route] = []
    if G_hubs.has_edge(source, target):
        cost = G[source][target]["weight"] if G.has_edge(source, target) else 0.0
        routes.append(Route([source, target], cost, G, median_lsci))

    # Phase 2 – Single hub (≤2 hops), geo-filtered, hub subgraph
    if len(routes) < k:
        phase2 = _routes_for_hops(G_hubs, cutoff, geo_filter=True)
        seen_paths = {tuple(r.path) for r in routes}
        routes += [r for r in phase2 if tuple(r.path) not in seen_paths]

    # Phase 3 – Two hubs (≤3 hops), geo-filtered, hub subgraph
    if len(routes) < k and cutoff < MAX_HOPS_FALLBACK:
        phase3 = _routes_for_hops(G_hubs, MAX_HOPS_FALLBACK, geo_filter=True)
        seen_paths = {tuple(r.path) for r in routes}
        routes += [r for r in phase3 if tuple(r.path) not in seen_paths]

    # Phase 4 & 5 are last resorts — only activate when phases 1-3 found
    # nothing at all.  This prevents cheaper-but-invalid phantom routes from
    # displacing a valid (if expensive) direct route found in Phase 1.
    if not routes:
        # Relax geo filter, still using hub subgraph
        phase4 = _routes_for_hops(G_hubs, MAX_HOPS_FALLBACK, geo_filter=False)
        routes = phase4[:k]

    if not routes:
        # Absolute last resort: full graph, no filters (extremely remote corridors)
        routes = _routes_for_hops(G, MAX_HOPS_FALLBACK, geo_filter=False)[:k]

    # Deduplicate and sort by cost, preserving up to k routes
    seen: set[tuple] = set()
    deduped: list[Route] = []
    for r in sorted(routes, key=lambda r: r.cost):
        key = tuple(r.path)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
        if len(deduped) == k:
            break
    routes = deduped

    if not routes:
        raise nx.NetworkXNoPath(
            f"No path from '{source}' to '{target}' within "
            f"{MAX_HOPS_FALLBACK} hops."
        )

    return routes


def find_multi_criteria_routes(
    G: nx.DiGraph,
    source: str,
    target: str,
    scorer,
    k_candidates: int = 20,
    cutoff: int = MAX_HOPS,
    median_lsci: float = 50.0,
) -> dict:
    """
    Find the best route for each of 3 criteria from a candidate pool.

    Returns
    -------
    dict with keys:
        "cheapest"       — lowest freight cost
        "fastest"        — lowest estimated lead time
        "most_resilient" — highest Resilience Score
    Each value is a Route object with an extra ``rs`` attribute (float).
    """
    candidates = find_k_routes(
        G, source, target, k=k_candidates, cutoff=cutoff, median_lsci=median_lsci
    )

    # Score every candidate; use the next-cheapest as the k2 comparator
    for i, r in enumerate(candidates):
        k2 = candidates[i + 1] if i + 1 < len(candidates) else None
        rs_result = scorer.score(
            path_k1=r.path,
            cost_k1=r.cost,
            G=G,
            path_k2=k2.path if k2 else None,
            cost_k2=k2.cost if k2 else None,
        )
        r.rs        = rs_result["score"]
        r.rs_detail = rs_result

    cheapest       = candidates[0]                                    # pool is cost-sorted
    fastest        = min(candidates, key=lambda r: r.lead_time_days)
    most_resilient = max(candidates, key=lambda r: r.rs)

    return {
        "cheapest":       cheapest,
        "fastest":        fastest,
        "most_resilient": most_resilient,
    }


def compare_scenarios(
    G_base: nx.DiGraph,
    G_scenario: nx.DiGraph,
    source: str,
    target: str,
    k: int = K_ROUTES,
    median_lsci: float = 50.0,
) -> dict:
    """
    Run routing on both baseline and scenario graphs and return comparison dict.

    Returns dict with keys: baseline_routes, scenario_routes, cost_premium_pct,
    lead_time_delta_days, rerouted (bool)
    """
    try:
        base_routes = find_k_routes(G_base, source, target, k=k,
                                    median_lsci=median_lsci)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        return {"error": f"Baseline routing failed: {e}"}

    scenario_routes: Optional[list[Route]] = None
    scenario_error: Optional[str] = None
    try:
        scenario_routes = find_k_routes(G_scenario, source, target, k=k,
                                        median_lsci=median_lsci)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        scenario_error = str(e)

    base_best = base_routes[0]
    result = {
        "baseline_routes":     [r.to_dict() for r in base_routes],
        "scenario_routes":     [r.to_dict() for r in scenario_routes] if scenario_routes else [],
        "scenario_error":      scenario_error,
        "rerouted":            False,
        "cost_premium_pct":    0.0,
        "lead_time_delta_days": 0.0,
    }

    if scenario_routes:
        scen_best = scenario_routes[0]
        result["rerouted"] = (scen_best.path != base_best.path)
        result["cost_premium_pct"] = round(
            (scen_best.cost - base_best.cost) / (base_best.cost + 1e-9) * 100, 1
        )
        result["lead_time_delta_days"] = round(
            scen_best.lead_time_days - base_best.lead_time_days, 1
        )

    return result
