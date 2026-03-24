"""
Routing engine — Dijkstra + Yen's K-shortest paths with scenario simulation.
"""

import functools
import itertools
import math
import sys
import os
from typing import Optional

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    K_ROUTES, MAX_HOPS, MAX_HOPS_FALLBACK, COUNTRY_COORDS,
    MARITIME_WAYPOINTS, MARITIME_EDGES, COUNTRY_PORT_WAYPOINT, CHOKEPOINT_WAYPOINTS,
)
from src.graph.chokepoints import get_countries_to_remove, chokepoint_exposure

# Average container ship speed: 15 knots = 27.78 km/h = 666.7 km/day
_SHIP_SPEED_KM_PER_DAY = 666.7
# Port handling time per intermediate transshipment stop
# (unloading ~0.5d + storage/customs ~1.5d + reloading ~0.5d)
_PORT_DAYS_PER_STOP = 2.5
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


# ─── Maritime graph utilities ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _get_maritime_graph(blocked_wps: frozenset = frozenset()) -> nx.Graph:
    """
    Build (and cache) a maritime waypoint graph, optionally with waypoints
    removed for chokepoint scenarios.  The base graph (no blockings) is built
    once and reused; each unique blocked_wps combination is cached separately.
    """
    MG = nx.Graph()
    for node in MARITIME_WAYPOINTS:
        MG.add_node(node)
    for u, v in MARITIME_EDGES:
        cu, cv = MARITIME_WAYPOINTS[u], MARITIME_WAYPOINTS[v]
        dist = _haversine_km_raw(cu[0], cu[1], cv[0], cv[1])
        MG.add_edge(u, v, weight=dist)
    for wp in blocked_wps:
        if MG.has_node(wp):
            MG.remove_node(wp)
    return MG


def _country_waypoint(country: str) -> str | None:
    """Return maritime entry waypoint key for a country."""
    if country in COUNTRY_PORT_WAYPOINT:
        return COUNTRY_PORT_WAYPOINT[country]
    coord = COUNTRY_COORDS.get(country)
    if not coord:
        return None
    return min(
        MARITIME_WAYPOINTS.items(),
        key=lambda kv: _haversine_km_raw(coord[0], coord[1], kv[1][0], kv[1][1]),
    )[0]


def _maritime_leg_km(u: str, v: str, blocked_wps: frozenset = frozenset()) -> float:
    """
    Realistic maritime sailing distance in km for a trade leg u→v.

    Uses the waypoint graph (Dijkstra) rather than a direct haversine that
    would cut straight across land masses.  If a chokepoint is blocked,
    passes ``blocked_wps`` to force the path around the closure.
    Falls back to direct haversine when country waypoints are unknown.
    """
    MG = _get_maritime_graph(blocked_wps)
    wp_u = _country_waypoint(u)
    wp_v = _country_waypoint(v)

    if not wp_u or not wp_v:
        cu, cv = COUNTRY_COORDS.get(u), COUNTRY_COORDS.get(v)
        return _haversine_km_raw(cu[0], cu[1], cv[0], cv[1]) if cu and cv else 5_000.0

    try:
        wp_dist = nx.shortest_path_length(MG, wp_u, wp_v, weight="weight")
        # Add legs: country centroid → port waypoint on each end
        cu = COUNTRY_COORDS.get(u)
        cv = COUNTRY_COORDS.get(v)
        cwp_u = MARITIME_WAYPOINTS[wp_u]
        cwp_v = MARITIME_WAYPOINTS[wp_v]
        if cu:
            wp_dist += _haversine_km_raw(cu[0], cu[1], cwp_u[0], cwp_u[1])
        if cv:
            wp_dist += _haversine_km_raw(cv[0], cv[1], cwp_v[0], cwp_v[1])
        return wp_dist
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 40_000.0   # chokepoint makes route impossible → very long distance


def _apply_detour_penalty(H: nx.DiGraph, blocked_chokepoints: list[str]) -> None:
    """
    Reprice every edge whose normal maritime path crosses a blocked chokepoint.

    ML-imputation creates direct edges between distant country pairs (e.g.
    China→Germany) that never explicitly pass through Egypt.  Removing Egypt
    alone doesn't raise their cost.  This function detects that the maritime
    path for such an edge would normally transit the blocked waypoints and
    multiplies the edge weight by the detour ratio
    (alternative_distance / normal_distance).

    Example: Suez blocked → China→Germany edge ×1.35 (Cape route is 35% longer).
    Modifies H in-place.
    """
    blocked_wps = frozenset(
        wp for cp in blocked_chokepoints
        for wp in CHOKEPOINT_WAYPOINTS.get(cp, [])
    )
    if not blocked_wps:
        return

    MG_normal  = _get_maritime_graph(frozenset())
    MG_blocked = _get_maritime_graph(blocked_wps)

    # Cache detour ratios per (wp_u, wp_v) pair — only ~34 waypoints, cheap
    _ratio_cache: dict[tuple, float] = {}

    def _ratio(wp_u: str, wp_v: str) -> float:
        key = (min(wp_u, wp_v), max(wp_u, wp_v))
        if key in _ratio_cache:
            return _ratio_cache[key]
        try:
            normal_path = nx.shortest_path(MG_normal, wp_u, wp_v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            _ratio_cache[key] = 1.0
            return 1.0
        # Only reprice if the normal path goes through a blocked waypoint
        if not any(wp in blocked_wps for wp in normal_path):
            _ratio_cache[key] = 1.0
            return 1.0
        try:
            nd = nx.shortest_path_length(MG_normal,  wp_u, wp_v, weight="weight")
            dd = nx.shortest_path_length(MG_blocked, wp_u, wp_v, weight="weight")
            r = dd / max(nd, 1.0)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            r = 2.5   # can't route at all → heavy penalty
        _ratio_cache[key] = r
        return r

    for u, v in list(H.edges()):
        wp_u = _country_waypoint(u)
        wp_v = _country_waypoint(v)
        if not wp_u or not wp_v or wp_u == wp_v:
            continue
        r = _ratio(wp_u, wp_v)
        if r > 1.001:
            H[u][v]["weight"] = H[u][v]["weight"] * r


# ─── Haversine (private name used before class definitions) ───────────────────

def _haversine_km_raw(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (used by maritime utilities above)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


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


# ─── Maritime chokepoint exposure ────────────────────────────────────────────

_ALL_CHOKEPOINT_WPS: frozenset = frozenset(
    wp for wps in CHOKEPOINT_WAYPOINTS.values() for wp in wps
)


def maritime_chokepoint_exposure(path: list[str]) -> float:
    """
    Fraction of added sailing distance if all major chokepoints were simultaneously
    blocked (Suez, Panama, Hormuz, Malacca).

    Measures actual maritime route dependency, not intermediate country nodes:
      - China → US (transpacific): ~0%  — Pacific route, no chokepoints used
      - China → Germany:          ~35%  — Malacca + Suez, detours around Cape
      - India → Germany:          ~15%  — Suez dependency
      - US → Japan (via Pacific): ~0%   — no chokepoints on Pacific crossing

    Returns a value in [0, 1] where higher = more exposed to chokepoint disruption.
    """
    if len(path) < 2:
        return 0.0
    legs = list(zip(path[:-1], path[1:]))
    normal_km  = sum(_maritime_leg_km(u, v, frozenset())         for u, v in legs)
    blocked_km = sum(_maritime_leg_km(u, v, _ALL_CHOKEPOINT_WPS) for u, v in legs)
    if normal_km <= 0:
        return 0.0
    return float(min((blocked_km - normal_km) / normal_km, 1.0))


# ─── Route result dataclass ───────────────────────────────────────────────────

class Route:
    """Represents a single route result."""

    def __init__(self, path: list[str], cost: float, graph: nx.DiGraph,
                 median_lsci: float = 50.0,
                 blocked_wps: frozenset = frozenset()):
        self.path           = path
        self.cost           = cost
        self.hops           = len(path) - 1
        self.chk_exposure   = maritime_chokepoint_exposure(path)
        self._blocked_wps   = blocked_wps
        self.lead_time_days = self._estimate_lead_time(graph, median_lsci)
        self.has_predicted  = self._check_predicted(graph)

    def _estimate_lead_time(self, G: nx.DiGraph, median_lsci: float) -> float:
        """
        Lead time = total maritime sailing distance / avg ship speed + port handling.

        Methodology
        -----------
        - Maritime waypoint graph distance for each leg, routing through actual
          sea lanes (straits, canals, ocean crossings) rather than great-circle
          haversine which cuts across land masses.
        - When a chokepoint is blocked, the maritime graph is adjusted so the
          distance reflects the detour (e.g. Cape of Good Hope when Suez is
          blocked), giving a physically accurate lead time.
        - Average container ship speed: 15 knots ≈ 667 km/day.
        - Port handling: 2.5 days per intermediate transshipment stop
          (unloading ~0.5d + storage/customs ~1.5d + reloading ~0.5d).
        """
        days = 0.0
        for u, v in zip(self.path[:-1], self.path[1:]):
            dist_km = _maritime_leg_km(u, v, self._blocked_wps)
            days += dist_km / _SHIP_SPEED_KM_PER_DAY
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
        # Reprice all edges whose maritime path crosses the blocked chokepoint.
        # This ensures ML-predicted direct edges (e.g. China→Germany) correctly
        # reflect the detour cost — the router can no longer "ignore" Suez by
        # using a straight imputed edge that doesn't pass through Egypt.
        _apply_detour_penalty(H, blocked_chokepoints)

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
    blocked_wps: frozenset = frozenset(),
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
            found.append(Route(path, cost, G, median_lsci, blocked_wps))
            if len(found) == k:
                break
        return found

    # Phase 1 – Direct (0 intermediate hubs)
    routes: list[Route] = []
    if G_hubs.has_edge(source, target):
        cost = G[source][target]["weight"] if G.has_edge(source, target) else 0.0
        routes.append(Route([source, target], cost, G, median_lsci, blocked_wps))

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
    blocked_wps: frozenset = frozenset(),
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
        G, source, target, k=k_candidates, cutoff=cutoff,
        median_lsci=median_lsci, blocked_wps=blocked_wps,
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
        "_candidates":    candidates,   # full scored pool for downstream filtering
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
