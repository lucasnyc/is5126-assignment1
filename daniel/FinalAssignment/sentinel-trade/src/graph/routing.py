"""
Routing engine — Dijkstra + Yen's K-shortest paths with scenario simulation.
"""

import itertools
import sys
import os
from typing import Optional

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import K_ROUTES, MAX_HOPS
from src.graph.chokepoints import get_countries_to_remove, chokepoint_exposure


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
        Heuristic lead time: 7 base days per hop, adjusted by destination LSCI.
        Higher destination LSCI → more frequent services → faster effective transit.
        """
        days = 0.0
        for v in self.path[1:]:
            node_lsci = G.nodes[v].get("lsci", median_lsci) or median_lsci
            days += 7.0 / (node_lsci / median_lsci + 0.5)
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

    Parameters
    ----------
    G           : NetworkX DiGraph (after any scenario modifications)
    source      : origin country name
    target      : destination country name
    k           : number of paths to return
    cutoff      : maximum number of hops (prevents exponential search)
    median_lsci : used for lead time heuristic

    Returns
    -------
    List of Route objects in non-decreasing cost order.

    Raises
    ------
    nx.NodeNotFound      if source or target not in graph
    nx.NetworkXNoPath    if no path exists between source and target
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source node '{source}' not in graph.")
    if target not in G:
        raise nx.NodeNotFound(f"Target node '{target}' not in graph.")

    routes = []
    path_gen = nx.shortest_simple_paths(G, source, target, weight="weight")

    for path in itertools.islice(path_gen, k * 5):  # over-fetch then filter by cutoff
        if len(path) - 1 > cutoff:
            continue
        cost = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
        routes.append(Route(path, cost, G, median_lsci))
        if len(routes) == k:
            break

    if not routes:
        raise nx.NetworkXNoPath(
            f"No path from '{source}' to '{target}' within {cutoff} hops."
        )

    return routes


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
