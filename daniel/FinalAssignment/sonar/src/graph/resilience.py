"""
Resilience Scorer for trade route corridors.

Formula:
    RS = 100 × (0.47 × Alt + 0.28 × Chk + 0.17 × Bil + 0.07 × Fleet)

Components:
    Alt   — Route Alternatives: normalised count of viable k-shortest paths
    Chk   — Chokepoint Exposure: 1 - fraction of route legs through high-BC nodes
    Bil   — Bilateral Connectivity: normalised bilateral LSCI score
    Fleet — Fleet Diversity: normalised fleet ownership diversity

Weights derived via AHP (Saaty 1980), CR = 0.019. See config.py for matrix.
"""

import os
import sys
import pickle

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RS_WEIGHT_ALT, RS_WEIGHT_BIL, RS_WEIGHT_CHK, RS_WEIGHT_FLEET,
    ALL_CHOKEPOINT_COUNTRIES, CONSTANTS_PATH
)


def _load_constants() -> dict:
    """Load normalization constants saved by the feature pipeline."""
    if os.path.exists(CONSTANTS_PATH):
        with open(CONSTANTS_PATH, "rb") as f:
            return pickle.load(f)
    # Fallback defaults if constants file not available
    return {
        "bilateral_lsci_p95": 0.5,
        "median_fleet_pct": 0.1,
        "median_lsci": 20.0,
    }


class ResilienceScorer:
    """
    Computes the Resilience Score for a trade corridor.

    Parameters
    ----------
    G : nx.DiGraph
        The shipping network graph for a specific (year, product_code).
    weights : dict, optional
        Override default component weights.
    """

    def __init__(self, G: nx.DiGraph, weights: dict | None = None):
        self.G = G
        self.constants = _load_constants()

        # Component weights
        self.w_alt   = RS_WEIGHT_ALT
        self.w_bil   = RS_WEIGHT_BIL
        self.w_chk   = RS_WEIGHT_CHK
        self.w_fleet = RS_WEIGHT_FLEET

        if weights:
            self.w_alt   = weights.get("alt",   self.w_alt)
            self.w_bil   = weights.get("bil",   self.w_bil)
            self.w_chk   = weights.get("chk",   self.w_chk)
            self.w_fleet = weights.get("fleet", self.w_fleet)

        # Pre-compute betweenness centrality (approximate, k=50) for Chk component
        self._bc: dict[str, float] = {}
        self._bc_computed = False

    def _ensure_bc(self):
        """Lazily compute betweenness centrality."""
        if not self._bc_computed:
            n = self.G.number_of_nodes()
            k = min(50, n)
            try:
                self._bc = nx.betweenness_centrality(
                    self.G, weight="weight", normalized=True, k=k
                )
            except Exception:
                self._bc = {}
            self._bc_computed = True

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def _alt_score(self, routes: list) -> float:
        """
        Normalised count of viable alternative routes.
        Score = min(n_routes / 5, 1.0), where routes with cost ≤ 2× optimal qualify.
        Returns 0 if only one route found (no alternatives).
        """
        if not routes:
            return 0.0
        optimal_cost = routes[0].cost if hasattr(routes[0], "cost") else routes[0].get("cost", 1.0)
        viable = sum(
            1 for r in routes
            if (r.cost if hasattr(r, "cost") else r.get("cost", 0)) <= optimal_cost * 2.0
        )
        # 0 alternatives → score 0; 1 alt → 0.2; 4+ alts → 1.0
        n_alternatives = max(0, viable - 1)
        return float(min(n_alternatives / 4.0, 1.0))

    def _bil_score(self, origin: str, dest: str, routes: list) -> float:
        """
        Normalised bilateral LSCI of the best route edge (origin→dest or via hub).
        Falls back to average edge bilateral_lsci along the best path.
        """
        if not routes:
            return 0.0

        best_path = routes[0].path if hasattr(routes[0], "path") else routes[0].get("path", [])
        bil_values = []
        for u, v in zip(best_path[:-1], best_path[1:]):
            if self.G.has_edge(u, v):
                val = self.G[u][v].get("bilateral_lsci", 0.0) or 0.0
                bil_values.append(float(val))

        if not bil_values:
            return 0.0

        mean_bil = float(np.mean(bil_values))
        p95 = self.constants.get("bilateral_lsci_p95", 0.5)
        if p95 <= 0:
            return 0.0
        return float(min(mean_bil / p95, 1.0))

    def _chk_score(self, routes: list) -> float:
        """
        Chokepoint exposure score: 1 - fraction of intermediate nodes
        that are high-betweenness chokepoints.
        Higher = less exposure = more resilient.
        """
        if not routes:
            return 1.0

        self._ensure_bc()

        best_path = routes[0].path if hasattr(routes[0], "path") else routes[0].get("path", [])
        intermediates = best_path[1:-1]

        if not intermediates:
            return 1.0

        # Two criteria for chokepoint: in ALL_CHOKEPOINT_COUNTRIES OR high betweenness
        bc_threshold = np.percentile(list(self._bc.values()), 75) if self._bc else 0.0
        chk_count = sum(
            1 for n in intermediates
            if n in ALL_CHOKEPOINT_COUNTRIES or self._bc.get(n, 0.0) >= bc_threshold
        )
        exposure = chk_count / len(intermediates)
        return float(1.0 - exposure)

    def _fleet_score(self, routes: list) -> float:
        """
        Fleet diversity: normalised fleet ownership share of economies on the route.
        Higher fleet share → more vessels → more resilient.
        """
        if not routes:
            return 0.0

        best_path = routes[0].path if hasattr(routes[0], "path") else routes[0].get("path", [])
        fleet_vals = []
        for node in best_path:
            if self.G.has_node(node):
                val = self.G.nodes[node].get("fleet_pct", 0.0) or 0.0
                fleet_vals.append(float(val))

        if not fleet_vals:
            return 0.0

        total_fleet = float(sum(fleet_vals))
        median_fleet = self.constants.get("median_fleet_pct", 0.1)
        if median_fleet <= 0:
            return 0.0
        return float(min(total_fleet / (median_fleet * len(fleet_vals) * 2), 1.0))

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute(
        self,
        origin: str,
        dest: str,
        routes: list,
        year: int,
        weights: dict | None = None,
    ) -> dict:
        """
        Compute Resilience Score for a corridor.

        Parameters
        ----------
        origin  : origin country name
        dest    : destination country name
        routes  : list of Route objects (from find_k_routes)
        year    : integer year
        weights : optional weight override dict {alt, bil, chk, fleet}

        Returns
        -------
        dict with keys: total_score, alt_score, bil_score, chk_score, fleet_score
        """
        w_alt   = self.w_alt
        w_bil   = self.w_bil
        w_chk   = self.w_chk
        w_fleet = self.w_fleet

        if weights:
            w_alt   = weights.get("alt",   w_alt)
            w_bil   = weights.get("bil",   w_bil)
            w_chk   = weights.get("chk",   w_chk)
            w_fleet = weights.get("fleet", w_fleet)

        alt_raw   = self._alt_score(routes)
        bil_raw   = self._bil_score(origin, dest, routes)
        chk_raw   = self._chk_score(routes)
        fleet_raw = self._fleet_score(routes)

        total = 100.0 * (
            w_alt   * alt_raw   +
            w_bil   * bil_raw   +
            w_chk   * chk_raw   +
            w_fleet * fleet_raw
        )
        total = float(np.clip(total, 0.0, 100.0))

        return {
            "total_score": total,
            "alt_score":   float(alt_raw   * 100),
            "bil_score":   float(bil_raw   * 100),
            "chk_score":   float(chk_raw   * 100),
            "fleet_score": float(fleet_raw * 100),
        }


# ------------------------------------------------------------------
# Sensitivity analysis
# ------------------------------------------------------------------

def sensitivity_analysis(
    scorer: ResilienceScorer,
    origin: str,
    dest: str,
    routes: list,
    year: int,
    base_weights: dict,
    perturbation: float = 0.10,
) -> dict:
    """
    Run sensitivity analysis by perturbing each weight ±perturbation.

    Parameters
    ----------
    scorer       : ResilienceScorer instance
    origin       : origin country
    dest         : destination country
    routes       : list of Route objects
    year         : integer year
    base_weights : dict {alt, bil, chk, fleet} with base weight values
    perturbation : fractional perturbation (e.g. 0.10 for ±10%)

    Returns
    -------
    dict mapping component name → {base_score, low_score, high_score}
    """
    results = {}

    # Base score
    base_rs = scorer.compute(origin, dest, routes, year, weights=base_weights)
    base_total = base_rs["total_score"]

    for comp in ["alt", "bil", "chk", "fleet"]:
        # Low weight
        low_w = dict(base_weights)
        low_w[comp] = max(0.0, base_weights[comp] * (1 - perturbation))
        low_rs = scorer.compute(origin, dest, routes, year, weights=low_w)

        # High weight
        high_w = dict(base_weights)
        high_w[comp] = base_weights[comp] * (1 + perturbation)
        high_rs = scorer.compute(origin, dest, routes, year, weights=high_w)

        results[comp] = {
            "base_score": base_total,
            "low_score":  low_rs["total_score"],
            "high_score": high_rs["total_score"],
        }

    return results
