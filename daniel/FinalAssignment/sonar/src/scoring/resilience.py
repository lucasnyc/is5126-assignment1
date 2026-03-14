"""
Resilience Score — a composite 0-100 index for supply chain route resilience.

Formula:
    RS = 100 × (0.47 × Alt + 0.28 × Chk + 0.17 × Bil + 0.07 × Fleet)

Components:
    Alt   — Route Redundancy: cost premium of 2nd-best path vs optimal
    Chk   — Chokepoint Avoidance: fraction of route avoiding high-BC nodes
    Bil   — Bilateral Connectivity: normalised bilateral LSCI along route
    Fleet — Fleet Availability: normalised fleet ownership on corridor

Weights derived via Analytic Hierarchy Process (AHP, Saaty 1980).
Consistency Ratio CR = 0.019 < 0.10.  See config.py for the full matrix.
"""

import os
import sys
import pickle

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RS_WEIGHT_ALT, RS_WEIGHT_BIL, RS_WEIGHT_CHK, RS_WEIGHT_FLEET,
    ALL_CHOKEPOINT_COUNTRIES, CONSTANTS_PATH,
)
from src.graph.chokepoints import chokepoint_exposure


def _load_constants() -> dict:
    if not os.path.exists(CONSTANTS_PATH):
        # Fallback defaults so the app doesn't crash before training
        return {
            "bilateral_lsci_p95": 0.30,
            "median_fleet_pct":   0.50,
            "median_lsci":        50.0,
        }
    with open(CONSTANTS_PATH, "rb") as f:
        return pickle.load(f)


class ResilienceScorer:
    """
    Computes the Resilience Score for a route on a given graph.

    Parameters
    ----------
    constants : dict with keys bilateral_lsci_p95, median_fleet_pct, median_lsci
                If None, loaded from CONSTANTS_PATH.
    """

    def __init__(self, constants: dict | None = None):
        if constants is None:
            constants = _load_constants()
        self.bilateral_lsci_p95 = max(constants.get("bilateral_lsci_p95", 0.30), 1e-6)
        self.median_fleet_pct   = max(constants.get("median_fleet_pct", 0.50), 1e-6)
        self.median_lsci        = max(constants.get("median_lsci", 50.0), 1e-6)

    # ── Component calculators ─────────────────────────────────────────────────

    def _alt_component(self, cost_k1: float, cost_k2: float | None) -> float:
        """
        Alt ∈ [0, 1].
        Measures how viable the second-best route is relative to the cheapest.

        With routes constrained to ≤2 hops (direct or single transshipment),
        alternatives route through geographically comparable hubs.  A cost
        premium of 100 % (k2 costs 2× k1) is therefore already a strong
        signal of poor redundancy — so the cap is set at 100 % rather than
        the 200 % used when long multi-hop detours are permitted.

        premium = 0 %   → Alt = 1.0  (perfect substitutes)
        premium = 50 %  → Alt = 0.5
        premium ≥ 100 % → Alt = 0.0  (effectively no viable alternative)
        No k2 route     → Alt = 0.0
        """
        if cost_k2 is None or cost_k2 <= 0:
            return 0.0
        premium = (cost_k2 - cost_k1) / (cost_k1 + 1e-9)
        return float(max(0.0, 1.0 - premium))

    def _bil_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Bil ∈ [0, 1].
        Average bilateral LSCI along path edges, normalized by 95th-percentile.
        """
        if len(path) < 2:
            return 0.0
        vals = []
        for u, v in zip(path[:-1], path[1:]):
            edge_data = G.edges[u, v] if G.has_edge(u, v) else {}
            vals.append(float(edge_data.get("bilateral_lsci", 0.0) or 0.0))
        return float(min(1.0, np.mean(vals) / self.bilateral_lsci_p95))

    def _chk_component(self, path: list[str]) -> float:
        """
        Chk ∈ [0, 1].
        1 − chokepoint_exposure(path).
        """
        return float(1.0 - chokepoint_exposure(path))

    def _fleet_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Fleet ∈ [0, 1].
        Average fleet_pct across path nodes, normalized by median_fleet_pct
        (capped at 1).
        """
        if not path:
            return 0.0
        vals = [float(G.nodes[n].get("fleet_pct", 0.0) or 0.0) for n in path]
        mean_fleet = np.mean(vals)
        return float(min(1.0, mean_fleet / self.median_fleet_pct))

    # ── Main scorer ───────────────────────────────────────────────────────────

    def score(
        self,
        path_k1:    list[str],
        cost_k1:    float,
        G:          nx.DiGraph,
        path_k2:    list[str] | None = None,
        cost_k2:    float | None = None,
    ) -> dict:
        """
        Compute Resilience Score and all component values.

        Parameters
        ----------
        path_k1  : best route (list of country names)
        cost_k1  : freight cost of best route
        G        : NetworkX DiGraph (with node/edge attributes)
        path_k2  : second-best route (optional)
        cost_k2  : cost of second-best route (optional)

        Returns
        -------
        dict with keys: score (0-100), alt, bil, chk, fleet, components_pct
        """
        alt   = self._alt_component(cost_k1, cost_k2)
        bil   = self._bil_component(path_k1, G)
        chk   = self._chk_component(path_k1)
        fleet = self._fleet_component(path_k1, G)

        raw_score = (
            RS_WEIGHT_ALT   * alt
            + RS_WEIGHT_BIL   * bil
            + RS_WEIGHT_CHK   * chk
            + RS_WEIGHT_FLEET * fleet
        )
        final_score = round(float(np.clip(raw_score * 100, 0, 100)), 1)

        return {
            "score":          final_score,
            "alt":            round(alt,   3),
            "bil":            round(bil,   3),
            "chk":            round(chk,   3),
            "fleet":          round(fleet, 3),
            "label":          _score_label(final_score),
            "components_pct": {
                "Redundancy":    round(alt   * RS_WEIGHT_ALT   * 100, 1),
                "Connectivity":  round(bil   * RS_WEIGHT_BIL   * 100, 1),
                "Chokepoint":    round(chk   * RS_WEIGHT_CHK   * 100, 1),
                "Fleet":         round(fleet * RS_WEIGHT_FLEET * 100, 1),
            },
        }

    def score_from_routes(
        self,
        routes: list,  # list of Route objects from routing.py
        G: nx.DiGraph,
    ) -> dict:
        """Convenience wrapper: compute score from a list of Route objects."""
        if not routes:
            return {"score": 0.0, "label": "Unknown"}
        k1 = routes[0]
        k2 = routes[1] if len(routes) >= 2 else None
        return self.score(
            path_k1=k1.path,
            cost_k1=k1.cost,
            G=G,
            path_k2=k2.path if k2 else None,
            cost_k2=k2.cost if k2 else None,
        )


def _score_label(score: float) -> str:
    if score >= 75:
        return "High Resilience"
    elif score >= 50:
        return "Moderate Resilience"
    elif score >= 25:
        return "Low Resilience"
    else:
        return "Critical Risk"


def sensitivity_analysis(scorer: ResilienceScorer,
                          path_k1: list[str],
                          cost_k1: float,
                          G: nx.DiGraph,
                          path_k2: list[str] | None = None,
                          cost_k2: float | None = None,
                          delta: float = 0.10) -> dict:
    """
    Vary each weight by ±delta and report the resulting score range.
    Used in notebook 05 to validate formula stability.

    Returns dict: {component: (score_minus, score_base, score_plus)}
    """
    base_result = scorer.score(path_k1, cost_k1, G, path_k2, cost_k2)
    base_score  = base_result["score"]
    weights     = {
        "alt":   RS_WEIGHT_ALT,
        "bil":   RS_WEIGHT_BIL,
        "chk":   RS_WEIGHT_CHK,
        "fleet": RS_WEIGHT_FLEET,
    }
    results = {"base_score": base_score, "perturbations": {}}
    for comp, w in weights.items():
        for sign, label in [(1.0, "plus"), (-1.0, "minus")]:
            # Perturb this weight, re-normalize others to sum to 1
            new_w     = w + sign * delta * w
            remaining = 1.0 - new_w
            other_sum = sum(v for k, v in weights.items() if k != comp)
            if other_sum < 1e-9:
                continue
            scale = remaining / other_sum
            new_weights = {k: (new_w if k == comp else v * scale)
                           for k, v in weights.items()}
            alt   = scorer._alt_component(cost_k1, cost_k2)
            bil   = scorer._bil_component(path_k1, G)
            chk   = scorer._chk_component(path_k1)
            fleet = scorer._fleet_component(path_k1, G)
            s = (new_weights["alt"] * alt + new_weights["bil"] * bil +
                 new_weights["chk"] * chk + new_weights["fleet"] * fleet) * 100
            if comp not in results["perturbations"]:
                results["perturbations"][comp] = {}
            results["perturbations"][comp][label] = round(float(s), 1)

    return results
