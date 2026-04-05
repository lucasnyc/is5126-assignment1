"""
Resilience Score — a composite 0-100 index for supply chain route resilience.

5-factor model aligned with maritime resilience literature:

    RS = 100 × (Reliability × Redundancy × Weather × Ports × Security)^0.20

Geometric mean aggregation (equal weights, 0.20 each). Non-compensatory:
if any single dimension approaches 0, the total score collapses to 0 regardless
of how well the other dimensions score.

Each factor uses a single input — no internal sub-weights:

    Reliability  = mean(on-time delivery rate across path countries)
    Redundancy   = mean(LSCI / LSCI_p95 across path countries)
    Weather      = 1 − mean(weather severity across path countries)
    Ports        = mean(TEU / TEU_p95 across path countries)
    Security     = 1 − mean(Geopolitical Risk Index across path countries)

See config.py for references.
"""

import os
import sys
import pickle

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RS_WEIGHT_REL, RS_WEIGHT_FLEX, RS_WEIGHT_ENV,
    RS_WEIGHT_PORT, RS_WEIGHT_SEC,
    CONSTANTS_PATH,
)
def _load_constants() -> dict:
    if not os.path.exists(CONSTANTS_PATH):
        return {
            "teu_p95":                 1e7,
            "lsci_p95":                100.0,
            "weather_severity_median": 0.10,
            "rel_median":              0.87,
            "sec_median_gri":          0.50,
        }
    with open(CONSTANTS_PATH, "rb") as f:
        return pickle.load(f)


class ResilienceScorer:
    """
    Computes the 5-factor Resilience Score for a route on a given graph.

    Parameters
    ----------
    constants : dict  (if None, loaded from CONSTANTS_PATH)
    """

    def __init__(self, constants: dict | None = None):
        if constants is None:
            constants = _load_constants()
        self.teu_p95                = max(constants.get("teu_p95",  1e7),   1e-6)
        self.lsci_p95               = max(constants.get("lsci_p95", 100.0), 1e-6)
        self.weather_severity_median = constants.get("weather_severity_median", 0.10)
        self.rel_median             = constants.get("rel_median",    0.87)
        self.sec_median_gri         = constants.get("sec_median_gri", 0.50)

    # ── Component calculators ─────────────────────────────────────────────────

    def _rel_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Reliability ∈ [0, 1] — mean on-time delivery rate across path countries.
        Countries not in the disruption dataset use the dataset median OTD rate.
        """
        if not path:
            return self.rel_median
        vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            otd = nd.get("otd_rate")
            vals.append(otd if otd is not None else self.rel_median)
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _flex_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Redundancy ∈ [0, 1] — mean normalised LSCI across path countries.

        LSCI (Liner Shipping Connectivity Index, UNCTAD) measures how many
        shipping services, companies, and vessel sizes serve each country.
        Higher LSCI = more real alternatives exist if the primary route fails.
        Normalised to the 95th-percentile LSCI observed in the dataset.
        """
        if not path:
            return 0.5
        vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            lsci = float(nd.get("lsci", 0.0) or 0.0)
            vals.append(min(lsci / self.lsci_p95, 1.0))
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _env_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Env ∈ [0, 1] — Weather Safety.
        1 − mean weather severity across path nodes.
        Higher = calmer weather along route.
        """
        if not path:
            return 0.5
        vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            sev = nd.get("weather_severity")
            vals.append(sev if sev is not None else self.weather_severity_median)
        return float(np.clip(1.0 - np.mean(vals), 0.0, 1.0))

    def _port_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Ports ∈ [0, 1] — mean normalised TEU capacity across path countries.
        TEU throughput (UNCTAD) proxies port infrastructure size.
        Normalised to the 95th-percentile TEU observed in the dataset.
        """
        if not path:
            return 0.5
        vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            teu = float(nd.get("teu", 0.0) or 0.0)
            vals.append(min(teu / self.teu_p95, 1.0))
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _sec_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Security ∈ [0, 1] — 1 − mean Geopolitical Risk Index across path countries.
        GRI (from the disruption dataset) captures conflict, political instability,
        and trade friction. Higher = safer route.
        """
        if not path:
            return 1.0 - self.sec_median_gri
        vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            gri = nd.get("mean_gri")
            vals.append(gri if gri is not None else self.sec_median_gri)
        return float(np.clip(1.0 - np.mean(vals), 0.0, 1.0))

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

        Returns
        -------
        dict with keys: score (0-100), rel, flex, env, port, sec,
                        label, components_pct
        """
        rel  = self._rel_component(path_k1, G)
        flex = self._flex_component(path_k1, G)
        env  = self._env_component(path_k1, G)
        port = self._port_component(path_k1, G)
        sec  = self._sec_component(path_k1, G)

        # Geometric mean: RS = 100 × ∏(C_i ^ w_i)
        # Non-compensatory — any component near 0 collapses the total score.
        # With equal weights (0.20 each) this reduces to the 5th-root of the product.
        raw_score = (
            (rel  ** RS_WEIGHT_REL)
            * (flex ** RS_WEIGHT_FLEX)
            * (env  ** RS_WEIGHT_ENV)
            * (port ** RS_WEIGHT_PORT)
            * (sec  ** RS_WEIGHT_SEC)
        )
        final_score = round(float(np.clip(raw_score * 100, 0, 100)), 1)

        return {
            "score":          final_score,
            "rel":            round(rel,  3),
            "flex":           round(flex, 3),
            "env":            round(env,  3),
            "port":           round(port, 3),
            "sec":            round(sec,  3),
            "label":          _score_label(final_score),
            "components_pct": {
                "Reliability": round(rel  * 100, 1),
                "Redundancy":  round(flex * 100, 1),
                "Weather":     round(env  * 100, 1),
                "Ports":       round(port * 100, 1),
                "Security":    round(sec  * 100, 1),
            },
        }

    def score_from_routes(
        self,
        routes: list,
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
    Returns dict: {component: {plus: score, minus: score}}
    """
    base_result = scorer.score(path_k1, cost_k1, G, path_k2, cost_k2)
    base_score  = base_result["score"]
    weights     = {
        "rel":  RS_WEIGHT_REL,
        "flex": RS_WEIGHT_FLEX,
        "env":  RS_WEIGHT_ENV,
        "port": RS_WEIGHT_PORT,
        "sec":  RS_WEIGHT_SEC,
    }

    # Compute raw component values
    rel  = scorer._rel_component(path_k1, G)
    flex = scorer._flex_component(path_k1, G)
    env  = scorer._env_component(path_k1, G)
    port = scorer._port_component(path_k1, G)
    sec  = scorer._sec_component(path_k1, G)
    comp_vals = {"rel": rel, "flex": flex, "env": env, "port": port, "sec": sec}

    results = {"base_score": base_score, "perturbations": {}}
    for comp, w in weights.items():
        for sign, label in [(1.0, "plus"), (-1.0, "minus")]:
            new_w     = w + sign * delta * w
            remaining = 1.0 - new_w
            other_sum = sum(v for k, v in weights.items() if k != comp)
            if other_sum < 1e-9:
                continue
            scale = remaining / other_sum
            new_weights = {k: (new_w if k == comp else v * scale)
                           for k, v in weights.items()}
            raw_s = 1.0
            for k in weights:
                raw_s *= comp_vals[k] ** new_weights[k]
            s = float(np.clip(raw_s * 100, 0, 100))
            if comp not in results["perturbations"]:
                results["perturbations"][comp] = {}
            results["perturbations"][comp][label] = round(float(s), 1)

    return results
