"""
Resilience Score — a composite 0-100 index for supply chain route resilience.

5-factor model aligned with maritime resilience literature:

    RS = 100 × (0.37×Rel + 0.21×Flex + 0.21×Env + 0.11×Port + 0.10×Sec)

Components (explainable user labels):
    Rel  — Delivery Confidence : historical on-time rate & delay severity
    Flex — Backup Options      : route redundancy & chokepoint avoidance
    Env  — Weather Safety      : environmental stability along route
    Port — Port Health         : port infrastructure capacity & congestion
    Sec  — Security Level      : geopolitical risk & conflict exposure

Equal weights (0.20 each) across all five factors.
See config.py for the weight constants.
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
from src.graph.routing import maritime_chokepoint_exposure


def _load_constants() -> dict:
    if not os.path.exists(CONSTANTS_PATH):
        return {
            "teu_p95":                1e7,
            "weather_severity_median": 0.10,
            "rel_median":             0.87,
            "sec_median_gri":         0.50,
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
        self.teu_p95                = max(constants.get("teu_p95", 1e7), 1e-6)
        self.weather_severity_median = constants.get("weather_severity_median", 0.10)
        self.rel_median             = constants.get("rel_median", 0.87)
        self.sec_median_gri         = constants.get("sec_median_gri", 0.50)

    # ── Component calculators ─────────────────────────────────────────────────

    def _rel_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Rel ∈ [0, 1] — Delivery Confidence.
        Combines on-time delivery rate (60%) and delay severity (40%).
        Countries not in the disruption dataset use the dataset median OTD.
        """
        if not path:
            return 0.5
        # delay_median is derived so the combined formula matches rel_median at default:
        # 0.60 * rel_median + 0.40 * (1 - delay_median) = rel_median
        # → delay_median = 1 - rel_median / 0.40 * (1 - 0.60) = (1 - rel_median) / 0.40 * 0.40
        # Simplified: delay_median = 1.0 - rel_median  (the complement of the OTD gap)
        _delay_median = round(1.0 - self.rel_median, 4)  # ≈ 0.13 when rel_median=0.87
        otd_vals = []
        delay_vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            otd = nd.get("otd_rate")
            delay = nd.get("mean_delay_norm")
            otd_vals.append(otd if otd is not None else self.rel_median)
            delay_vals.append(delay if delay is not None else _delay_median)
        otd_score = float(np.mean(otd_vals))
        delay_score = float(1.0 - np.mean(delay_vals))
        return float(np.clip(0.60 * otd_score + 0.40 * delay_score, 0.0, 1.0))

    def _flex_component(
        self, cost_k1: float, cost_k2: float | None, path: list[str]
    ) -> float:
        """
        Flex ∈ [0, 1] — Backup Options.
        Combines route redundancy (60%) and chokepoint avoidance (40%).

        Route redundancy: cost premium of 2nd-best route vs best.
        premium = 0%   → alt = 1.0 (perfect substitutes)
        premium ≥ 100% → alt = 0.0 (no viable alternative)
        No k2 route    → alt = 0.0
        """
        # Alt sub-component
        if cost_k2 is None or cost_k2 <= 0:
            alt = 0.0
        else:
            premium = (cost_k2 - cost_k1) / (cost_k1 + 1e-9)
            alt = float(max(0.0, 1.0 - premium))

        # Chk sub-component — maritime detour ratio if all chokepoints blocked
        chk = float(1.0 - maritime_chokepoint_exposure(path))

        return float(np.clip(0.60 * alt + 0.40 * chk, 0.0, 1.0))

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
        Port ∈ [0, 1] — Port Health.
        Combines TEU capacity (60%, proxy for infrastructure) and
        congestion avoidance (40%, from disruption data).
        """
        if not path:
            return 0.5
        teu_vals = []
        cong_vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            teu = float(nd.get("teu", 0.0) or 0.0)
            teu_vals.append(min(teu / self.teu_p95, 1.0))
            cong = nd.get("congestion_rate")
            cong_vals.append(cong if cong is not None else 0.0)
        teu_score = float(np.mean(teu_vals))
        cong_score = float(1.0 - np.mean(cong_vals))
        return float(np.clip(0.60 * teu_score + 0.40 * cong_score, 0.0, 1.0))

    def _sec_component(self, path: list[str], G: nx.DiGraph) -> float:
        """
        Sec ∈ [0, 1] — Security Level.
        Combines geopolitical conflict rate (50%) and mean geopolitical
        risk index (50%).  Higher = safer route.
        """
        if not path:
            return 0.5
        conflict_vals = []
        gri_vals = []
        for n in path:
            nd = G.nodes.get(n, {})
            conflict = nd.get("geo_conflict_rate")
            gri = nd.get("mean_gri")
            conflict_vals.append(conflict if conflict is not None else 0.0)
            gri_vals.append(gri if gri is not None else self.sec_median_gri)
        conflict_score = float(1.0 - np.mean(conflict_vals))
        gri_score = float(1.0 - np.mean(gri_vals))
        return float(np.clip(0.50 * conflict_score + 0.50 * gri_score, 0.0, 1.0))

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
        flex = self._flex_component(cost_k1, cost_k2, path_k1)
        env  = self._env_component(path_k1, G)
        port = self._port_component(path_k1, G)
        sec  = self._sec_component(path_k1, G)

        raw_score = (
            RS_WEIGHT_REL  * rel
            + RS_WEIGHT_FLEX * flex
            + RS_WEIGHT_ENV  * env
            + RS_WEIGHT_PORT * port
            + RS_WEIGHT_SEC  * sec
        )
        # Normalise to 0–100 using a floor of 0.50.
        # The raw weighted sum clusters between 0.70–0.95 for real-world routes.
        # Mapping [0.50, 1.00] → [0, 100] spreads scores across the full scale:
        #   raw=0.50 → 0   (Critical),  raw=0.75 → 50 (Moderate),
        #   raw=0.87 → 74  (High),      raw=0.95 → 90 (High),  raw=1.00 → 100
        _FLOOR = 0.50
        final_score = round(float(np.clip((raw_score - _FLOOR) / (1.0 - _FLOOR) * 100, 0, 100)), 1)

        return {
            "score":          final_score,
            "rel":            round(rel,  3),
            "flex":           round(flex, 3),
            "env":            round(env,  3),
            "port":           round(port, 3),
            "sec":            round(sec,  3),
            "label":          _score_label(final_score),
            "components_pct": {
                "Delivery Confidence": round(rel  * RS_WEIGHT_REL  * 100, 1),
                "Backup Options":      round(flex * RS_WEIGHT_FLEX * 100, 1),
                "Weather Safety":      round(env  * RS_WEIGHT_ENV  * 100, 1),
                "Port Health":         round(port * RS_WEIGHT_PORT * 100, 1),
                "Security Level":      round(sec  * RS_WEIGHT_SEC  * 100, 1),
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
    flex = scorer._flex_component(cost_k1, cost_k2, path_k1)
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
            raw_s = sum(new_weights[k] * comp_vals[k] for k in weights)
            s = float(np.clip((raw_s - 0.50) / 0.50 * 100, 0, 100))
            if comp not in results["perturbations"]:
                results["perturbations"][comp] = {}
            results["perturbations"][comp][label] = round(float(s), 1)

    return results
