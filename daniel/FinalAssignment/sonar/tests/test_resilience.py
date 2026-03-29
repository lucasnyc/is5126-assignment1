"""
Tests for the 5-factor Resilience Score formula.
Run with: pytest tests/test_resilience.py -v
"""

import os
import sys
import pytest
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring.resilience import ResilienceScorer, sensitivity_analysis
from src.graph.chokepoints import chokepoint_exposure
from config import (
    ALL_CHOKEPOINT_COUNTRIES,
    RS_WEIGHT_REL, RS_WEIGHT_FLEX, RS_WEIGHT_ENV,
    RS_WEIGHT_PORT, RS_WEIGHT_SEC,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    constants = {
        "teu_p95":                 1e7,
        "weather_severity_median": 0.10,
        "rel_median":              0.87,
        "sec_median_gri":          0.50,
    }
    return ResilienceScorer(constants=constants)


@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    for u, v, bl in [("A", "B", 0.25), ("B", "C", 0.20)]:
        G.add_edge(u, v, weight=2.0, bilateral_lsci=bl)
        for n in (u, v):
            G.nodes[n].setdefault("fleet_pct", 0.5)
            G.nodes[n].setdefault("lsci", 50.0)
            G.nodes[n].setdefault("teu", 5e6)
            G.nodes[n].setdefault("weather_severity", 0.10)
            G.nodes[n].setdefault("otd_rate", 0.90)
            G.nodes[n].setdefault("mean_delay_norm", 0.05)
            G.nodes[n].setdefault("congestion_rate", 0.05)
            G.nodes[n].setdefault("geo_conflict_rate", 0.02)
            G.nodes[n].setdefault("mean_gri", 0.40)
    return G


# ── Weight consistency ───────────────────────────────────────────────────────

def test_weights_sum_to_one():
    total = RS_WEIGHT_REL + RS_WEIGHT_FLEX + RS_WEIGHT_ENV + RS_WEIGHT_PORT + RS_WEIGHT_SEC
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ── Score bounds ─────────────────────────────────────────────────────────────

def test_score_in_bounds(scorer, simple_graph):
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph,
                          path_k2=["A", "C"], cost_k2=5.0)
    assert 0.0 <= result["score"] <= 100.0, f"Score out of bounds: {result['score']}"


def test_score_components_non_negative(scorer, simple_graph):
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph)
    for comp in ["rel", "flex", "env", "port", "sec"]:
        assert result[comp] >= 0.0, f"{comp} component is negative"
        assert result[comp] <= 1.0, f"{comp} component exceeds 1.0"


def test_score_has_all_component_labels(scorer, simple_graph):
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph)
    expected_labels = {
        "Reliability", "Redundancy", "Weather",
        "Ports", "Security",
    }
    assert set(result["components_pct"].keys()) == expected_labels


# ── Flex component tests ─────────────────────────────────────────────────────

def test_no_alternative_gives_lower_flex(scorer, simple_graph):
    """When no second route exists, flex is driven only by chk (alt=0)."""
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph,
                          path_k2=None, cost_k2=None)
    # flex should be < 1.0 since alt=0
    assert result["flex"] < 1.0


def test_identical_cost_k2_gives_max_alt(scorer, simple_graph):
    """When k2 costs same as k1, alt premium=0 → alt sub-component = 1."""
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph,
                          path_k2=["A", "D", "C"], cost_k2=4.0)
    # flex = 0.6 * 1.0 (alt) + 0.4 * chk
    assert result["flex"] >= 0.6  # at least the alt contribution


# ── Chokepoint exposure tests ────────────────────────────────────────────────

def test_chokepoint_exposure_zero_for_direct_route():
    path = ["China", "Germany"]
    assert chokepoint_exposure(path) == 0.0


def test_chokepoint_exposure_increases_with_chokepoints():
    path_clean = ["China", "India", "Germany"]
    path_risky = ["China", "Egypt", "Germany"]
    assert chokepoint_exposure(path_risky) > chokepoint_exposure(path_clean)


# ── Weather component tests ──────────────────────────────────────────────────

def test_bad_weather_lowers_env(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n, sev in [("A", 0.80), ("B", 0.70)]:
        G.nodes[n]["weather_severity"] = sev
    env = scorer._env_component(["A", "B"], G)
    assert env < 0.35, f"Expected low Env for bad weather, got {env}"


def test_good_weather_gives_high_env(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["weather_severity"] = 0.05
    env = scorer._env_component(["A", "B"], G)
    assert env > 0.90, f"Expected high Env for good weather, got {env}"


# ── Reliability component tests ──────────────────────────────────────────────

def test_high_otd_gives_high_rel(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["otd_rate"] = 0.98
        G.nodes[n]["mean_delay_norm"] = 0.01
    rel = scorer._rel_component(["A", "B"], G)
    assert rel > 0.90, f"Expected high Rel for good OTD, got {rel}"


def test_low_otd_gives_low_rel(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["otd_rate"] = 0.60
        G.nodes[n]["mean_delay_norm"] = 0.50
    rel = scorer._rel_component(["A", "B"], G)
    assert rel < 0.60, f"Expected low Rel for poor OTD, got {rel}"


# ── Security component tests ────────────────────────────────────────────────

def test_high_geo_risk_lowers_sec(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["geo_conflict_rate"] = 0.30
        G.nodes[n]["mean_gri"] = 0.80
    sec = scorer._sec_component(["A", "B"], G)
    assert sec < 0.50, f"Expected low Sec for high geo risk, got {sec}"


# ── Port component tests ────────────────────────────────────────────────────

def test_large_port_gives_high_port(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["teu"] = 8e6
        G.nodes[n]["congestion_rate"] = 0.0
    port = scorer._port_component(["A", "B"], G)
    assert port > 0.70, f"Expected high Port for large port, got {port}"


# ── Sensitivity analysis tests ───────────────────────────────────────────────

def test_sensitivity_returns_results(scorer, simple_graph):
    result = sensitivity_analysis(scorer, ["A", "B", "C"], 4.0, simple_graph,
                                  ["A", "D", "C"], 5.0)
    assert "base_score" in result
    assert "perturbations" in result
    assert len(result["perturbations"]) == 5


def test_sensitivity_score_stable(scorer, simple_graph):
    """Score under ±10% weight perturbation should not vary more than 15 points."""
    result = sensitivity_analysis(scorer, ["A", "B", "C"], 4.0, simple_graph,
                                  ["A", "D", "C"], 5.0, delta=0.10)
    base = result["base_score"]
    for comp, perturbs in result["perturbations"].items():
        for direction, perturbed_score in perturbs.items():
            d = abs(perturbed_score - base)
            assert d <= 15.0, (
                f"Score changed by {d:.1f} pts when perturbing {comp} {direction} "
                f"(base={base:.1f}, perturbed={perturbed_score:.1f})"
            )


# ── Integration: real graph ─────────────────────────────────────────────────

def test_real_route_score(scorer):
    """Score from real routing should produce a valid result."""
    try:
        from src.graph.builder import load_graphs_cache
        from src.graph.routing import find_k_routes
        graphs = load_graphs_cache()
        G = graphs[(2021, 8517)]
        routes = find_k_routes(G, "China", "Germany", k=3)
        result = scorer.score_from_routes(routes, G)
        assert 0.0 <= result["score"] <= 100.0
        assert result["label"] in [
            "High Resilience", "Moderate Resilience",
            "Low Resilience", "Critical Risk"
        ]
    except (FileNotFoundError, nx.NetworkXNoPath, nx.NodeNotFound):
        pytest.skip("Graph cache or route unavailable; run pipeline first.")
