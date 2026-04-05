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
from config import (
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
            G.nodes[n].setdefault("lsci", 50.0)
            G.nodes[n].setdefault("teu", 5e6)
            G.nodes[n].setdefault("weather_severity", 0.10)
            G.nodes[n].setdefault("otd_rate", 0.90)
            G.nodes[n].setdefault("mean_gri", 0.40)
    return G


# ── Weight consistency ───────────────────────────────────────────────────────

def test_weights_sum_to_one():
    total = RS_WEIGHT_REL + RS_WEIGHT_FLEX + RS_WEIGHT_ENV + RS_WEIGHT_PORT + RS_WEIGHT_SEC
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ── Score bounds ─────────────────────────────────────────────────────────────

def test_score_in_bounds(scorer, simple_graph):
    result = scorer.score(["A", "B", "C"], 4.0, simple_graph)
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


# ── Redundancy component tests ────────────────────────────────────────────────

def test_high_lsci_gives_high_flex(scorer):
    """Countries with high LSCI (well-connected shipping) → high redundancy."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["lsci"] = 90.0   # close to lsci_p95=100
    flex = scorer._flex_component(["A", "B"], G)
    assert flex >= 0.85, f"Expected high redundancy for high LSCI, got {flex}"


def test_low_lsci_gives_low_flex(scorer):
    """Countries with low LSCI → low redundancy."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["lsci"] = 5.0    # very low connectivity
    flex = scorer._flex_component(["A", "B"], G)
    assert flex <= 0.10, f"Expected low redundancy for low LSCI, got {flex}"


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
    rel = scorer._rel_component(["A", "B"], G)
    assert rel > 0.95, f"Expected high Rel for good OTD, got {rel}"


def test_low_otd_gives_low_rel(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["otd_rate"] = 0.60
    rel = scorer._rel_component(["A", "B"], G)
    assert rel < 0.65, f"Expected low Rel for poor OTD, got {rel}"


# ── Security component tests ────────────────────────────────────────────────

def test_high_geo_risk_lowers_sec(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["mean_gri"] = 0.80
    sec = scorer._sec_component(["A", "B"], G)
    assert sec < 0.25, f"Expected low Sec for high GRI, got {sec}"


# ── Port component tests ────────────────────────────────────────────────────

def test_large_port_gives_high_port(scorer):
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    for n in ("A", "B"):
        G.nodes[n]["teu"] = 8e6
    port = scorer._port_component(["A", "B"], G)
    assert port > 0.70, f"Expected high Port for large port, got {port}"


# ── Sensitivity analysis tests ───────────────────────────────────────────────

def test_sensitivity_returns_results(scorer, simple_graph):
    result = sensitivity_analysis(scorer, ["A", "B", "C"], 4.0, simple_graph)
    assert "base_score" in result
    assert "perturbations" in result
    assert len(result["perturbations"]) == 5


def test_sensitivity_score_stable(scorer, simple_graph):
    """Score under ±10% weight perturbation should not vary more than 15 points."""
    result = sensitivity_analysis(scorer, ["A", "B", "C"], 4.0, simple_graph,
                                  delta=0.10)
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
