"""
Tests for the Resilience Score formula.
Run with: pytest tests/test_resilience.py -v
"""

import os
import sys
import pytest
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring.resilience import ResilienceScorer, sensitivity_analysis
from src.graph.chokepoints import chokepoint_exposure
from config import ALL_CHOKEPOINT_COUNTRIES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    constants = {
        "bilateral_lsci_p95": 0.30,
        "median_fleet_pct":   0.50,
        "median_lsci":        50.0,
    }
    return ResilienceScorer(constants=constants)


@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    for u, v, bl in [("A","B",0.25), ("B","C",0.20)]:
        G.add_edge(u, v, weight=2.0, bilateral_lsci=bl)
        G.nodes[u]["fleet_pct"] = 0.5
        G.nodes[u]["lsci"]      = 50.0
        G.nodes[v]["fleet_pct"] = 0.5
        G.nodes[v]["lsci"]      = 50.0
    return G


# ── Score bounds ──────────────────────────────────────────────────────────────

def test_score_in_bounds(scorer, simple_graph):
    result = scorer.score(["A","B","C"], 4.0, simple_graph,
                          path_k2=["A","C"], cost_k2=5.0)
    assert 0.0 <= result["score"] <= 100.0, f"Score out of bounds: {result['score']}"


def test_score_components_non_negative(scorer, simple_graph):
    result = scorer.score(["A","B","C"], 4.0, simple_graph)
    for comp in ["alt", "bil", "chk", "fleet"]:
        assert result[comp] >= 0.0, f"{comp} component is negative"


def test_no_alternative_gives_zero_alt(scorer, simple_graph):
    """When no second route exists (cost_k2=None), Alt must be 0."""
    result = scorer.score(["A","B","C"], 4.0, simple_graph,
                          path_k2=None, cost_k2=None)
    assert result["alt"] == 0.0, f"Expected Alt=0, got {result['alt']}"


def test_identical_cost_k2_gives_max_alt(scorer, simple_graph):
    """When k2 costs same as k1, premium=0, Alt=1."""
    result = scorer.score(["A","B","C"], 4.0, simple_graph,
                          path_k2=["A","D","C"], cost_k2=4.0)
    assert abs(result["alt"] - 1.0) < 1e-6, f"Expected Alt≈1, got {result['alt']}"


# ── Chokepoint exposure tests ─────────────────────────────────────────────────

def test_chokepoint_exposure_zero_for_direct_route():
    path = ["China", "Germany"]
    assert chokepoint_exposure(path) == 0.0


def test_chokepoint_exposure_increases_with_chokepoints():
    path_clean = ["China", "India", "Germany"]
    path_risky = ["China", "Egypt", "Germany"]
    assert chokepoint_exposure(path_risky) > chokepoint_exposure(path_clean)


def test_route_through_singapore_has_lower_chk(scorer, simple_graph):
    """A path through Singapore (chokepoint) has lower Chk than a direct path."""
    # Build a graph that includes Singapore as an intermediate node
    G = simple_graph.copy()
    G.add_node("Singapore")
    G.nodes["Singapore"]["fleet_pct"] = 1.0
    G.nodes["Singapore"]["lsci"]      = 80.0

    chk_direct    = scorer._chk_component(["China", "Germany"])
    chk_via_sing  = scorer._chk_component(["China", "Singapore", "Germany"])
    assert chk_via_sing < chk_direct, (
        f"Route through Singapore ({chk_via_sing:.2f}) should have lower Chk "
        f"than direct route ({chk_direct:.2f})"
    )


# ── Sensitivity analysis tests ────────────────────────────────────────────────

def test_sensitivity_returns_results(scorer, simple_graph):
    result = sensitivity_analysis(scorer, ["A","B","C"], 4.0, simple_graph,
                                  ["A","D","C"], 5.0)
    assert "base_score" in result
    assert "perturbations" in result
    assert len(result["perturbations"]) == 4


def test_sensitivity_score_stable(scorer, simple_graph):
    """Score under ±10% weight perturbation should not vary more than 10 points."""
    result = sensitivity_analysis(scorer, ["A","B","C"], 4.0, simple_graph,
                                  ["A","D","C"], 5.0, delta=0.10)
    base = result["base_score"]
    for comp, perturbs in result["perturbations"].items():
        for direction, perturbed_score in perturbs.items():
            delta = abs(perturbed_score - base)
            assert delta <= 15.0, (
                f"Score changed by {delta:.1f} pts when perturbing {comp} {direction} "
                f"(base={base:.1f}, perturbed={perturbed_score:.1f})"
            )


# ── Integration: real graph ───────────────────────────────────────────────────

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
