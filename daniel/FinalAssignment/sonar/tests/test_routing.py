"""
Tests for the graph routing engine.
Run with: pytest tests/test_routing.py -v
"""

import os
import sys
import pytest
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph.routing import find_k_routes, apply_scenario, Route
from src.graph.chokepoints import get_countries_to_remove, get_tariff_multipliers
from config import CHOKEPOINTS


# ── Toy graph fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def toy_graph():
    """Simple A→B→C graph with known weights."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0, bilateral_lsci=0.2, is_predicted=False)
    G.add_edge("B", "C", weight=2.0, bilateral_lsci=0.3, is_predicted=False)
    G.add_edge("A", "C", weight=10.0, bilateral_lsci=0.1, is_predicted=True)
    # Add node attributes
    for n in ["A", "B", "C"]:
        G.nodes[n]["lsci"]      = 50.0
        G.nodes[n]["fleet_pct"] = 0.5
        G.nodes[n]["lat"]       = 0.0
        G.nodes[n]["lon"]       = 0.0
    return G


@pytest.fixture
def world_graph():
    """Load real graph (2021, 8517) if pipeline has been run."""
    try:
        from src.graph.builder import load_graphs_cache
        graphs = load_graphs_cache()
        return graphs.get((2021, 8517))
    except FileNotFoundError:
        pytest.skip("Graph cache not built yet; run pipeline first.")


# ── Routing tests ─────────────────────────────────────────────────────────────

def test_dijkstra_correct_cost(toy_graph):
    """Dijkstra must prefer A→B→C (cost 3) over A→C directly (cost 10)."""
    routes = find_k_routes(toy_graph, "A", "C", k=1)
    assert len(routes) == 1
    best = routes[0]
    assert best.path == ["A", "B", "C"], f"Expected ['A','B','C'], got {best.path}"
    assert abs(best.cost - 3.0) < 1e-9, f"Expected cost 3.0, got {best.cost}"


def test_k_routes_non_decreasing(toy_graph):
    """Routes returned by Yen's must be in non-decreasing cost order."""
    routes = find_k_routes(toy_graph, "A", "C", k=2)
    assert len(routes) == 2
    assert routes[0].cost <= routes[1].cost


def test_source_not_found(toy_graph):
    """NodeNotFound raised when source is not in graph."""
    with pytest.raises(nx.NodeNotFound):
        find_k_routes(toy_graph, "Z", "C", k=1)


def test_no_path(toy_graph):
    """NetworkXNoPath raised when no path exists."""
    # Add isolated node
    toy_graph.add_node("Isolated")
    toy_graph.nodes["Isolated"]["lsci"] = 0.0
    toy_graph.nodes["Isolated"]["fleet_pct"] = 0.0
    toy_graph.nodes["Isolated"]["lat"] = 0.0
    toy_graph.nodes["Isolated"]["lon"] = 0.0
    with pytest.raises(nx.NetworkXNoPath):
        find_k_routes(toy_graph, "A", "Isolated", k=1)


# ── Scenario tests ────────────────────────────────────────────────────────────

def test_scenario_does_not_mutate_original(toy_graph):
    """apply_scenario must return a copy; original must be unchanged."""
    node_count_before = toy_graph.number_of_nodes()
    toy_graph_copy = toy_graph.copy()
    # Add "B" as a chokepoint-like target (hack for test)
    H = toy_graph_copy.copy()
    H.remove_node("B")
    assert toy_graph_copy.number_of_nodes() == node_count_before


def test_chokepoint_removal_removes_node(world_graph):
    """After blocking Suez Canal, Egypt must not be in the graph."""
    H = apply_scenario(world_graph, ["Suez Canal"], {})
    assert "Egypt" not in H.nodes(), "Egypt should be removed when Suez is blocked"


def test_chokepoint_does_not_mutate_base(world_graph):
    """apply_scenario must not mutate the base graph."""
    base_nodes = world_graph.number_of_nodes()
    _ = apply_scenario(world_graph, list(CHOKEPOINTS.keys()), {})
    assert world_graph.number_of_nodes() == base_nodes


def test_tariff_increases_edge_weight(toy_graph):
    """Applying a tariff on country B should increase edge weights involving B."""
    weight_before = toy_graph["A"]["B"]["weight"]
    H = apply_scenario(toy_graph, [], {"B": 1.5})  # 50% tariff on B
    weight_after = H["A"]["B"]["weight"]
    assert weight_after == pytest.approx(weight_before * 1.5, rel=1e-6)


# ── get_tariff_multipliers tests ──────────────────────────────────────────────

def test_tariff_multipliers_zero():
    """Zero tariffs → no countries in multiplier dict."""
    result = get_tariff_multipliers(0, 0, 0, 0)
    assert result == {}


def test_tariff_multipliers_us():
    """US 10% tariff → United States has multiplier 1.10."""
    result = get_tariff_multipliers(us_pct=10.0)
    assert "United States" in result
    assert abs(result["United States"] - 1.10) < 1e-6


# ── Route object tests ────────────────────────────────────────────────────────

def test_route_to_dict(toy_graph):
    routes = find_k_routes(toy_graph, "A", "C", k=1)
    d = routes[0].to_dict()
    assert "path" in d
    assert "cost" in d
    assert "hops" in d
    assert "lead_time_days" in d
    assert "chk_exposure" in d
    assert d["hops"] == 2
