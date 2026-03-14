"""
NetworkX graph builder.
Constructs a directed weighted graph from the complete edge matrix.

Primary path: 5 graphs (one per product, latest year only) for the dashboard.
Legacy path: build_all_graphs() for notebooks / full historical analysis.
"""

import os
import sys
import pickle

import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    EDGES_PATH, YEARS, LATEST_YEAR, PRODUCT_CODES, PROCESSED_DIR,
    COUNTRY_COORDS
)
from src.data.loaders import load_country_lsci, load_merchant_fleet


GRAPHS_CACHE_PATH        = os.path.join(PROCESSED_DIR, "graphs_cache.pkl")
LATEST_GRAPHS_CACHE_PATH = os.path.join(PROCESSED_DIR, "graphs_latest.pkl")


def _load_node_attributes() -> dict[str, dict]:
    """
    Build a dict mapping country → {lsci_by_year, fleet_pct_by_year, lat, lon}.
    Used to annotate graph nodes.
    """
    clsci = load_country_lsci()
    fleet = load_merchant_fleet()

    attrs: dict[str, dict] = {}

    for _, row in clsci.iterrows():
        c = row["country"]
        if c not in attrs:
            attrs[c] = {"lsci": {}, "fleet_pct": {}, "lat": None, "lon": None}
        attrs[c]["lsci"][int(row["year"])] = float(row["country_lsci"] or 0)

    for _, row in fleet.iterrows():
        c = row["country"]
        if c not in attrs:
            attrs[c] = {"lsci": {}, "fleet_pct": {}, "lat": None, "lon": None}
        attrs[c]["fleet_pct"][int(row["year"])] = float(row["fleet_pct"] or 0)

    for country, (lat, lon) in COUNTRY_COORDS.items():
        if country not in attrs:
            attrs[country] = {"lsci": {}, "fleet_pct": {}, "lat": None, "lon": None}
        attrs[country]["lat"] = lat
        attrs[country]["lon"] = lon

    return attrs


def build_graph(
    edges_df: pd.DataFrame,
    year: int,
    product_code: int,
    node_attrs: dict | None = None,
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph for a specific (year, product_code).

    Parameters
    ----------
    edges_df    : complete edge matrix (from graph_edges_full.parquet)
    year        : integer year (2016-2021)
    product_code: HS product code
    node_attrs  : optional pre-computed node attribute dict (for speed)

    Returns
    -------
    nx.DiGraph with:
      - edge attrs: weight (freight_rate), bilateral_lsci, is_predicted
      - node attrs: lsci, fleet_pct, lat, lon
    """
    subset = edges_df[
        (edges_df["year"] == year) &
        (edges_df["product_code"] == product_code) &
        edges_df["freight_rate"].notna() &
        (edges_df["freight_rate"] >= 0)
    ].copy()

    G = nx.DiGraph()

    for _, row in subset.iterrows():
        origin = str(row["origin"])
        dest   = str(row["destination"])
        if origin == dest:
            continue
        G.add_edge(
            origin, dest,
            weight=float(row["freight_rate"]),
            bilateral_lsci=float(row.get("bilateral_lsci", 0.0) or 0.0),
            is_predicted=bool(row.get("is_predicted", False)),
        )

    # Annotate nodes
    if node_attrs is None:
        node_attrs = _load_node_attributes()

    for node in G.nodes():
        na = node_attrs.get(node, {})
        G.nodes[node]["lsci"]      = na.get("lsci", {}).get(year, 0.0)
        G.nodes[node]["fleet_pct"] = na.get("fleet_pct", {}).get(year, 0.0)
        G.nodes[node]["lat"]       = na.get("lat")
        G.nodes[node]["lon"]       = na.get("lon")

    return G


def build_latest_graphs(
    edges_df: pd.DataFrame | None = None,
    save_cache: bool = True,
) -> dict[tuple[int, int], nx.DiGraph]:
    """
    Build 5 graphs — one per product for LATEST_YEAR only.
    This is the primary path used by the Streamlit dashboard.

    Returns dict keyed by (LATEST_YEAR, product_code).
    """
    if edges_df is None:
        if not os.path.exists(EDGES_PATH):
            raise FileNotFoundError(
                f"Edge matrix not found at {EDGES_PATH}. "
                "Run src/models/train_xgb.py first."
            )
        edges_df = pd.read_parquet(EDGES_PATH)

    print(f"Building node attribute lookup (year={LATEST_YEAR})...")
    node_attrs = _load_node_attributes()

    graphs = {}
    for prod in PRODUCT_CODES:
        G = build_graph(edges_df, LATEST_YEAR, prod, node_attrs=node_attrs)
        graphs[(LATEST_YEAR, prod)] = G
        print(f"  ({LATEST_YEAR}, {prod:4d}) → {G.number_of_nodes():3d} nodes, "
              f"{G.number_of_edges():6d} edges")

    if save_cache:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(LATEST_GRAPHS_CACHE_PATH, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Latest graphs cached → {LATEST_GRAPHS_CACHE_PATH}")

    return graphs


def load_latest_graphs_cache() -> dict[tuple[int, int], nx.DiGraph]:
    """Load pre-built latest-year graph cache (5 graphs) from disk."""
    if not os.path.exists(LATEST_GRAPHS_CACHE_PATH):
        raise FileNotFoundError(
            f"Latest graph cache not found at {LATEST_GRAPHS_CACHE_PATH}. "
            "Run build_latest_graphs() first."
        )
    with open(LATEST_GRAPHS_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def build_all_graphs(
    edges_df: pd.DataFrame | None = None,
    save_cache: bool = True,
) -> dict[tuple[int, int], nx.DiGraph]:
    """
    Build all 30 graphs (6 years × 5 products).
    Used by historical notebooks. Dashboard uses build_latest_graphs() instead.

    Returns dict keyed by (year, product_code).
    """
    if edges_df is None:
        if not os.path.exists(EDGES_PATH):
            raise FileNotFoundError(
                f"Edge matrix not found at {EDGES_PATH}. "
                "Run src/models/train_xgb.py first."
            )
        edges_df = pd.read_parquet(EDGES_PATH)

    print("Building node attribute lookup...")
    node_attrs = _load_node_attributes()

    graphs = {}
    for yr in YEARS:
        for prod in PRODUCT_CODES:
            G = build_graph(edges_df, yr, prod, node_attrs=node_attrs)
            graphs[(yr, prod)] = G
            print(f"  ({yr}, {prod:4d}) → {G.number_of_nodes():3d} nodes, "
                  f"{G.number_of_edges():6d} edges")

    if save_cache:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(GRAPHS_CACHE_PATH, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Graphs cached → {GRAPHS_CACHE_PATH}")

    return graphs


def load_graphs_cache() -> dict[tuple[int, int], nx.DiGraph]:
    """Load pre-built full graph cache from disk (30 graphs)."""
    if not os.path.exists(GRAPHS_CACHE_PATH):
        raise FileNotFoundError(
            f"Graph cache not found at {GRAPHS_CACHE_PATH}. "
            "Run build_all_graphs() first (notebook 04 or builder.py)."
        )
    with open(GRAPHS_CACHE_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_latest_graphs(save_cache=True)
