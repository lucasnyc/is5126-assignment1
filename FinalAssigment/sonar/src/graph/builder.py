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
from src.data.loaders import (
    load_country_lsci, load_merchant_fleet, load_port_throughput,
    load_weather_severity, load_disruption_metrics,
)


GRAPHS_CACHE_PATH        = os.path.join(PROCESSED_DIR, "graphs_cache.pkl")
LATEST_GRAPHS_CACHE_PATH = os.path.join(PROCESSED_DIR, "graphs_latest.pkl")


def _get_year_attr(d: dict, year: int, fallback_year: int = 2021) -> float:
    """
    Look up a year-keyed attribute dict, falling back to fallback_year when the
    requested year has no data.  Needed for 2022 graphs where UNCTAD data ends at 2021.
    """
    if year in d:
        return float(d[year])
    if fallback_year in d:
        return float(d[fallback_year])
    return 0.0


def _load_node_attributes() -> dict[str, dict]:
    """
    Build a dict mapping country → {lsci_by_year, fleet_pct_by_year, teu_by_year,
    weather_severity, otd_rate, mean_delay_norm, congestion_rate,
    geo_conflict_rate, mean_gri, lat, lon}.
    Used to annotate graph nodes.
    """
    clsci = load_country_lsci()
    fleet = load_merchant_fleet()
    teu   = load_port_throughput()
    weather   = load_weather_severity()
    disruption = load_disruption_metrics()

    _defaults = {
        "lsci": {}, "fleet_pct": {}, "teu": {},
        "weather_severity": None,
        "otd_rate": None, "mean_delay_norm": None,
        "congestion_rate": None, "geo_conflict_rate": None, "mean_gri": None,
        "lat": None, "lon": None,
    }

    attrs: dict[str, dict] = {}

    def _ensure(c):
        if c not in attrs:
            attrs[c] = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _defaults.items()}

    for _, row in clsci.iterrows():
        c = row["country"]
        _ensure(c)
        attrs[c]["lsci"][int(row["year"])] = float(row["country_lsci"] or 0)

    for _, row in fleet.iterrows():
        c = row["country"]
        _ensure(c)
        attrs[c]["fleet_pct"][int(row["year"])] = float(row["fleet_pct"] or 0)

    for _, row in teu.iterrows():
        c = row["country"]
        _ensure(c)
        attrs[c]["teu"][int(row["year"])] = float(row["teu"]) if pd.notna(row["teu"]) else 0.0

    for _, row in weather.iterrows():
        c = row["country"]
        _ensure(c)
        attrs[c]["weather_severity"] = float(row["weather_severity"])

    for _, row in disruption.iterrows():
        c = row["country"]
        _ensure(c)
        attrs[c]["otd_rate"]          = float(row["otd_rate"])
        attrs[c]["mean_delay_norm"]   = float(row["mean_delay_norm"])
        attrs[c]["congestion_rate"]   = float(row["congestion_rate"])
        attrs[c]["geo_conflict_rate"] = float(row["geo_conflict_rate"])
        attrs[c]["mean_gri"]          = float(row["mean_gri"])

    for country, (lat, lon) in COUNTRY_COORDS.items():
        _ensure(country)
        attrs[country]["lat"] = lat
        attrs[country]["lon"] = lon

    return attrs


def build_graph(
    edges_df: pd.DataFrame,
    year: int,
    product_code: int,
    node_attrs: dict | None = None,
    historical_edge_set: set | None = None,
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph for a specific (year, product_code).

    Parameters
    ----------
    edges_df    : complete edge matrix (from graph_edges_full.parquet)
    year        : integer year (2016-2021)
    product_code: HS product code
    node_attrs  : optional pre-computed node attribute dict (for speed)
    historical_edge_set : optional set of historically observed
                          (origin, destination, product_code) tuples

    Returns
    -------
    nx.DiGraph with:
      - edge attrs: weight (freight_rate), bilateral_lsci, is_predicted,
                    has_historical_support
      - node attrs: lsci, fleet_pct, lat, lon
    """
    if historical_edge_set is None:
        hist_df = edges_df[
            (edges_df["year"] < year) &
            (edges_df["product_code"] == product_code) &
            edges_df["freight_rate"].notna() &
            (edges_df["freight_rate"] >= 0)
        ][["origin", "destination", "product_code"]].drop_duplicates()

        historical_edge_set = set(
            hist_df.itertuples(index=False, name=None)
        )

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

        edge_key = (origin, dest, product_code)
        has_historical_support = edge_key in historical_edge_set

        G.add_edge(
            origin, dest,
            weight=float(row["freight_rate"]),
            bilateral_lsci=float(row.get("bilateral_lsci", 0.0) or 0.0),
            is_predicted=bool(row.get("is_predicted", False)),
            has_historical_support=has_historical_support,
        )

    # Annotate nodes
    if node_attrs is None:
        node_attrs = _load_node_attributes()

    for node in G.nodes():
        na = node_attrs.get(node, {})
        G.nodes[node]["lsci"]              = _get_year_attr(na.get("lsci", {}), year)
        G.nodes[node]["fleet_pct"]         = _get_year_attr(na.get("fleet_pct", {}), year)
        G.nodes[node]["teu"]               = _get_year_attr(na.get("teu", {}), year)
        G.nodes[node]["weather_severity"]  = na.get("weather_severity")
        G.nodes[node]["otd_rate"]          = na.get("otd_rate")
        G.nodes[node]["mean_delay_norm"]   = na.get("mean_delay_norm")
        G.nodes[node]["congestion_rate"]   = na.get("congestion_rate")
        G.nodes[node]["geo_conflict_rate"] = na.get("geo_conflict_rate")
        G.nodes[node]["mean_gri"]          = na.get("mean_gri")
        G.nodes[node]["lat"]               = na.get("lat")
        G.nodes[node]["lon"]               = na.get("lon")

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
