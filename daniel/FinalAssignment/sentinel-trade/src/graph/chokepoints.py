"""
Chokepoint definitions and helper utilities.
Imported by builder.py, routing.py, and the Streamlit app.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CHOKEPOINTS, ALL_CHOKEPOINT_COUNTRIES, TARIFF_REGIONS


def get_countries_to_remove(blocked_chokepoints: list[str]) -> set[str]:
    """
    Given a list of chokepoint display names (e.g. ['Suez Canal', 'Panama Canal']),
    return the set of country node names to remove from the graph.
    """
    countries = set()
    for cp in blocked_chokepoints:
        countries.update(CHOKEPOINTS.get(cp, []))
    return countries


def get_tariff_multipliers(
    us_pct: float = 0.0,
    eu_pct: float = 0.0,
    china_pct: float = 0.0,
    asean_pct: float = 0.0,
) -> dict[str, float]:
    """
    Build a country → multiplier dict from regional tariff sliders.
    The multiplier is 1 + tariff_rate (e.g. 25% tariff → 1.25).
    Countries in multiple regions get the maximum applicable multiplier.
    """
    region_rates = {
        "United States":  us_pct / 100.0,
        "European Union": eu_pct / 100.0,
        "China":          china_pct / 100.0,
        "ASEAN":          asean_pct / 100.0,
    }
    multipliers: dict[str, float] = {}
    for region, rate in region_rates.items():
        if rate <= 0:
            continue
        for country in TARIFF_REGIONS.get(region, []):
            existing = multipliers.get(country, 0.0)
            multipliers[country] = max(existing, rate)

    # Convert rate → multiplier
    return {c: 1.0 + r for c, r in multipliers.items()}


def chokepoint_exposure(path: list[str]) -> float:
    """
    Fraction of chokepoint countries present among the intermediate nodes of path.
    0.0 = no chokepoint countries on path (best)
    1.0 = all chokepoint countries on path (worst)
    """
    if len(path) <= 2:
        return 0.0
    intermediates = set(path[1:-1])
    cp_set        = set(ALL_CHOKEPOINT_COUNTRIES)
    exposure = len(intermediates & cp_set) / max(len(cp_set), 1)
    return float(exposure)
