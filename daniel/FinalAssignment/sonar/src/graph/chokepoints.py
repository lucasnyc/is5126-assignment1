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
    DEPRECATED — no longer called by apply_scenario(). Country nodes are not
    removed when a chokepoint is blocked; only _apply_detour_penalty() reprices
    affected edges. Kept for backward compatibility.

    Given a list of chokepoint display names (e.g. ['Suez Canal', 'Panama Canal']),
    return the set of country node names associated with each chokepoint.
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
    Fraction of *intermediate* nodes that are chokepoint countries.
    0.0 = no chokepoint countries among transshipment hubs (best)
    1.0 = every transshipment hub is a strategic chokepoint (worst)

    Divides by the number of actual intermediate stops, not by the total
    count of all known chokepoint countries.  This ensures:
      - Direct routes always score 0.0 (no intermediates)
      - Routes through 1 hub where that hub is a chokepoint → 1.0
      - Routes through 2 hubs where 1 is a chokepoint → 0.5
    """
    if len(path) <= 2:
        return 0.0
    intermediates = path[1:-1]
    cp_set        = set(ALL_CHOKEPOINT_COUNTRIES)
    hits = sum(1 for n in intermediates if n in cp_set)
    return float(hits / len(intermediates))
