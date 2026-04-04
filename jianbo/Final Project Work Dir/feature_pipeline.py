"""
Feature engineering pipeline.
Produces the master features_long.parquet used for ML training and inference.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    FEATURES_PATH, CONSTANTS_PATH, ARTIFACTS_DIR, YEARS,
    PRODUCT_CODES, ML_FEATURES
)
from src.data.loaders import load_all


def build_features(save: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Load all 7 raw datasets (already cleaned / long-format by loaders).
      2. Build the observation table from transport_cost (all years, all OD pairs).
      3. Join in all supporting features.
      4. Engineer derived features.
      5. Null-fill with documented strategy; add is_imputed flags.
      6. Optionally save as parquet and persist normalization constants.

    Returns
    -------
    pd.DataFrame with columns:
        origin, destination, product_code, product_label, year, freight_rate,
        [25 ML features], product_cat (int code)
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading raw datasets...")
    data = load_all()
    tc   = data["transport_cost"]     # origin, destination, product_code, year, freight_rate
    bil  = data["bilateral_lsci"]     # origin, destination, year, bilateral_lsci
    clsci = data["country_lsci"]      # country, year, country_lsci
    teu   = data["port_throughput"]   # country, year, teu
    fleet = data["merchant_fleet"]    # country, year, fleet_pct
    trade = data["seaborne_trade"]    # country, year, loaded_kt, discharged_kt
    vpct  = data["vessel_pct"]        # country, year, vessel_pct, vessel_pct_extrapolated

    print(f"  Transport cost rows: {len(tc):,}")

    # ── Filter to 5 products ──────────────────────────────────────────────────
    tc = tc[tc["product_code"].isin(PRODUCT_CODES)].copy()
    print(f"  After product filter: {len(tc):,} rows")

    # ── Join bilateral LSCI ───────────────────────────────────────────────────
    df = tc.merge(
        bil.rename(columns={"bilateral_lsci": "bilateral_lsci"}),
        on=["origin", "destination", "year"],
        how="left",
    )
    df["bilateral_lsci_is_imputed"] = df["bilateral_lsci"].isna().astype(int)
    df["bilateral_lsci"] = df["bilateral_lsci"].fillna(0.0)

    # ── Join country LSCI (origin) ────────────────────────────────────────────
    df = df.merge(
        clsci.rename(columns={"country": "origin", "country_lsci": "origin_lsci"}),
        on=["origin", "year"],
        how="left",
    )
    df = df.merge(
        clsci.rename(columns={"country": "destination", "country_lsci": "dest_lsci"}),
        on=["destination", "year"],
        how="left",
    )

    # ── Join TEU ──────────────────────────────────────────────────────────────
    df = df.merge(
        teu.rename(columns={"country": "origin", "teu": "origin_teu"}),
        on=["origin", "year"],
        how="left",
    )
    df["origin_teu_is_imputed"] = df["origin_teu"].isna().astype(int)
    df = df.merge(
        teu.rename(columns={"country": "destination", "teu": "dest_teu"}),
        on=["destination", "year"],
        how="left",
    )
    df["dest_teu_is_imputed"] = df["dest_teu"].isna().astype(int)

    # ── Join merchant fleet ───────────────────────────────────────────────────
    df = df.merge(
        fleet.rename(columns={"country": "origin", "fleet_pct": "origin_fleet_pct"}),
        on=["origin", "year"],
        how="left",
    )
    df = df.merge(
        fleet.rename(columns={"country": "destination", "fleet_pct": "dest_fleet_pct"}),
        on=["destination", "year"],
        how="left",
    )

    # ── Join seaborne trade ───────────────────────────────────────────────────
    df = df.merge(
        trade.rename(columns={
            "country": "origin",
            "loaded_kt": "origin_loaded_kt",
            "discharged_kt": "origin_discharged_kt"
        }),
        on=["origin", "year"],
        how="left",
    )
    df = df.merge(
        trade.rename(columns={
            "country": "destination",
            "loaded_kt": "dest_loaded_kt",
            "discharged_kt": "dest_discharged_kt"
        }),
        on=["destination", "year"],
        how="left",
    )

    # ── Join vessel % ─────────────────────────────────────────────────────────
    df = df.merge(
        vpct[["country", "year", "vessel_pct"]].rename(
            columns={"country": "origin", "vessel_pct": "origin_vessel_pct"}
        ),
        on=["origin", "year"],
        how="left",
    )
    df = df.merge(
        vpct[["country", "year", "vessel_pct"]].rename(
            columns={"country": "destination", "vessel_pct": "dest_vessel_pct"}
        ),
        on=["destination", "year"],
        how="left",
    )

    # ── Historical mean rate ──────────────────────────────────────────────────
    # Use only pre-2020 observed rates to compute the group mean.
    # This avoids leaking 2020 (val) and 2021 (test) targets into features.
    historical_obs = df[
        df["freight_rate"].notna() & df["year"].isin([2016, 2017, 2018, 2019])
    ]
    grp_mean = (
        historical_obs
        .groupby(["origin", "destination", "product_code"])["freight_rate"]
        .mean()
        .rename("historical_mean_rate")
    )
    df = df.join(grp_mean, on=["origin", "destination", "product_code"])

    # For (OD, product) with zero observed rows: fill with product median
    product_medians = (
        df[df["freight_rate"].notna()]
        .groupby("product_code")["freight_rate"]
        .median()
    )
    df["historical_mean_rate_is_imputed"] = df["historical_mean_rate"].isna().astype(int)
    for prod, med in product_medians.items():
        mask = (df["product_code"] == prod) & df["historical_mean_rate"].isna()
        df.loc[mask, "historical_mean_rate"] = med

    # ── Null-fill numeric features ────────────────────────────────────────────
    _fill_zero = [
        "bilateral_lsci",
        "origin_teu", "dest_teu",
        "origin_fleet_pct", "dest_fleet_pct",
        "origin_loaded_kt", "dest_loaded_kt",
        "origin_discharged_kt", "dest_discharged_kt",
        "origin_vessel_pct", "dest_vessel_pct",
    ]
    df[_fill_zero] = df[_fill_zero].fillna(0.0)

    # country_lsci: fill with year median
    for yr in YEARS:
        mask = df["year"] == yr
        med_o = df.loc[mask & df["origin_lsci"].notna(), "origin_lsci"].median()
        med_d = df.loc[mask & df["dest_lsci"].notna(), "dest_lsci"].median()
        df.loc[mask & df["origin_lsci"].isna(), "origin_lsci"] = med_o
        df.loc[mask & df["dest_lsci"].isna(), "dest_lsci"]     = med_d

    # Final fallback
    df["origin_lsci"] = df["origin_lsci"].fillna(0.0)
    df["dest_lsci"]   = df["dest_lsci"].fillna(0.0)

    # ── Derived / engineered features ─────────────────────────────────────────
    df["lsci_asymmetry"]  = (df["origin_lsci"] - df["dest_lsci"]).abs()
    df["trade_imbalance"] = (
        (df["origin_loaded_kt"] - df["dest_loaded_kt"])
        / (df["origin_loaded_kt"] + df["dest_loaded_kt"] + 1e-9)
    )
    df["teu_log_product"] = (
        np.log1p(df["origin_teu"]) * np.log1p(df["dest_teu"])
    )
    df["fleet_supply"]  = df["origin_fleet_pct"] + df["dest_fleet_pct"]
    df["post_covid"]    = (df["year"] >= 2020).astype(int)
    df["year_int"]      = df["year"].astype(int)
    df["product_cat"]   = (
        df["product_code"]
        .astype("category")
        .cat.set_categories(PRODUCT_CODES)
        .cat.codes
        .astype(int)
    )

    # ── Persist normalization constants used by Resilience Score ──────────────
    bilateral_lsci_p95 = float(
        df.loc[df["bilateral_lsci"] > 0, "bilateral_lsci"].quantile(0.95)
    )
    median_fleet_pct = float(
        df.loc[df["fleet_supply"] > 0, "fleet_supply"].median()
    )
    median_lsci = float(df.loc[df["origin_lsci"] > 0, "origin_lsci"].median())

    # TEU 95th percentile for Port Health normalization
    teu_p95 = float(
        df.loc[df["origin_teu"] > 0, "origin_teu"].quantile(0.95)
    )

    # Weather and disruption medians (for countries not in the datasets)
    weather_data    = data.get("weather_severity")
    disruption_data = data.get("disruption_metrics")
    weather_severity_median = float(
        weather_data["weather_severity"].median()
    ) if weather_data is not None and len(weather_data) > 0 else 0.10
    rel_median = float(
        disruption_data["otd_rate"].median()
    ) if disruption_data is not None and len(disruption_data) > 0 else 0.87
    sec_median_gri = float(
        disruption_data["mean_gri"].median()
    ) if disruption_data is not None and len(disruption_data) > 0 else 0.50

    constants = {
        "bilateral_lsci_p95":     bilateral_lsci_p95,
        "median_fleet_pct":       median_fleet_pct,
        "median_lsci":            median_lsci,
        "teu_p95":                teu_p95,
        "weather_severity_median": weather_severity_median,
        "rel_median":             rel_median,
        "sec_median_gri":         sec_median_gri,
    }
    if save:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(CONSTANTS_PATH, "wb") as f:
            pickle.dump(constants, f)
        print(f"  Saved normalization constants → {CONSTANTS_PATH}")

    print(f"Feature table shape: {df.shape}")
    print(f"  Observed (non-null) freight rates: {df['freight_rate'].notna().sum():,}")
    print(f"  Missing freight rates (targets):   {df['freight_rate'].isna().sum():,}")

    if save:
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        df.to_parquet(FEATURES_PATH, index=False)
        print(f"  Saved → {FEATURES_PATH}")

    return df


def load_features() -> pd.DataFrame:
    """Load pre-built features from parquet (fast path)."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature table not found at {FEATURES_PATH}. "
            "Run build_features() first (notebook 02 or train_xgb.py)."
        )
    return pd.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    build_features(save=True)
