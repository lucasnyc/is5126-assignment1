"""
Feature engineering pipeline.
Produces the master features_long.parquet used for ML training and inference.
"""

import os
import sys
import pickle
import math

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    FEATURES_PATH, CONSTANTS_PATH, ARTIFACTS_DIR, YEARS,
    PRODUCT_CODES, ML_FEATURES, COUNTRY_COORDS
)
from src.data.loaders import load_all


# ─────────────────────────────────────────────────────────────────────────────
# 🌍 Distance utility
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def compute_distance(origin, destination):
    c1 = COUNTRY_COORDS.get(origin)
    c2 = COUNTRY_COORDS.get(destination)
    if c1 and c2:
        return haversine_km(c1[0], c1[1], c2[0], c2[1])
    return np.nan


def build_features(save: bool = True) -> pd.DataFrame:
    print("Loading raw datasets...")
    data = load_all()

    tc   = data["transport_cost"]
    bil  = data["bilateral_lsci"]
    clsci = data["country_lsci"]
    teu   = data["port_throughput"]
    fleet = data["merchant_fleet"]
    trade = data["seaborne_trade"]
    vpct  = data["vessel_pct"]

    tc = tc[tc["product_code"].isin(PRODUCT_CODES)].copy()

    # ── Base merge ────────────────────────────────────────────────────
    df = tc.merge(bil, on=["origin", "destination", "year"], how="left")
    df["bilateral_lsci_is_imputed"] = df["bilateral_lsci"].isna().astype(int)
    df["bilateral_lsci"] = df["bilateral_lsci"].fillna(0.0)

    df = df.merge(clsci.rename(columns={"country": "origin", "country_lsci": "origin_lsci"}), on=["origin", "year"], how="left")
    df = df.merge(clsci.rename(columns={"country": "destination", "country_lsci": "dest_lsci"}), on=["destination", "year"], how="left")

    df = df.merge(teu.rename(columns={"country": "origin", "teu": "origin_teu"}), on=["origin", "year"], how="left")
    df["origin_teu_is_imputed"] = df["origin_teu"].isna().astype(int)

    df = df.merge(teu.rename(columns={"country": "destination", "teu": "dest_teu"}), on=["destination", "year"], how="left")
    df["dest_teu_is_imputed"] = df["dest_teu"].isna().astype(int)

    df = df.merge(fleet.rename(columns={"country": "origin", "fleet_pct": "origin_fleet_pct"}), on=["origin", "year"], how="left")
    df = df.merge(fleet.rename(columns={"country": "destination", "fleet_pct": "dest_fleet_pct"}), on=["destination", "year"], how="left")

    df = df.merge(trade.rename(columns={"country": "origin", "loaded_kt": "origin_loaded_kt", "discharged_kt": "origin_discharged_kt"}), on=["origin", "year"], how="left")
    df = df.merge(trade.rename(columns={"country": "destination", "loaded_kt": "dest_loaded_kt", "discharged_kt": "dest_discharged_kt"}), on=["destination", "year"], how="left")

    df = df.merge(vpct.rename(columns={"country": "origin", "vessel_pct": "origin_vessel_pct"}), on=["origin", "year"], how="left")
    df = df.merge(vpct.rename(columns={"country": "destination", "vessel_pct": "dest_vessel_pct"}), on=["destination", "year"], how="left")

    # ─────────────────────────────────────────────────────────────────
    # 🔥 DISTANCE FEATURES (MOST IMPORTANT)
    # ─────────────────────────────────────────────────────────────────
    print("Computing distance features...")
    df["distance_km"] = df.apply(
        lambda x: compute_distance(x["origin"], x["destination"]), axis=1
    )
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    df["log_distance"] = np.log1p(df["distance_km"])

    # ─────────────────────────────────────────────────────────────────
    # TIME SERIES FEATURES
    # ─────────────────────────────────────────────────────────────────
    df = df.sort_values(by=["origin", "destination", "product_code", "year"])

    df["freight_lag_1"] = df.groupby(
        ["origin", "destination", "product_code"]
    )["freight_rate"].shift(1)

    df["freight_lag_2"] = df.groupby(
        ["origin", "destination", "product_code"]
    )["freight_rate"].shift(2)

    df["freight_roll_mean_2"] = df.groupby(
        ["origin", "destination", "product_code"]
    )["freight_rate"].transform(lambda x: x.shift(1).rolling(2).mean())

    df["trade_exists"] = (
        df[["origin_loaded_kt", "dest_loaded_kt"]]
        .fillna(0)
        .sum(axis=1) > 0
    ).astype(int)

    # ─────────────────────────────────────────────────────────────────
    # HISTORICAL MEAN (NO LEAKAGE)
    # ─────────────────────────────────────────────────────────────────
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

    product_medians = (
        df[df["freight_rate"].notna()]
        .groupby("product_code")["freight_rate"]
        .median()
    )

    df["historical_mean_rate_is_imputed"] = df["historical_mean_rate"].isna().astype(int)

    for prod, med in product_medians.items():
        mask = (df["product_code"] == prod) & df["historical_mean_rate"].isna()
        df.loc[mask, "historical_mean_rate"] = med

    # ─────────────────────────────────────────────────────────────────
    # FILL LAGS
    # ─────────────────────────────────────────────────────────────────
    for col in ["freight_lag_1", "freight_lag_2", "freight_roll_mean_2"]:
        df[col + "_is_imputed"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df["historical_mean_rate"])

    # ─────────────────────────────────────────────────────────────────
    # FILL NUMERIC
    # ─────────────────────────────────────────────────────────────────
    fill_zero_cols = [
        "bilateral_lsci",
        "origin_teu", "dest_teu",
        "origin_fleet_pct", "dest_fleet_pct",
        "origin_loaded_kt", "dest_loaded_kt",
        "origin_vessel_pct", "dest_vessel_pct",
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0.0)

    for yr in YEARS:
        mask = df["year"] == yr
        df.loc[mask & df["origin_lsci"].isna(), "origin_lsci"] = df.loc[mask, "origin_lsci"].median()
        df.loc[mask & df["dest_lsci"].isna(), "dest_lsci"] = df.loc[mask, "dest_lsci"].median()

    df["origin_lsci"] = df["origin_lsci"].fillna(0)
    df["dest_lsci"]   = df["dest_lsci"].fillna(0)

    # ─────────────────────────────────────────────────────────────────
    # 🔥 NEW DERIVED FEATURES
    # ─────────────────────────────────────────────────────────────────
    df["lsci_asymmetry"] = (df["origin_lsci"] - df["dest_lsci"]).abs()

    df["trade_imbalance"] = (
        (df["origin_loaded_kt"] - df["dest_loaded_kt"])
        / (df["origin_loaded_kt"] + df["dest_loaded_kt"] + 1e-9)
    )

    df["teu_log_product"] = np.log1p(df["origin_teu"]) * np.log1p(df["dest_teu"])
    df["fleet_supply"] = df["origin_fleet_pct"] + df["dest_fleet_pct"]

    df["post_covid"] = (df["year"] >= 2020).astype(int)
    df["year_int"] = df["year"].astype(int)

    # 🔥 INTERACTION FEATURES
    df["distance_lsci_interaction"] = df["distance_km"] * df["bilateral_lsci"]
    df["distance_fleet_interaction"] = df["distance_km"] * df["fleet_supply"]
    df["lsci_product"] = df["origin_lsci"] * df["dest_lsci"]

    # 🔥 IMPUTATION STRENGTH
    impute_cols = [c for c in df.columns if "_is_imputed" in c]
    df["num_imputed"] = df[impute_cols].sum(axis=1)

    df["trade_intensity"] = (df["origin_loaded_kt"] + df["dest_loaded_kt"]) / (df["origin_teu"] + df["dest_teu"] + 1e-6)
    df["port_pressure"] = (df["origin_loaded_kt"] / (df["origin_teu"] + 1e-6) + df["dest_loaded_kt"] / (df["dest_teu"] + 1e-6))
    df["directional_imbalance"] = (df["origin_loaded_kt"] - df["dest_discharged_kt"])
    df["capacity_gap"] = (df["fleet_supply"] - df["trade_intensity"])

    # ─────────────────────────────────────────────────────────────────
    # PRODUCT CATEGORY
    # ─────────────────────────────────────────────────────────────────
    df["product_cat"] = (
        df["product_code"]
        .astype("category")
        .cat.set_categories(PRODUCT_CODES)
        .cat.codes
        .astype(int)
    )

    # ─────────────────────────────────────────────────────────────────
    # SAVE CONSTANTS
    # ─────────────────────────────────────────────────────────────────
    constants = {
        "bilateral_lsci_p95": float(df["bilateral_lsci"].quantile(0.95)),
        "median_fleet_pct": float(df["fleet_supply"].median()),
        "median_lsci": float(df["origin_lsci"].median()),
    }

    if save:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(CONSTANTS_PATH, "wb") as f:
            pickle.dump(constants, f)

    print(f"Feature table shape: {df.shape}")

    if save:
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        df.to_parquet(FEATURES_PATH, index=False)

    return df


def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Run build_features() first.")
    return pd.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    build_features(save=True)