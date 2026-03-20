"""
Data loaders — one function per raw CSV.
Each function returns a clean, normalized DataFrame with canonical country names.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RAW_FILES, NAME_CANONICAL, YEARS,
    CITY_COUNTRY_MAP, WEATHER_SEVERITY_MAP, WEATHER_SEVERITY_DEFAULT,
)


def _normalize_country(series: pd.Series) -> pd.Series:
    """Apply NAME_CANONICAL mapping to a country-name column."""
    return series.map(lambda x: NAME_CANONICAL.get(x, x) if isinstance(x, str) else x)


def _melt_years(df: pd.DataFrame, id_vars: list, value_col_pattern: str,
                new_col: str) -> pd.DataFrame:
    """
    Melt year-wide columns that contain `value_col_pattern` into long format.
    Extracts the integer year from the column name prefix (e.g. '2016_...' → 2016).
    Returns columns: id_vars + ['year', new_col]
    """
    year_cols = [c for c in df.columns if value_col_pattern in c and
                 c[:4].isdigit()]
    melted = df.melt(id_vars=id_vars, value_vars=year_cols,
                     var_name="_raw_col", value_name=new_col)
    melted["year"] = melted["_raw_col"].str[:4].astype(int)
    return melted.drop(columns=["_raw_col"])


# ─── 1. Transport cost (core) ─────────────────────────────────────────────────

def load_transport_cost() -> pd.DataFrame:
    """
    Returns long-format transport cost DataFrame.
    Columns: origin, destination, product_code, product_label, year, freight_rate
    Rows with missing freight_rate are kept — these are the ML prediction targets.
    """
    df = pd.read_csv(RAW_FILES["transport_cost"])
    df["Origin_Label"]      = _normalize_country(df["Origin_Label"])
    df["Destination_Label"] = _normalize_country(df["Destination_Label"])

    df = _melt_years(
        df,
        id_vars=["Origin_Label", "Destination_Label", "Product_Code", "Product_Label"],
        value_col_pattern="Advalorem_freight_rate_Value",
        new_col="freight_rate",
    )
    df = df.rename(columns={
        "Origin_Label":      "origin",
        "Destination_Label": "destination",
        "Product_Code":      "product_code",
        "Product_Label":     "product_label",
    })
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    return df


# ─── 2. Bilateral LSCI ────────────────────────────────────────────────────────

def load_bilateral_lsci() -> pd.DataFrame:
    """
    Returns long-format bilateral connectivity index.
    Columns: origin, destination, year, bilateral_lsci
    Only Q1 of each year is retained (already Q1 structure in source data).
    """
    df = pd.read_csv(RAW_FILES["bilateral_lsci"])

    # Keep only Q1 rows (the dataset has one row per economy per year at Q1)
    df = df[df["Quarter_Label"].str.startswith("Q1")].copy()
    df["year"] = df["Quarter_Label"].str.extract(r"(\d{4})").astype(int)
    df = df[df["year"].isin(YEARS)]

    # Melt partner columns
    partner_cols = [c for c in df.columns if "_Index_Value" in c]
    df = df.melt(
        id_vars=["Economy_Label", "year"],
        value_vars=partner_cols,
        var_name="_partner_raw",
        value_name="bilateral_lsci",
    )
    df["destination"] = df["_partner_raw"].str.replace("_Index_Value", "", regex=False)
    df = df.rename(columns={"Economy_Label": "origin"})

    # Normalize names
    df["origin"]      = _normalize_country(df["origin"])
    df["destination"] = _normalize_country(df["destination"])

    df = df[["origin", "destination", "year", "bilateral_lsci"]].reset_index(drop=True)
    return df


# ─── 3. Country LSCI ──────────────────────────────────────────────────────────

def load_country_lsci() -> pd.DataFrame:
    """
    Returns long-format country-level LSCI connectivity index.
    Columns: country, year, country_lsci
    """
    df = pd.read_csv(RAW_FILES["country_lsci"])
    df = df.rename(columns={"Economy_Label": "country"})
    df["country"] = _normalize_country(df["country"])

    # Melt Q1 year columns: 'Q1 2016_Index_Average_Q1_2023__100_Value'
    year_cols = [c for c in df.columns if c.startswith("Q1 ") and "_Value" in c]
    df = df.melt(
        id_vars=["country"],
        value_vars=year_cols,
        var_name="_raw_col",
        value_name="country_lsci",
    )
    df["year"] = df["_raw_col"].str.extract(r"Q1 (\d{4})").astype(int)
    df = df[df["year"].isin(YEARS)][["country", "year", "country_lsci"]].reset_index(drop=True)
    return df


# ─── 4. Container port throughput ─────────────────────────────────────────────

def load_port_throughput() -> pd.DataFrame:
    """
    Returns long-format container port throughput in TEU.
    Columns: country, year, teu
    Drops header/summary rows (Economy_Label == 'Individual economies').
    """
    df = pd.read_csv(RAW_FILES["port_throughput"])
    df = df[df["Economy_Label"] != "Individual economies"].copy()
    df = df.rename(columns={"Economy_Label": "country"})
    df["country"] = _normalize_country(df["country"])

    df = _melt_years(
        df,
        id_vars=["country"],
        value_col_pattern="TEU_Twenty_foot_Equivalent_Unit_Value",
        new_col="teu",
    )
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    return df


# ─── 5. Merchant fleet ────────────────────────────────────────────────────────

def load_merchant_fleet() -> pd.DataFrame:
    """
    Returns long-format merchant fleet share (% of world total).
    Columns: country, year, fleet_pct
    """
    df = pd.read_csv(RAW_FILES["merchant_fleet"])
    df = df[df["Economy_Label"] != "Individual economies"].copy()
    df = df.rename(columns={"Economy_Label": "country"})
    df["country"] = _normalize_country(df["country"])

    df = _melt_years(
        df,
        id_vars=["country"],
        value_col_pattern="Percentage_of_total_world_Value",
        new_col="fleet_pct",
    )
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    return df


# ─── 6. Seaborne trade ────────────────────────────────────────────────────────

def load_seaborne_trade() -> pd.DataFrame:
    """
    Returns wide-format seaborne trade with separate loaded/discharged columns.
    Columns: country, year, loaded_kt, discharged_kt
    """
    df = pd.read_csv(RAW_FILES["seaborne_trade"])
    df = df[df["Economy_Label"] != "Individual economies"].copy()
    df = df.rename(columns={"Economy_Label": "country", "CargoType_Label": "cargo_type"})
    df["country"] = _normalize_country(df["country"])

    df = _melt_years(
        df,
        id_vars=["country", "cargo_type"],
        value_col_pattern="Metric_tons_in_thousands_Value",
        new_col="volume_kt",
    )
    df = df[df["year"].isin(YEARS)]

    # Pivot cargo type to columns
    df_pivot = df.pivot_table(
        index=["country", "year"],
        columns="cargo_type",
        values="volume_kt",
        aggfunc="sum",
    ).reset_index()
    df_pivot.columns.name = None

    # Standardize column names
    rename_map = {}
    for c in df_pivot.columns:
        if "loaded" in c.lower() and "discharged" not in c.lower():
            rename_map[c] = "loaded_kt"
        elif "discharged" in c.lower():
            rename_map[c] = "discharged_kt"
    df_pivot = df_pivot.rename(columns=rename_map)

    for col in ["loaded_kt", "discharged_kt"]:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan

    return df_pivot[["country", "year", "loaded_kt", "discharged_kt"]].reset_index(drop=True)


# ─── 7. Vessel percent ────────────────────────────────────────────────────────

def load_vessel_pct() -> pd.DataFrame:
    """
    Returns long-format vessel % of global fleet (2019-2021 only).
    For 2016-2018, values are back-filled from 2019 (fleet flags change slowly).
    Columns: country, year, vessel_pct, vessel_pct_extrapolated (bool)
    """
    df = pd.read_csv(RAW_FILES["vessel_pct"])
    df = df[df["FlagOfRegistration_Label"] != "Individual economies"].copy()
    df = df.rename(columns={"FlagOfRegistration_Label": "country"})
    df["country"] = _normalize_country(df["country"])

    df = _melt_years(
        df,
        id_vars=["country"],
        value_col_pattern="Percentage_of_global_fleet_value_Value",
        new_col="vessel_pct",
    )
    df = df[df["year"].isin([2019, 2020, 2021])].copy()
    df["vessel_pct_extrapolated"] = False

    # Back-fill to 2016-2018 using 2019 values
    df_2019 = df[df["year"] == 2019][["country", "vessel_pct"]].copy()
    extra_rows = []
    for yr in [2016, 2017, 2018]:
        tmp = df_2019.copy()
        tmp["year"] = yr
        tmp["vessel_pct_extrapolated"] = True
        extra_rows.append(tmp)
    df = pd.concat([df] + extra_rows, ignore_index=True)
    df = df[df["year"].isin(YEARS)].sort_values(["country", "year"]).reset_index(drop=True)
    return df


# ─── 8. Weather severity (per country) ────────────────────────────────────────

def load_weather_severity() -> pd.DataFrame:
    """
    Per-country mean weather severity from daily weather observations.
    Columns: country, weather_severity (float 0-1, higher = worse weather)

    Source: country_date_conditions.csv (129K rows, 211 countries, 669 dates).
    Each condition string is mapped to a numeric severity via WEATHER_SEVERITY_MAP.
    """
    df = pd.read_csv(RAW_FILES["weather"])
    df["severity"] = df["condition_text"].map(WEATHER_SEVERITY_MAP).fillna(
        WEATHER_SEVERITY_DEFAULT
    )
    result = (
        df.groupby("country")["severity"]
        .mean()
        .reset_index()
        .rename(columns={"severity": "weather_severity"})
    )
    result["country"] = _normalize_country(result["country"])
    return result


# ─── 9. Disruption metrics (per country) ─────────────────────────────────────

def load_disruption_metrics() -> pd.DataFrame:
    """
    Per-country disruption metrics derived from shipment-level data.
    Columns: country, otd_rate, delay_cv, congestion_rate, geo_conflict_rate, mean_gri

    Source: global_supply_chain_disruption_v1.csv (10K shipments, 6 OD pairs).
    Cities are mapped to countries via CITY_COUNTRY_MAP, then metrics are
    aggregated per country (appearing as either origin or destination).
    """
    df = pd.read_csv(RAW_FILES["disruption"])

    # Map cities to countries
    df["origin_country"] = df["Origin_City"].map(CITY_COUNTRY_MAP)
    df["dest_country"]   = df["Destination_City"].map(CITY_COUNTRY_MAP)

    # Stack origin and destination into a single "country" column
    # so each shipment contributes to both endpoints
    origin_df = df.rename(columns={"origin_country": "country"})
    dest_df   = df.rename(columns={"dest_country":   "country"})
    stacked   = pd.concat([
        origin_df[["country", "Delivery_Status", "Delay_Days",
                   "Disruption_Event", "Geopolitical_Risk_Index"]],
        dest_df[["country", "Delivery_Status", "Delay_Days",
                 "Disruption_Event", "Geopolitical_Risk_Index"]],
    ], ignore_index=True)

    # Drop rows with unmapped cities
    stacked = stacked.dropna(subset=["country"])

    def _agg(group):
        n = len(group)
        otd_rate = (group["Delivery_Status"] == "On Time").sum() / n
        # Normalize mean delay against a 10-day benchmark (max expected delay)
        mean_delay_norm = min(group["Delay_Days"].mean() / 10.0, 1.0)

        congestion_rate = (
            (group["Disruption_Event"] == "Port Congestion").sum() / n
        )
        geo_conflict_rate = (
            (group["Disruption_Event"] == "Geopolitical Conflict (Route Diversion)").sum() / n
        )
        mean_gri = group["Geopolitical_Risk_Index"].mean()

        return pd.Series({
            "otd_rate":            otd_rate,
            "mean_delay_norm":     mean_delay_norm,
            "congestion_rate":     congestion_rate,
            "geo_conflict_rate":   geo_conflict_rate,
            "mean_gri":            mean_gri,
        })

    result = stacked.groupby("country").apply(_agg, include_groups=False).reset_index()
    return result


# ─── Convenience: load all ────────────────────────────────────────────────────

def load_all() -> dict:
    """Load and return all datasets as a dict of DataFrames."""
    return {
        "transport_cost":      load_transport_cost(),
        "bilateral_lsci":      load_bilateral_lsci(),
        "country_lsci":        load_country_lsci(),
        "port_throughput":     load_port_throughput(),
        "merchant_fleet":      load_merchant_fleet(),
        "seaborne_trade":      load_seaborne_trade(),
        "vessel_pct":          load_vessel_pct(),
        "weather_severity":    load_weather_severity(),
        "disruption_metrics":  load_disruption_metrics(),
    }
