"""
Offline batch script: generate TFT 2022 freight rate forecasts for all SONAR routes.

Run once before starting the dashboard:
    python3 /home/daniel/is5126-assignment1/daniel/FinalAssignment/sonar/src/models/generate_tft_predictions.py

What it does:
1. Loads the existing graph_edges_full.parquet (2016-2021 rows)
2. Runs TFT inference for every unique (origin, destination, product_code) for year 2022
3. Appends 2022 rows (with quantile columns) to graph_edges_full.parquet
4. Saves a separate tft_predictions_2022.parquet for the Model Insights page
5. Regenerates graphs_latest.pkl for year 2022

Usage:
    python3 src/models/generate_tft_predictions.py [--products 8517 2106]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    EDGES_PATH,
    TFT_BUNDLE_PATH,
    TFT_DF_MODEL_PATH,
    TFT_PREDICTIONS_PATH,
    FORECAST_YEAR,
    PRODUCT_CODES,
    ARTIFACTS_DIR,
)


def main(products: list[int] | None = None):
    t0 = time.time()

    # ── Validate paths ───────────────────────────────────────────────────────
    steps = [
        ("TFT bundle", TFT_BUNDLE_PATH),
        ("df_model",   TFT_DF_MODEL_PATH),
        ("edges",      EDGES_PATH),
    ]
    for label, path in tqdm(steps, desc="Checking paths", leave=False):
        if not os.path.exists(path):
            tqdm.write(f"ERROR: {label} not found at:\n  {path}")
            sys.exit(1)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Load existing edges ───────────────────────────────────────────────────
    print(f"\nLoading edges from {EDGES_PATH} ...")
    edges = pd.read_parquet(EDGES_PATH)
    print(f"  Shape: {edges.shape}")

    if FORECAST_YEAR in edges["year"].values:
        print(f"  Year {FORECAST_YEAR} rows already present. Removing them for re-generation.")
        edges = edges[edges["year"] != FORECAST_YEAR].copy()

    # Determine which products to run
    target_products = products or PRODUCT_CODES

    # Extract distinct routes from 2021 (most complete year)
    routes_2021 = edges[
        (edges["year"] == FORECAST_YEAR - 1) &
        (edges["product_code"].isin(target_products))
    ][["origin", "destination", "product_code"]].drop_duplicates()
    all_routes = routes_2021.to_dict("records")
    print(f"  Routes to forecast: {len(all_routes)} for products {target_products}")

    # ── Run TFT inference (skip if predictions already saved) ────────────────
    if os.path.exists(TFT_PREDICTIONS_PATH):
        tqdm.write(f"\nFound existing predictions at {TFT_PREDICTIONS_PATH} — skipping TFT inference.")
        preds_df = pd.read_parquet(TFT_PREDICTIONS_PATH)
        tqdm.write(f"  Loaded {len(preds_df)} rows from existing predictions.")
    else:
        from src.models.tft_predictor import TFTPredictor

        tqdm.write("\nLoading TFT model ...")
        predictor = TFTPredictor(TFT_BUNDLE_PATH, TFT_DF_MODEL_PATH)
        tqdm.write(f"\nRunning TFT batch inference for year {FORECAST_YEAR} ...")
        preds_df = predictor.predict_batch(all_routes, forecast_year=FORECAST_YEAR)
        tqdm.write(f"\nPredictions generated: {len(preds_df)} rows")

        if preds_df.empty:
            print("ERROR: No predictions generated. Aborting.")
            sys.exit(1)

    # ── Spot-check against known demo result ─────────────────────────────────
    demo = preds_df[
        (preds_df["origin"] == "Japan") &
        (preds_df["destination"] == "China") &
        (preds_df["product_code"] == 8517)
    ]
    if not demo.empty:
        q50 = demo["q50"].iloc[0]
        print(f"\n  Spot-check Japan→China (8517, 2022): q50={q50:.4f} "
              f"(expect ≈0.15 per notebook demo)")

    # ── Save standalone predictions parquet ──────────────────────────────────
    preds_df.to_parquet(TFT_PREDICTIONS_PATH, index=False)
    print(f"\nSaved standalone predictions → {TFT_PREDICTIONS_PATH}")

    # ── Build 2022 rows for the full edges parquet ────────────────────────────
    # Join 2021 feature columns (best available proxy for 2022 covariates)
    features_2021 = edges[edges["year"] == FORECAST_YEAR - 1].copy()
    feat_cols = [c for c in features_2021.columns
                 if c not in ("year", "freight_rate", "is_predicted",
                              "origin", "destination", "product_code")]

    rows_2022 = preds_df.merge(
        features_2021[feat_cols + ["origin", "destination", "product_code"]],
        on=["origin", "destination", "product_code"],
        how="left",
    )
    rows_2022["year"]         = FORECAST_YEAR
    rows_2022["year_int"]     = FORECAST_YEAR
    rows_2022["post_covid"]   = 1
    rows_2022["is_predicted"] = True
    # Ensure freight_rate column exists (= q50)
    rows_2022["freight_rate"] = rows_2022["q50"]

    # ── Add quantile + model_source columns to historical rows ────────────────
    q_cols = ["q10", "q20", "q30", "q50", "q70", "q80", "q90"]
    with tqdm(total=4, desc="Assembling edges parquet", leave=True) as pbar:
        for col in q_cols:
            if col not in edges.columns:
                edges[col] = np.nan
        if "model_source" not in edges.columns:
            edges["model_source"] = np.where(edges["is_predicted"], "xgb", "observed")
        rows_2022["model_source"] = "tft"
        pbar.update(1)

        all_cols = list(edges.columns)
        for col in q_cols + ["model_source"]:
            if col not in all_cols:
                all_cols.append(col)
        for col in all_cols:
            if col not in rows_2022.columns:
                rows_2022[col] = np.nan
        rows_2022 = rows_2022[all_cols]
        pbar.update(1)

        full_edges = pd.concat([edges, rows_2022], ignore_index=True)
        pbar.update(1)

        full_edges.to_parquet(EDGES_PATH, index=False)
        pbar.update(1)

    tqdm.write(f"Saved updated edges → {EDGES_PATH}")
    tqdm.write(f"  Final shape: {full_edges.shape}")
    tqdm.write(f"  Year counts:\n{full_edges['year'].value_counts().sort_index().to_string()}")

    # ── Regenerate graph cache for year 2022 ──────────────────────────────────
    tqdm.write(f"\nRegenerating graph cache for year {FORECAST_YEAR} ...")
    from src.graph.builder import build_latest_graphs
    with tqdm(total=1, desc="Building graph cache", leave=True) as pbar:
        build_latest_graphs(edges_df=full_edges, save_cache=True)
        pbar.update(1)

    elapsed = time.time() - t0
    tqdm.write(f"\nDone in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TFT 2022 freight rate forecasts")
    parser.add_argument(
        "--products", nargs="+", type=int, default=None,
        help="Product codes to process (default: all 5). E.g. --products 8517 2106"
    )
    args = parser.parse_args()
    main(products=args.products)
