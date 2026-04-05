"""
TFT (Temporal Fusion Transformer) predictor.
Wraps the groupmate's trained TFT bundle and provides batch inference
for 2022 freight rate forecasts across all SONAR routes.

Run generate_tft_predictions.py offline to produce tft_predictions_2022.parquet.
This module is NOT imported by the live Streamlit app — it is used only during
the offline prediction generation step.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TFTPredictor:
    """
    Wraps the trained TFT bundle for batch freight-rate forecasting.

    Handles:
    - Loading the pytorch_forecasting model + training dataset from a .pt bundle
    - SONAR ↔ TFT country name translation (6 differing names)
    - Known-route direct inference (batch per product)
    - Unknown-route proxy inference (similarity-based proxy selection)
    """

    # Country name mappings between SONAR and TFT's UNCTAD names
    SONAR_TO_TFT: dict[str, str] = {
        "United States":               "United States of America",
        "Republic of Korea":           "Korea, Republic of",
        "Dem. People's Rep. of Korea": "Korea, Dem. People's Rep. of",
        "Republic of Moldova":         "Moldova, Republic of",
        "United Republic of Tanzania": "Tanzania, United Republic of",
        "Dem. Rep. of the Congo":      "Congo, Dem. Rep. of the",
    }
    TFT_TO_SONAR: dict[str, str] = {v: k for k, v in SONAR_TO_TFT.items()}

    SIMILARITY_COLS = [
        "dist_km",
        "dest_teu",
        "lsci_d",
        "Destination_GDP_US_at_current_prices_in_millions_Value",
        "trade_imbalance",
    ]
    QUANTILE_NAMES = ["q10", "q20", "q30", "q50", "q70", "q80", "q90"]

    def __init__(self, bundle_path: str, df_model_path: str):
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        print(f"Loading TFT bundle from {bundle_path} ...")
        bundle = torch.load(bundle_path, weights_only=False)

        # The bundle stores the checkpoint path as a Windows relative path
        # (e.g. "saved_models\tft_model2_all_products.ckpt").
        # Reconstruct it using the known models directory so it works on Linux.
        raw_ckpt = bundle["model_ckpt"]
        ckpt_filename = os.path.basename(raw_ckpt.replace("\\", "/"))
        ckpt_path = os.path.join(os.path.dirname(bundle_path), ckpt_filename)
        print(f"  Checkpoint: {ckpt_path}")

        self.tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        self.tft.eval()

        self.training_ds = bundle["training_ds"]
        self.min_yr = int(bundle["min_year"])
        self.products = bundle["products"]

        # Covariate columns: all time-varying unknown reals except the target
        trained_tvur = list(self.training_ds.time_varying_unknown_reals)
        self.covariate_cols = [c for c in trained_tvur if c != "y_log"]

        print(f"  Model loaded — products: {self.products}, min_year: {self.min_yr}")
        print(f"  Covariates: {len(self.covariate_cols)}")

        # Load ground-truth covariate source
        print(f"Loading df_model from {df_model_path} ...")
        self.df_raw = pd.read_parquet(df_model_path).copy()
        self.df_raw["Year"] = self.df_raw["Year"].astype(int)
        self.df_raw["group_id"] = (
            self.df_raw["Origin_Label"].astype(str) + "||" +
            self.df_raw["Destination_Label"].astype(str) + "||" +
            self.df_raw["Product_Code"].astype(str)
        )

        # Determine known groups (routes the TFT encoder has seen)
        self.known_groups = self._extract_known_groups()
        print(f"  Known routes: {len(self.known_groups)}")

        # Pre-compute product category lookup for proxy selection
        if "product_cat" in self.df_raw.columns:
            self._product_cat_map = (
                self.df_raw[["Product_Code", "product_cat"]]
                .drop_duplicates()
                .set_index("Product_Code")["product_cat"]
                .to_dict()
            )
        else:
            self._product_cat_map = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_known_groups(self) -> set[str]:
        """Extract known group_ids from the TFT encoder's label encoder."""
        for attr_name in ("categorical_encoders", "_categorical_encoders", "scalers"):
            mapping = getattr(self.training_ds, attr_name, None)
            if not isinstance(mapping, dict):
                continue
            enc = mapping.get("group_id")
            if enc is None:
                continue
            classes = getattr(enc, "classes_", None)
            if classes is None:
                continue
            return set(classes.keys()) if hasattr(classes, "keys") else set(classes)

        # Fallback: approximate from df_raw (routes with enough history)
        min_len = (self.training_ds.max_encoder_length +
                   self.training_ds.max_prediction_length + 1)
        grp = self.df_raw.groupby("group_id")["Year"].agg(["count", "max"])
        mask = (grp["count"] >= min_len) & (grp["max"] > int(self.df_raw["Year"].max()) - 2)
        return set(grp[mask].index.tolist())

    def _to_tft(self, name: str) -> str:
        return self.SONAR_TO_TFT.get(name, name)

    def _to_sonar(self, name: str) -> str:
        return self.TFT_TO_SONAR.get(name, name)

    def _build_target_profile(self, origin_tft: str, dest_tft: str, product: int) -> pd.Series:
        """Build a logistics similarity profile for proxy matching."""
        gid = f"{origin_tft}||{dest_tft}||{product}"
        exact = self.df_raw[self.df_raw["group_id"] == gid]
        if not exact.empty:
            profile = exact[self.SIMILARITY_COLS].mean()
            if not profile.isna().all():
                return profile

        dest_rows   = self.df_raw[self.df_raw["Destination_Label"] == dest_tft]
        origin_rows = self.df_raw[self.df_raw["Origin_Label"] == origin_tft]
        profile: dict = {}

        for col in ["dest_teu", "lsci_d",
                    "Destination_GDP_US_at_current_prices_in_millions_Value",
                    "trade_imbalance"]:
            profile[col] = dest_rows[col].mean() if not dest_rows.empty else np.nan

        od_rows = self.df_raw[
            (self.df_raw["Origin_Label"] == origin_tft) &
            (self.df_raw["Destination_Label"] == dest_tft)
        ]
        if not od_rows.empty:
            profile["dist_km"] = od_rows["dist_km"].mean()
        elif not origin_rows.empty:
            profile["dist_km"] = origin_rows["dist_km"].mean()
        else:
            profile["dist_km"] = self.df_raw["dist_km"].mean()

        series = pd.Series({c: profile.get(c, np.nan) for c in self.SIMILARITY_COLS})
        for col in self.SIMILARITY_COLS:
            if pd.isna(series[col]):
                series[col] = self.df_raw[col].mean()
        return series

    def _find_proxy(self, origin_tft: str, dest_tft: str, product: int) -> str:
        """Find the best known-route proxy using normalised Euclidean distance."""
        from sklearn.preprocessing import StandardScaler

        target_profile = self._build_target_profile(origin_tft, dest_tft, product)
        pool_mask = self.df_raw["group_id"].isin(self.known_groups)

        req_product_cat = self._product_cat_map.get(product)

        for extra_mask in [
            pool_mask & (self.df_raw["Product_Code"] == product),
            pool_mask & (self.df_raw.get("product_cat", pd.Series(dtype="object")) == req_product_cat)
            if req_product_cat is not None else None,
            pool_mask,
        ]:
            if extra_mask is None:
                continue
            pool = self.df_raw[extra_mask]
            if pool["group_id"].nunique() == 0:
                continue

            candidates = (
                pool.groupby("group_id")[self.SIMILARITY_COLS]
                .mean()
                .dropna(how="all")
            )
            if candidates.empty:
                continue

            combined = pd.concat([target_profile.to_frame().T, candidates], ignore_index=False)
            combined = combined.fillna(combined.mean())
            scaler = StandardScaler()
            scaled = scaler.fit_transform(combined)
            dists = np.linalg.norm(scaled[1:] - scaled[0], axis=1)
            return candidates.index[int(np.argmin(dists))]

        # Last resort: first known group for this product
        for gid in sorted(self.known_groups):
            if gid.endswith(f"||{product}"):
                return gid
        return sorted(self.known_groups)[0]

    def _build_context_df(
        self,
        active_group: str,
        covariate_source_rows: pd.DataFrame,
        forecast_year: int,
    ) -> pd.DataFrame:
        """
        Build the encoder context + forecast row for one (active_group, forecast_year).
        active_group     : the group_id the TFT encoder knows
        covariate_source_rows : rows from df_raw to use for covariate values
        """
        proxy_origin, proxy_dest, proxy_prod = active_group.split("||")
        context_end = forecast_year - 1

        proxy_rows = self.df_raw[self.df_raw["group_id"] == active_group].copy()
        context_src = covariate_source_rows if not covariate_source_rows.empty else proxy_rows
        context_rows = context_src[context_src["Year"] <= context_end].copy()

        if context_rows.empty:
            context_rows = proxy_rows[proxy_rows["Year"] <= context_end].copy()

        if context_rows.empty:
            return pd.DataFrame()

        context_rows = context_rows.copy()
        context_rows["group_id"]          = active_group
        context_rows["Origin_Label"]      = proxy_origin
        context_rows["Destination_Label"] = proxy_dest
        context_rows["Product_Code"]      = int(proxy_prod)
        context_rows["time_idx"]          = (context_rows["Year"] - self.min_yr).astype(int)

        # Build 2022 forecast row
        last_ctx = context_rows.iloc[[-1]].copy()
        fcst_row = last_ctx.copy()
        fcst_row["Year"]     = forecast_year
        fcst_row["time_idx"] = int(forecast_year - self.min_yr)

        # Fill 2022 covariates with product-year means (no actual 2022 data)
        prod_yr_rows = self.df_raw[
            (self.df_raw["Product_Code"].astype(str) == proxy_prod) &
            (self.df_raw["Year"] == forecast_year - 1)   # use last available year
        ]
        if not prod_yr_rows.empty:
            prod_mean = prod_yr_rows[self.covariate_cols].mean()
            for col in self.covariate_cols:
                if col in prod_mean.index and not pd.isna(prod_mean[col]):
                    fcst_row[col] = prod_mean[col]

        fcst_row["y_log"] = last_ctx["y_log"].iloc[0]
        fcst_row["group_id"]          = active_group
        fcst_row["Origin_Label"]      = proxy_origin
        fcst_row["Destination_Label"] = proxy_dest
        fcst_row["Product_Code"]      = int(proxy_prod)

        return (
            pd.concat([context_rows, fcst_row], ignore_index=True)
            .sort_values("time_idx")
            .reset_index(drop=True)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        routes: list[dict],
        forecast_year: int = 2022,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Predict freight rates for a list of SONAR routes.

        Parameters
        ----------
        routes : list of {"origin": str, "destination": str, "product_code": int}
                 using SONAR country names
        forecast_year : year to forecast (default 2022, out-of-sample)
        batch_size    : DataLoader batch size

        Returns
        -------
        DataFrame with columns:
            origin, destination, product_code, q10..q90, freight_rate (= q50), is_proxy
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from tqdm import tqdm

        results = []

        # Process per-product to limit memory
        products = sorted({r["product_code"] for r in routes})
        prod_bar = tqdm(products, desc="Products", unit="product", position=0)

        for prod in prod_bar:
            prod_routes = [r for r in routes if r["product_code"] == prod]
            prod_bar.set_postfix(product=prod, routes=len(prod_routes))

            # ── Classify routes as known or proxy ────────────────────────
            known_batch: list[tuple[str, dict]] = []
            proxy_batch: list[tuple[str, str, dict]] = []

            classify_bar = tqdm(
                prod_routes, desc=f"  [{prod}] Classifying", unit="route",
                position=1, leave=False,
            )
            for route in classify_bar:
                orig_tft = self._to_tft(route["origin"])
                dest_tft = self._to_tft(route["destination"])
                gid = f"{orig_tft}||{dest_tft}||{prod}"
                if gid in self.known_groups:
                    known_batch.append((gid, route))
                else:
                    proxy_gid = self._find_proxy(orig_tft, dest_tft, prod)
                    proxy_batch.append((proxy_gid, gid, route))
            classify_bar.close()

            tqdm.write(f"  [{prod}] Known: {len(known_batch)}, Proxy: {len(proxy_batch)}")

            # ── Known routes ─────────────────────────────────────────────
            if known_batch:
                all_frames = []
                valid_routes = []
                build_bar = tqdm(
                    known_batch, desc=f"  [{prod}] Building known contexts",
                    unit="route", position=1, leave=False,
                )
                for gid, route in build_bar:
                    src_rows = self.df_raw[self.df_raw["group_id"] == gid]
                    df_demo = self._build_context_df(gid, src_rows, forecast_year)
                    if df_demo.empty:
                        continue
                    all_frames.append(df_demo)
                    valid_routes.append((gid, route))
                build_bar.close()

                if all_frames:
                    tqdm.write(f"  [{prod}] Running TFT on {len(valid_routes)} known routes ...")
                    combined = pd.concat(all_frames, ignore_index=True)
                    preds = self._run_tft_on_df(combined, batch_size)
                    for i, (gid, route) in enumerate(valid_routes):
                        if i < len(preds):
                            q_vals = np.clip(np.expm1(preds[i]), 0, None)
                            results.append({
                                "origin":       route["origin"],
                                "destination":  route["destination"],
                                "product_code": prod,
                                **{q: float(q_vals[j]) for j, q in enumerate(self.QUANTILE_NAMES)},
                                "freight_rate": float(q_vals[3]),  # q50 = index 3
                                "is_proxy":     False,
                            })

            # ── Proxy routes (group by unique proxy) ──────────────────────
            if proxy_batch:
                proxy_groups: dict[str, list[tuple[str, dict]]] = {}
                for proxy_gid, req_gid, route in proxy_batch:
                    proxy_groups.setdefault(proxy_gid, []).append((req_gid, route))

                proxy_list = list(proxy_groups.items())
                chunk_size = 200
                n_chunks = (len(proxy_list) + chunk_size - 1) // chunk_size

                chunk_bar = tqdm(
                    range(0, len(proxy_list), chunk_size),
                    total=n_chunks,
                    desc=f"  [{prod}] Proxy chunks",
                    unit="chunk", position=1, leave=False,
                )
                for chunk_start in chunk_bar:
                    chunk = proxy_list[chunk_start:chunk_start + chunk_size]
                    chunk_bar.set_postfix(
                        proxies=len(chunk),
                        routes=sum(len(v) for _, v in chunk),
                    )
                    chunk_frames = []
                    chunk_routes = []
                    for proxy_gid, req_route_pairs in chunk:
                        proxy_src = self.df_raw[self.df_raw["group_id"] == proxy_gid]
                        df_demo = self._build_context_df(proxy_gid, proxy_src, forecast_year)
                        if df_demo.empty:
                            continue
                        chunk_frames.append(df_demo)
                        chunk_routes.append((proxy_gid, req_route_pairs))

                    if not chunk_frames:
                        continue

                    combined = pd.concat(chunk_frames, ignore_index=True)
                    preds = self._run_tft_on_df(combined, batch_size)

                    for i, (proxy_gid, req_route_pairs) in enumerate(chunk_routes):
                        if i >= len(preds):
                            continue
                        q_vals = np.clip(np.expm1(preds[i]), 0, None)
                        for req_gid, route in req_route_pairs:
                            results.append({
                                "origin":       route["origin"],
                                "destination":  route["destination"],
                                "product_code": prod,
                                **{q: float(q_vals[j]) for j, q in enumerate(self.QUANTILE_NAMES)},
                                "freight_rate": float(q_vals[3]),  # q50
                                "is_proxy":     True,
                            })
                chunk_bar.close()

            tqdm.write(f"  [{prod}] Done — {len([r for r in results if r['product_code'] == prod])} predictions")

        prod_bar.close()
        return pd.DataFrame(results)

    def _run_tft_on_df(self, df_demo: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """
        Run TFT quantile prediction on a combined multi-group DataFrame.
        Returns array of shape (N_groups, 7) — one row per group, 7 quantiles.
        """
        from pytorch_forecasting import TimeSeriesDataSet

        try:
            demo_ds = TimeSeriesDataSet.from_dataset(
                self.training_ds,
                df_demo,
                predict=True,
                stop_randomization=True,
                allow_missing_timesteps=True,
            )
        except Exception as e:
            print(f"    Warning: TimeSeriesDataSet construction failed: {e}")
            return np.array([])

        n_workers = min(4, os.cpu_count() or 1)
        loader = demo_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=n_workers)

        pred_q = self.tft.predict(
            loader,
            mode="quantiles",
            trainer_kwargs={
                "accelerator": "auto",
                "logger": False,
                "enable_progress_bar": False,
            },
        )
        # pred_q shape: (N, prediction_length, n_quantiles)
        # We want the last prediction step: pred_q[:, -1, :]
        result = pred_q[:, -1, :].numpy() if hasattr(pred_q, "numpy") else np.array(pred_q)[:, -1, :]
        return result
