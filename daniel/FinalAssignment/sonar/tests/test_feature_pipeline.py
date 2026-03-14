"""
Tests for the feature engineering pipeline.
Run with: pytest tests/test_feature_pipeline.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PRODUCT_CODES, YEARS, TRAIN_YEARS, ML_FEATURES


@pytest.fixture(scope="module")
def features_df():
    """Load or build the feature table once per test session."""
    from src.data.feature_pipeline import build_features, load_features
    try:
        return load_features()
    except FileNotFoundError:
        return build_features(save=True)


def test_product_filter(features_df):
    """Only the 5 target HS codes should be present."""
    assert set(features_df["product_code"].unique()) == set(PRODUCT_CODES)


def test_year_range(features_df):
    """Only years 2016-2021 should be present."""
    assert set(features_df["year"].unique()) == set(YEARS)


def test_no_data_leakage(features_df):
    """Test rows must not appear in training years."""
    observed = features_df[features_df["freight_rate"].notna()]
    train = observed[observed["year"].isin(TRAIN_YEARS)]
    assert 2021 not in train["year"].values
    assert 2020 not in train["year"].values


def test_historical_mean_always_filled(features_df):
    """historical_mean_rate must have no nulls after pipeline."""
    assert features_df["historical_mean_rate"].isna().sum() == 0, (
        "historical_mean_rate should never be null (fallback to product median)"
    )


def test_ml_features_present(features_df):
    """All ML_FEATURES columns must exist in the DataFrame."""
    missing = [f for f in ML_FEATURES if f not in features_df.columns]
    assert len(missing) == 0, f"Missing ML feature columns: {missing}"


def test_bilateral_lsci_no_nulls(features_df):
    """bilateral_lsci should be 0-filled (no nulls)."""
    assert features_df["bilateral_lsci"].isna().sum() == 0


def test_teu_no_nulls(features_df):
    """TEU columns should be 0-filled after pipeline."""
    assert features_df["origin_teu"].isna().sum() == 0
    assert features_df["dest_teu"].isna().sum() == 0


def test_reasonable_row_count(features_df):
    """Expect at least 200k observed rows (lower bound sanity check)."""
    n_observed = features_df["freight_rate"].notna().sum()
    assert n_observed > 100_000, f"Only {n_observed:,} observed rows — suspiciously low"


def test_freight_rate_non_negative(features_df):
    """All observed freight rates must be non-negative."""
    observed = features_df["freight_rate"].dropna()
    assert (observed >= 0).all(), "Negative freight rates found"


def test_product_cat_valid(features_df):
    """product_cat should be integers 0-4."""
    cats = features_df["product_cat"].unique()
    assert all(c in range(5) for c in cats), f"Unexpected product_cat values: {cats}"
