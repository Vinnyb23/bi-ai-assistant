import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.forecast import FEATURE_COLS, add_features


def _fake_daily_sales(n_days=90):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"order_date": dates, "total_sales": range(1, n_days + 1)})


def test_add_features_produces_expected_columns():
    daily = _fake_daily_sales()
    features = add_features(daily)
    for col in FEATURE_COLS:
        assert col in features.columns


def test_add_features_drops_rows_without_enough_history():
    daily = _fake_daily_sales(n_days=90)
    features = add_features(daily)
    # first 28 rows should be dropped because lag_28 / rolling_mean_28 are NaN
    assert len(features) == len(daily) - 28
    assert not features[FEATURE_COLS].isna().any().any()
