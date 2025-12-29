"""Tests for feature engineering"""

import warnings

import pandas as pd
import pytest

from src.features.build_features import create_features

warnings.filterwarnings("ignore")


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    params = {
        "features": {
            "age_bins": [18, 30, 45, 60, 100],
            "age_labels": ["young", "middle", "senior", "elderly"],
        }
    }

    df = pd.DataFrame(
        {
            "limit_bal": [20000, 30000, 40000],
            "age": [25, 35, 55],
            "pay_0": [2, 0, -1],
            "pay_2": [2, 1, 0],
            "pay_3": [1, 0, -1],
            "pay_4": [0, -1, -2],
            "pay_5": [-1, -2, -2],
            "pay_6": [-2, -2, -2],
            "bill_amt1": [10000, 15000, 20000],
            "bill_amt2": [9000, 14000, 19000],
            "bill_amt3": [8000, 13000, 18000],
            "bill_amt4": [7000, 12000, 17000],
            "bill_amt5": [6000, 11000, 16000],
            "bill_amt6": [5000, 10000, 15000],
            "pay_amt1": [1000, 2000, 3000],
            "pay_amt2": [1100, 2100, 3100],
            "pay_amt3": [1200, 2200, 3200],
            "pay_amt4": [1300, 2300, 3300],
            "pay_amt5": [1400, 2400, 3400],
            "pay_amt6": [1500, 2500, 3500],
            "target": [1, 0, 0],
        }
    )

    return df, params


def test_age_bin_creation(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert "age_bin" in df_features.columns
    assert str(df_features["age_bin"].iloc[0]) == "young"
    assert df_features["age_bin"].dtype.name == "category"


def test_utilization_last(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    util_expected = 10000 / 20000
    assert abs(df_features["utilization_last"].iloc[0] - util_expected) < 1e-3
    assert df_features["utilization_last"].between(0, 2).all()


def test_pay_delay_sum(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert df_features["pay_delay_sum"].iloc[0] == 3
    assert df_features["pay_delay_sum"].iloc[1] == 1
    assert (df_features["pay_delay_sum"] >= 0).all()


def test_pay_delay_max(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert df_features["pay_delay_max"].iloc[0] == 2
    assert (df_features["pay_delay_max"] >= -2).all()


def test_bill_trend(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert df_features["bill_trend"].between(-5, 5).all()


def test_pay_trend(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert df_features["pay_trend"].between(-5, 5).all()


def test_bill_avg(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    expected_avg = (10000 + 9000 + 8000 + 7000 + 6000 + 5000) / 6
    assert abs(df_features["bill_avg"].iloc[0] - expected_avg) < 1e-3


def test_pay_amt_avg(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    expected_avg = (1000 + 1100 + 1200 + 1300 + 1400 + 1500) / 6
    assert abs(df_features["pay_amt_avg"].iloc[0] - expected_avg) < 1e-3


def test_pay_to_bill_ratio(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    assert df_features["pay_to_bill_ratio"].between(0, 5).all()


def test_original_columns_preserved(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    for col in df.columns:
        assert col in df_features.columns


def test_no_nan_in_new_features(sample_data):
    df, params = sample_data
    df_features = create_features(df, params)

    new_features = [
        "utilization_last",
        "pay_delay_sum",
        "pay_delay_max",
        "bill_trend",
        "pay_trend",
        "bill_avg",
        "pay_amt_avg",
        "pay_to_bill_ratio",
    ]

    for feat in new_features:
        assert not df_features[feat].isna().any(), f"NaN found in {feat}"
