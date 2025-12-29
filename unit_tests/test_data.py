"""Tests for data preparation and validation"""

import warnings

import pandas as pd
import pytest

from src.data.make_dataset import optimize_dtypes, preprocess_data, organize_columns
from src.data.validation import validate_dataframe

warnings.filterwarnings("ignore")


@pytest.fixture
def raw_data():
    """Small valid raw-like dataset before cleaning"""
    return pd.DataFrame(
        {
            "limit_bal": [20000, 30000],
            "sex": [1, 2],
            "education": [2, 5],
            "marriage": [1, 0],
            "age": [25, 45],
            "pay_0": [0, 2],
            "pay_2": [2, 1],
            "pay_3": [0, 0],
            "pay_4": [0, 0],
            "pay_5": [0, 0],
            "pay_6": [0, 0],
            "bill_amt1": [10000, 20000],
            "bill_amt2": [9000, 19000],
            "bill_amt3": [8000, 18000],
            "bill_amt4": [7000, 17000],
            "bill_amt5": [6000, 16000],
            "bill_amt6": [5000, 15000],
            "pay_amt1": [500, 1000],
            "pay_amt2": [600, 1100],
            "pay_amt3": [700, 1200],
            "pay_amt4": [800, 1300],
            "pay_amt5": [900, 1400],
            "pay_amt6": [1000, 1500],
            "default_payment_next_month": [0, 1],
        }
    )


def test_clean_data(raw_data):
    df = preprocess_data(raw_data)

    assert "education" in df.columns
    assert sorted(df["education"].unique()) == [2, 4]

    assert "marriage" in df.columns
    assert sorted(df["marriage"].unique()) == [1, 3]

    assert df["age"].between(18, 100).all()


def test_cast_dtypes(raw_data):
    df = preprocess_data(raw_data)
    df = optimize_dtypes(df)

    assert df["sex"].dtype == "int16"
    assert df["limit_bal"].dtype == "float64"
    assert df["default_payment"].dtype == "int8"


def test_reorder_columns(raw_data):
    df = preprocess_data(raw_data)
    df = organize_columns(df)

    assert list(df.columns)[-1] == "default_payment"
    assert "pay_amt1" in df.columns  # порядок сохранён


def test_validation_passes_on_clean_data(raw_data):
    df = preprocess_data(raw_data)
    df = optimize_dtypes(df)
    df = organize_columns(df)

    validate_dataframe(df)  # должно пройти без ошибок


def test_validation_fails_on_wrong_values(raw_data):
    df = raw_data.copy()
    df.loc[0, "sex"] = 99  # некорректное значение

    with pytest.raises(AssertionError):
        validate_dataframe(df)
