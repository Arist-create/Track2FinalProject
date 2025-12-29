"""Data validation with Great Expectations (fallback to pandas checks).

Если GE доступен — используем PandasDataset (GX 0.18.x совместимый слой).
Если GE недоступен — строгие проверки на pandas. При провале бросаем AssertionError.
Также есть CLI-режим: валидировать train/test из params.yaml (удобно для CI).
"""

from __future__ import annotations

import math
import sys

import pandas as pd
import yaml

# Пытаемся использовать Great Expectations, но не завязываемся на DataContext
try:
    from great_expectations.dataset import PandasDataset  # deprecated, но в 0.18.x ещё есть

    _HAS_GE = True
except Exception:
    _HAS_GE = False

# ---- Единая схема, чтобы не дублировать ----
REQ_BASE_COLS = [
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "target",
]
PAY_STATUS_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
PAY_AMT_COLS = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]


def _ge_validate(df: pd.DataFrame) -> None:
    """Валидация через Great Expectations. Бросает AssertionError при провале."""
    ds = PandasDataset(df.copy())

    # Состав колонок
    ds.expect_table_columns_to_match_set(REQ_BASE_COLS)

    # Нет пропусков
    for c in REQ_BASE_COLS:
        ds.expect_column_values_to_not_be_null(c)

    # Категориальные
    ds.expect_column_values_to_be_in_set("sex", [1, 2])
    ds.expect_column_values_to_be_in_set("education", [1, 2, 3, 4])
    ds.expect_column_values_to_be_in_set("marriage", [1, 2, 3])
    ds.expect_column_values_to_be_in_set("target", [0, 1])

    # Диапазоны
    ds.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    for c in PAY_STATUS_COLS:
        ds.expect_column_values_to_be_between(c, min_value=-2, max_value=9)

    # Деньги/счета:
    # limit_bal >= 0; bill_amt* могут быть отрицательными; pay_amt* >= 0
    ds.expect_column_values_to_be_between(
        "limit_bal", min_value=0, max_value=None, allow_cross_type_comparisons=True
    )
    for c in BILL_COLS:
        ds.expect_column_values_to_be_between(
            c, min_value=-1e9, max_value=1e12, allow_cross_type_comparisons=True
        )
    for c in PAY_AMT_COLS:
        ds.expect_column_values_to_be_between(
            c, min_value=0, max_value=1e12, allow_cross_type_comparisons=True
        )

    res = ds.validate(result_format="SUMMARY")
    if not res.get("success", False):
        failed = []
        for r in res.get("results", []):
            if r.get("success") is False:
                failed.append(
                    r.get("expectation_config", {}).get("expectation_type", "unknown_expectation")
                )
        raise AssertionError(
            f"Data validation failed (Great Expectations). Failed expectations: {sorted(set(failed))}"
        )


def _pandas_validate(df: pd.DataFrame) -> None:
    """Fallback-валидатор на pandas. Бросает AssertionError при провале."""
    # 1) Колонки и пропуски
    missing = [c for c in REQ_BASE_COLS if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    assert df[REQ_BASE_COLS].isna().sum().sum() == 0, "Missing values in required columns"

    # 2) Категориальные домены
    assert set(df["sex"].unique()).issubset({1, 2}), "Unexpected values in 'sex'"
    assert set(df["education"].unique()).issubset({1, 2, 3, 4}), "Unexpected values in 'education'"
    assert set(df["marriage"].unique()).issubset({1, 2, 3}), "Unexpected values in 'marriage'"
    assert set(df["target"].unique()).issubset({0, 1}), "Unexpected values in 'target'"

    # 3) Диапазоны
    assert df["age"].between(18, 100).all(), "Age out of range [18,100]"
    for c in PAY_STATUS_COLS:
        assert df[c].between(-2, 9).all(), f"{c} out of range [-2,9]"

    # 4) Деньги/счета
    assert (df["limit_bal"] >= 0).all(), "limit_bal has negative values"
    for c in BILL_COLS:
        assert (df[c].between(-1e9, 1e12)).all(), f"{c} out of expected range"
    for c in PAY_AMT_COLS:
        assert (df[c] >= 0).all(), f"{c} has negative values"

    # 5) Нет бесконечностей
    assert (
        ~df[REQ_BASE_COLS].applymap(lambda x: isinstance(x, float) and (math.isinf(x))).any().any()
    ), "Inf detected"


def validate_dataframe(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """Главная функция для пайплайна. При провале бросает AssertionError."""
    if _HAS_GE:
        _ge_validate(df)
        print(f"[validation] Great Expectations passed for: {dataset_name}")
    else:
        _pandas_validate(df)
        print(f"[validation] Fallback pandas validation passed for: {dataset_name}")


# -------- CLI (удобно для CI) --------
def _load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def _cli():
    try:
        P = _load_params()
        train_path = P["data"]["train_raw_path"]
        test_path = P["data"]["test_raw_path"]

        tr = pd.read_csv(train_path)
        ts = pd.read_csv(test_path)

        validate_dataframe(tr, "train_clean")
        validate_dataframe(ts, "test_clean")
        print("[validation] ✅ OK")
        return 0
    except Exception as e:
        print(f"[validation] ❌ FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
