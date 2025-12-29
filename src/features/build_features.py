"""Stage: FEATURES
Читает clean train/test CSV (после PREPARE) и добавляет инженерные признаки.
Входы:  params.dataset.train_raw_location / test_raw_location
Выходы: params.dataset.train_features_location / test_features_location
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_configuration() -> dict:
    with open("params.yaml", "r") as config_file:
        return yaml.safe_load(config_file)


REQUIRED_BASE_COLUMNS = [
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
    "default_payment",
]
BILLING_COLUMNS = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
PAYMENT_AMOUNT_COLUMNS = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]
PAYMENT_STATUS_COLUMNS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]


def validate_required_columns(df: pd.DataFrame, location: str) -> None:
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[features] Missing columns in {location}: {missing}")


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Добавляет признаки. НЕ трогает default_payment, не делает лики."""
    processed_df = df.copy()

    # 1) Возраст в категории
    age_bins = config["feature_engineering"]["age_bins"]
    age_labels = config["feature_engineering"]["age_labels"]
    processed_df["age_bin"] = pd.cut(processed_df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    # 2) Utilization last month (bill_amt1 / limit_bal)
    #    Безопасное деление, клип на [0, 2]
    denominator = np.where(processed_df["limit_bal"] > 0, processed_df["limit_bal"], np.nan)
    processed_df["utilization_last"] = (processed_df["bill_amt1"] / denominator).fillna(0.0).clip(0, 2)

    # 3) Сумма задержек (сколько месяцев с просрочкой > 0)
    processed_df["pay_delay_sum"] = (processed_df[PAYMENT_STATUS_COLUMNS].values > 0).sum(axis=1).astype(np.int16)

    # 4) Максимальная задержка
    processed_df["pay_delay_max"] = processed_df[PAYMENT_STATUS_COLUMNS].max(axis=1)

    # 5) Тренд задолженности за 3 месяца: (bill1 - bill3) / (|bill3| + 1)
    processed_df["bill_trend"] = (
        (processed_df["bill_amt1"] - processed_df["bill_amt3"]) / (processed_df["bill_amt3"].abs() + 1.0)
    ).clip(-5, 5)

    # 6) Тренд платежей за 3 месяца: (pay1 - pay3) / (pay3 + 1)
    processed_df["pay_trend"] = ((processed_df["pay_amt1"] - processed_df["pay_amt3"]) / (processed_df["pay_amt3"] + 1.0)).clip(-5, 5)

    # 7) Средние значения
    processed_df["bill_avg"] = processed_df[BILLING_COLUMNS].mean(axis=1)
    processed_df["pay_amt_avg"] = processed_df[PAYMENT_AMOUNT_COLUMNS].mean(axis=1)

    # 8) Отношение платежей к среднему долгу, клип [0,5]
    processed_df["pay_to_bill_ratio"] = np.where(
        processed_df["bill_avg"] > 0, processed_df["pay_amt_avg"] / processed_df["bill_avg"], 0.0
    )
    processed_df["pay_to_bill_ratio"] = processed_df["pay_to_bill_ratio"].clip(0, 5)

    # Стабильный порядок: базовые → новые признаки → default_payment в конце
    new_feature_columns = [
        "age_bin",
        "utilization_last",
        "pay_delay_sum",
        "pay_delay_max",
        "bill_trend",
        "pay_trend",
        "bill_avg",
        "pay_amt_avg",
        "pay_to_bill_ratio",
    ]
    base_columns_no_target = [c for c in REQUIRED_BASE_COLUMNS if c != "default_payment" and c in processed_df.columns]
    columns = base_columns_no_target + new_feature_columns + (["default_payment"] if "default_payment" in processed_df.columns else [])
    return processed_df.loc[:, columns]


def main() -> None:
    config = load_configuration()

    raw_train = config["dataset"]["train_raw_location"]
    raw_test = config["dataset"]["test_raw_location"]
    feat_train = config["dataset"]["train_features_location"]
    feat_test = config["dataset"]["test_features_location"]

    Path(feat_train).parent.mkdir(parents=True, exist_ok=True)

    print("[features] Loading train:", raw_train)
    train_df = pd.read_csv(raw_train)
    print("[features] Loading test :", raw_test)
    test_df = pd.read_csv(raw_test)

    validate_required_columns(train_df, "train")
    validate_required_columns(test_df, "test")

    print("[features] Creating features (train)...")
    train_features = engineer_features(train_df, config)
    print("[features] Creating features (test)...")
    test_features = engineer_features(test_df, config)

    train_features.to_csv(feat_train, index=False)
    test_features.to_csv(feat_test, index=False)

    print(f"[features] Saved: {feat_train} {train_features.shape}")
    print(f"[features] Saved: {feat_test} {test_features.shape}")
    print(
        "[features] New feature columns:",
        [
            "age_bin",
            "utilization_last",
            "pay_delay_sum",
            "pay_delay_max",
            "bill_trend",
            "pay_trend",
            "bill_avg",
            "pay_amt_avg",
            "pay_to_bill_ratio",
        ],
    )


if __name__ == "__main__":
    main()
