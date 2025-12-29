"""
Stage: PREPARE
Сырый CSV -> строгая очистка -> raw_data/processed/train.csv, test.csv
(без feature engineering)
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

EXPECTED_RAW_COLUMNS = {
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default.payment.next.month",
}
ALTERNATIVE_COL_A = "PAY_0"   # часто встречается
ALTERNATIVE_COL_B = "PAY_1"   # альтернативное имя

def load_configuration() -> dict:
    with open("params.yaml", "r") as config_file:
        return yaml.safe_load(config_file)

def validate_required_columns(df: pd.DataFrame):
    columns = set(df.columns)
    has_alternative = (ALTERNATIVE_COL_A in columns) or (ALTERNATIVE_COL_B in columns)
    missing = sorted((EXPECTED_RAW_COLUMNS - {ALTERNATIVE_COL_A, ALTERNATIVE_COL_B}) - columns)
    if missing or not has_alternative:
        raise ValueError(
            f"Missing columns: {missing}; PAY_0 exists={ALTERNATIVE_COL_A in columns}; PAY_1 exists={ALTERNATIVE_COL_B in columns}"
        )

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(".", "_", regex=False)
    return cleaned_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df.copy()

    # target rename
    if "default_payment_next_month" in processed_df.columns:
        processed_df = processed_df.rename(columns={"default_payment_next_month": "default_payment"})

    # PAY_1 -> PAY_0 (если нужно)
    if "pay_1" in processed_df.columns and "pay_0" not in processed_df.columns:
        processed_df = processed_df.rename(columns={"pay_1": "pay_0"})

    # нормализация категорий
    if "education" in processed_df.columns:
        processed_df["education"] = processed_df["education"].replace({0: 4, 5: 4, 6: 4})
    if "marriage" in processed_df.columns:
        processed_df["marriage"] = processed_df["marriage"].replace({0: 3})

    # фильтры по здравому смыслу
    if "sex" in processed_df.columns:
        processed_df = processed_df[processed_df["sex"].isin([1, 2])]
    if "age" in processed_df.columns:
        processed_df = processed_df[(processed_df["age"] >= 18) & (processed_df["age"] <= 100)]

    # клип статусов просрочек
    for column in ["pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]:
        if column in processed_df.columns:
            processed_df[column] = processed_df[column].clip(-2, 9)

    # drop ID
    if "id" in processed_df.columns:
        processed_df = processed_df.drop(columns=["id"])

    if processed_df.empty:
        raise ValueError("After cleaning the dataframe is empty — check filters.")
    if "default_payment" not in processed_df.columns:
        raise ValueError("Column 'default_payment' is missing after renaming.")

    return processed_df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    optimized_df = df.copy()
    for column in ["sex","education","marriage","age","pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]:
        if column in optimized_df.columns:
            optimized_df[column] = optimized_df[column].astype("int16")
    for column in ["limit_bal", *[f"bill_amt{i}" for i in range(1,7)], *[f"pay_amt{i}" for i in range(1,7)]]:
        if column in optimized_df.columns:
            optimized_df[column] = optimized_df[column].astype("float64")
    if "default_payment" in optimized_df.columns:
        optimized_df["default_payment"] = optimized_df["default_payment"].astype("int8")
    return optimized_df

def organize_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "limit_bal","sex","education","marriage","age",
        "pay_0","pay_2","pay_3","pay_4","pay_5","pay_6",
        "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
        "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6",
    ]
    columns = [col for col in base_columns if col in df.columns]
    if "default_payment" in df.columns:
        columns.append("default_payment")
    return df.loc[:, columns]

def main():
    config = load_configuration()
    raw_path     = config["dataset"]["raw_location"]
    train_path   = config["dataset"]["train_raw_location"]
    test_path    = config["dataset"]["test_raw_location"]
    processed_dir = Path(config["dataset"]["processed_location"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data…")
    raw_dataframe = pd.read_csv(raw_path)
    validate_required_columns(raw_dataframe)

    print("Cleaning…")
    dataframe = normalize_column_names(raw_dataframe)
    dataframe = preprocess_data(dataframe)
    dataframe = optimize_dtypes(dataframe)
    dataframe = organize_columns(dataframe)
    print("Cleaned shape:", dataframe.shape)

    print("Splitting…")
    training_dataframe, testing_dataframe = train_test_split(
        dataframe,
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"],
        stratify=dataframe["default_payment"],
    )

    training_dataframe.to_csv(train_path, index=False)
    testing_dataframe.to_csv(test_path, index=False)
    print(f"✅ Saved: {train_path} {training_dataframe.shape}")
    print(f"✅ Saved: {test_path} {testing_dataframe.shape}")
    print("Train target balance:", training_dataframe["default_payment"].value_counts(normalize=True).to_dict())

if __name__ == "__main__":
    main()
