"""ML Pipeline creation for credit scoring — FINAL PRODUCTION VERSION"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns():
    """
    Expected feature set after FEATURES stage (train_features.csv/test_features.csv).

    Includes:
    - Base numeric features
    - Engineered behavioral/time-series features
    - Low-cardinality categorical features
    """

    numeric_features = [
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio",
    ]

    categorical_features = ["sex", "education", "marriage", "age_bin"]

    return numeric_features, categorical_features


def create_preprocessor():
    """
    Preprocessing block:
    - numeric → median imputation + StandardScaler
    - categorical → most frequent + OneHotEncoder
        with handle_unknown='ignore' to avoid runtime failures
    """

    numeric_features, categorical_features = get_feature_columns()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ✅ no sparse warnings, future-proof API
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
                sparse_output=False  # <-- replaces deprecated `sparse`
            )),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # cleaner feature names in get_feature_names_out()
    )


def create_pipeline(model_type: str = "logistic_regression", params: dict | None = None):
    """Full sklearn Pipeline: preprocessing → classifier."""
    if params is None:
        params = {}

    preprocessor = create_preprocessor()

    if model_type == "logistic_regression":
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def get_model_params(model_type: str, config: dict) -> dict:
    """Extract model hyperparameters from params.yaml config[model]."""
    return config.get(model_type, {})
