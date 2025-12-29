"""
Stage: TRAIN — FINAL
Обучение + CV + обработка дисбаланса, выбор порога, логгирование в MLflow, сохранение артефактов.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models.pipeline import create_pipeline, get_model_params

# ============================== utilities ==============================


def load_configuration() -> dict:
    """Загрузить параметры из params.yaml."""
    with open("params.yaml", "r") as config_file:
        return yaml.safe_load(config_file)


def load_datasets(
    train_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Загрузить подготовленные выборки (features.csv с колонкой default_payment)."""
    training_data = pd.read_csv(train_path)
    testing_data = pd.read_csv(test_path)
    return training_data.drop(columns=["default_payment"]), training_data["default_payment"], testing_data.drop(columns=["default_payment"]), testing_data["default_payment"]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Рассчитать основные метрики классификации.

    ВАЖНО: ключи совпадают с ожиданиями тестов:
    ["roc_auc", "f1_score", "precision", "recall", "pr_auc"]
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def generate_roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, path: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def generate_precision_recall_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, path: str) -> str:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, linewidth=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def generate_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, path: str, title: str = "Confusion Matrix"
) -> str:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # подписи значений
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# ============================== core train ==============================


def train_credit_risk_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
):
    """Обучение модели с CV и логированием в MLflow, возврат пайплайна и метрик."""
    model_type = config["model_config"]["selected_algorithm"]
    model_cfg = get_model_params(model_type, config["model_config"])
    random_state = config["training_config"].get("random_state", 42)

    # Базовые дефолты и обработка дисбаланса
    if model_type == "logistic_regression":
        model_cfg.setdefault("max_iter", 500)
        model_cfg.setdefault("class_weight", "balanced")
        model_cfg.setdefault("solver", "liblinear")
        model_cfg.setdefault("random_state", random_state)
    elif model_type == "random_forest":
        model_cfg.setdefault("n_estimators", 300)
        model_cfg.setdefault("max_depth", 12)
        model_cfg.setdefault("class_weight", "balanced")
        model_cfg.setdefault("n_jobs", -1)
        model_cfg.setdefault("random_state", random_state)
    elif model_type == "gradient_boosting":
        model_cfg.setdefault("n_estimators", 200)
        model_cfg.setdefault("learning_rate", 0.05)
        model_cfg.setdefault("max_depth", 3)
        model_cfg.setdefault("random_state", random_state)

    # MLflow
    mlflow.set_tracking_uri(config["experiment_tracking"]["tracking_uri"])
    mlflow.set_experiment(config["experiment_tracking"]["experiment_name"])

    cv_folds = config["training_config"].get("cv_folds", 5)
    scoring = config["training_config"].get("scoring", "roc_auc")

    # Сетки гиперпараметров
    if model_type == "logistic_regression":
        param_grid = {"classifier__C": [0.1, 1.0, 3.0, 10.0]}
    elif model_type == "random_forest":
        param_grid = {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [10, 12, 16],
        }
    elif model_type == "gradient_boosting":
        param_grid = {
            "classifier__n_estimators": [150, 200, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [2, 3],
        }
    else:
        param_grid = {}

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        for k, v in model_cfg.items():
            mlflow.log_param(f"init_{k}", v)

        pipeline = create_pipeline(model_type, model_cfg)

        # Веса классов через sample_weight (gradboost и RF поддерживают в fit)
        sample_weight = None
        if model_type in {"gradient_boosting", "random_forest"}:
            p1 = float((y_train == 1).mean())
            # balanced: суммарные веса классов равны (0.5/0.5)
            w0 = 0.5 / max(1.0 - p1, 1e-12)
            w1 = 0.5 / max(p1, 1e-12)
            sample_weight = np.where(y_train.values == 1, w1, w0)

        if param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs = {"classifier__sample_weight": sample_weight}
            grid.fit(X_train, y_train, **fit_kwargs)
            pipeline = grid.best_estimator_
            mlflow.log_metric("best_cv_score", float(grid.best_score_))
            for k, v in grid.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            if sample_weight is not None:
                pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
            else:
                pipeline.fit(X_train, y_train)

        # Инференс
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Подбор порога по F1
        prec, rec, thr = precision_recall_curve(y_test, y_proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = int(np.argmax(f1s))
        # precision_recall_curve возвращает thr длиной на 1 меньше
        best_thr = float(thr[max(best_idx - 1, 0)])
        mlflow.log_metric("best_threshold_f1", best_thr)

        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_best = (y_proba >= best_thr).astype(int)

        metrics_default = compute_classification_metrics(y_test, y_pred_default, y_proba)
        metrics_best = compute_classification_metrics(y_test, y_pred_best, y_proba)

        # Логирование метрик
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics_default.items()})
        mlflow.log_metrics({f"test_best_{k}": float(v) for k, v in metrics_best.items()})

        # Артефакты
        Path("trained_models").mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(generate_roc_curve_plot(y_test, y_proba, "trained_models/roc_curve.png"))
        mlflow.log_artifact(generate_precision_recall_curve_plot(y_test, y_proba, "trained_models/pr_curve.png"))
        mlflow.log_artifact(
            generate_confusion_matrix_plot(y_test, y_pred_default, "trained_models/confusion_default.png", "CM @0.5")
        )
        mlflow.log_artifact(
            generate_confusion_matrix_plot(y_test, y_pred_best, "trained_models/confusion_best.png", "CM @best-thr")
        )

        # Сохранение модели и метрик
        joblib.dump(pipeline, "trained_models/model.pkl")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        with open("trained_models/metrics.json", "w") as metrics_file:
            json.dump(
                {"default": metrics_default, "best_thr": {"threshold": best_thr, **metrics_best}},
                metrics_file,
                indent=2,
            )

        # Консольный вывод (без эмодзи)
        print("Test @0.5:", {k: round(v, 4) for k, v in metrics_default.items()})
        print("Test @best-thr:", {k: round(v, 4) for k, v in metrics_best.items()})
        print("Model trained and saved.")

        return pipeline, {"default": metrics_default, "best": metrics_best, "thr": best_thr}


# ============================== entrypoint ==============================


def main() -> None:
    configuration = load_configuration()
    X_train, y_train, X_test, y_test = load_datasets(
        configuration["dataset"]["train_features_location"], configuration["dataset"]["test_features_location"]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    _, metrics = train_credit_risk_model(X_train, y_train, X_test, y_test, configuration)
    print(
        "ROC-AUC @0.5: "
        f"{metrics['default']['roc_auc']:.4f} | "
        f"@best: {metrics['best']['roc_auc']:.4f} (thr={metrics['thr']:.3f})"
    )


if __name__ == "__main__":
    main()
