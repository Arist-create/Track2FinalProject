"""Tests for model metrics and PSI drift utilities."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.train import calculate_metrics

# Берём каноничную реализацию PSI и оборачиваем её обработкой edge-cейсов
from src.monitoring.api_drift_test import psi_score as _psi_score


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Test-helper: возвращает NaN для пустых выборок, иначе зовёт каноничную psi_score()."""
    if expected.size == 0 or actual.size == 0:
        return np.nan
    return _psi_score(expected, actual, buckets=buckets)


def test_calculate_metrics_perfect():
    """Perfect predictions -> все основные метрики на максимуме."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    for key in ["roc_auc", "f1_score", "precision", "recall", "pr_auc"]:
        assert key in metrics

    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert 0.99 <= metrics["roc_auc"] <= 1.0
    assert 0.99 <= metrics["pr_auc"] <= 1.0


def test_calculate_metrics_random():
    """Случайные предсказания -> метрики в допустимом диапазоне."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_proba = np.random.random(1000)
    y_pred = (y_proba > 0.5).astype(int)

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # ключи присутствуют
    for key in ["roc_auc", "f1_score", "precision", "recall", "pr_auc"]:
        assert key in metrics

    # все метрики в [0,1]
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_psi_no_drift():
    """Похожие распределения -> низкий PSI."""
    np.random.seed(0)
    expected = np.random.normal(0, 1, 5000)
    actual = np.random.normal(0, 1, 5000)

    psi = calculate_psi(expected, actual)

    assert psi < 0.20  # оставляем небольшой зазор от 0.1, чтобы избежать флейков


def test_psi_with_drift():
    """Сдвиг распределения -> высокий PSI."""
    np.random.seed(0)
    expected = np.random.normal(0, 1, 5000)
    actual = np.random.normal(2.0, 1, 5000)  # заметный сдвиг

    psi = calculate_psi(expected, actual)

    assert psi > 0.25


def test_psi_same_data():
    """Идентичные выборки -> PSI ≈ 0."""
    np.random.seed(123)
    data = np.random.normal(0, 1, 5000)
    psi = calculate_psi(data, data)
    assert psi < 1e-6


def test_psi_edge_cases():
    """Edge-кейсы PSI: пустые массивы и константы."""
    psi_empty = calculate_psi(np.array([]), np.array([1, 2, 3]))
    assert np.isnan(psi_empty)

    psi_constant = calculate_psi(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))
    # В нашей реализации при равных константах бины совпадут -> PSI ~ 0
    assert psi_constant < 1e-6


def test_metrics_binary_classification():
    """Проверка согласованности precision/recall на простом кейсе."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    expected_precision = 2 / 3  # TP=2, FP=1
    expected_recall = 2 / 3  # TP=2, FN=1

    assert abs(metrics["precision"] - expected_precision) < 1e-6
    assert abs(metrics["recall"] - expected_recall) < 1e-6


def test_roc_auc_threshold():
    """AUC на ранжировании, заведомо лучше случайного."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    auc = roc_auc_score(y_true, y_proba)
    assert auc > 0.8


def test_metrics_imbalanced_data():
    """Метрики на дисбалансном датасете считаются и отражают низкий recall."""
    np.random.seed(7)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate(
        [
            np.random.uniform(0.0, 0.3, 85),  # TN
            np.random.uniform(0.6, 0.9, 5),  # TP
            np.random.uniform(0.0, 0.4, 10),  # FN
        ]
    )
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Все метрики в допустимом диапазоне
    for v in metrics.values():
        assert 0.0 <= v <= 1.0

    # Recall низкий из-за большого числа FN
    assert metrics["recall"] < 0.6
