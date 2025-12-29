"""
PSI-мониторинг дрифта по инженерным признакам.
Берёт пути и параметры из params.yaml:
- data.train_features_path / data.test_features_path
- monitoring.features_to_monitor, monitoring.psi_threshold, monitoring.buckets
Пишет отчёт в models/drift_report.json и завершает процесс кодом:
  0 — дрифтов нет; 1 — есть 'drift' или 'missing' фичи.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

# ---------------- utils ----------------


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _uniform_bins(min_val: float, max_val: float, buckets: int) -> np.ndarray:
    """Равномерные границы бинов. Если диапазон вырожден — чуть расширяем."""
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        min_val, max_val = 0.0, 1.0
    if max_val <= min_val:
        max_val = float(min_val) + 1e-6
    return np.linspace(min_val, max_val, buckets + 1)


def _hist_proportions(values: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Доли наблюдений по бинам с защитой от нулей/деления."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        props = np.full_like(counts, fill_value=eps, dtype=float)
        return props / props.sum()
    props = counts.astype(float) / float(total)
    props = np.where(props == 0.0, eps, props)
    return props / props.sum()


def psi_score(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """PSI между train (expected) и test (actual) по общим границам бинов."""
    min_val = float(np.nanmin([np.nanmin(expected), np.nanmin(actual)]))
    max_val = float(np.nanmax([np.nanmax(expected), np.nanmax(actual)]))
    edges = _uniform_bins(min_val, max_val, buckets)
    p = _hist_proportions(expected, edges)
    q = _hist_proportions(actual, edges)
    return float(np.sum((p - q) * np.log(p / q)))


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Жёстко приводим признак к float; нечисловое → NaN."""
    return pd.to_numeric(series, errors="coerce").astype(float)


# ---------------- main ----------------


def main() -> int:
    print("📊 Running PSI drift monitoring...")
    P = load_params()
    mon = P.get("monitoring", {})

    features_to_monitor: List[str] = mon.get("features_to_monitor", [])
    psi_threshold: float = float(mon.get("psi_threshold", 0.25))
    buckets: int = int(mon.get("buckets", 10))

    # пути к файлам с ФИЧАМИ из params.yaml
    train_path = Path(P["data"]["train_features_path"])
    test_path = Path(P["data"]["test_features_path"])

    if not train_path.exists() or not test_path.exists():
        print("❌ Feature files not found. Run pipeline to generate features.")
        return 1

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    report: Dict[str, dict] = {}
    n_ok = n_watch = n_drift = 0
    missing: List[str] = []

    for feat in features_to_monitor:
        if feat not in train.columns or feat not in test.columns:
            report[feat] = {"status": "missing", "reason": "feature not found in feature files"}
            missing.append(feat)
            continue

        tr = _coerce_numeric(train[feat]).dropna().to_numpy()
        te = _coerce_numeric(test[feat]).dropna().to_numpy()
        if tr.size == 0 or te.size == 0:
            report[feat] = {"status": "skipped", "reason": "no valid numeric values"}
            continue

        score = psi_score(tr, te, buckets=buckets)
        if score >= psi_threshold:
            status = "drift"
            n_drift += 1
        elif score >= 0.10:
            status = "watch"
            n_watch += 1
        else:
            status = "ok"
            n_ok += 1

        report[feat] = {
            "status": status,
            "psi": round(score, 4),
            "threshold": psi_threshold,
            "bins": buckets,
            "n_train": int(tr.size),
            "n_test": int(te.size),
        }

    # агрегированный статус
    global_status = "ok"
    exit_code = 0
    if n_drift > 0 or len(missing) > 0:
        global_status = "drift"
        exit_code = 1
    elif n_watch > 0:
        global_status = "watch"

    summary = {
        "_summary": {
            "status": global_status,
            "psi_threshold": psi_threshold,
            "bins": buckets,
            "ok": n_ok,
            "watch": n_watch,
            "drift": n_drift,
            "missing": len(missing),
        }
    }
    full_report = {**summary, **report}

    out_path = Path("models/drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Drift report saved to {out_path}")
    print(json.dumps(full_report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
