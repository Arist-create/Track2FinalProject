"""
Stage: PREDICT — FINAL
Оффлайн-инференс обученного Pipeline.
Требует вход в СХЕМЕ FEATURES (как train_features.csv/test_features.csv).
Поддерживает:
  --input  data/features.csv  -> predictions.csv
  --single '{"limit_bal":20000, "sex":2, ...}' (все features-поля)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

# используем список исходных (preprocessor-input) фичей из пайплайна
from .pipeline import get_feature_columns

DEFAULT_MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"

def load_params() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_threshold(default_thr: float = 0.5) -> float:
    p = Path(METRICS_PATH)
    if not p.exists():
        return default_thr
    try:
        obj = json.loads(p.read_text())
        # train.py писал: {"best_thr":{"threshold": ...}}
        return float(obj.get("best_thr", {}).get("threshold", default_thr))
    except Exception:
        return default_thr

def load_model(path: str = DEFAULT_MODEL_PATH):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = get_feature_columns()
    expected = num + cat
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "Input must be in FEATURES schema (after build_features). "
            f"Missing columns: {missing}"
        )
    # жёстко упорядочим колонки под pipeline
    return df.loc[:, expected]

def predict_df(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = ensure_schema(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["probability"] = proba
    out["prediction"]  = pred
    out["threshold"]   = threshold
    return out

def main():
    parser = argparse.ArgumentParser(description="Offline credit scoring predictions (FEATURES schema)")
    parser.add_argument("--input", type=str, help="CSV with FEATURES columns")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--single", type=str, help="Single row as JSON with FEATURES columns")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model)
    threshold = load_threshold(default_thr=0.5)

    if args.single:
        row = json.loads(args.single)
        df = pd.DataFrame([row])
        out = predict_df(model, df, threshold)
        print(json.dumps({
            "prediction": int(out.loc[0, "prediction"]),
            "probability": float(out.loc[0, "probability"]),
            "threshold": float(threshold)
        }, indent=2))
        return 0

    if args.input:
        df = pd.read_csv(args.input)
        out = predict_df(model, df, threshold)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"✅ Saved: {args.output}  |  n={len(out)}  |  thr={threshold:.3f}")
        return 0

    print("❌ Provide --input CSV (FEATURES schema) or --single JSON.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
