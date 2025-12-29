# scripts/model_training/export_scaler_params.py

import json
from pathlib import Path

import joblib
import numpy as np


SCALER_PATH = Path("trained_models/nn/scaler.joblib")
FEATURES_PATH = Path("trained_models/nn/feature_columns.json")
OUT_PATH = Path("trained_models/nn/scaler_params.json")


def main():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing: {SCALER_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_PATH}")

    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    mean = np.asarray(getattr(scaler, "mean_", None), dtype=float)
    scale = np.asarray(getattr(scaler, "scale_", None), dtype=float)

    if mean.size == 0 or scale.size == 0:
        raise ValueError("Scaler does not have mean_/scale_ attributes")

    if len(features) != mean.shape[0] or len(features) != scale.shape[0]:
        raise ValueError(
            f"Features mismatch: features={len(features)} mean={mean.shape[0]} scale={scale.shape[0]}"
        )

    payload = {
        "features": features,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "eps": 1e-12,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK: saved {OUT_PATH} (n_features={len(features)})")


if __name__ == "__main__":
    main()
