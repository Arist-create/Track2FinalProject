# src/api_onnx.py

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"

# In container: /app/models/... (copied by Dockerfile)
# Locally: models/... in repo root
CANDIDATE_ROOTS = [
    Path("models"),  # local run from repo root
    Path("/app/models"),  # docker
    MODELS_DIR,  # fallback
]

ONNX_PATH: Optional[Path] = None
SCALER_JSON_PATH: Optional[Path] = None
FEATURES_PATH: Optional[Path] = None

for root in CANDIDATE_ROOTS:
    p1 = root / "onnx" / "credit_nn.onnx"
    p2 = root / "nn" / "scaler_params.json"
    p3 = root / "nn" / "feature_columns.json"
    if p1.exists() and p2.exists() and p3.exists():
        ONNX_PATH = p1
        SCALER_JSON_PATH = p2
        FEATURES_PATH = p3
        break

if ONNX_PATH is None or SCALER_JSON_PATH is None or FEATURES_PATH is None:
    raise RuntimeError(
        "Model files not found. Expected:\n"
        "- models/onnx/credit_nn.onnx\n"
        "- models/nn/scaler_params.json\n"
        "- models/nn/feature_columns.json"
    )

# Load feature order
with FEATURES_PATH.open("r", encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)

# Load scaler params (NO sklearn)
with SCALER_JSON_PATH.open("r", encoding="utf-8") as f:
    scaler_params = json.load(f)

MEAN = np.asarray(scaler_params["mean"], dtype=np.float32)
SCALE = np.asarray(scaler_params["scale"], dtype=np.float32)
EPS = float(scaler_params.get("eps", 1e-12))

# ONNX session
sess = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature dictionary")


app = FastAPI(title="Credit Scoring ONNX API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _vectorize_and_scale(features: Dict[str, Any]) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(f"Missing features: {missing[:10]} " f"(total={len(missing)})"),
        )

    x = np.array(
        [float(features[c]) for c in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    x = (x - MEAN) / (SCALE + EPS)
    return x.reshape(1, -1)


@app.post("/predict")
def predict(req: PredictRequest):
    x = _vectorize_and_scale(req.features)

    y = sess.run([output_name], {input_name: x})[0]

    # y can be shape (1,1) probability or logits depending on export.
    prob = float(np.asarray(y).reshape(-1)[0])

    threshold = 0.5
    pred_class = int(prob >= threshold)

    return {
        "class": pred_class,
        "probability": prob,
        "threshold": threshold,
    }
