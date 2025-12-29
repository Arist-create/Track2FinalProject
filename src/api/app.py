"""
FastAPI scoring service — FINAL
-----------------------------------
→ получает engineered-features (как в train_features.csv)
→ pipeline сам делает только preprocessing + инференс
→ использует сохранённый лучший порог из metrics.json
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")

# =========================================================
# API и модель
# =========================================================
app = FastAPI(
    title="Credit Scoring API",
    description="Predict probability of default (PD-model)",
    version="1.0.0",
)
model = None
best_threshold = 0.5


@app.on_event("startup")
async def _load_model():
    """Load trained sklearn pipeline and threshold on startup."""
    global model, best_threshold

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")

    if METRICS_PATH.exists():
        try:
            obj = json.loads(METRICS_PATH.read_text())
            best_threshold = float(obj.get("best_thr", {}).get("threshold", 0.5))
            print(f"✅ Threshold loaded: {best_threshold:.3f}")
        except Exception:
            best_threshold = 0.5


# =========================================================
# SCHEMA (engineered features only)
# =========================================================
class CreditFeatures(BaseModel):
    limit_bal: float
    sex: int = Field(..., ge=1, le=2)
    education: int = Field(..., ge=1, le=4)
    marriage: int = Field(..., ge=1, le=3)
    age: int = Field(..., ge=18, le=100)

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

    # engineered features
    age_bin: str
    utilization_last: float
    pay_delay_sum: int
    pay_delay_max: int
    bill_trend: float
    pay_trend: float
    bill_avg: float
    pay_amt_avg: float
    pay_to_bill_ratio: float


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_proba: float
    risk_level: str
    threshold: float


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
async def root():
    return {"message": "Credit Scoring API", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "ok" if model else "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: CreditFeatures):
    """Infer using trained sklearn Pipeline."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload.dict()])

        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= best_threshold)

        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.6 else "High"

        return PredictionResponse(
            predicted_class=pred,
            predicted_proba=proba,
            risk_level=risk,
            threshold=best_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
