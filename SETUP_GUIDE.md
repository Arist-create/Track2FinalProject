SETUP GUIDE — PD-MODEL (CREDIT CARD DEFAULT PREDICTION)
──────────────────────────────────────────────────────────────
This document provides step-by-step instructions for setting up, running,
and verifying the full ML pipeline for the credit-card default prediction model.
The project includes automated stages for data preparation, model training,
experiment tracking, API deployment, and drift monitoring.
──────────────────────────────────────────────────────────────


1. ENVIRONMENT SETUP
──────────────────────────────────────────────────────────────
Option A — Conda (recommended)
--------------------------------
conda create -n pdmodel-310 python=3.10 -y
conda activate pdmodel-310
pip install -r requirements.txt

Option B — Virtual Environment
--------------------------------
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Verify installation:
python --version
dvc --version
mlflow --version


2. RUNNING THE FULL ML PIPELINE
──────────────────────────────────────────────────────────────
Execute the entire automated workflow:
dvc repro

Pipeline stages:
1. prepare  — data loading, cleaning, and train/test split
2. features — feature engineering and dataset enrichment
3. train    — model training with hyperparameter tuning and MLflow logging
4. drift    — PSI drift monitoring and report generation

Generated artifacts:
models/model.pkl
models/metrics.json
models/drift_report.json


4. EXPERIMENT TRACKING (MLFLOW)
──────────────────────────────────────────────────────────────
Launch the local MLflow interface:
mlflow ui --port 5000

Access via browser:
http://localhost:5000


5. FASTAPI SCORING SERVICE
──────────────────────────────────────────────────────────────
Run the scoring API locally:
uvicorn src.api.app:app --reload --port 8000

Health check:
curl http://localhost:8000/health

Example prediction request:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 30,
    "pay_0": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
  }'


6. DOCKER DEPLOYMENT
──────────────────────────────────────────────────────────────
Build and run the containerized version:
docker build -t credit-scoring-api .
docker run -d -p 8000:8000 credit-scoring-api

Verify container health:
curl http://localhost:8000/health


7. DRIFT MONITORING
──────────────────────────────────────────────────────────────
Run the PSI drift analysis manually:
python -m src.monitoring.drift

Output file:
models/drift_report.json

If PSI < 0.25, the drift is considered insignificant.


8. UNIT TESTS
──────────────────────────────────────────────────────────────
Execute integrity and functionality tests:
pytest -q


9. VERIFICATION CHECKLIST
──────────────────────────────────────────────────────────────
Component           | Description                                | Status
──────────────────────────────────────────────────────────────
Data preparation    : DVC stage "prepare" completes successfully |   ✓
Feature engineering : Derived features saved correctly           |   ✓
Model training      : Model and metrics generated                |   ✓
MLflow tracking     : Experiment visible in UI                   |   ✓
API service         : Returns valid predictions                  |   ✓
Docker container    : Passes health check                        |   ✓
Drift monitoring    : Report generated successfully              |   ✓


10. REFERENCES
──────────────────────────────────────────────────────────────
DVC Documentation       → https://dvc.org/doc
MLflow Documentation    → https://mlflow.org/
FastAPI Documentation   → https://fastapi.tiangolo.com/
Scikit-learn User Guide → https://scikit-learn.org/stable/user_guide.html
──────────────────────────────────────────────────────────────
