# scripts/model_training/onnx_conversion.py

import os
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn

DATA_PATH = "raw_data/processed/train.csv"

PT_MODEL_PATH = "trained_models/nn/credit_nn.pth"
SCALER_PATH = "trained_models/nn/scaler.joblib"
FEATURES_PATH = "trained_models/nn/feature_columns.json"

ONNX_DIR = "trained_models/onnx"
ONNX_MODEL_PATH = f"{ONNX_DIR}/credit_nn.onnx"

os.makedirs(ONNX_DIR, exist_ok=True)

# --- load feature columns ---
with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)

# --- load scaler (just to ensure it exists; not used in export itself) ---
_ = joblib.load(SCALER_PATH)

# --- input dim from features list ---
input_dim = len(feature_cols)

# --- same architecture as train_nn.py ---
class CreditNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = CreditNN(input_dim)
model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, input_dim)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

print(f"ONNX модель сохранена: {ONNX_MODEL_PATH}")
print("Важно: вход в ONNX должен быть уже scale() и в порядке feature_columns.json")
