# scripts/model_training/train_nn.py
# ---------------------------------
# Train neural network model for credit scoring
# Saves:
#   - PyTorch model weights
#   - feature columns order
#   - fitted scaler
# ---------------------------------

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =============================
# CONFIG
# =============================
DATA_PATH = "raw_data/processed/train.csv"
TARGET_COL = "default_payment"

MODEL_DIR = "trained_models/nn"
MODEL_PATH = f"{MODEL_DIR}/credit_nn.pth"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
FEATURES_PATH = f"{MODEL_DIR}/feature_columns.json"

RANDOM_STATE = 42
EPOCHS = 10
LR = 1e-3

# =============================
# REPRODUCIBILITY
# =============================
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =============================
# PREPARE DIR
# =============================
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
print("Loading data...")
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

feature_cols = [c for c in df.columns if c != TARGET_COL]

X = df[feature_cols]
y = df[TARGET_COL]

# save feature order
with open(FEATURES_PATH, "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"Features count: {len(feature_cols)}")

# =============================
# SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# =============================
# SCALE
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
print("Scaler saved")

# =============================
# MODEL
# =============================
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

model = CreditNN(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =============================
# TRAIN
# =============================
print("Training NN model...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(
    y_train.values, dtype=torch.float32
).view(-1, 1)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    preds = model(X_train_tensor)
    loss = criterion(preds, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - loss: {loss.item():.6f}")

# =============================
# EVALUATION
# =============================
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_preds = model(X_test_tensor).numpy().reshape(-1)

roc_auc = roc_auc_score(y_test, test_preds)
print(f"ROC-AUC (NN): {roc_auc:.4f}")

# =============================
# SAVE MODEL
# =============================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

print("Training finished successfully")
