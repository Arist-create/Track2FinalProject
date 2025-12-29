# scripts/model_training/benchmark_inference.py

import json
import time
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
import torch
import torch.nn as nn

DATA_PATH = "raw_data/processed/train.csv"

FEATURES_PATH = "trained_models/nn/feature_columns.json"
SCALER_PATH = "trained_models/nn/scaler.joblib"

PT_MODEL_PATH = "trained_models/nn/credit_nn.pth"

ONNX_FP32 = "trained_models/onnx/credit_nn.onnx"
ONNX_INT8 = "trained_models/onnx/credit_nn_quant.onnx"

N_REQUESTS = 5000
WARMUP = 300
SEED = 123


def load_feature_cols(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json must be non-empty list")
    return cols


def build_scaled_input(df: pd.DataFrame, feature_cols: list[str], scaler) -> np.ndarray:
    X_df = df[feature_cols].copy()
    X_scaled = scaler.transform(X_df)
    return np.asarray(X_scaled, dtype=np.float32)


def make_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def benchmark_onnx(onnx_path: str, X_scaled: np.ndarray):
    sess = make_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    feed = {input_name: X_scaled}

    for _ in range(WARMUP):
        sess.run([out_name], feed)

    t0 = time.perf_counter()
    for _ in range(N_REQUESTS):
        sess.run([out_name], feed)
    t1 = time.perf_counter()

    total = t1 - t0
    rps = N_REQUESTS / total if total > 0 else float("inf")
    latency_ms = (total / N_REQUESTS) * 1000.0 if N_REQUESTS > 0 else 0.0

    print(f"\nМодель: {onnx_path} (ONNXRuntime CPU)")
    print(f"Время ({N_REQUESTS} запросов): {total:.4f} сек")
    print(f"Скорость: {rps:.2f} req/s")
    print(f"Latency: {latency_ms:.4f} ms/req")


class CreditNN(nn.Module):
    # ДОЛЖНО СОВПАДАТЬ с train_nn.py: 64 -> 32 -> 1
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def benchmark_pytorch(pt_path: str, X_scaled: np.ndarray):
    device = torch.device("cpu")

    n_features = int(X_scaled.shape[1])
    model = CreditNN(n_features).to(device)

    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = torch.from_numpy(X_scaled.astype(np.float32)).to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_REQUESTS):
            _ = model(x)
    t1 = time.perf_counter()

    total = t1 - t0
    rps = N_REQUESTS / total if total > 0 else float("inf")
    latency_ms = (total / N_REQUESTS) * 1000.0 if N_REQUESTS > 0 else 0.0

    print(f"\nМодель: {pt_path} (PyTorch CPU)")
    print(f"Время ({N_REQUESTS} запросов): {total:.4f} сек")
    print(f"Скорость: {rps:.2f} req/s")
    print(f"Latency: {latency_ms:.4f} ms/req")


def main():
    feature_cols = load_feature_cols(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(DATA_PATH)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    rng = np.random.default_rng(SEED)
    i = int(rng.integers(0, len(df)))
    row = df.iloc[i : i + 1].reset_index(drop=True)

    X_scaled = build_scaled_input(row, feature_cols, scaler)

    print("== Benchmark (CPU) ==")
    benchmark_pytorch(PT_MODEL_PATH, X_scaled)
    benchmark_onnx(ONNX_FP32, X_scaled)
    benchmark_onnx(ONNX_INT8, X_scaled)


if __name__ == "__main__":
    main()
