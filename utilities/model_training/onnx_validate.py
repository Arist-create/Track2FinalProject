# scripts/model_training/onnx_validate.py

import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
import torch
import torch.nn as nn

DATA_PATH = "data/processed/train.csv"

FEATURES_PATH = "models/nn/feature_columns.json"
SCALER_PATH = "models/nn/scaler.joblib"
TORCH_PATH = "models/nn/credit_nn.pth"

ONNX_PATH = "models/onnx/credit_nn.onnx"

N_SAMPLES = 256
SEED = 42
TOL = 1e-4


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_feature_cols(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json must be a non-empty list")
    return cols


def build_scaled_input(df: pd.DataFrame, feature_cols: list[str], scaler) -> np.ndarray:
    X_df = df[feature_cols].copy()
    X_scaled = scaler.transform(X_df)  # DataFrame -> no sklearn warning
    return np.asarray(X_scaled, dtype=np.float32)


def _extract_layer_index(key: str) -> int:
    parts = key.replace("/", ".").split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return 10**9


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    Remap keys like:
      net.0.weight -> 0.weight
      model.net.2.bias -> 2.bias
    So it can be loaded into nn.Sequential with layer indices 0/2/4...
    """
    prefixes_to_strip = ["net.", "model.", "module."]

    def strip_known_prefixes(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in prefixes_to_strip:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    new_sd = {}
    for k, v in state_dict.items():
        k2 = strip_known_prefixes(k)
        # if there is still "net." inside (e.g. model.net.0.weight), strip again
        if k2.startswith("net."):
            k2 = k2[len("net."):]
        new_sd[k2] = v
    return new_sd


def build_mlp_from_state_dict(state_dict: dict) -> nn.Module:
    """
    Reconstruct MLP structure from 2D weight tensors, then load weights
    after normalizing key names (net.* -> * for Sequential).
    """
    # Build layer list from original keys (still includes net.*)
    weight_keys = [k for k, v in state_dict.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if not weight_keys:
        raise ValueError("No linear weight tensors found in checkpoint (ndim==2).")

    weight_keys = sorted(weight_keys, key=_extract_layer_index)

    linears = []
    for wk in weight_keys:
        w = state_dict[wk]
        out_features, in_features = int(w.shape[0]), int(w.shape[1])

        bk = wk[:-6] + "bias"  # weight -> bias
        has_bias = bk in state_dict and getattr(state_dict[bk], "ndim", 0) == 1

        linears.append(nn.Linear(in_features, out_features, bias=has_bias))

    layers = []
    for i, lin in enumerate(linears):
        layers.append(lin)
        if i < len(linears) - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    # Normalize keys to match Sequential
    sd_norm = normalize_state_dict_keys(state_dict)

    # Load strictly now that keys should align
    missing, unexpected = model.load_state_dict(sd_norm, strict=False)

    # Ensure core weights are loaded
    expected_weight_keys = [f"{i}.weight" for i in range(0, len(model), 2)]  # 0,2,4,...
    loaded_weights = sum(1 for k in expected_weight_keys if k in sd_norm)
    if loaded_weights == 0:
        raise ValueError(f"Failed to map keys. missing={missing}, unexpected={unexpected}")

    model.eval()
    return model


def main():
    feature_cols = load_feature_cols(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(DATA_PATH)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_SAMPLES, len(df)), replace=False)
    df_batch = df.iloc[idx].reset_index(drop=True)

    X_scaled = build_scaled_input(df_batch, feature_cols, scaler)

    ckpt = torch.load(TORCH_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and any(str(k).endswith("weight") for k in ckpt.keys()):
        state_dict = ckpt
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format in credit_nn.pth")

    torch_model = build_mlp_from_state_dict(state_dict)

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(X_scaled)).numpy().reshape(-1).astype(np.float32)
        torch_probs = sigmoid(torch_logits).astype(np.float32)

    # --- ONNX ---
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    onnx_out = sess.run([out_name], {input_name: X_scaled})[0]
    onnx_vec = np.asarray(onnx_out, dtype=np.float32).reshape(-1)

    # ONNX may output logits or probs -> check both
    onnx_as_logits_probs = sigmoid(onnx_vec).astype(np.float32)
    diff_logits = np.abs(torch_probs - onnx_as_logits_probs)

    diff_probs = np.abs(torch_probs - onnx_vec)

    if diff_logits.mean() <= diff_probs.mean():
        mode = "onnx_output=logits (applied sigmoid)"
        diff = diff_logits
    else:
        mode = "onnx_output=probability (no sigmoid)"
        diff = diff_probs

    print(f"Samples: {len(torch_probs)}")
    print(f"Mode: {mode}")
    print(f"MAX diff:  {diff.max():.10f}")
    print(f"MEAN diff: {diff.mean():.10f}")

    if diff.max() < TOL:
        print(f"OK: ONNX совпадает с PyTorch (diff < {TOL})")
    else:
        raise SystemExit(f"FAIL: ONNX != PyTorch (max diff = {diff.max()})")


if __name__ == "__main__":
    main()
