# Benchmark report: NN → ONNX → Quantization

## Files
- PyTorch model: `models/nn/credit_nn.pth`
- ONNX model (FP32): `models/onnx/credit_nn.onnx`
- Quantized ONNX model (INT8): `models/onnx/credit_nn_quant.onnx`

---

## 1) ONNX conversion validation
Script: `scripts/model_training/onnx_validate.py`

Result:
- MAX diff: **0.0000000596**
- MEAN diff: **0.0000000151**
- Status: **OK (diff < 1e-4)**

This confirms numerical equivalence between PyTorch and ONNX models.

---

## 2) Model size comparison
Script: `scripts/model_training/quantize_onnx.py`

- ONNX FP32 size: **15.13 KB**
- ONNX INT8 size: **7.85 KB**
- Compression ratio: **1.93x smaller**

Quantization significantly reduces the model size.

---

## 3) Inference performance (CPU)
Script: `scripts/model_training/benchmark_inference.py`  
Settings: **5000 requests**, **batch_size = 1**, **CPUExecutionProvider**

| Model | Total time (s) | Throughput (req/s) | Latency (ms/req) |
|---|---:|---:|---:|
| PyTorch (`credit_nn.pth`) | **0.0879** | **56 882.66** | **0.0176** |
| ONNX FP32 (`credit_nn.onnx`) | **0.0225** | **222 159.28** | **0.0045** |
| ONNX INT8 (`credit_nn_quant.onnx`) | **0.0313** | **159 702.53** | **0.0063** |

---

## Conclusion
- ONNX conversion preserves model accuracy (negligible numerical difference).
- ONNX FP32 inference is **~4× faster** than native PyTorch on CPU.
- Quantization reduces model size by **~1.93×**, but on this small neural network introduces additional overhead, making INT8 inference slower than FP32 ONNX.
- ONNX FP32 is the optimal choice for CPU-based production deployment in this setup.
