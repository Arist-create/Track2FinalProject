# scripts/model_training/quantize_onnx.py

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

ONNX_MODEL_PATH = "models/onnx/credit_nn.onnx"
ONNX_QUANT_PATH = "models/onnx/credit_nn_quant.onnx"

# Квантование (динамическое) — самое простое и стабильное
quantize_dynamic(
    model_input=ONNX_MODEL_PATH,
    model_output=ONNX_QUANT_PATH,
    weight_type=QuantType.QInt8
)

print(f"Квантованная ONNX сохранена: {ONNX_QUANT_PATH}")

# Печатаем размеры файлов
size_fp = os.path.getsize(ONNX_MODEL_PATH)
size_q = os.path.getsize(ONNX_QUANT_PATH)

print(f"Размер обычной ONNX: {size_fp/1024:.2f} KB")
print(f"Размер quant ONNX:  {size_q/1024:.2f} KB")
print(f"Сжатие: {(size_fp/size_q):.2f}x")
