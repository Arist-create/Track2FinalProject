#!/usr/bin/env bash

echo "========== SYSTEM =========="
python --version || exit 1
docker --version || exit 1
kubectl version --client || exit 1
terraform version || exit 1

echo "========== PYTEST =========="
pytest -q || exit 1

echo "========== DVC =========="
dvc repro || exit 1

echo "========== ONNX VALIDATION =========="
python utilities/model_training/onnx_validate.py || exit 1

echo "========== QUANTIZATION =========="
python utilities/model_training/quantize_onnx.py || exit 1

echo "========== BENCHMARK =========="
python utilities/model_training/benchmark_inference.py || exit 1

echo "========== EVIDENTLY =========="
python -m src.monitoring.evidently_drift || exit 1
test -f reports/evidently_drift_report.html || exit 1

echo "========== FASTAPI =========="
uvicorn src.api_onnx:app --port 8000 >/tmp/api.log 2>&1 &
API_PID=$!
sleep 5
curl -f http://localhost:8000/health || exit 1
kill $API_PID

echo "========== DOCKER =========="
docker build -f deploy/docker/Dockerfile.api_onnx.prod -t credit-api:prod . || exit 1
docker run -d -p 8000:8000 --name credit-api-test credit-api:prod || exit 1
sleep 5
curl -f http://localhost:8000/health || exit 1
docker stop credit-api-test
docker rm credit-api-test

echo "========== K8S =========="
kubectl apply -f deploy/kubernetes/ || exit 1

echo "========== TERRAFORM =========="
terraform -chdir=infrastructure/environments/staging init -backend=false || exit 1
terraform -chdir=infrastructure/environments/staging validate || exit 1

echo "========== AIRFLOW =========="
test -f retraining/airflow/dags/retraining_dag.py || exit 1

echo "========== CI/CD =========="
test -f .github/workflows/ci-cd.yml || exit 1
test -f .github/workflows/track2-onnx-security.yml || exit 1

echo ""
echo "====================================="
echo "✅ ALL CHECKS PASSED"
echo "✅ READY FOR SUBMISSION"
echo "====================================="


