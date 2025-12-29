set -e

echo "== 1) Общая инфа =="
pwd
git status -sb
echo

echo "== 2) В Git локально =="
git ls-files | sort > /tmp/local_tracked.txt
wc -l /tmp/local_tracked.txt
echo

echo "== 3) На GitHub (origin/main) =="
git fetch origin
git ls-tree -r --name-only origin/main | sort > /tmp/remote_tracked.txt
wc -l /tmp/remote_tracked.txt
echo

echo "== 4) Сравнение лок./удал. отслеживаемых файлов =="
if diff -u /tmp/local_tracked.txt /tmp/remote_tracked.txt; then
  echo "✅ Списки совпадают"
else
  echo "⚠️ Есть различия (см. diff выше)"
fi
echo

echo "== 5) Обязательные файлы в Git =="
required=(
  ".github/workflows/ci-cd.yml" "Dockerfile" "README.md" "SETUP_GUIDE.md" "mlflow_tracking.md"
  "dvc.yaml" "dvc.lock" "params.yaml" "pyproject.toml" "requirements.txt" "analysis/01_eda.ipynb"
  "src/__init__.py" "src/data/make_dataset.py" "src/data/validation.py" "src/features/build_features.py"
  "src/models/pipeline.py" "src/models/train.py" "src/models/predict.py" "src/api/app.py"
  "src/monitoring/drift.py" "src/monitoring/api_drift_test.py" "unit_tests/test_data.py"
  "unit_tests/test_features.py" "unit_tests/test_metrics.py"
)
missing=0
for f in "${required[@]}"; do
  if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "MISSING: $f"; missing=$((missing+1))
  fi
done
if [ $missing -eq 0 ]; then echo "✅ Все обязательные файлы присутствуют"; fi
echo

echo "== 6) Не должно быть в Git =="
echo "- raw CSV:"; git ls-files | grep -E '^raw_data/raw/.*\.csv$' || echo "✅ ок"
echo "- processed CSV:"; git ls-files | grep -E '^raw_data/processed/.*\.csv$' || echo "✅ ок"
echo "- модели/артефакты:"; git ls-files | grep -E '^trained_models/.*\.(pkl|joblib|png|json)$' || echo "✅ ок"
echo "- mlruns:"; git ls-files | grep -E '^mlruns/' || echo "✅ ок"
echo "- дубликат workflow с пробелом:"; git ls-files | grep -E '\.github/workflows/ci-cd\.yml\s$' || echo "✅ ок"
echo

echo "== 7) DVC статус =="
dvc status || true
echo "== 8) DVC файл для raw CSV =="
git ls-files | grep 'raw_data/raw/UCI_Credit_Card.csv.dvc' && echo "✅ DVC ok" || echo "❌ нет .dvc файла"
