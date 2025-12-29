# ======================================================
# ✅ Production-ready Dockerfile for Credit Risk Assessment API
# ======================================================

FROM python:3.10-slim AS base

# -------------------------------
# 1. Base system setup
# -------------------------------
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  curl \
  && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 2. Install Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 3. Copy project files
# -------------------------------
COPY core ./core
COPY trained_models ./trained_models
COPY params.yaml .

# Create working dirs used by DVC / MLflow
RUN mkdir -p raw_data/raw raw_data/processed mlruns

# -------------------------------
# 4. Security (non-root execution)
# -------------------------------
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# -------------------------------
# 5. Expose port & healthcheck
# -------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

# -------------------------------
# 6. Start FastAPI app
# -------------------------------
CMD ["uvicorn", "core.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
