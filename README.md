# CREDIT-RISK-PREDICTOR

## Credit Card Default Risk Assessment

### Industrial MLOps Pipeline (Task 1 + Task 2)

---

## Overview

This repository contains a **full industrial-grade MLOps implementation** of a credit risk assessment system for predicting credit card default.
The project was developed as part of two academic tracks and represents an evolution from a classical ML pipeline to a **production-ready ML system** with automated delivery, deployment, and monitoring.

The solution covers the complete ML lifecycle:

- data preparation and validation
- model training and experimentation
- model optimization and export (ONNX)
- containerized inference services
- Kubernetes-based deployment
- Infrastructure as Code (IaC)
- CI/CD automation
- online service monitoring

---

## Project Structure & Tracks

The project consists of two logically separated but fully compatible parts.

### Track 1 — Automation of ML Development and Testing

Focuses on:

- reproducible ML pipelines (DVC)
- experiment tracking (MLflow)
- data and model validation
- drift detection

### Track 2 — Automation of Delivery and Deployment

Extends Track 1 to production by adding:

- ONNX optimization and INT8 quantization
- containerized inference services
- Kubernetes orchestration
- Infrastructure as Code (Terraform)
- CI/CD pipelines
- online monitoring and observability

---

## Repository Structure

The repository follows a **cookiecutter-data-science–style layout** with strict separation of concerns and full reproducibility.

```
.
├── raw_data/                # Raw and processed datasets (DVC tracked)
├── trained_models/          # Trained models, metrics, reports
├── analysis/                # EDA and exploratory analysis
├── core/                    # Core application code
│   ├── data/                # Data loading, cleaning, validation
│   ├── features/            # Feature engineering
│   ├── models/              # Training and inference logic
│   ├── api/                 # FastAPI inference service
│   └── monitoring/          # Data and prediction drift detection
├── utilities/               # Training, ONNX conversion, benchmarks
├── unit_tests/              # Unit tests (pytest)
├── deploy/
│   ├── docker/              # Dockerfiles (development / production)
│   └── kubernetes/          # Kubernetes manifests
│       └── monitoring/      # Prometheus / Grafana configuration
├── infrastructure/          # Terraform IaC (modules and environments)
├── .github/workflows/       # CI/CD pipelines
├── dvc.yaml                 # DVC pipeline definition
├── params.yaml              # Centralized configuration
├── README.md                # Project documentation
└── SETUP_GUIDE.md           # Extended setup and launch instructions
```

---

## Track 2 — Industrial Deployment & Delivery

### Model Preparation

- Neural network trained in **PyTorch**
- Exported to **ONNX**
- Numerical validation: PyTorch vs ONNX
- **INT8 quantization**
- CPU inference benchmarking

**Artifacts:**

```
trained_models/nn/credit_nn.pth
trained_models/onnx/credit_nn.onnx
trained_models/onnx/credit_nn_quant.onnx
reports/benchmarks.md
```

---

### API & Containerization

- **FastAPI** inference service
- **ONNX Runtime** for inference
- Endpoints:
  - `GET /health`
  - `GET /metrics`
  - `POST /predict`
- Separate Docker images for development and production

---

### Online Monitoring

The inference service exposes **Prometheus-compatible metrics** via the `/metrics` endpoint.

Collected metrics include:

- HTTP request rate and status codes
- request latency
- model inference latency

Monitoring stack:

- **Prometheus** for metrics collection
- **Grafana** dashboards (JSON definitions stored in the repository)

This setup provides real-time observability of service health and model performance.

---

### Kubernetes Deployment

- Dedicated namespaces:
  - `credit-staging`
  - `credit-prod`
- Rolling updates
- Liveness and readiness probes
- Resource limits and requests
- Horizontal Pod Autoscaler (HPA)
- Service and Ingress configuration

---

### Infrastructure as Code

- **Terraform** used for infrastructure provisioning
- Modular structure:
  - network
  - kubernetes
  - storage
  - monitoring
- Environments:
  - staging
  - production
- Configuration validated using `terraform validate`

---

### CI/CD

Automated pipelines include:

- code linting and unit tests
- ONNX validation and benchmarking
- Docker image build
- security scanning
- reproducible deployment artifacts

---

## Track 1 — ML Pipeline (Original Project)

This repository preserves the original ML pipeline developed in Track 1.

**Dataset:**  
Default of Credit Card Clients  
UCI Machine Learning Repository (Taiwan, 2005)

**Pipeline stages:**

- data preparation
- feature engineering
- model training
- experiment tracking (MLflow)
- API service
- drift monitoring
