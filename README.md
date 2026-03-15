# 🔄 Telco Customer Churn - MLOps Pipeline

![CI](https://github.com/Eva270201/telco-churn-mlops/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/Eva270201/telco-churn-mlops/actions/workflows/cd.yml/badge.svg)

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for predicting customer churn in a telecom company. It covers the full lifecycle from data preprocessing and model training to API serving, containerization, CI/CD automation, and monitoring.

## 2. Problem Definition & Data

**Task:** Binary classification — predict whether a customer will churn (Yes/No).

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features including contract type, monthly charges, tenure, and services subscribed.

**Evaluation metrics:** Accuracy, F1-score, Precision, Recall.

**Results:**
| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | 79.6% | 0.57 |
| Logistic Regression | 79.9% | 0.59 |

## 3. System Architecture
```
Raw Data → Preprocessing → Training (MLflow) → Model Artifact
                                                      ↓
                                              FastAPI Inference API
                                                      ↓
                                              Docker Container
                                                      ↓
                                              CI/CD (GitHub Actions)
```

**Project structure:**
```
telco-churn-mlops/
├── src/
│   ├── data/        # preprocessing pipeline
│   ├── models/      # training script
│   └── api/         # FastAPI application
├── tests/           # unit tests
├── config/          # hyperparameters
├── .github/workflows/ # CI/CD
├── Dockerfile
└── pyproject.toml
```

## 4. MLOps Practices

- **UV** for reproducible Python environment management
- **Pre-commit hooks** (black, ruff) for code quality enforcement
- **MLflow** for experiment tracking — parameters, metrics, and model artifacts logged for every run
- **pytest** with 61% code coverage
- **Docker** for containerization of the inference service
- **GitHub Actions** for CI/CD automation

## 5. Monitoring & Reliability

The FastAPI application includes:
- `/health` endpoint — returns API and model status
- `/metrics` endpoint — tracks total requests, predictions, and churn rate
- Structured logging middleware — logs method, path, status code, and response time for every request

## 6. Team Collaboration

With Maizane Ali Ahamada project. All development tracked through Git with feature branches, atomic commits, and meaningful commit messages following conventional commits format (`feat:`, `fix:`, `chore:`).

## 7. Limitations & Future Work

**Current limitations:**
- Model served from local pickle file (not a model registry)
- In-memory metrics reset on restart (no persistent monitoring)
- No data drift detection

**Future work:**
- Integrate Prometheus + Grafana for persistent monitoring
- Add data versioning with DVC
- Deploy to cloud (Azure/Railway)
- Implement model retraining pipeline triggered by data drift

## 🎥 Demo Video

> Link to demo video: https://youtu.be/jrDY22teCrY

## 🚀 How to Run

**Local:**
```bash
uv sync
uv run python src/models/train.py --model random_forest
uv run uvicorn src.api.main:app --reload
```

**Docker:**
```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

**API Endpoints:**
- `GET /health` — health check
- `GET /metrics` — monitoring metrics
- `POST /predict` — churn prediction
- `GET /docs` — interactive Swagger UI
