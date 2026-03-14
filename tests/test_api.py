"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Create a simple trained mock model
def create_mock_model():
    X = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] * 100
    )
    y = np.array([0, 1] * 50)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


MOCK_MODEL = create_mock_model()

SAMPLE_INPUT = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
}


@pytest.fixture
def client():
    with patch("src.api.main.load_model"):
        from src.api.main import app
        import src.api.main as main_module

        main_module.model = MOCK_MODEL
        return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_predictions" in data
    assert "churn_predictions" in data
    assert "model_loaded" in data


def test_predict_endpoint(client):
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "churn_probability" in data
    assert isinstance(data["churn"], bool)
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_no_model():
    with patch("src.api.main.load_model"):
        from src.api.main import app
        import src.api.main as main_module

        main_module.model = None
        test_client = TestClient(app)
        response = test_client.post("/predict", json=SAMPLE_INPUT)
        assert response.status_code == 503
