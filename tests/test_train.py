"""
Unit tests for training script.
"""

import pytest
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.train import evaluate_model, load_config, train_model


def test_load_config():
    config = load_config()
    assert "data" in config
    assert "model" in config
    assert "mlflow" in config


def test_evaluate_model():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


@patch("mlflow.start_run")
@patch("mlflow.log_metrics")
@patch("mlflow.log_params")
@patch("mlflow.log_param")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
def test_train_random_forest(
    mock_uri,
    mock_exp,
    mock_log_model,
    mock_log_param,
    mock_log_params,
    mock_log_metrics,
    mock_run,
):
    mock_run.return_value.__enter__ = lambda s: s
    mock_run.return_value.__exit__ = lambda s, *a: None
    model, metrics = train_model(model_type="random_forest")
    assert isinstance(model, RandomForestClassifier)
    assert "accuracy" in metrics


@patch("mlflow.start_run")
@patch("mlflow.log_metrics")
@patch("mlflow.log_params")
@patch("mlflow.log_param")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
def test_train_logistic_regression(
    mock_uri,
    mock_exp,
    mock_log_model,
    mock_log_param,
    mock_log_params,
    mock_log_metrics,
    mock_run,
):
    mock_run.return_value.__enter__ = lambda s: s
    mock_run.return_value.__exit__ = lambda s, *a: None
    model, metrics = train_model(model_type="logistic_regression")
    assert isinstance(model, LogisticRegression)
    assert "accuracy" in metrics


def test_train_invalid_model():
    with pytest.raises(ValueError):
        train_model(model_type="invalid_model")
