"""
Training script for Telco Customer Churn prediction.
"""

import argparse
import logging

import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data.preprocess import preprocess_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def train_model(model_type: str = "random_forest"):
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        filepath=config["data"]["raw_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    if model_type == "random_forest":
        params = config["model"]["random_forest"]
        model = RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        params = config["model"]["logistic_regression"]
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    with mlflow.start_run(run_name=model_type):
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"Model: {model_type}")
        logger.info(f"Metrics: {metrics}")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
    )
    args = parser.parse_args()
    train_model(model_type=args.model)
