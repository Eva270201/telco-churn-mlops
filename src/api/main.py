"""
FastAPI application for Telco Customer Churn prediction.
"""

import logging
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn for Telco dataset",
    version="1.0.0",
)

model = None


def load_model():
    """Load model from pickle file."""
    global model
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from models/model.pkl")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")


class CustomerFeatures(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn: bool
    churn_probability: float
    model_version: str = "1.0.0"


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_data = np.array(
        [
            [
                features.gender,
                features.SeniorCitizen,
                features.Partner,
                features.Dependents,
                features.tenure,
                features.PhoneService,
                features.MultipleLines,
                features.InternetService,
                features.OnlineSecurity,
                features.OnlineBackup,
                features.DeviceProtection,
                features.TechSupport,
                features.StreamingTV,
                features.StreamingMovies,
                features.Contract,
                features.PaperlessBilling,
                features.PaymentMethod,
                features.MonthlyCharges,
                features.TotalCharges,
            ]
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return PredictionResponse(
        churn=bool(prediction),
        churn_probability=float(probability),
    )
