"""
Data loading and preprocessing for Telco Customer Churn dataset.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from given filepath."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe: fix types, handle missing values."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    logger.info("Data cleaned successfully")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features with LabelEncoder."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
        logger.debug(f"Encoded column: {col}")
    return df


def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def scale_features(X_train, X_test):
    """Fit scaler on train, transform both train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline: load → clean → encode → split → scale."""
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, scaler = scale_features(X_train.values, X_test.values)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler