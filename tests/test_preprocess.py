"""
Unit tests for preprocessing pipeline.
"""

import pandas as pd
import pytest

from src.data.preprocess import (
    clean_data,
    encode_features,
    scale_features,
    split_features_target,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4", "5"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "tenure": [12, 24, 1, 36, 5],
            "MonthlyCharges": [50.0, 70.0, 30.0, 90.0, 45.0],
            "TotalCharges": ["600", "1680", " ", "3240", "225"],
            "Churn": ["No", "Yes", "No", "No", "Yes"],
        }
    )


def test_clean_data_removes_customer_id(sample_df):
    cleaned = clean_data(sample_df)
    assert "customerID" not in cleaned.columns


def test_clean_data_fixes_total_charges(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["TotalCharges"].dtype == float
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_encode_features_no_object_columns(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_features(cleaned)
    assert len(encoded.select_dtypes(include="object").columns) == 0


def test_split_features_target(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_features(cleaned)
    X, y = split_features_target(encoded)
    assert "Churn" not in X.columns
    assert len(X) == len(y)


def test_scale_features_output_shape(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_features(cleaned)
    X, y = split_features_target(encoded)
    X_train = X.iloc[:4].values
    X_test = X.iloc[4:].values
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
