"""
Tests for the ML model training and prediction pipeline.

Tests:
- Data preprocessing
- Model creation
- Prediction output format
- Feature importance extraction
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_sample_data(n_rows: int = 100) -> pd.DataFrame:
    """Create sample Telco Churn data for testing."""
    np.random.seed(42)
    
    data = {
        "customerID": [f"CUST-{i:04d}" for i in range(n_rows)],
        "gender": np.random.choice(["Male", "Female"], n_rows),
        "SeniorCitizen": np.random.choice([0, 1], n_rows),
        "Partner": np.random.choice(["Yes", "No"], n_rows),
        "Dependents": np.random.choice(["Yes", "No"], n_rows),
        "tenure": np.random.randint(0, 72, n_rows),
        "PhoneService": np.random.choice(["Yes", "No"], n_rows),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_rows),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_rows),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_rows),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_rows),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n_rows),
        "MonthlyCharges": np.round(np.random.uniform(18, 118, n_rows), 2),
        "TotalCharges": np.round(np.random.uniform(18, 8600, n_rows), 2),
        "Churn": np.random.choice(["Yes", "No"], n_rows, p=[0.27, 0.73]),
    }
    
    return pd.DataFrame(data)


class TestDataPreprocessing:
    """Test suite for data preprocessing pipeline."""
    
    def test_clean_data(self):
        """Test that clean_data handles missing values and converts types."""
        from src.data.preprocess import clean_data
        
        df = create_sample_data()
        df["TotalCharges"] = df["TotalCharges"].astype(object)
        # Introduce a blank TotalCharges
        df.loc[0, "TotalCharges"] = ""
        
        cleaned = clean_data(df)
        
        # Check customerID is removed
        assert "customerID" not in cleaned.columns
        
        # Check Churn is binary
        assert cleaned["Churn"].dtype in [np.int64, np.int32, int]
        assert set(cleaned["Churn"].unique()).issubset({0, 1})
        
        # Check no missing values
        assert cleaned.isnull().sum().sum() == 0
    
    def test_engineer_features(self):
        """Test that feature engineering creates expected columns."""
        from src.data.preprocess import clean_data, engineer_features
        
        df = create_sample_data()
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        
        # Check new features exist
        assert "tenure_bucket" in engineered.columns
        assert "monthly_to_total_ratio" in engineered.columns
        assert "service_count" in engineered.columns
        assert "avg_monthly_spend" in engineered.columns
    
    def test_encode_features(self):
        """Test that encoding produces numeric features."""
        from src.data.preprocess import clean_data, engineer_features, encode_features
        
        df = create_sample_data()
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        encoded, encoding_map = encode_features(engineered)
        
        # Check all columns are numeric
        for col in encoded.columns:
            if col != "Churn":
                assert encoded[col].dtype in [np.int64, np.int32, np.float64, int, float], \
                    f"Column {col} is not numeric: {encoded[col].dtype}"
        
        # Check encoding map is populated
        assert len(encoding_map) > 0


class TestModelTraining:
    """Test suite for model training."""
    
    def test_create_model(self):
        """Test model creation with default params."""
        from src.model.train import create_model
        from xgboost import XGBClassifier
        
        model = create_model()
        assert isinstance(model, XGBClassifier)
    
    def test_model_prediction_format(self):
        """Test that model predictions have correct format."""
        from src.data.preprocess import clean_data, engineer_features, encode_features
        from src.model.train import create_model
        
        df = create_sample_data(200)
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        encoded, _ = encode_features(engineered)
        
        X = encoded.drop(columns=["Churn"])
        y = encoded["Churn"]
        
        model = create_model({"n_estimators": 10, "max_depth": 3, "random_state": 42, "eval_metric": "logloss"})
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
        
        # Test probabilities
        probabilities = model.predict_proba(X[:5])
        assert probabilities.shape == (5, 2)
        assert all(0 <= p <= 1 for row in probabilities for p in row)


class TestModelEvaluation:
    """Test suite for model evaluation."""
    
    def test_classification_metrics(self):
        """Test that metrics are computed correctly."""
        from src.data.preprocess import clean_data, engineer_features, encode_features
        from src.model.evaluate import get_classification_metrics
        from src.model.train import create_model
        
        df = create_sample_data(200)
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        encoded, _ = encode_features(engineered)
        
        X = encoded.drop(columns=["Churn"])
        y = encoded["Churn"]
        
        model = create_model({"n_estimators": 10, "max_depth": 3, "random_state": 42, "eval_metric": "logloss"})
        model.fit(X, y)
        
        metrics = get_classification_metrics(model, X, y)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        
        # All metrics should be between 0 and 1
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} = {value} is out of range"
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        from src.data.preprocess import clean_data, engineer_features, encode_features
        from src.model.evaluate import get_feature_importance
        from src.model.train import create_model
        
        df = create_sample_data(200)
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        encoded, _ = encode_features(engineered)
        
        X = encoded.drop(columns=["Churn"])
        y = encoded["Churn"]
        
        model = create_model({"n_estimators": 10, "max_depth": 3, "random_state": 42, "eval_metric": "logloss"})
        model.fit(X, y)
        
        fi = get_feature_importance(model, list(X.columns), top_n=5)
        
        assert "top_features" in fi
        assert len(fi["top_features"]) == 5
        assert fi["top_features"][0]["rank"] == 1
