"""
Tests for FastAPI endpoints.

Tests:
- Root endpoint
- Prediction endpoints
- Health check
- API schema validation
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_customer():
    """Sample customer data for testing."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20,
    }


class TestAPISchemas:
    """Test API schema validation."""
    
    def test_customer_prediction_schema(self, sample_customer):
        """Test that valid customer data passes schema validation."""
        from src.api.schemas import CustomerPredictionRequest
        
        request = CustomerPredictionRequest(**sample_customer)
        assert request.gender == "Female"
        assert request.tenure == 12
        assert request.MonthlyCharges == 29.85
    
    def test_invalid_senior_citizen(self, sample_customer):
        """Test that invalid SeniorCitizen value is rejected."""
        from src.api.schemas import CustomerPredictionRequest
        from pydantic import ValidationError
        
        sample_customer["SeniorCitizen"] = 5  # Invalid
        
        with pytest.raises(ValidationError):
            CustomerPredictionRequest(**sample_customer)
    
    def test_negative_charges_rejected(self, sample_customer):
        """Test that negative charges are rejected."""
        from src.api.schemas import CustomerPredictionRequest
        from pydantic import ValidationError
        
        sample_customer["MonthlyCharges"] = -10  # Invalid
        
        with pytest.raises(ValidationError):
            CustomerPredictionRequest(**sample_customer)
    
    def test_batch_request_schema(self, sample_customer):
        """Test batch prediction request schema."""
        from src.api.schemas import BatchPredictionRequest, CustomerPredictionRequest
        
        batch = BatchPredictionRequest(
            customers=[
                CustomerPredictionRequest(**sample_customer),
                CustomerPredictionRequest(**sample_customer),
            ]
        )
        assert len(batch.customers) == 2
    
    def test_prediction_response_schema(self):
        """Test prediction response schema."""
        from src.api.schemas import PredictionResponse
        
        response = PredictionResponse(
            prediction=1,
            churn_label="Yes",
            churn_probability=0.85,
            risk_level="HIGH"
        )
        assert response.prediction == 1
        assert response.risk_level == "HIGH"
    
    def test_health_response_schema(self):
        """Test health response schema."""
        from src.api.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            database_connected=True,
            model_version="v_20250618",
            uptime_seconds=120.5
        )
        assert response.status == "healthy"


class TestRootEndpoint:
    """Test the API root endpoint."""
    
    def test_root_returns_welcome(self):
        """Test that root endpoint returns welcome message."""
        from src.api.main import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
