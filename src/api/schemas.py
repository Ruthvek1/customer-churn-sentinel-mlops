"""
Pydantic schemas for FastAPI request/response models.

Ensures type-safe API contracts with automatic OpenAPI documentation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --- Prediction Schemas ---

class CustomerPredictionRequest(BaseModel):
    """Request body for single customer prediction."""
    
    model_config = {"json_schema_extra": {"examples": [{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}]}}
    
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    
    prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_label: str = Field(..., description="Human-readable prediction")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    
    customers: List[CustomerPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    predictions: List[PredictionResponse]
    count: int


class PredictionHistoryResponse(BaseModel):
    """Response for prediction history."""
    
    predictions: List[Dict]
    total: int


# --- Monitoring Schemas ---

class HealthResponse(BaseModel):
    """Health check response."""
    
    model_config = {"json_schema_extra": {"examples": [{"status": "healthy", "model_loaded": True, "database_connected": True, "model_version": "v1.0", "uptime_seconds": 120.5}]}}
    
    status: str
    model_loaded: bool
    database_connected: bool
    model_version: str = Field(default="unknown")
    uptime_seconds: float = Field(default=0)


class DriftFeatureDetail(BaseModel):
    """Drift details for a single feature."""
    
    drifted: bool
    drift_score: float
    stattest_name: str
    stattest_threshold: float


class DriftReportResponse(BaseModel):
    """Response for drift detection report."""
    
    is_drifted: bool
    drift_score: float
    drifted_features: List[str]
    feature_drift_details: Dict[str, Dict]
    report_path: Optional[str] = None
    timestamp: str
    error: Optional[str] = None


class DriftStatusResponse(BaseModel):
    """Quick drift status check."""
    
    status: str = Field(..., description="stable, warning, or critical")
    last_check: Optional[str] = None
    drift_score: float = 0.0
    drifted_features_count: int = 0
    total_predictions: int = 0


class DriftHistoryResponse(BaseModel):
    """Response for drift check history."""
    
    history: List[Dict]
    total: int


# --- Retraining Schemas ---

class RetrainRequest(BaseModel):
    """Request to trigger retraining."""
    
    reason: str = Field(default="manual", description="Reason for retraining")


class RetrainResponse(BaseModel):
    """Response for retraining request."""
    
    status: str = Field(..., description="success, failed, or in_progress")
    timestamp: str
    new_model_version: Optional[str] = None
    metrics: Optional[Dict] = None
    training_time: Optional[float] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response for model information."""
    
    model_path: str
    model_type: str
    n_features: int
    feature_names: List[str]
    version: str = "unknown"
    trained_at: str = "unknown"
    metrics: Dict = {}


class FeatureImportanceResponse(BaseModel):
    """Response for feature importance."""
    
    features: List[Dict]


# --- Statistics Schemas ---

class PredictionStatsResponse(BaseModel):
    """Response for prediction statistics."""
    
    total: int
    avg_probability: float = 0.0
    churn_count: int = 0
    no_churn_count: int = 0
    churn_rate: float = 0.0
    high_risk: int = 0
    medium_risk: int = 0
    low_risk: int = 0
