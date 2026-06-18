"""
Pydantic schemas for FastAPI request/response models.

Ensures type-safe API contracts with automatic OpenAPI documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --- Prediction Schemas ---

class CustomerPredictionRequest(BaseModel):
    """Request body for single customer prediction."""
    
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=29.85)
    TotalCharges: float = Field(..., ge=0, example=29.85)


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
    
    status: str = Field(..., example="healthy")
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
