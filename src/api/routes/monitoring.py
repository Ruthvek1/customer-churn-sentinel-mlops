"""
Monitoring API routes.

Endpoints:
- GET /health           — Health check
- GET /drift/report     — Run drift detection
- GET /drift/status     — Quick drift status
- GET /drift/history    — Drift check history
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DriftHistoryResponse,
    DriftReportResponse,
    DriftStatusResponse,
    HealthResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check.
    
    Verifies model is loaded and database is accessible.
    Used by Docker health checks and load balancers.
    """
    import time
    from src.api.main import get_predictor, get_prediction_logger, app_start_time
    
    try:
        predictor = get_predictor()
        pred_logger = get_prediction_logger()
        
        model_loaded = predictor.model is not None
        
        # Test DB connection
        try:
            pred_logger.get_prediction_count()
            db_connected = True
        except Exception:
            db_connected = False
        
        model_info = predictor.get_model_info() if model_loaded else {}
        
        status = "healthy" if (model_loaded and db_connected) else "degraded"
        
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            database_connected=db_connected,
            model_version=model_info.get("version", "unknown"),
            uptime_seconds=round(time.time() - app_start_time, 2)
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            database_connected=False,
            model_version="error",
            uptime_seconds=0
        )


@router.get("/drift/report", response_model=DriftReportResponse)
async def run_drift_report():
    """
    Run data drift detection.
    
    Compares recent prediction inputs against the training data
    distribution using statistical tests (KS, chi-squared).
    
    This is the key endpoint that demonstrates production ML monitoring.
    """
    from src.api.main import get_drift_detector, get_prediction_logger
    
    try:
        drift_detector = get_drift_detector()
        pred_logger = get_prediction_logger()
        
        # Get recent prediction features
        current_data = pred_logger.get_prediction_features_df(limit=500)
        
        if current_data.empty:
            return DriftReportResponse(
                is_drifted=False,
                drift_score=0.0,
                drifted_features=[],
                feature_drift_details={},
                timestamp="",
                error="No predictions logged yet. Make some predictions first."
            )
        
        # Run drift detection
        drift_result = drift_detector.detect_drift(current_data)
        
        # Log the drift check
        pred_logger.log_drift_check(
            is_drifted=drift_result.get("is_drifted", False),
            drift_score=drift_result.get("drift_score", 0.0),
            drifted_features=drift_result.get("drifted_features", []),
            total_features=len(drift_result.get("feature_drift_details", {})),
            report_path=drift_result.get("report_path")
        )
        
        return DriftReportResponse(**drift_result)
    
    except Exception as e:
        logger.error("drift_report_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@router.get("/drift/status", response_model=DriftStatusResponse)
async def get_drift_status():
    """
    Quick drift status check.
    
    Returns the current drift status based on the most recent check.
    """
    from src.api.main import get_prediction_logger
    
    try:
        pred_logger = get_prediction_logger()
        
        # Get latest drift check
        history = pred_logger.get_drift_history(limit=1)
        total_predictions = pred_logger.get_prediction_count()
        
        if not history:
            return DriftStatusResponse(
                status="unknown",
                last_check=None,
                drift_score=0.0,
                drifted_features_count=0,
                total_predictions=total_predictions
            )
        
        latest = history[0]
        
        if latest["is_drifted"]:
            status = "critical" if latest["drift_score"] >= 0.5 else "warning"
        else:
            status = "stable"
        
        return DriftStatusResponse(
            status=status,
            last_check=latest["timestamp"],
            drift_score=latest["drift_score"],
            drifted_features_count=latest["drifted_feature_count"],
            total_predictions=total_predictions
        )
    
    except Exception as e:
        logger.error("drift_status_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/history", response_model=DriftHistoryResponse)
async def get_drift_history(limit: int = 20):
    """
    Get drift check history.
    
    Returns past drift detection results for trend analysis.
    """
    from src.api.main import get_prediction_logger
    
    pred_logger = get_prediction_logger()
    history = pred_logger.get_drift_history(limit)
    
    return DriftHistoryResponse(
        history=history,
        total=len(history)
    )


@router.post("/drift/reset")
async def reset_drift_data():
    """
    Reset all prediction logs and drift check history.
    
    Clears the database so drift detection starts fresh.
    """
    from src.api.main import get_prediction_logger
    
    try:
        pred_logger = get_prediction_logger()
        result = pred_logger.reset_database()
        
        logger.info("drift_data_reset", **result)
        
        return {
            "status": "success",
            "message": "All prediction logs and drift history cleared.",
            **result,
        }
    except Exception as e:
        logger.error("reset_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
