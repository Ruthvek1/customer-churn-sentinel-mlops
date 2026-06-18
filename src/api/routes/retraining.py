"""
Retraining API routes.

Endpoints:
- POST /retrain          — Trigger model retraining
- GET  /retrain/history  — Retraining event history
- GET  /model/info       — Current model information
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.schemas import (
    ModelInfoResponse,
    RetrainRequest,
    RetrainResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Retraining"])

# Track retraining state
_retrain_status = {"status": "idle", "result": None}


def _run_retrain(reason: str):
    """Background retraining task."""
    global _retrain_status
    
    from src.api.main import get_alert_manager, get_predictor, get_prediction_logger, get_drift_detector
    
    try:
        _retrain_status = {"status": "in_progress", "result": None}
        
        alert_manager = get_alert_manager()
        predictor = get_predictor()
        pred_logger = get_prediction_logger()
        
        # Get old model info
        old_info = predictor.get_model_info()
        
        # Log retrain event
        pred_logger.log_retrain_event(
            trigger=reason,
            old_model_version=old_info.get("version", "unknown"),
            status="started"
        )
        
        # Run retraining
        result = alert_manager.trigger_retrain()
        
        if result["status"] == "success":
            # Reload model in predictor
            predictor.reload_model()
            
            # Reload reference data in drift detector
            drift_detector = get_drift_detector()
            drift_detector.reload_reference()
            
            # Log completion
            pred_logger.log_retrain_event(
                trigger=reason,
                old_model_version=old_info.get("version", "unknown"),
                new_model_version=result.get("new_model_version", "unknown"),
                old_metrics=old_info.get("metrics", {}),
                new_metrics=result.get("metrics", {}),
                status="completed"
            )
        
        _retrain_status = {"status": "completed", "result": result}
        
    except Exception as e:
        logger.error("retrain_background_error", error=str(e))
        _retrain_status = {
            "status": "failed",
            "result": {"status": "failed", "error": str(e)}
        }


@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(
    request: RetrainRequest = RetrainRequest(),
    background_tasks: BackgroundTasks = None
):
    """
    Trigger model retraining.
    
    Retrains the model with the current dataset and automatically
    hot-reloads the new model into the prediction service.
    
    Can be triggered manually or automatically when drift is detected.
    """
    global _retrain_status
    
    if _retrain_status["status"] == "in_progress":
        return RetrainResponse(
            status="in_progress",
            timestamp="",
            error="Retraining already in progress"
        )
    
    try:
        from datetime import datetime
        
        if background_tasks:
            # Run in background for non-blocking API
            background_tasks.add_task(_run_retrain, request.reason)
            
            return RetrainResponse(
                status="started",
                timestamp=datetime.now().isoformat(),
            )
        else:
            # Run synchronously
            _run_retrain(request.reason)
            result = _retrain_status.get("result", {})
            
            return RetrainResponse(
                status=result.get("status", "unknown"),
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                new_model_version=result.get("new_model_version"),
                metrics=result.get("metrics"),
                training_time=result.get("training_time"),
                error=result.get("error")
            )
    
    except Exception as e:
        logger.error("retrain_trigger_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrain/status")
async def get_retrain_status():
    """Get current retraining status."""
    return _retrain_status


@router.get("/retrain/history")
async def get_retrain_history(limit: int = 20):
    """Get retraining event history."""
    from src.api.main import get_prediction_logger
    
    pred_logger = get_prediction_logger()
    history = pred_logger.get_retrain_history(limit)
    
    return {"history": history, "total": len(history)}


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns model version, training date, performance metrics,
    and feature list.
    """
    from src.api.main import get_predictor
    
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    
    except Exception as e:
        logger.error("model_info_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
