"""
Prediction API routes.

Endpoints:
- POST /predict          — Single customer churn prediction
- POST /predict/batch    — Batch predictions
- GET  /predictions/history — Recent prediction history
- GET  /predictions/stats   — Aggregate prediction statistics
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerPredictionRequest,
    FeatureImportanceResponse,
    PredictionHistoryResponse,
    PredictionResponse,
    PredictionStatsResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: CustomerPredictionRequest):
    """
    Predict customer churn for a single customer.
    
    Returns the prediction (churn/no churn), probability score,
    and risk level (LOW/MEDIUM/HIGH).
    """
    from src.api.main import get_predictor, get_prediction_logger
    
    try:
        predictor = get_predictor()
        pred_logger = get_prediction_logger()
        
        # Convert request to dict
        customer_data = request.model_dump()
        
        # Make prediction
        result = predictor.predict_single(customer_data)
        
        # Log prediction
        model_info = predictor.get_model_info()
        pred_logger.log_prediction(
            input_features=customer_data,
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            model_version=model_info.get("version", "unknown")
        )
        
        return PredictionResponse(**result)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers at once.
    
    Accepts a list of customer records and returns predictions for each.
    """
    from src.api.main import get_predictor, get_prediction_logger
    
    try:
        predictor = get_predictor()
        pred_logger = get_prediction_logger()
        
        results = []
        model_info = predictor.get_model_info()
        
        for customer in request.customers:
            customer_data = customer.model_dump()
            result = predictor.predict_single(customer_data)
            
            # Log each prediction
            pred_logger.log_prediction(
                input_features=customer_data,
                prediction=result["prediction"],
                churn_probability=result["churn_probability"],
                risk_level=result["risk_level"],
                model_version=model_info.get("version", "unknown")
            )
            
            results.append(PredictionResponse(**result))
        
        return BatchPredictionResponse(predictions=results, count=len(results))
    
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(limit: int = 50):
    """
    Get recent prediction history.
    
    Returns the last N predictions with their inputs and results.
    """
    from src.api.main import get_prediction_logger
    
    pred_logger = get_prediction_logger()
    predictions = pred_logger.get_recent_predictions(limit)
    
    return PredictionHistoryResponse(
        predictions=predictions,
        total=len(predictions)
    )


@router.get("/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats():
    """
    Get aggregate prediction statistics.
    
    Returns counts, averages, and risk distribution.
    """
    from src.api.main import get_prediction_logger
    
    pred_logger = get_prediction_logger()
    stats = pred_logger.get_prediction_stats()
    
    return PredictionStatsResponse(**stats)


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(top_n: int = 10):
    """
    Get global feature importance ranking.
    
    Shows which features the model considers most important
    for predicting churn.
    """
    from src.api.main import get_predictor
    
    predictor = get_predictor()
    features = predictor.get_feature_importance(top_n)
    
    return FeatureImportanceResponse(features=features)
