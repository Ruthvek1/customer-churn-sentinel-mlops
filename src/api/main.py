"""
FastAPI application — the production ML serving layer.

Serves churn predictions via REST API with:
- Automatic Swagger/OpenAPI documentation
- CORS for dashboard integration
- Model loading on startup (not per-request)
- Structured JSON logging
- Health checks for container orchestration

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import monitoring, predictions, retraining
from src.model.predict import ChurnPredictor
from src.monitoring.alerts import AlertManager
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.prediction_logger import PredictionLogger
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# --- Global state (loaded once on startup) ---
_predictor = None
_prediction_logger = None
_drift_detector = None
_alert_manager = None
app_start_time = time.time()


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_predictor() -> ChurnPredictor:
    """Get the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


def get_prediction_logger() -> PredictionLogger:
    """Get the global prediction logger instance."""
    global _prediction_logger
    if _prediction_logger is None:
        _prediction_logger = PredictionLogger()
    return _prediction_logger


def get_drift_detector() -> DriftDetector:
    """Get the global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Loads all services into memory on startup for fast inference.
    """
    global _predictor, _prediction_logger, _drift_detector, _alert_manager, app_start_time
    
    setup_logging()
    app_start_time = time.time()
    
    logger.info("starting_application")
    
    try:
        # Load model into memory (once, not per-request)
        _predictor = ChurnPredictor()
        logger.info("model_loaded_on_startup")
    except FileNotFoundError:
        logger.warning("model_not_found_will_load_on_first_request")
        _predictor = None
    
    # Initialize monitoring services
    _prediction_logger = PredictionLogger()
    _drift_detector = DriftDetector()
    _alert_manager = AlertManager()
    
    logger.info("all_services_initialized")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("shutting_down_application")


# --- Create FastAPI app ---
config = load_config()
api_config = config.get("api", {})

app = FastAPI(
    title=api_config.get("title", "Churn Prediction API"),
    description=api_config.get("description", "Production ML API with drift monitoring"),
    version=api_config.get("version", "1.0.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predictions.router)
app.include_router(monitoring.router)
app.include_router(retraining.router)


@app.get("/", tags=["Root"])
async def root():
    """API root — welcome message and useful links."""
    return {
        "message": "🔮 Churn Prediction API — Production ML System with Drift Monitoring",
        "version": api_config.get("version", "1.0.0"),
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predictions/predict",
            "batch_predict": "POST /predictions/predict/batch",
            "drift_report": "GET /drift/report",
            "retrain": "POST /retrain",
            "model_info": "GET /model/info",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=True,
        log_level="info"
    )
