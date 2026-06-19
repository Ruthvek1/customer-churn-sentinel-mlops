"""
Inference module for customer churn prediction.

Handles:
- Loading trained model artifacts
- Single and batch predictions
- Probability scores
- Feature importance for individual predictions
"""

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import yaml

from src.data.preprocess import preprocess_single_input
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ChurnPredictor:
    """
    Production-grade churn prediction service.
    
    Loads model once on initialization, then serves predictions
    without reloading — designed for API server lifecycle.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to model artifact. If None, loads latest.
        """
        self.config = load_config()
        
        if model_path is None:
            model_path = (
                Path(self.config["model"]["save_dir"]) /
                f"{self.config['model']['model_name']}.joblib"
            )
        
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.encoding_map = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run training first: python -m src.model.train"
            )
        
        self.model = joblib.load(self.model_path)
        
        # Load preprocessing metadata
        metadata_path = Path(self.config["data"]["processed_dir"]) / "preprocessing_metadata.joblib"
        if metadata_path.exists():
            self.metadata = joblib.load(metadata_path)
            self.feature_names = self.metadata.get("feature_names", [])
            self.encoding_map = self.metadata.get("encoding_map", {})
        
        logger.info(
            "predictor_initialized",
            model_path=str(self.model_path),
            n_features=len(self.feature_names) if self.feature_names else "unknown"
        )
    
    def predict_single(self, customer_data: dict) -> Dict:
        """
        Make a prediction for a single customer.
        
        Args:
            customer_data: Dictionary of customer features
            
        Returns:
            Dictionary with prediction, probability, and risk level
        """
        # Preprocess input
        X = preprocess_single_input(
            customer_data,
            self.feature_names,
            self.encoding_map
        )
        
        # Predict
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        result = {
            "prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
        }
        
        logger.info("prediction_made", **result)
        return result
    
    def predict_batch(self, customer_list: List[dict]) -> List[Dict]:
        """
        Make predictions for multiple customers.
        
        Args:
            customer_list: List of customer feature dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for customer in customer_list:
            result = self.predict_single(customer)
            results.append(result)
        
        logger.info("batch_prediction", count=len(results))
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        """
        Get global feature importance ranking.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of {feature, importance} dictionaries
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        result = []
        for i in range(min(top_n, len(self.feature_names))):
            idx = indices[i]
            result.append({
                "feature": self.feature_names[idx],
                "importance": round(float(importances[idx]), 4)
            })
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        import json
        
        info = {
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names or [],
        }
        
        # Load latest experiment for metrics
        experiment_path = Path("logs") / "latest_experiment.json"
        if experiment_path.exists():
            with open(experiment_path, "r") as f:
                experiment = json.load(f)
                info["version"] = experiment.get("model_version", "unknown")
                info["trained_at"] = experiment.get("timestamp", "unknown")
                info["metrics"] = experiment.get("metrics", {})
        
        return info
    
    def reload_model(self, model_path: Optional[str] = None):
        """
        Hot-reload the model (e.g., after retraining).
        
        Args:
            model_path: New model path. If None, reloads from same path.
        """
        if model_path:
            self.model_path = Path(model_path)
        
        self._load_artifacts()
        logger.info("model_reloaded", path=str(self.model_path))
