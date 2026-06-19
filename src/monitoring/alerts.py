"""
Alert and retraining trigger logic.

Monitors drift detection results and automatically triggers
model retraining when data drift exceeds configured thresholds.

This closes the MLOps loop:
Data Drift Detected → Alert → Retrain → Deploy → Monitor → ...
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AlertManager:
    """
    Manages drift alerts and retraining triggers.
    
    Implements:
    - Drift threshold checking
    - Cooldown period between retrains
    - Alert history logging
    - Automatic retraining orchestration
    """
    
    def __init__(self):
        self.config = load_config()
        self.retrain_config = self.config["monitoring"]["retrain"]
        self.drift_config = self.config["monitoring"]["drift"]
        self.last_retrain_time = 0
        self._alert_history = []
    
    def check_and_alert(self, drift_result: Dict) -> Dict:
        """
        Evaluate drift results and determine if action is needed.
        
        Args:
            drift_result: Output from DriftDetector.detect_drift()
            
        Returns:
            Alert response with action recommendation
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_result.get("is_drifted", False),
            "drift_score": drift_result.get("drift_score", 0.0),
            "drifted_features": drift_result.get("drifted_features", []),
            "action": "none",
            "message": "",
        }
        
        if not drift_result.get("is_drifted", False):
            alert["action"] = "none"
            alert["message"] = "No drift detected. Model is stable."
            alert["severity"] = "info"
        else:
            drift_score = drift_result.get("drift_score", 0.0)
            
            if drift_score >= self.drift_config["feature_drift_threshold"]:
                alert["severity"] = "critical"
                alert["action"] = self._determine_action()
                alert["message"] = (
                    f"CRITICAL: {drift_score:.0%} of features have drifted. "
                    f"Drifted: {', '.join(drift_result.get('drifted_features', [])[:5])}"
                )
            else:
                alert["severity"] = "warning"
                alert["action"] = "monitor"
                alert["message"] = (
                    f"WARNING: Mild drift detected ({drift_score:.0%} features). "
                    f"Monitoring closely."
                )
        
        self._alert_history.append(alert)
        logger.info("alert_generated", severity=alert["severity"], action=alert["action"])
        
        return alert
    
    def _determine_action(self) -> str:
        """
        Determine the appropriate action based on configuration and cooldown.
        
        Returns:
            Action string: 'retrain', 'monitor', or 'none'
        """
        if not self.retrain_config["auto_retrain"]:
            return "notify"
        
        # Check cooldown
        time_since_retrain = time.time() - self.last_retrain_time
        if time_since_retrain < self.retrain_config["cooldown_seconds"]:
            remaining = self.retrain_config["cooldown_seconds"] - time_since_retrain
            logger.info(
                "retrain_cooldown_active",
                remaining_seconds=round(remaining)
            )
            return "monitor"
        
        return "retrain"
    
    def trigger_retrain(self) -> Dict:
        """
        Trigger model retraining.
        
        Returns:
            Retraining status dictionary
        """
        from src.data.preprocess import run_preprocessing_pipeline
        from src.model.train import train_model
        
        logger.info("retraining_triggered")
        self.last_retrain_time = time.time()
        
        try:
            # Re-run preprocessing
            X_train, X_test, y_train, y_test, metadata = run_preprocessing_pipeline()
            
            # Train new model
            model, experiment = train_model(
                X_train, y_train, X_test, y_test,
                tune=True, save=True
            )
            
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "new_model_version": experiment["model_version"],
                "metrics": experiment["metrics"],
                "training_time": experiment["training_time_seconds"],
            }
            
            logger.info("retraining_complete", **result)
            return result
            
        except Exception as e:
            error_result = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            logger.error("retraining_failed", error=str(e))
            return error_result
    
    def get_alert_history(self, limit: int = 50) -> list:
        """Get recent alert history."""
        return self._alert_history[-limit:]
    
    def should_check_drift(self, prediction_count: int) -> bool:
        """
        Determine if it's time to run a drift check.
        
        Args:
            prediction_count: Total predictions since last check
            
        Returns:
            True if drift check should be performed
        """
        check_interval = self.drift_config["check_interval"]
        return prediction_count > 0 and prediction_count % check_interval == 0
