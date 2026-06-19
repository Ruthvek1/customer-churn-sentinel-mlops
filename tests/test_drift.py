"""
Tests for drift detection and monitoring.

Tests:
- Prediction logger (SQLite operations)
- Drift detector initialization
- Alert manager logic
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPredictionLogger:
    """Test suite for the SQLite prediction logger."""
    
    @pytest.fixture
    def logger(self, tmp_path):
        """Create a temporary prediction logger."""
        from src.monitoring.prediction_logger import PredictionLogger
        
        db_path = str(tmp_path / "test_predictions.db")
        return PredictionLogger(db_path=db_path)
    
    def test_log_prediction(self, logger):
        """Test logging a single prediction."""
        logger.log_prediction(
            input_features={"tenure": 12, "MonthlyCharges": 50.0},
            prediction=1,
            churn_probability=0.75,
            risk_level="HIGH",
            model_version="v_test"
        )
        
        count = logger.get_prediction_count()
        assert count == 1
    
    def test_get_recent_predictions(self, logger):
        """Test retrieving recent predictions."""
        # Log multiple predictions
        for i in range(5):
            logger.log_prediction(
                input_features={"tenure": i * 10, "MonthlyCharges": 50.0 + i},
                prediction=i % 2,
                churn_probability=0.5 + (i * 0.1),
                risk_level="HIGH" if i % 2 else "LOW",
                model_version="v_test"
            )
        
        predictions = logger.get_recent_predictions(limit=3)
        assert len(predictions) == 3
        
        # Most recent first
        assert predictions[0]["input_features"]["tenure"] == 40
    
    def test_prediction_stats(self, logger):
        """Test aggregate statistics."""
        for i in range(10):
            logger.log_prediction(
                input_features={"tenure": i},
                prediction=1 if i < 3 else 0,
                churn_probability=0.8 if i < 3 else 0.2,
                risk_level="HIGH" if i < 3 else "LOW",
                model_version="v_test"
            )
        
        stats = logger.get_prediction_stats()
        assert stats["total"] == 10
        assert stats["churn_count"] == 3
        assert stats["no_churn_count"] == 7
    
    def test_log_drift_check(self, logger):
        """Test logging drift check results."""
        logger.log_drift_check(
            is_drifted=True,
            drift_score=0.65,
            drifted_features=["tenure", "MonthlyCharges"],
            total_features=20,
            report_path="/tmp/report.html"
        )
        
        history = logger.get_drift_history(limit=1)
        assert len(history) == 1
        assert history[0]["is_drifted"] is True
        assert history[0]["drift_score"] == 0.65
        assert "tenure" in history[0]["drifted_features"]
    
    def test_log_retrain_event(self, logger):
        """Test logging retraining events."""
        logger.log_retrain_event(
            trigger="drift_detected",
            old_model_version="v_001",
            new_model_version="v_002",
            old_metrics={"accuracy": 0.8},
            new_metrics={"accuracy": 0.85},
            status="completed"
        )
        
        history = logger.get_retrain_history(limit=1)
        assert len(history) == 1
        assert history[0]["trigger"] == "drift_detected"
        assert history[0]["new_metrics"]["accuracy"] == 0.85


class TestDriftDetector:
    """Test suite for drift detection."""
    
    def test_detector_with_no_reference(self, tmp_path):
        """Test drift detector when no reference data exists."""
        from src.monitoring.drift_detector import DriftDetector
        
        detector = DriftDetector(
            reference_path=str(tmp_path / "nonexistent.csv")
        )
        
        current = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        })
        
        result = detector.detect_drift(current)
        assert result["is_drifted"] is False
        assert "error" in result or result["drift_score"] == 0.0
    
    def test_detector_with_reference(self, tmp_path):
        """Test drift detector with matching reference data."""
        from src.monitoring.drift_detector import DriftDetector
        
        # Create reference data
        np.random.seed(42)
        reference = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 200),
            "feature2": np.random.normal(5, 2, 200),
        })
        ref_path = str(tmp_path / "reference.csv")
        reference.to_csv(ref_path, index=False)
        
        detector = DriftDetector(reference_path=ref_path)
        
        # Same distribution — should NOT drift
        current_same = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
        })
        
        result = detector.detect_drift(current_same, generate_report=False)
        assert isinstance(result["is_drifted"], bool)
        assert isinstance(result["drift_score"], float)
    
    def test_detector_detects_drift(self, tmp_path):
        """Test that detector identifies significant drift."""
        from src.monitoring.drift_detector import DriftDetector
        
        # Create reference data
        np.random.seed(42)
        reference = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 500),
            "feature2": np.random.normal(5, 2, 500),
        })
        ref_path = str(tmp_path / "reference.csv")
        reference.to_csv(ref_path, index=False)
        
        detector = DriftDetector(reference_path=ref_path)
        
        # Very different distribution — SHOULD drift
        current_drifted = pd.DataFrame({
            "feature1": np.random.normal(10, 1, 100),  # Mean shifted from 0 to 10
            "feature2": np.random.normal(50, 2, 100),  # Mean shifted from 5 to 50
        })
        
        result = detector.detect_drift(current_drifted, generate_report=False)
        assert result["is_drifted"] is True
        assert len(result["drifted_features"]) > 0
    
    def test_min_samples_check(self, tmp_path):
        """Test that small datasets are rejected."""
        from src.monitoring.drift_detector import DriftDetector
        
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})
        ref_path = str(tmp_path / "reference.csv")
        reference.to_csv(ref_path, index=False)
        
        detector = DriftDetector(reference_path=ref_path)
        
        # Only 5 samples — below min threshold
        current = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        result = detector.detect_drift(current, generate_report=False)
        
        assert result["is_drifted"] is False
        assert "error" in result


class TestAlertManager:
    """Test suite for alert management."""
    
    def test_no_drift_alert(self):
        """Test alert when no drift detected."""
        from src.monitoring.alerts import AlertManager
        
        manager = AlertManager()
        
        alert = manager.check_and_alert({
            "is_drifted": False,
            "drift_score": 0.05,
            "drifted_features": [],
        })
        
        assert alert["action"] == "none"
        assert alert["severity"] == "info"
    
    def test_critical_drift_alert(self):
        """Test alert for critical drift."""
        from src.monitoring.alerts import AlertManager
        
        manager = AlertManager()
        
        alert = manager.check_and_alert({
            "is_drifted": True,
            "drift_score": 0.75,
            "drifted_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        })
        
        assert alert["severity"] == "critical"
        assert alert["action"] in ["retrain", "notify"]
    
    def test_warning_drift_alert(self):
        """Test alert for mild drift."""
        from src.monitoring.alerts import AlertManager
        
        manager = AlertManager()
        
        alert = manager.check_and_alert({
            "is_drifted": True,
            "drift_score": 0.3,
            "drifted_features": ["tenure"],
        })
        
        assert alert["severity"] == "warning"
        assert alert["action"] == "monitor"
    
    def test_alert_history(self):
        """Test alert history tracking."""
        from src.monitoring.alerts import AlertManager
        
        manager = AlertManager()
        
        for i in range(5):
            manager.check_and_alert({
                "is_drifted": i % 2 == 0,
                "drift_score": 0.1 * i,
                "drifted_features": [],
            })
        
        history = manager.get_alert_history()
        assert len(history) == 5
