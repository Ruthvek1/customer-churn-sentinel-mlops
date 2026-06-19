"""
Prediction logger using SQLite.

Logs every prediction to a local database for:
- Drift detection (comparing recent predictions to training data)
- Dashboard visualization (prediction history, trends)
- Audit trail (who predicted what, when)

Uses SQLite for zero-config, file-based storage — perfect for
portfolio projects and local development.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PredictionLogger:
    """
    SQLite-based prediction logging service.
    
    Stores prediction inputs, outputs, and metadata for
    downstream monitoring and drift detection.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the prediction logger.
        
        Args:
            db_path: Path to SQLite database. If None, uses config.
        """
        config = load_config()
        if db_path is None:
            db_path = config["monitoring"]["db_path"]
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create the predictions table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_features TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                is_drifted INTEGER NOT NULL,
                drift_score REAL,
                drifted_features TEXT,
                total_features INTEGER,
                drifted_feature_count INTEGER,
                report_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger TEXT NOT NULL,
                old_model_version TEXT,
                new_model_version TEXT,
                old_metrics TEXT,
                new_metrics TEXT,
                status TEXT DEFAULT 'started',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("database_initialized", path=str(self.db_path))
    
    def log_prediction(
        self,
        input_features: dict,
        prediction: int,
        churn_probability: float,
        risk_level: str,
        model_version: str = "unknown"
    ):
        """
        Log a single prediction to the database.
        
        Args:
            input_features: Raw input features
            prediction: Binary prediction (0/1)
            churn_probability: Churn probability score
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            model_version: Model version string
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, input_features, prediction, churn_probability, risk_level, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(input_features),
            prediction,
            churn_probability,
            risk_level,
            model_version
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get recent predictions from the database.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            result["input_features"] = json.loads(result["input_features"])
            results.append(result)
        
        return results
    
    def get_prediction_features_df(self, limit: int = 500) -> pd.DataFrame:
        """
        Get recent prediction input features as a DataFrame.
        
        Used by the drift detector to compare against reference data.
        
        Args:
            limit: Maximum number of predictions to include
            
        Returns:
            DataFrame of input features
        """
        predictions = self.get_recent_predictions(limit)
        
        if not predictions:
            return pd.DataFrame()
        
        features = [p["input_features"] for p in predictions]
        return pd.DataFrame(features)
    
    def get_prediction_count(self) -> int:
        """Get total number of logged predictions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_prediction_stats(self) -> Dict:
        """
        Get aggregate statistics about predictions.
        
        Returns:
            Dictionary with prediction statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(churn_probability) as avg_probability,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as churn_count,
                SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as no_churn_count,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as medium_risk,
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) as low_risk
            FROM predictions
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row[0] == 0:
            return {"total": 0}
        
        return {
            "total": row[0],
            "avg_probability": round(row[1], 4) if row[1] else 0,
            "churn_count": row[2],
            "no_churn_count": row[3],
            "churn_rate": round(row[2] / row[0], 4) if row[0] > 0 else 0,
            "high_risk": row[4],
            "medium_risk": row[5],
            "low_risk": row[6],
        }
    
    def log_drift_check(
        self,
        is_drifted: bool,
        drift_score: float,
        drifted_features: List[str],
        total_features: int,
        report_path: Optional[str] = None
    ):
        """Log a drift check result."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO drift_checks
            (timestamp, is_drifted, drift_score, drifted_features, 
             total_features, drifted_feature_count, report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            int(is_drifted),
            drift_score,
            json.dumps(drifted_features),
            total_features,
            len(drifted_features),
            report_path
        ))
        
        conn.commit()
        conn.close()
    
    def get_drift_history(self, limit: int = 50) -> List[Dict]:
        """Get recent drift check results."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM drift_checks 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            result["drifted_features"] = json.loads(result["drifted_features"])
            result["is_drifted"] = bool(result["is_drifted"])
            results.append(result)
        
        return results
    
    def log_retrain_event(
        self,
        trigger: str,
        old_model_version: str,
        new_model_version: str = "",
        old_metrics: dict = None,
        new_metrics: dict = None,
        status: str = "started"
    ):
        """Log a retraining event."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO retrain_events
            (timestamp, trigger, old_model_version, new_model_version,
             old_metrics, new_metrics, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            trigger,
            old_model_version,
            new_model_version,
            json.dumps(old_metrics) if old_metrics else "{}",
            json.dumps(new_metrics) if new_metrics else "{}",
            status
        ))
        
        conn.commit()
        conn.close()
    
    def get_retrain_history(self, limit: int = 20) -> List[Dict]:
        """Get retraining event history."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM retrain_events 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            result["old_metrics"] = json.loads(result["old_metrics"])
            result["new_metrics"] = json.loads(result["new_metrics"])
            results.append(result)
        
        return results

    def reset_database(self) -> Dict:
        """
        Clear all predictions and drift check history.
        
        Returns:
            Dictionary with counts of deleted records.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM drift_checks")
        drift_count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM drift_checks")
        conn.commit()
        conn.close()
        
        logger.info(
            "database_reset",
            predictions_cleared=predictions_count,
            drift_checks_cleared=drift_count,
        )
        
        return {
            "predictions_cleared": predictions_count,
            "drift_checks_cleared": drift_count,
        }
