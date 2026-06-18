"""
Data drift detection using Evidently AI.

This is the core differentiator of the project — it demonstrates
understanding that models degrade in production when data distributions shift.

Implements:
- Feature-level drift detection (KS test for numerical, chi-squared for categorical)
- Overall drift scoring
- HTML report generation
- Structured drift results for API/dashboard consumption

Why Evidently?
- Industry-leading open-source drift detection library
- Built-in statistical tests (KS, PSI, chi-squared)
- Beautiful HTML reports that can be embedded in dashboards
- Active community and enterprise adoption
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DriftDetector:
    """
    Production drift detection service using Evidently AI.
    
    Compares incoming prediction data against the training reference
    dataset to detect distributional shifts that could degrade model
    performance.
    
    Statistical Tests Used:
    - Kolmogorov-Smirnov (KS) test: For numerical features
      Measures the maximum distance between two CDFs
    - Chi-squared test: For categorical features
      Tests independence of frequency distributions
    - Population Stability Index (PSI): Alternative drift metric
      Quantifies distribution shift magnitude
    """
    
    def __init__(self, reference_path: Optional[str] = None):
        """
        Initialize drift detector with reference dataset.
        
        Args:
            reference_path: Path to reference CSV. If None, uses config.
        """
        self.config = load_config()
        
        if reference_path is None:
            reference_path = self.config["data"]["reference_path"]
        
        self.reference_path = Path(reference_path)
        self.reference_data = None
        self.drift_config = self.config["monitoring"]["drift"]
        
        # Create report directory
        self.report_dir = Path(self.drift_config["report_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_reference()
    
    def _load_reference(self):
        """Load reference dataset (training data distribution)."""
        if self.reference_path.exists():
            self.reference_data = pd.read_csv(self.reference_path)
            logger.info(
                "reference_data_loaded",
                shape=self.reference_data.shape,
                path=str(self.reference_path)
            )
        else:
            logger.warning(
                "reference_data_not_found",
                path=str(self.reference_path)
            )
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        generate_report: bool = True
    ) -> Dict:
        """
        Run drift detection comparing current data against reference.
        
        Args:
            current_data: Recent prediction input features
            generate_report: Whether to save HTML report
            
        Returns:
            Dictionary with drift detection results:
            {
                "is_drifted": bool,
                "drift_score": float,
                "drifted_features": [...],
                "feature_drift_details": {...},
                "report_path": str or None,
                "timestamp": str
            }
        """
        if self.reference_data is None:
            return {
                "is_drifted": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "feature_drift_details": {},
                "report_path": None,
                "timestamp": datetime.now().isoformat(),
                "error": "No reference data available"
            }
        
        if len(current_data) < self.drift_config["min_samples"]:
            return {
                "is_drifted": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "feature_drift_details": {},
                "report_path": None,
                "timestamp": datetime.now().isoformat(),
                "error": f"Need at least {self.drift_config['min_samples']} samples, got {len(current_data)}"
            }
        
        # Align columns between reference and current data
        common_columns = list(
            set(self.reference_data.columns) & set(current_data.columns)
        )
        
        if not common_columns:
            return {
                "is_drifted": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "feature_drift_details": {},
                "report_path": None,
                "timestamp": datetime.now().isoformat(),
                "error": "No common columns between reference and current data"
            }
        
        ref_aligned = self.reference_data[common_columns].copy()
        cur_aligned = current_data[common_columns].copy()
        
        logger.info(
            "running_drift_detection",
            reference_shape=ref_aligned.shape,
            current_shape=cur_aligned.shape,
            n_features=len(common_columns)
        )
        
        # Run Evidently drift report
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        drift_report.run(
            reference_data=ref_aligned,
            current_data=cur_aligned,
        )
        
        # Extract results
        report_dict = drift_report.as_dict()
        
        # Parse drift results from Evidently output
        drift_result = self._parse_drift_results(report_dict, common_columns)
        
        # Generate HTML report if requested
        report_path = None
        if generate_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = str(self.report_dir / f"drift_report_{timestamp}.html")
            drift_report.save_html(report_path)
            drift_result["report_path"] = report_path
            logger.info("drift_report_saved", path=report_path)
        
        drift_result["timestamp"] = datetime.now().isoformat()
        
        logger.info(
            "drift_detection_complete",
            is_drifted=drift_result["is_drifted"],
            drift_score=drift_result["drift_score"],
            drifted_features_count=len(drift_result["drifted_features"])
        )
        
        return drift_result
    
    def _parse_drift_results(self, report_dict: dict, feature_names: list) -> Dict:
        """
        Parse Evidently report dictionary into structured results.
        
        Args:
            report_dict: Evidently report as dictionary
            feature_names: List of feature names
            
        Returns:
            Parsed drift results
        """
        results = {
            "is_drifted": False,
            "drift_score": 0.0,
            "drifted_features": [],
            "feature_drift_details": {},
            "report_path": None,
        }
        
        try:
            metrics = report_dict.get("metrics", [])
            
            # Dataset-level drift
            for metric in metrics:
                metric_result = metric.get("result", {})
                
                # DatasetDriftMetric
                if "dataset_drift" in metric_result:
                    results["is_drifted"] = metric_result["dataset_drift"]
                    results["drift_score"] = metric_result.get(
                        "share_of_drifted_columns", 0.0
                    )
                
                # DataDriftTable — per-feature details
                if "drift_by_columns" in metric_result:
                    drift_by_columns = metric_result["drift_by_columns"]
                    
                    for col_name, col_data in drift_by_columns.items():
                        is_col_drifted = col_data.get("drift_detected", False)
                        
                        results["feature_drift_details"][col_name] = {
                            "drifted": is_col_drifted,
                            "drift_score": col_data.get("drift_score", 0.0),
                            "stattest_name": col_data.get("stattest_name", "unknown"),
                            "stattest_threshold": col_data.get("stattest_threshold", 0.05),
                            "current_distribution": col_data.get("current", {}).get("small_distribution", {}),
                            "reference_distribution": col_data.get("reference", {}).get("small_distribution", {}),
                        }
                        
                        if is_col_drifted:
                            results["drifted_features"].append(col_name)
        
        except Exception as e:
            logger.error("drift_parsing_error", error=str(e))
            # Fallback: try to extract basic info
            results["error"] = str(e)
        
        return results
    
    def get_drift_summary(self, drift_result: Dict) -> str:
        """
        Generate a human-readable drift summary.
        
        Args:
            drift_result: Output from detect_drift()
            
        Returns:
            Formatted summary string
        """
        if drift_result.get("error"):
            return f"⚠️ Drift check skipped: {drift_result['error']}"
        
        if drift_result["is_drifted"]:
            features = ", ".join(drift_result["drifted_features"][:5])
            return (
                f"🚨 DRIFT DETECTED! "
                f"Score: {drift_result['drift_score']:.2%} | "
                f"Drifted features: {features}"
            )
        else:
            return (
                f"✅ No significant drift detected. "
                f"Score: {drift_result['drift_score']:.2%}"
            )
    
    def reload_reference(self, reference_path: Optional[str] = None):
        """
        Reload reference data (e.g., after retraining).
        
        Args:
            reference_path: New reference data path
        """
        if reference_path:
            self.reference_path = Path(reference_path)
        self._load_reference()
        logger.info("reference_data_reloaded")
