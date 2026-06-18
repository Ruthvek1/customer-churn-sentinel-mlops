"""
Model evaluation utilities.

Provides comprehensive metrics computation for classification models:
- Standard metrics (accuracy, precision, recall, F1)
- ROC-AUC curve data
- Confusion matrix
- Feature importance ranking
- Performance comparison across model versions
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_classification_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        model: Trained classifier with predict() and predict_proba()
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    
    logger.info("evaluation_metrics", **metrics)
    return metrics


def get_confusion_matrix_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Get confusion matrix data for visualization.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary with confusion matrix values
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "labels": ["Not Churned", "Churned"]
    }


def get_roc_curve_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Get ROC curve data points for plotting.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary with FPR, TPR arrays and AUC score
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc_score": float(auc_score)
    }


def get_classification_report_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> str:
    """
    Get formatted classification report.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        
    Returns:
        Formatted classification report string
    """
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred,
        target_names=["Not Churned", "Churned"]
    )


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15
) -> Dict:
    """
    Extract and rank feature importances.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        Dictionary with sorted feature importance data
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_features = []
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        top_features.append({
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
            "rank": i + 1
        })
    
    return {
        "top_features": top_features,
        "all_importances": dict(zip(
            [feature_names[i] for i in indices],
            [float(importances[i]) for i in indices]
        ))
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Run complete model evaluation.
    
    Aggregates all evaluation metrics, confusion matrix, ROC data,
    and feature importance into a single evaluation report.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        feature_names: Optional feature names for importance ranking
        
    Returns:
        Complete evaluation report dictionary
    """
    if feature_names is None:
        feature_names = list(X_test.columns)
    
    report = {
        "metrics": get_classification_metrics(model, X_test, y_test),
        "confusion_matrix": get_confusion_matrix_data(model, X_test, y_test),
        "roc_curve": get_roc_curve_data(model, X_test, y_test),
        "feature_importance": get_feature_importance(model, feature_names),
        "classification_report": get_classification_report_data(model, X_test, y_test),
    }
    
    logger.info("full_evaluation_complete", auc=report["metrics"]["roc_auc"])
    return report
