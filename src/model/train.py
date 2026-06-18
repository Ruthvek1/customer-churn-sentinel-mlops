"""
Model training pipeline for customer churn prediction.

Implements:
- XGBoost classifier training with hyperparameter tuning
- Stratified k-fold cross-validation
- Experiment logging (metrics, params, artifacts)
- Model versioning with timestamps
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.model.evaluate import evaluate_model, get_classification_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(params: Optional[dict] = None) -> XGBClassifier:
    """
    Create an XGBoost classifier with given or default parameters.
    
    Args:
        params: Model hyperparameters. If None, uses config defaults.
        
    Returns:
        Configured XGBClassifier instance
    """
    config = load_config()
    model_params = params or config["model"]["params"]
    
    model = XGBClassifier(**model_params)
    logger.info("model_created", params=model_params)
    return model


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[XGBClassifier, dict]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Uses stratified k-fold to handle class imbalance properly.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of random parameter combinations to try
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric for optimization
        
    Returns:
        Tuple of (best model, best parameters)
    """
    config = load_config()
    tuning_config = config["model"]["tuning"]
    
    param_distributions = tuning_config["param_distributions"]
    
    base_model = XGBClassifier(
        random_state=config["model"]["params"]["random_state"],
        eval_metric="logloss",
        scale_pos_weight=config["model"]["params"].get("scale_pos_weight", 1.0),
    )
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    logger.info(
        "starting_hyperparameter_tuning",
        n_iter=n_iter,
        cv_folds=cv_folds,
        scoring=scoring
    )
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    
    search.fit(X_train, y_train)
    
    logger.info(
        "tuning_complete",
        best_score=float(search.best_score_),
        best_params=search.best_params_
    )
    
    return search.best_estimator_, search.best_params_


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune: bool = True,
    save: bool = True
) -> Tuple[XGBClassifier, Dict]:
    """
    Execute the full training pipeline.
    
    Steps:
    1. Optionally tune hyperparameters
    2. Train the model
    3. Evaluate on test set
    4. Save model artifact + experiment log
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tune: Whether to perform hyperparameter tuning
        save: Whether to save model to disk
        
    Returns:
        Tuple of (trained model, experiment metadata)
    """
    config = load_config()
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("starting_training_pipeline", tune=tune)
    
    # Step 1: Create or tune model
    if tune and config["model"]["tuning"]["enabled"]:
        model, best_params = tune_hyperparameters(
            X_train, y_train,
            n_iter=config["model"]["tuning"]["n_iter"],
            cv_folds=config["model"]["tuning"]["cv_folds"],
            scoring=config["model"]["tuning"]["scoring"]
        )
    else:
        model = create_model()
        best_params = config["model"]["params"]
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Step 2: Evaluate
    metrics = get_classification_metrics(model, X_test, y_test)
    feature_importance = dict(zip(
        X_train.columns,
        [float(x) for x in model.feature_importances_]
    ))
    
    # Sort feature importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    
    # Step 3: Create experiment log
    experiment = {
        "timestamp": timestamp,
        "model_version": f"v_{timestamp}",
        "model_type": "XGBClassifier",
        "training_time_seconds": round(training_time, 2),
        "hyperparameters": {k: str(v) for k, v in best_params.items()},
        "metrics": metrics,
        "feature_importance": feature_importance,
        "dataset": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(X_train.columns),
            "feature_names": list(X_train.columns),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test": float(y_test.mean()),
        }
    }
    
    logger.info(
        "training_complete",
        accuracy=metrics["accuracy"],
        roc_auc=metrics["roc_auc"],
        f1=metrics["f1_score"],
        training_time=round(training_time, 2)
    )
    
    # Step 4: Save
    if save:
        model_dir = Path(config["model"]["save_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{config['model']['model_name']}.joblib"
        joblib.dump(model, model_path)
        
        # Save versioned copy
        versioned_path = model_dir / f"{config['model']['model_name']}_{timestamp}.joblib"
        joblib.dump(model, versioned_path)
        
        # Save experiment log
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Append to experiment history
        history_path = log_dir / "experiment_history.json"
        history = []
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
        
        history.append(experiment)
        
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save latest experiment
        with open(log_dir / "latest_experiment.json", "w") as f:
            json.dump(experiment, f, indent=2, default=str)
        
        logger.info(
            "model_saved",
            model_path=str(model_path),
            versioned_path=str(versioned_path)
        )
    
    return model, experiment


def load_trained_model(model_path: Optional[str] = None) -> XGBClassifier:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model file. If None, loads latest.
        
    Returns:
        Loaded XGBClassifier
    """
    config = load_config()
    if model_path is None:
        model_path = Path(config["model"]["save_dir"]) / f"{config['model']['model_name']}.joblib"
    
    model = joblib.load(model_path)
    logger.info("model_loaded", path=str(model_path))
    return model


if __name__ == "__main__":
    """Run training as standalone script."""
    from src.utils.logger import setup_logging
    from src.data.preprocess import run_preprocessing_pipeline
    
    setup_logging()
    
    print("🔄 Step 1: Preprocessing data...")
    X_train, X_test, y_train, y_test, metadata = run_preprocessing_pipeline()
    
    print("🧠 Step 2: Training model...")
    model, experiment = train_model(X_train, y_train, X_test, y_test, tune=True)
    
    print(f"\n✅ Training complete!")
    print(f"   Accuracy:  {experiment['metrics']['accuracy']:.4f}")
    print(f"   ROC-AUC:   {experiment['metrics']['roc_auc']:.4f}")
    print(f"   F1 Score:  {experiment['metrics']['f1_score']:.4f}")
    print(f"   Precision: {experiment['metrics']['precision']:.4f}")
    print(f"   Recall:    {experiment['metrics']['recall']:.4f}")
    print(f"   Time:      {experiment['training_time_seconds']}s")
