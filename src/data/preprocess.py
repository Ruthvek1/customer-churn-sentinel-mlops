"""
Data preprocessing pipeline for Telco Customer Churn dataset.

Handles:
- Missing value imputation
- Feature engineering (tenure buckets, ratios, service counts)
- Categorical encoding (label + one-hot)
- Train/test splitting with stratification
- Reference dataset creation for drift detection
- Reproducible sklearn Pipeline construction
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Telco Churn dataset.
    
    Args:
        data_path: Path to CSV file. If None, uses config path.
        
    Returns:
        Raw DataFrame
    """
    config = load_config()
    if data_path is None:
        data_path = config["data"]["raw_path"]
    
    logger.info("loading_raw_data", path=data_path)
    df = pd.read_csv(data_path)
    logger.info("raw_data_loaded", rows=len(df), columns=len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: handle missing values and type conversions.
    
    - TotalCharges has blank strings for new customers → convert to float, fill with 0
    - Drop customerID (not a feature)
    - Convert Churn to binary (Yes=1, No=0)
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    logger.info("cleaning_data", initial_shape=df.shape)
    
    # TotalCharges: convert blanks to NaN, then to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing TotalCharges with MonthlyCharges * tenure (or 0 for new customers)
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    # Convert target to binary
    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
    
    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    missing_count = df.isnull().sum().sum()
    logger.info("data_cleaned", final_shape=df.shape, remaining_missing=int(missing_count))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features that add predictive power.
    
    New features:
    - tenure_bucket: Categorize tenure into groups (0-12, 12-24, 24-48, 48-60, 60+)
    - monthly_to_total_ratio: MonthlyCharges / TotalCharges (spending consistency)
    - service_count: Number of services the customer has
    - avg_monthly_spend: TotalCharges / max(tenure, 1)
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    logger.info("engineering_features")
    
    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)
    
    # Monthly to total ratio (spending pattern indicator)
    df["monthly_to_total_ratio"] = np.where(
        df["TotalCharges"] > 0,
        df["MonthlyCharges"] / df["TotalCharges"],
        0
    )
    
    # Count of services subscribed
    service_columns = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Count 'Yes' values (service is active)
    service_count = pd.DataFrame()
    for col in service_columns:
        if col in df.columns:
            service_count[col] = (df[col] == "Yes").astype(int)
    
    df["service_count"] = service_count.sum(axis=1)
    
    # Average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    
    logger.info("features_engineered", new_features=["tenure_bucket", "monthly_to_total_ratio", "service_count", "avg_monthly_spend"])
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features for model input.
    
    - Binary features: Label encode (Yes/No → 1/0)
    - Multi-class features: One-hot encode
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Tuple of (encoded DataFrame, encoding mappings for inverse transform)
    """
    df = df.copy()
    config = load_config()
    
    binary_columns = []
    multiclass_columns = []
    
    categorical_features = config["data"]["categorical_features"]
    
    for col in categorical_features:
        if col not in df.columns:
            continue
        unique_vals = df[col].nunique()
        if unique_vals <= 2:
            binary_columns.append(col)
        else:
            multiclass_columns.append(col)
    
    encoding_map = {}
    
    # Label encode binary features
    for col in binary_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoding_map[col] = {
            "type": "label",
            "classes": list(le.classes_)
        }
    
    # One-hot encode multi-class features
    if multiclass_columns:
        df = pd.get_dummies(df, columns=multiclass_columns, drop_first=True, dtype=int)
        for col in multiclass_columns:
            encoding_map[col] = {"type": "onehot"}
    
    logger.info(
        "features_encoded",
        binary_encoded=len(binary_columns),
        onehot_encoded=len(multiclass_columns),
        final_features=len(df.columns)
    )
    return df, encoding_map


def run_preprocessing_pipeline(
    data_path: Optional[str] = None,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Execute the full preprocessing pipeline.
    
    Pipeline steps:
    1. Load raw data
    2. Clean data (missing values, type conversions)
    3. Engineer features
    4. Encode categorical features
    5. Split into train/test
    6. Save reference dataset for drift detection
    
    Args:
        data_path: Optional path to raw CSV
        save: Whether to save processed data to disk
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    config = load_config()
    
    # Step 1: Load
    df = load_raw_data(data_path)
    
    # Step 2: Clean
    df = clean_data(df)
    
    # Step 3: Feature engineering
    df = engineer_features(df)
    
    # Step 4: Encode
    df, encoding_map = encode_features(df)
    
    # Step 5: Split
    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )
    
    logger.info(
        "data_split",
        train_size=len(X_train),
        test_size=len(X_test),
        churn_rate_train=float(y_train.mean()),
        churn_rate_test=float(y_test.mean()),
        features=len(X_train.columns)
    )
    
    # Store feature names for later use
    feature_names = list(X_train.columns)
    
    metadata = {
        "encoding_map": encoding_map,
        "feature_names": feature_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "churn_rate": float(y.mean()),
        "n_features": len(feature_names),
    }
    
    if save:
        processed_dir = Path(config["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(processed_dir / "X_train.csv", index=False)
        X_test.to_csv(processed_dir / "X_test.csv", index=False)
        y_train.to_csv(processed_dir / "y_train.csv", index=False)
        y_test.to_csv(processed_dir / "y_test.csv", index=False)
        
        # Save reference dataset for drift detection (training data)
        reference_path = Path(config["data"]["reference_path"])
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(reference_path, index=False)
        
        # Save metadata
        joblib.dump(metadata, processed_dir / "preprocessing_metadata.joblib")
        
        logger.info("processed_data_saved", path=str(processed_dir))
    
    return X_train, X_test, y_train, y_test, metadata


def preprocess_single_input(
    input_data: dict,
    feature_names: list,
    encoding_map: dict
) -> pd.DataFrame:
    """
    Preprocess a single customer input for prediction.
    
    Used by the API to transform incoming prediction requests
    into the same format the model was trained on.
    
    Args:
        input_data: Dictionary of customer features
        feature_names: Expected feature names from training
        encoding_map: Encoding mappings from training
        
    Returns:
        Single-row DataFrame ready for model.predict()
    """
    df = pd.DataFrame([input_data])
    
    # Clean
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode
    df, _ = encode_features(df)
    
    # Align columns with training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    
    return df


if __name__ == "__main__":
    """Run preprocessing as standalone script."""
    from src.utils.logger import setup_logging
    setup_logging()
    
    X_train, X_test, y_train, y_test, metadata = run_preprocessing_pipeline()
    print(f"\n✅ Preprocessing complete!")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Features: {metadata['n_features']}")
    print(f"   Churn rate: {metadata['churn_rate']:.2%}")
