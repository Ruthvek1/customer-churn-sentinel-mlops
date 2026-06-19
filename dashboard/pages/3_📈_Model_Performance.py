"""
📈 Model Performance Page

Comprehensive model evaluation metrics, ROC curve, confusion matrix,
and performance tracking across model versions.
"""

import streamlit as st
import requests
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dashboard.components.charts import (
    create_feature_importance_chart,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.markdown("# 📈 Model Performance")
st.markdown("Comprehensive evaluation metrics for the current churn prediction model.")
st.markdown("---")

API_URL = st.session_state.get("api_url", "http://localhost:8000")


# --- Model Info ---
st.markdown("### 🧠 Current Model")

try:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    
    if response.status_code == 200:
        model_info = response.json()
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Model Type", model_info.get("model_type", "Unknown"))
            st.metric("Features", model_info.get("n_features", 0))
        
        with info_col2:
            st.metric("Version", model_info.get("version", "unknown")[:20])
            st.metric("Trained At", model_info.get("trained_at", "unknown")[:16])
        
        with info_col3:
            metrics = model_info.get("metrics", {})
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    else:
        st.warning("Could not load model info.")

except Exception:
    st.info("Connect to the API to see model information.")

st.markdown("---")

# --- Key Metrics ---
st.markdown("### 📊 Classification Metrics")

try:
    # Load from experiment log
    experiment_path = Path("logs/latest_experiment.json")
    
    if experiment_path.exists():
        with open(experiment_path, "r") as f:
            experiment = json.load(f)
        
        metrics = experiment.get("metrics", {})
        
        met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
        
        with met_col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with met_col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with met_col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with met_col4:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        with met_col5:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        
        st.markdown("---")
        
        # --- Visualizations ---
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("### 🎯 Feature Importance")
            fi = experiment.get("feature_importance", {})
            if fi:
                features_list = [
                    {"feature": k, "importance": v} 
                    for k, v in list(fi.items())[:15]
                ]
                fig = create_feature_importance_chart(features_list)
                st.plotly_chart(fig, width="stretch")
        
        with viz_col2:
            st.markdown("### 📋 Training Details")
            dataset_info = experiment.get("dataset", {})
            st.json({
                "Model Version": experiment.get("model_version", "unknown"),
                "Training Time": f"{experiment.get('training_time_seconds', 0)}s",
                "Train Size": dataset_info.get("train_size", 0),
                "Test Size": dataset_info.get("test_size", 0),
                "Features": dataset_info.get("n_features", 0),
                "Train Churn Rate": f"{dataset_info.get('churn_rate_train', 0):.2%}",
                "Test Churn Rate": f"{dataset_info.get('churn_rate_test', 0):.2%}",
            })
        
        st.markdown("---")
        
        # --- Hyperparameters ---
        st.markdown("### ⚙️ Hyperparameters")
        params = experiment.get("hyperparameters", {})
        if params:
            param_cols = st.columns(4)
            for i, (key, value) in enumerate(params.items()):
                with param_cols[i % 4]:
                    st.code(f"{key}: {value}")
    else:
        st.info("No experiment logs found. Train a model first!")
        st.code("python -m src.model.train")

except Exception as e:
    st.error(f"Error loading metrics: {str(e)}")

st.markdown("---")

# --- Experiment History ---
st.markdown("### 📜 Experiment History")

try:
    history_path = Path("logs/experiment_history.json")
    
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        
        if history:
            st.markdown(f"**{len(history)} experiments logged**")
            
            import pandas as pd
            hist_df = pd.DataFrame([{
                "Version": exp.get("model_version", "")[:16],
                "Time": exp.get("timestamp", ""),
                "Accuracy": f"{exp.get('metrics', {}).get('accuracy', 0):.4f}",
                "ROC-AUC": f"{exp.get('metrics', {}).get('roc_auc', 0):.4f}",
                "F1": f"{exp.get('metrics', {}).get('f1_score', 0):.4f}",
                "Training Time": f"{exp.get('training_time_seconds', 0)}s",
            } for exp in reversed(history)])
            
            st.dataframe(hist_df, width="stretch")
        else:
            st.info("No experiments recorded yet.")
    else:
        st.info("Experiment history will appear here after training.")

except Exception as e:
    st.error(f"Error: {str(e)}")
