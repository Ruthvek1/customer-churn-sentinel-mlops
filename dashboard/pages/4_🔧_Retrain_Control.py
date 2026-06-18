"""
🔧 Retrain Control Page

Manual model retraining controls, version management,
and before/after comparison.
"""

import streamlit as st
import requests
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Retrain Control", page_icon="🔧", layout="wide")

st.markdown("# 🔧 Retrain Control Center")
st.markdown("Manage model retraining — manually or automatically triggered by drift detection.")
st.markdown("---")

API_URL = st.session_state.get("api_url", "http://localhost:8000")


# --- Current Model Status ---
st.markdown("### 🧠 Current Model")

try:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    
    if response.status_code == 200:
        model_info = response.json()
        metrics = model_info.get("metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Version", model_info.get("version", "unknown")[:16])
        with col2:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        with col3:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        with col4:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    else:
        st.warning("Could not load model info.")
except Exception:
    st.info("Connect to the API to see model status.")

st.markdown("---")

# --- Retrain Controls ---
st.markdown("### 🔄 Manual Retraining")

st.markdown("""
Trigger a full model retrain cycle. This will:
1. Re-run the data preprocessing pipeline
2. Perform hyperparameter tuning (RandomizedSearchCV)
3. Train a new XGBoost model
4. Evaluate against the test set
5. Hot-reload the new model into the API
6. Update the reference dataset for drift detection
""")

retrain_reason = st.text_input("Reason for retraining", value="manual", 
                                placeholder="e.g., drift detected, new data available")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Retrain Model", use_container_width=True, type="primary"):
        try:
            with st.spinner("🧠 Retraining model... This may take a minute."):
                response = requests.post(
                    f"{API_URL}/retrain",
                    json={"reason": retrain_reason},
                    timeout=300  # 5 min timeout for training
                )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                
                if status in ("success", "completed", "started"):
                    st.success("✅ **Model retrained successfully!**")
                    
                    new_metrics = result.get("metrics", {})
                    if new_metrics:
                        st.markdown("#### New Model Metrics")
                        met_cols = st.columns(4)
                        with met_cols[0]:
                            st.metric("Accuracy", f"{new_metrics.get('accuracy', 0):.4f}")
                        with met_cols[1]:
                            st.metric("ROC-AUC", f"{new_metrics.get('roc_auc', 0):.4f}")
                        with met_cols[2]:
                            st.metric("F1 Score", f"{new_metrics.get('f1_score', 0):.4f}")
                        with met_cols[3]:
                            st.metric("Training Time", f"{result.get('training_time', 0):.1f}s")
                    
                    if result.get("new_model_version"):
                        st.info(f"📦 New version: `{result['new_model_version']}`")
                    
                    st.balloons()
                else:
                    st.error(f"Retraining failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"Retraining failed: {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API.")
        except requests.exceptions.Timeout:
            st.warning("⏳ Retraining is taking longer than expected. Check the API logs.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    if st.button("📋 Check Retrain Status", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/retrain/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                st.json(status)
            else:
                st.error("Could not get retrain status.")
        except Exception:
            st.info("Connect to the API.")

st.markdown("---")

# --- Retraining History ---
st.markdown("### 📜 Retraining History")

try:
    response = requests.get(f"{API_URL}/retrain/history?limit=10", timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        history = data.get("history", [])
        
        if history:
            import pandas as pd
            hist_df = pd.DataFrame([{
                "Time": h.get("timestamp", "")[:19],
                "Trigger": h.get("trigger", ""),
                "Old Version": h.get("old_model_version", "")[:12],
                "New Version": h.get("new_model_version", "")[:12],
                "Status": h.get("status", ""),
            } for h in history])
            
            st.dataframe(hist_df, use_container_width=True)
            
            # Compare old vs new metrics
            with st.expander("📊 Before/After Comparison"):
                for h in history[:3]:
                    old_m = h.get("old_metrics", {})
                    new_m = h.get("new_metrics", {})
                    
                    if old_m and new_m:
                        st.markdown(f"**{h.get('timestamp', '')[:19]}** — Triggered by: {h.get('trigger', 'unknown')}")
                        comp_cols = st.columns(4)
                        
                        for i, metric_name in enumerate(["accuracy", "roc_auc", "f1_score", "precision"]):
                            old_val = old_m.get(metric_name, 0)
                            new_val = new_m.get(metric_name, 0)
                            delta = new_val - old_val
                            with comp_cols[i]:
                                st.metric(
                                    metric_name.replace("_", " ").title(),
                                    f"{new_val:.4f}",
                                    delta=f"{delta:+.4f}",
                                    delta_color="normal"
                                )
                        st.markdown("---")
        else:
            st.info("No retraining events yet.")
    else:
        st.info("Retraining history will appear here.")
except Exception:
    st.info("Connect to the API to see retraining history.")

st.markdown("---")

# --- Model Versions ---
st.markdown("### 📦 Model Artifacts")

models_dir = Path("models")
if models_dir.exists():
    model_files = sorted(models_dir.glob("*.joblib"), reverse=True)
    
    if model_files:
        import pandas as pd
        files_df = pd.DataFrame([{
            "File": f.name,
            "Size": f"{f.stat().st_size / 1024:.1f} KB",
            "Modified": str(pd.Timestamp(f.stat().st_mtime, unit='s'))[:19],
        } for f in model_files])
        
        st.dataframe(files_df, use_container_width=True)
    else:
        st.info("No model artifacts found. Train a model first!")
else:
    st.info("Models directory not found.")
