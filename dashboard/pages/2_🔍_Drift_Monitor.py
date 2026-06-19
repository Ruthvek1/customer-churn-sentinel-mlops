"""
🔍 Drift Monitor Page

The star feature — monitors data drift between training and production data.
"""

import streamlit as st
import requests
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dashboard.components.charts import (
    create_drift_heatmap,
    create_drift_trend_chart,
)

st.set_page_config(page_title="Drift Monitor", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Data Drift Monitor")
st.markdown("""
Monitor data drift between your training data and incoming predictions.
**This is the key differentiator** — it demonstrates understanding that models decay in production
when the data distribution shifts from what the model was trained on.
""")
st.markdown("---")

API_URL = st.session_state.get("api_url", "http://localhost:8000")


# --- Drift Status Overview ---
st.markdown("### 📡 Current Drift Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

try:
    status_resp = requests.get(f"{API_URL}/drift/status", timeout=5)
    
    if status_resp.status_code == 200:
        status = status_resp.json()
        
        drift_status = status.get("status", "unknown")
        status_map = {
            "stable": ("✅ STABLE", "success"),
            "warning": ("⚠️ WARNING", "warning"),
            "critical": ("🚨 CRITICAL", "error"),
            "unknown": ("❓ UNKNOWN", "info"),
        }
        label, style = status_map.get(drift_status, ("❓", "info"))
        
        with status_col1:
            st.metric("Status", label)
        with status_col2:
            st.metric("Drift Score", f"{status.get('drift_score', 0):.2%}")
        with status_col3:
            st.metric("Drifted Features", status.get("drifted_features_count", 0))
        with status_col4:
            st.metric("Total Predictions", status.get("total_predictions", 0))
        
        if drift_status == "critical":
            st.error("🚨 **CRITICAL DRIFT DETECTED!** Model predictions may be unreliable. Consider retraining immediately.")
        elif drift_status == "warning":
            st.warning("⚠️ **Mild drift detected.** Monitor closely — the model may need retraining soon.")
        elif drift_status == "stable":
            st.success("✅ **No significant drift.** Model is performing with data similar to training distribution.")
    else:
        for col in [status_col1, status_col2, status_col3, status_col4]:
            with col:
                st.metric("—", "—")
except Exception:
    for col in [status_col1, status_col2, status_col3, status_col4]:
        with col:
            st.metric("—", "—")
    st.info("Connect to the API to see drift status.")

st.markdown("---")

# --- Run Drift Detection ---
st.markdown("### 🔬 Run Drift Detection")
st.markdown("""
Click below to analyze the current prediction data against the training reference dataset.
Uses **Kolmogorov-Smirnov** test for numerical features and **Chi-squared** test for categorical features.
""")

if st.button("🔬 Run Drift Analysis", width="stretch"):
    try:
        with st.spinner("Running drift detection... This compares your recent predictions against the training data distribution."):
            response = requests.get(f"{API_URL}/drift/report", timeout=30)
        
        if response.status_code == 200:
            drift_result = response.json()
            
            if drift_result.get("error"):
                st.warning(f"⚠️ {drift_result['error']}")
            else:
                # Summary
                if drift_result["is_drifted"]:
                    st.error(f"""
                    ### 🚨 Drift Detected!
                    
                    **Drift Score:** {drift_result['drift_score']:.2%} of features have drifted  
                    **Drifted Features:** {', '.join(drift_result['drifted_features'][:10])}
                    """)
                else:
                    st.success(f"""
                    ### ✅ No Significant Drift
                    
                    **Drift Score:** {drift_result['drift_score']:.2%}  
                    The incoming data distribution is consistent with training data.
                    """)
                
                # Feature-level drift details
                feature_details = drift_result.get("feature_drift_details", {})
                
                if feature_details:
                    st.markdown("### 📊 Per-Feature Drift Analysis")
                    
                    fig = create_drift_heatmap(feature_details)
                    st.plotly_chart(fig, width="stretch")
                    
                    # Detailed table
                    with st.expander("📋 Detailed Feature Drift Scores"):
                        drift_df = pd.DataFrame([
                            {
                                "Feature": name,
                                "Drifted": "🔴 YES" if info.get("drifted") else "🟢 NO",
                                "Drift Score": f"{info.get('drift_score', 0):.6f}",
                                "Test Used": info.get("stattest_name", "—"),
                                "Threshold": info.get("stattest_threshold", "—"),
                            }
                            for name, info in feature_details.items()
                        ])
                        st.dataframe(drift_df, width="stretch")
                
                # Report link
                if drift_result.get("report_path"):
                    st.info(f"📄 Full Evidently HTML report saved to: `{drift_result['report_path']}`")
        else:
            st.error(f"Drift detection failed: {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")

# --- Drift History ---
st.markdown("### 📈 Drift History")
st.markdown("Track how data drift evolves over time.")

try:
    response = requests.get(f"{API_URL}/drift/history?limit=20", timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        history = data.get("history", [])
        
        if history:
            fig = create_drift_trend_chart(history)
            st.plotly_chart(fig, width="stretch")
            
            # History table
            with st.expander("📋 Drift Check Log"):
                hist_df = pd.DataFrame([{
                    "Time": h.get("timestamp", "")[:19],
                    "Drifted": "🔴 YES" if h.get("is_drifted") else "🟢 NO",
                    "Score": f"{h.get('drift_score', 0):.2%}",
                    "Features Drifted": h.get("drifted_feature_count", 0),
                    "Total Features": h.get("total_features", 0),
                } for h in history])
                st.dataframe(hist_df, width="stretch")
        else:
            st.info("No drift checks yet. Run a drift analysis above!")
    else:
        st.info("Drift history will appear here after running drift checks.")
except Exception:
    st.info("Connect to the API to see drift history.")

st.markdown("---")

# --- Educational Section ---
with st.expander("📚 Understanding Data Drift"):
    st.markdown("""
    ### What is Data Drift?
    
    Data drift occurs when the statistical distribution of input data changes over time,
    causing model performance to degrade. This is one of the most common reasons ML models
    fail in production.
    
    ### Why It Matters
    
    - A model trained on 2023 customer data may not work well on 2025 customers
    - Seasonal trends, market changes, and business shifts all cause drift
    - Without monitoring, you won't know your model is degrading until it's too late
    
    ### Statistical Tests Used
    
    | Test | For | What It Measures |
    |:---|:---|:---|
    | **Kolmogorov-Smirnov** | Numerical features | Max distance between two cumulative distributions |
    | **Chi-squared** | Categorical features | Independence of frequency distributions |
    | **PSI** | Both | Population Stability Index — magnitude of distribution shift |
    
    ### Drift Thresholds
    
    - **Drift Score < 0.1:** No significant drift — model is stable
    - **Drift Score 0.1 - 0.5:** Warning — monitor closely
    - **Drift Score > 0.5:** Critical — retraining recommended
    """)

st.markdown("---")

# --- Reset Section ---
st.markdown("### 🗑️ Reset Data")
st.markdown("Clear all prediction logs and drift history to start fresh.")

reset_col1, reset_col2 = st.columns([3, 1])

with reset_col1:
    confirm_reset = st.checkbox("I understand this will permanently delete all prediction logs and drift history.", key="confirm_reset")

with reset_col2:
    reset_disabled = not confirm_reset
    if st.button("🗑️ Reset All Data", width="stretch", type="primary", disabled=reset_disabled):
        try:
            with st.spinner("Resetting database..."):
                response = requests.post(f"{API_URL}/drift/reset", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ **Database reset!** Cleared {result.get('predictions_cleared', 0)} predictions and {result.get('drift_checks_cleared', 0)} drift checks.")
                st.rerun()
            else:
                st.error(f"Reset failed: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
