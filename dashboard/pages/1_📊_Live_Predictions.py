"""
📊 Live Predictions Page

Make real-time customer churn predictions and view prediction history.
"""

import streamlit as st
import requests
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dashboard.components.charts import (
    create_churn_gauge,
    create_feature_importance_chart,
    create_prediction_history_chart,
    create_risk_distribution_chart,
)

st.set_page_config(page_title="Live Predictions", page_icon="📊", layout="wide")

st.markdown("# 📊 Live Predictions")
st.markdown("Make real-time churn predictions for individual customers.")
st.markdown("---")

API_URL = st.session_state.get("api_url", "http://localhost:8000")


# --- Prediction Form ---
st.markdown("### 🔮 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.markdown("**Services**")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.markdown("**Billing**")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, step=50.0)

st.markdown("---")

# --- Predict Button ---
if st.button("🔮 Predict Churn", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    
    try:
        with st.spinner("Making prediction..."):
            response = requests.post(f"{API_URL}/predictions/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result
            res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
            
            with res_col1:
                fig = create_churn_gauge(result["churn_probability"])
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                st.markdown("### Result")
                
                if result["prediction"] == 1:
                    st.error(f"⚠️ **WILL CHURN**")
                else:
                    st.success(f"✅ **WON'T CHURN**")
                
                st.metric("Probability", f"{result['churn_probability']:.1%}")
                st.metric("Risk Level", result["risk_level"])
            
            with res_col3:
                st.markdown("### Risk Level")
                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.markdown(f"## {risk_colors.get(result['risk_level'], '⚪')} {result['risk_level']}")
                
                if result["risk_level"] == "HIGH":
                    st.warning("**Action Required:** This customer has a high probability of churning. Consider retention offers.")
                elif result["risk_level"] == "MEDIUM":
                    st.info("**Monitor:** This customer shows some churn risk. Keep an eye on engagement.")
                else:
                    st.success("**Stable:** This customer appears satisfied.")
        else:
            st.error(f"Prediction failed: {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")

# --- Feature Importance ---
st.markdown("### 📊 Feature Importance")
try:
    response = requests.get(f"{API_URL}/predictions/feature-importance?top_n=15", timeout=5)
    if response.status_code == 200:
        data = response.json()
        fig = create_feature_importance_chart(data.get("features", []))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance will appear after the model is loaded.")
except Exception:
    st.info("Connect to the API to see feature importance.")

st.markdown("---")

# --- Prediction History ---
st.markdown("### 📜 Recent Predictions")

try:
    response = requests.get(f"{API_URL}/predictions/history?limit=20", timeout=5)
    if response.status_code == 200:
        data = response.json()
        predictions = data.get("predictions", [])
        
        if predictions:
            hist_col1, hist_col2 = st.columns([2, 1])
            
            with hist_col1:
                fig = create_prediction_history_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)
            
            with hist_col2:
                stats_resp = requests.get(f"{API_URL}/predictions/stats", timeout=5)
                if stats_resp.status_code == 200:
                    stats = stats_resp.json()
                    fig = create_risk_distribution_chart(stats)
                    st.plotly_chart(fig, use_container_width=True)
            
            # History table
            with st.expander("View Raw Data"):
                df = pd.DataFrame([{
                    "Time": p.get("timestamp", "")[:19],
                    "Prediction": "Churn" if p.get("prediction") == 1 else "No Churn",
                    "Probability": f"{p.get('churn_probability', 0):.2%}",
                    "Risk": p.get("risk_level", ""),
                } for p in predictions])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No predictions yet. Make a prediction above!")
    else:
        st.info("Prediction history will appear here.")
except Exception:
    st.info("Connect to the API to see prediction history.")
