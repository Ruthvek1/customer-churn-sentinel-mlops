"""
🔮 Churn Prediction System — Dashboard

Multi-page Streamlit dashboard for monitoring and interacting
with the ML prediction system.

Pages:
1. 📊 Live Predictions — Make predictions and view history
2. 🔍 Drift Monitor — Data drift detection and alerts
3. 📈 Model Performance — Metrics, ROC curve, confusion matrix
4. 🔧 Retrain Control — Manual retraining and model management

Launch with:
    streamlit run dashboard/app.py
"""

import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="🔮 Churn Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Dark Theme ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .status-stable { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .status-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .status-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    
    /* Card container */
    .dashboard-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("# 🔮 Churn Predictor")
    st.markdown("---")
    st.markdown("""
    **End-to-End ML System**  
    with Drift Monitoring
    
    ---
    
    ### 🏗️ System Components
    - ⚡ FastAPI Backend
    - 🧠 XGBoost Model
    - 🔍 Evidently AI Drift Detection
    - 📊 Streamlit Dashboard
    - 💾 SQLite Prediction Logging
    
    ---
    """)
    
    # API connection settings
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="FastAPI backend URL"
    )
    
    st.session_state["api_url"] = api_url
    
    # Connection status
    import requests
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            health = response.json()
            st.success(f"✅ Connected | {health.get('model_version', 'unknown')}")
        else:
            st.warning("⚠️ API responding with errors")
    except Exception:
        st.error("❌ API not connected")
        st.caption("Start the API first:\n```\nuvicorn src.api.main:app\n```")
    
    st.markdown("---")
    st.caption("Built with ❤️ for ML Engineering")


# --- Main Page ---
st.markdown("# 🔮 Customer Churn Prediction System")
st.markdown("### End-to-End ML Pipeline with Production Drift Monitoring")

st.markdown("---")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

try:
    stats_response = requests.get(f"{api_url}/predictions/stats", timeout=3)
    drift_response = requests.get(f"{api_url}/drift/status", timeout=3)
    model_response = requests.get(f"{api_url}/model/info", timeout=3)
    
    if stats_response.status_code == 200:
        stats = stats_response.json()
        with col1:
            st.metric("Total Predictions", stats.get("total", 0))
        with col2:
            st.metric("Churn Rate", f"{stats.get('churn_rate', 0):.1%}")
    else:
        with col1:
            st.metric("Total Predictions", "—")
        with col2:
            st.metric("Churn Rate", "—")
    
    if drift_response.status_code == 200:
        drift = drift_response.json()
        status = drift.get("status", "unknown")
        emoji = {"stable": "✅", "warning": "⚠️", "critical": "🚨"}.get(status, "❓")
        with col3:
            st.metric("Drift Status", f"{emoji} {status.upper()}")
    else:
        with col3:
            st.metric("Drift Status", "—")
    
    if model_response.status_code == 200:
        model_info = model_response.json()
        with col4:
            st.metric("Model Version", model_info.get("version", "unknown")[:12])
    else:
        with col4:
            st.metric("Model Version", "—")

except Exception:
    with col1:
        st.metric("Total Predictions", "—")
    with col2:
        st.metric("Churn Rate", "—")
    with col3:
        st.metric("Drift Status", "—")
    with col4:
        st.metric("Model Version", "—")

st.markdown("---")

# Navigation cards
st.markdown("### 📌 Navigate to")

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    st.markdown("""
    <div class="dashboard-card">
        <h3>📊 Live Predictions</h3>
        <p style="color: #94a3b8;">Make predictions and view history</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div class="dashboard-card">
        <h3>🔍 Drift Monitor</h3>
        <p style="color: #94a3b8;">Detect data distribution shifts</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div class="dashboard-card">
        <h3>📈 Model Performance</h3>
        <p style="color: #94a3b8;">Metrics, ROC, confusion matrix</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col4:
    st.markdown("""
    <div class="dashboard-card">
        <h3>🔧 Retrain Control</h3>
        <p style="color: #94a3b8;">Manual retraining & versioning</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col5:
    st.markdown("""
    <div class="dashboard-card">
        <h3>💡 Layman Guide</h3>
        <p style="color: #94a3b8;">Simple explanation of the model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
> **💡 Getting Started:** Use the sidebar to navigate between pages.
> Make predictions on the **Live Predictions** page, then check the
> **Drift Monitor** to see if your data has shifted from training distribution.
""")
