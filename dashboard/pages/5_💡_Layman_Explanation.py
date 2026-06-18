import streamlit as st
import requests

st.set_page_config(
    page_title="Layman Explanation — Churn Predictor",
    page_icon="💡",
    layout="wide",
)

# Reuse the same style class structure
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .layman-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .highlight-purple {
        color: #a78bfa;
        font-weight: 700;
    }
    .highlight-green {
        color: #34d399;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 💡 What Exactly Does This Model Do?")
st.markdown("### A Layman's Guide to Customer Churn & MLOps")
st.markdown("---")

st.markdown("""
Machine Learning can sound like magic, but at its heart, it's just about **recognizing patterns** and **anticipating behavior**. Here is a breakdown of how this system works in plain English.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="layman-card">
            <h2>🏃‍♂️ What is "Customer Churn"?</h2>
            <p>Imagine you run a subscription business (like Netflix or a mobile carrier). <b>"Churn"</b> is when a customer decides to cancel their service and leave the company.</p>
            <p>Keeping an existing customer is <b>5x cheaper</b> than trying to find and convince a new one. Therefore, if we can predict who is thinking about leaving <i>before</i> they actually click "Cancel", we can save them.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="layman-card">
            <h2>🧠 How the Brain (XGBoost) Works</h2>
            <p>Our model is built using an algorithm called <b>XGBoost</b>. Think of it as a team of 100+ smart advisors playing a game of "20 Questions".</p>
            <p>When customer data comes in, the advisors ask questions like:</p>
            <ul>
                <li><i>Is the contract Month-to-Month?</i></li>
                <li><i>Has the customer been with us for less than a year?</i></li>
                <li><i>Is their monthly bill unusually high?</i></li>
            </ul>
            <p>By combining all these answers, the team calculates a <b>Churn Probability</b> (e.g., 85% chance of leaving).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("### 🔄 The Life Cycle: How the System Works in Real Life")

step_col1, step_col2, step_col3, step_col4 = st.columns(4)

with step_col1:
    st.markdown(
        """
        <div class="layman-card" style="min-height: 280px;">
            <h4 class="highlight-purple">Step 1: The Input</h4>
            <p style="font-size: 14px; color: #cbd5e1;">A customer's details (like contract type, tenure, and services used) are fed into the system.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step_col2:
    st.markdown(
        """
        <div class="layman-card" style="min-height: 280px;">
            <h4 class="highlight-purple">Step 2: The Prediction</h4>
            <p style="font-size: 14px; color: #cbd5e1;">The model instantly computes a probability and labels them as <b>Low, Medium, or High Risk</b>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step_col3:
    st.markdown(
        """
        <div class="layman-card" style="min-height: 280px;">
            <h4 class="highlight-purple">Step 3: The Drift Alert</h4>
            <p style="font-size: 14px; color: #cbd5e1;">As weeks go by, the system monitors incoming data. If customer behavior patterns shift, a <b>Drift Detector</b> flags it.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step_col4:
    st.markdown(
        """
        <div class="layman-card" style="min-height: 280px;">
            <h4 class="highlight-purple">Step 4: Self-Healing</h4>
            <p style="font-size: 14px; color: #cbd5e1;">When a shift is detected, the model automatically triggers a <b>retraining process</b> to learn the new patterns.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Add an interactive component to make it fun!
st.markdown("### 🎮 Try it Out: Layman's Simulator")
st.markdown("Let's simulate a customer profile to understand why they might churn:")

sim_col1, sim_col2 = st.columns([1, 1.5])

with sim_col1:
    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        help="Month-to-month contracts are much easier to cancel, making them high risk."
    )
    tenure_months = st.slider(
        "Tenure (Months)",
        1, 72, 12,
        help="Newer customers are generally less loyal and more likely to churn."
    )
    has_tech_support = st.radio(
        "Has Tech Support?",
        ["No", "Yes"],
        help="Customers with tech support are less frustrated and more likely to stay."
    )

with sim_col2:
    # Rule of thumb heuristic to simulate explanation
    score = 0
    reasons = []
    
    if contract_type == "Month-to-month":
        score += 45
        reasons.append("• **Month-to-Month Contract**: No long-term commitment makes it very easy to leave at any time.")
    else:
        score += 10
        
    if tenure_months < 6:
        score += 35
        reasons.append("• **New Customer**: In the first 6 months, customers are still evaluating the service and have low loyalty.")
    elif tenure_months < 24:
        score += 15
        reasons.append("• **Medium Tenure**: Has been with us for a while, but hasn't established deep habits yet.")
    else:
        score -= 15
        
    if has_tech_support == "No":
        score += 20
        reasons.append("• **No Tech Support**: If issues arise, the customer doesn't have an easy way to resolve them, leading to frustration.")
    else:
        score -= 10
        
    score = max(5, min(95, score))
    
    st.markdown("#### 📊 Risk Analysis:")
    if score >= 70:
        st.error(f"🚨 **High Risk Profile (estimated {score}% probability of leaving)**")
    elif score >= 40:
        st.warning(f"⚠️ **Medium Risk Profile (estimated {score}% probability of leaving)**")
    else:
        st.success(f"✅ **Low Risk Profile (estimated {score}% probability of leaving)**")
        
    st.markdown("##### Why this rating?")
    if reasons:
        st.markdown("\n".join(reasons))
    else:
        st.markdown("• The customer has a long-term commitment, solid loyalty history, and active support options.")
