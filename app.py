# ============================================
# AI Customer Churn Intelligence Dashboard
# ============================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Customer Churn Intelligence",
    layout="wide"
)

st.title("AI Customer Churn Intelligence System")

st.markdown(
"""
This dashboard predicts customer churn risk using a Machine Learning model,
explains key drivers, estimates revenue exposure, and recommends retention actions.
"""
)

# ---------------------------
# Load Trained Model
# ---------------------------
model = pickle.load(open("logistic_churn_model.pkl", "rb"))

# ---------------------------
# Load Dataset Structure
# ---------------------------
data = pd.read_csv("data/customer_churnn.csv")

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['Churn'] = data['Churn'].map({'Yes':1,'No':0})
data.drop('customerID', axis=1, inplace=True)

data = pd.get_dummies(data, drop_first=True)
data.fillna(0, inplace=True)

X_columns = data.drop('Churn', axis=1).columns

# ---------------------------
# Input Panel
# ---------------------------
st.subheader("Customer Input Panel")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges", 0.0, 200.0, 70.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges", 0.0, 10000.0, 1000.0
    )

st.divider()

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Churn Risk"):

    # Create full feature structure
    input_df = pd.DataFrame(columns=X_columns)
    input_df.loc[0] = 0

    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges

    probability = model.predict_proba(input_df)[0][1]

    # ---------------------------
    # Prediction Output
    # ---------------------------
    st.metric(
        label="Churn Probability",
        value=f"{round(probability*100,2)} %"
    )

    st.progress(float(probability))

    # Risk Classification
    if probability < 0.3:
        st.success("Low Risk Customer")
        risk_level = "low"

    elif probability < 0.7:
        st.warning("Medium Risk Customer")
        risk_level = "medium"

    else:
        st.error("High Risk Customer")
        risk_level = "high"

    # ---------------------------
    # Revenue Risk Estimation
    # ---------------------------
    estimated_loss = monthly_charges * probability

    st.metric(
        "Estimated Revenue Risk ($ / month)",
        round(estimated_loss, 2)
    )

    st.divider()

    # ---------------------------
    # Explainable AI Section
    # ---------------------------
    st.subheader("Top Risk Drivers (Explainable AI)")

    coefficients = model.coef_[0]

    explain_df = pd.DataFrame({
        "Feature": X_columns,
        "Impact": coefficients
    })

    top_features = explain_df.sort_values(
        by="Impact",
        key=abs,
        ascending=False
    ).head(5)

    st.table(top_features)

    st.info(
        "These features contributed most strongly to the churn prediction."
    )

    st.divider()

    # ---------------------------
    # AI Retention Recommendations
    # ---------------------------
    st.subheader("AI Retention Recommendations")

    if risk_level == "low":
        st.success("""
        ✅ Customer is stable.

        Recommended Actions:
        - Maintain engagement
        - Offer loyalty rewards
        - Promote premium services
        """)

    elif risk_level == "medium":
        st.warning("""
        ⚠ Moderate churn risk detected.

        Recommended Actions:
        - Personalized marketing offers
        - Customer satisfaction surveys
        - Small promotional discounts
        """)

    else:
        st.error("""
        🚨 High churn risk detected.

        Recommended Actions:
        - Immediate retention campaign
        - Dedicated customer support
        - Pricing or contract incentives
        - Proactive outreach
        """)

    st.divider()

    # ---------------------------
    # Cloud Deployment Note
    # ---------------------------
    st.info(
        "This AI system is cloud-ready and deployable using Streamlit Cloud or Hugging Face Spaces."
    )