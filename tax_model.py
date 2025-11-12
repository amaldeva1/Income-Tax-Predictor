# ======================================================
# INCOME TAX PREDICTION APP (HOME PAGE)
# ------------------------------------------------------
# Description:
#   This Streamlit web app predicts the Income Tax amount
#   based on Annual Income, Investment, and Deduction.
#   It uses a pre-trained Random Forest Regressor model.
# ------------------------------------------------------

import streamlit as st
import joblib
import numpy as np

# ------------------------------------------------------
# LOAD MODEL & SCALER
# ------------------------------------------------------
model = joblib.load("tax_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="Income Tax Prediction App",
    layout="centered"
)

# ------------------------------------------------------
# PAGE STYLE (Minimal Professional Look)
# ------------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 2rem;
            border-radius: 1rem;
        }
        h1 {
            text-align: center;
            color: #003366;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #555;
            font-size: 14px;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0055a5;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------
st.title("Income Tax Prediction App")
st.markdown("This application predicts the estimated tax amount based on your income, investments, and deductions.")

# ------------------------------------------------------
# USER INPUT FORM
# ------------------------------------------------------
with st.form("tax_form"):
    st.subheader("Enter Your Financial Details:")

    annual_income = st.number_input("Annual Income (₹)", min_value=100000.0, step=10000.0)
    investment = st.number_input("Total Investment (₹)", min_value=0.0, step=1000.0)
    deduction = st.number_input("Tax Deductions (₹)", min_value=0.0, step=1000.0)

    submitted = st.form_submit_button("Predict Tax")

# ------------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------------
if submitted:
    # Prepare and scale input
    user_data = np.array([[annual_income, investment, deduction]])
    user_data_scaled = scaler.transform(user_data)

    # Predict
    predicted_tax = model.predict(user_data_scaled)[0]

    st.success(f"Estimated Tax Payable: ₹{predicted_tax:,.2f}")

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("<div class='footer'>Developed by Amal Dev | Machine Learning Project</div>", unsafe_allow_html=True)
