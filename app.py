# ======================================================
# INCOME TAX PREDICTION APP (FINAL COMBINED VERSION)
# ------------------------------------------------------
# Author   : Amal Dev
# Purpose  : ML-based web app to predict income tax amount
# Version  : 3.5 (With Background Image from Google)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------
# PAGE CONFIG (must come first)
# ------------------------------------------------------
st.set_page_config(page_title="Income Tax Predictor", layout="wide")

# ------------------------------------------------------
# 🌆 ADD BACKGROUND IMAGE (From Google/Unsplash URL)
# ------------------------------------------------------
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# for background image
add_bg_from_url("https://wallpapers.com/images/hd/black-blur-background-1okwgennjzjn28y9.jpg")

# ------------------------------------------------------
# HELPER FUNCTION – TRAIN MODEL IF FILES MISSING/CORRUPTED
# ------------------------------------------------------
def train_model_if_needed():
    if not os.path.exists("tax_model.pkl") or not os.path.exists("scaler.pkl"):
        st.warning("Model files not found or corrupted. Re-training the model automatically...")
        if not os.path.exists("tax_dataset.csv"):
            st.error("Dataset 'tax_dataset.csv' not found! Please add it to your project folder.")
            st.stop()

        df = pd.read_csv("tax_dataset.csv")
        X = df[["Annual_Income", "Investment", "Deduction"]]
        y = df["Tax"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, "tax_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model retrained and saved successfully!")
        return model, scaler
    else:
        try:
            model = joblib.load("tax_model.pkl")
            scaler = joblib.load("scaler.pkl")
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Re-training model automatically...")
            os.remove("tax_model.pkl") if os.path.exists("tax_model.pkl") else None
            os.remove("scaler.pkl") if os.path.exists("scaler.pkl") else None
            return train_model_if_needed()

# ------------------------------------------------------
# LOAD OR RETRAIN MODEL
# ------------------------------------------------------
model, scaler = train_model_if_needed()

# ------------------------------------------------------
# NAVIGATION MENU
# ------------------------------------------------------
page = st.sidebar.selectbox(
    "Navigate",
    ("🏠 Home", "📊 Tax Predictor", "ℹ️ About Model")
)

# ======================================================
#  PAGE 1: HOME PAGE
# ======================================================
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#003366;'>Income Tax Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:gray;'>Machine Learning Based Tax Estimation System</h4>", unsafe_allow_html=True)
    st.write("---")

    st.subheader("Overview")
    st.write("""
    The **Income Tax Prediction App** is a machine learning-based web application that helps individuals estimate 
    their **annual tax amount** based on their **income, investment,** and **deductions**.  
    It is designed to assist users in understanding how financial decisions affect their taxable income.
    """)

    st.subheader("Key Features")
    st.markdown("""
    - Predicts tax based on income, investment, and deduction values.  
    - Automatically retrains model if files are missing or corrupted.  
    - Displays results with pie and bar charts.  
    - Uses Random Forest Regressor for accurate predictions.  
    - Built with Streamlit for a fast and interactive user experience.
    """)

    st.subheader("Workflow")
    st.markdown("""
    1. **Data Collection:** Dataset includes annual income, investment, deduction, and actual tax.  
    2. **Preprocessing:** Data is scaled using StandardScaler.  
    3. **Model Training:** Random Forest Regressor is trained to predict tax.  
    4. **Prediction:** User inputs are scaled and fed to the model for instant results.  
    5. **Visualization:** Pie and bar charts display tax breakdown clearly.
    """)

    st.subheader("How to Use")
    st.markdown("""
    1. Go to the **Tax Predictor** page from the sidebar.  
    2. Enter your annual income, investment, and deductions.  
    3. Click **Predict Tax** to get estimated tax and charts.  
    4. Check the **About Model** page to understand the ML process.
    """)

    st.write("---")
    st.markdown("<p style='text-align:center;color:gray;'>Developed by <b>Amal Dev</b> | Machine Learning Project | © 2025</p>", unsafe_allow_html=True)

# ======================================================
#  PAGE 2: TAX PREDICTOR
# ======================================================
elif page == "📊 Tax Predictor":
    st.markdown("<h1 style='text-align:center; color:#003366;'>Tax Prediction</h1>", unsafe_allow_html=True)
    st.write("Enter your financial details below to estimate your income tax:")

    # USER INPUTS
    col1, col2, col3 = st.columns(3)
    with col1:
        annual_income = st.number_input("Annual Income (₹)", min_value=100000, max_value=5000000, value=500000, step=5000)
    with col2:
        investment = st.number_input("Investment (₹)", min_value=0, max_value=500000, value=50000, step=5000)
    with col3:
        deduction = st.number_input("Deduction (₹)", min_value=0, max_value=300000, value=20000, step=5000)

    # PREDICTION
    if st.button("Predict Tax"):
        input_data = np.array([[annual_income, investment, deduction]])
        input_scaled = scaler.transform(input_data)
        predicted_tax = model.predict(input_scaled)[0]
        remaining_income = annual_income - investment - deduction - predicted_tax

        st.subheader("Prediction Results")
        st.write(f"**Predicted Tax:** ₹{predicted_tax:,.2f}")
        st.write(f"**Remaining Income (After Tax & Deductions):** ₹{remaining_income:,.2f}")

        # PIE + BAR CHARTS
        col_pie, col_bar = st.columns(2)
        labels = ["Investment", "Deduction", "Tax", "Remaining Income"]
        values = [investment, deduction, predicted_tax, remaining_income]

        with col_pie:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            ax1.axis("equal")
            st.pyplot(fig1)

        with col_bar:
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.bar(labels, values)
            ax2.set_title("Income Breakdown", fontsize=10)
            ax2.set_ylabel("Amount (₹)", fontsize=8)
            ax2.tick_params(axis='x', labelrotation=20, labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("<hr><p style='text-align:center;color:gray;'>Developed by <b>Amal Dev</b></p>", unsafe_allow_html=True)

# ======================================================
#  PAGE 3: ABOUT MODEL
# ======================================================
elif page == "ℹ️ About Model":
    st.markdown("<h1 style='text-align:center; color:#003366;'>About the Model</h1>", unsafe_allow_html=True)

    st.write("""
    The **Income Tax Prediction Model** is built using a **Random Forest Regressor**, 
    a powerful ensemble algorithm that improves accuracy by combining multiple decision trees.

    ### Model Details:
    - **Algorithm Used:** Random Forest Regressor  
    - **Library:** scikit-learn  
    - **Scaler Used:** StandardScaler  
    - **Evaluation Metrics:** R² Score and Mean Absolute Error (MAE)  

    ### Why Random Forest?
    - Provides high accuracy  
    - Handles large and small datasets effectively  
    - Reduces overfitting compared to single decision trees  

    ### Notes:
    If model or scaler files are missing, the app automatically retrains the model from the dataset.
    """)

    st.markdown("<hr><p style='text-align:center;color:gray;'>Developed by <b>Amal Dev</b> | ML-Based Tax Estimation System</p>", unsafe_allow_html=True)
