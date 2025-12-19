import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Churn Telco")

st.title("Prediksi Churn Pelanggan Telco")

model = joblib.load("best_churn_model.pkl")

st.write("Masukkan data pelanggan:")

# INPUT NUMERIK
tenure = st.number_input("Tenure (bulan)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# DATAFRAME LENGKAP (DEFAULT VALUE)
input_df = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [tenure],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total]
})

st.write("Data input:")
st.dataframe(input_df)

if st.button("Prediksi"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"Pelanggan DIPREDIKSI CHURN (Probabilitas: {prob:.2f})")
        else:
            st.success(f"Pelanggan TIDAK CHURN (Probabilitas: {prob:.2f})")

    except Exception as e:
        st.error("Terjadi error saat prediksi")
        st.write(e)
