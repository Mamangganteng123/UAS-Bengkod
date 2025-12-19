import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_churn_model.pkl")

st.set_page_config(page_title="Prediksi Churn Telco", layout="centered")

st.title("📊 Prediksi Churn Pelanggan Telco")
st.write("""
Aplikasi ini digunakan untuk memprediksi apakah seorang pelanggan
berpotensi **churn (berhenti berlangganan)** atau **tidak churn**
berdasarkan karakteristik layanan dan pelanggan.
""")

st.divider()

st.subheader("🔧 Input Data Pelanggan")

tenure = st.number_input("Tenure (lama berlangganan dalam bulan)", min_value=0)
monthly = st.number_input("Monthly Charges (biaya bulanan)", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total]
})

st.divider()

if st.button("🔍 Prediksi Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"❌ Pelanggan **BERPOTENSI CHURN**\n\nProbabilitas: {prob:.2f}")
    else:
        st.success(f"✅ Pelanggan **TIDAK CHURN**\n\nProbabilitas: {prob:.2f}")

st.divider()

st.caption("UAS Bengkel Koding Data Science – Telco Customer Churn")
