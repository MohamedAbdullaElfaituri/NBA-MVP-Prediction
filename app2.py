import joblib
import streamlit as st
import numpy as np

# Load the pre-trained model
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏀 MVP Tahmin Aracı (XGBoost)")
st.write("Oyuncunun sezon istatistiklerini girin:")

# Girdi alanları
ws = st.number_input("Win Shares (WS)", min_value=0.0, step=0.1)
vorp = st.number_input("Value Over Replacement Player (VORP)", min_value=0.0, step=0.1)
ows = st.number_input("Offensive Win Shares (OWS)", min_value=0.0, step=0.1)

if st.button("MVP Durumunu Tahmin Et"):
    input_data = np.array([[ws, vorp, ows]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"🌟 Bu oyuncunun MVP olması olasılığı yüksek! (Güven: {proba:.2%})")
    else:
        st.info(f"❌ Bu oyuncunun MVP olması beklenmiyor. (Güven: {proba:.2%})")
