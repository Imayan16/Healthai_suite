# =========================================================
# HEALTHAI – FINAL STABLE MULTILINGUAL CLINICAL AI SYSTEM
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import json, os, io, base64
import joblib
import tensorflow as tf
import xgboost as xgb
from PIL import Image
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from groq import Groq

# =========================================================
# ENV
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI – Multilingual Clinical AI System")
st.caption("Clinical ML • Imaging Diagnosis • Medical RAG • Sentiment • Translator")

# =========================================================
# MODEL LOADING (HF)
# =========================================================
@st.cache_resource
def load_models():
    repo = "imayan/healthai_models"

    los_model = joblib.load(hf_hub_download(repo, "los_model.pkl", token=HF_TOKEN))
    los_scaler = joblib.load(hf_hub_download(repo, "los_scaler.pkl", token=HF_TOKEN))

    kmeans = joblib.load(hf_hub_download(repo, "kmeans_cluster_model.pkl", token=HF_TOKEN))
    cluster_scaler = joblib.load(hf_hub_download(repo, "cluster_scaler_final.pkl", token=HF_TOKEN))

    xgb_model = xgb.Booster()
    xgb_model.load_model(hf_hub_download(repo, "xgboost_disease_model.json", token=HF_TOKEN))

    with open(hf_hub_download(repo, "association_rules.json", token=HF_TOKEN)) as f:
        association_rules = json.load(f)

    cnn_model = tf.keras.models.load_model(
        hf_hub_download(repo, "pneumonia_cnn_model.h5", token=HF_TOKEN)
    )

    return los_model, los_scaler, kmeans, cluster_scaler, xgb_model, association_rules, cnn_model


los_model, los_scaler, kmeans, cluster_scaler, xgb_model, association_rules, cnn_model = load_models()

# =========================================================
# LANGUAGE LIST (10 LANGUAGES)
# =========================================================
LANGUAGES = [
    "English",
    "Tamil",
    "Hindi",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Bengali",
    "Spanish",
    "French",
    "German"
]

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "Clinical Assessment",
    "Image Diagnosis",
    "Medical RAG",
    "Sentiment",
    "Translator",
    "About"
])

# =========================================================
# TAB 1 – CLINICAL ASSESSMENT
# =========================================================
with tabs[0]:
    st.subheader("🩺 Clinical Assessment")

    with st.form("clinical"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 45)
            bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
            hr = st.number_input("Heart Rate", 40, 160, 78)
            sugar = st.number_input("Blood Sugar", 60, 400, 110)
        with c2:
            sbp = st.number_input("Systolic BP", 80, 240, 120)
            dbp = st.number_input("Diastolic BP", 40, 150, 80)
            chol = st.number_input("Cholesterol", 100, 400, 180)
            temp = st.number_input("Body Temp (°C)", 35.0, 42.0, 36.8)

        submit = st.form_submit_button("Analyze")

    if submit:
        X = np.array([[age, bmi, hr, sugar, sbp, dbp, chol, temp]])
        X_scaled = los_scaler.transform(X)

        los_days = round(float(los_model.predict(X_scaled)[0]), 2)

        cluster = int(kmeans.predict(cluster_scaler.transform(X))[0])
        cluster_desc = {
            0: "Low Risk – Stable",
            1: "Medium Risk – Observation",
            2: "High Risk – Critical"
        }[cluster]

        dmatrix = xgb.DMatrix(X)
        risk_score = xgb_model.predict(dmatrix)[0]
        disease_risk = ["LOW", "MEDIUM", "HIGH"][int(np.argmax(risk_score))]

        st.success(f"🕒 **Predicted LOS:** {los_days} days")
        st.info(f"🧩 **Patient Cluster:** {cluster_desc}")
        st.warning(f"⚠️ **Disease Risk:** {disease_risk}")

        st.subheader("📌 Association Insights")
        st.json(association_rules[:3])

        notes = f"""
Age: {age}
BMI: {bmi}
LOS: {los_days}
Cluster: {cluster_desc}
Disease Risk: {disease_risk}
"""
        st.download_button("⬇️ Download Clinical Notes", notes, "clinical_notes.txt")

# =========================================================
# TAB 2 – IMAGE DIAGNOSIS
# =========================================================
with tabs[1]:
    st.subheader("🖼️ Pneumonia Detection")

    img = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
    if img:
        image = Image.open(img).convert("RGB")
        st.image(image, width=300)

        img_arr = np.array(image.resize((224, 224))) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        pred = cnn_model.predict(img_arr)[0][0]
        label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

        st.success(f"🩻 Diagnosis: **{label}**")

        rag_prompt = f"Patient X-ray diagnosis is {label}. Give symptoms and precautions."
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rag_prompt}]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# TAB 3 – MEDICAL RAG (10 LANGUAGES)
# =========================================================
with tabs[2]:
    st.subheader("💬 Medical RAG Chat")

    lang = st.selectbox("Language", LANGUAGES)
    q = st.text_input("Ask a medical question")

    if q:
        prompt = f"Answer in {lang}. Medical question: {q}"
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# TAB 4 – SENTIMENT
# =========================================================
with tabs[3]:
    st.subheader("🧠 Clinical Sentiment")

    txt = st.text_area("Clinical Notes")
    if txt:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Analyze clinical sentiment: {txt}"}]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# TAB 5 – TRANSLATOR (10 LANGUAGES)
# =========================================================
with tabs[4]:
    st.subheader("🌍 Medical Translator")

    t = st.text_input("Text")
    tgt = st.selectbox("Translate to", LANGUAGES)

    if t:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Translate to {tgt}: {t}"}]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# ABOUT
# =========================================================
with tabs[5]:
    st.write("""
**HealthAI – Production Ready Clinical AI**
- Real ML Predictions
- HF Hosted Models
- Groq LLM APIs
- Streamlit Cloud Compatible
""")
