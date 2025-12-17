# =========================================================
# HEALTHAI – FINAL STABLE MULTILINGUAL CLINICAL AI SYSTEM
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import json, os
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
st.caption("Clinical ML • X-ray Analysis • Medical RAG • Sentiment • Translator")

# =========================================================
# LOAD MODELS FROM HUGGINGFACE
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
# LANGUAGES (10)
# =========================================================
LANGUAGES = [
    "English", "Tamil", "Hindi", "Telugu", "Malayalam",
    "Kannada", "Bengali", "Spanish", "French", "German"
]

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "Disease Assessment",
    "X-ray Detection",
    "Medical RAG",
    "Sentiment",
    "Translator",
    "About"
])

# =========================================================
# TAB 1 – DISEASE ASSESSMENT (12 FEATURES ONLY)
# =========================================================
with tabs[0]:
    st.subheader("🩺 Disease Assessment")

    with st.form("clinical_form"):
        age = st.number_input("Age", 1, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
        sbp = st.number_input("Systolic BP", 80, 240, 120)
        dbp = st.number_input("Diastolic BP", 40, 150, 80)
        hr = st.number_input("Heart Rate", 40, 160, 78)
        chol = st.number_input("Cholesterol", 100, 400, 180)
        sugar = st.number_input("Blood Sugar", 60, 400, 110)

        submit = st.form_submit_button("Analyze")

    if submit:
        # ----- Derived features (EXACT training logic) -----
        age_group = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        bp_category = 0 if sbp < 120 and dbp < 80 else 1 if sbp < 140 or dbp < 90 else 2
        metabolic_risk = 1 if (bmi >= 30 or sugar >= 126 or chol >= 240) else 0

        df = pd.DataFrame([{
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "bmi": bmi,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "heart_rate": hr,
            "cholesterol": chol,
            "blood_sugar": sugar,
            "age_group": age_group,
            "bmi_category": bmi_category,
            "bp_category": bp_category,
            "metabolic_risk": metabolic_risk
        }])

        # ----- Predictions -----
        los_days = round(float(los_model.predict(los_scaler.transform(df))[0]), 2)

        cluster_id = int(kmeans.predict(cluster_scaler.transform(df))[0])
        cluster_info = {
            0: "Low Risk Cluster – Stable vitals, minimal intervention needed",
            1: "Medium Risk Cluster – Moderate abnormalities, needs monitoring",
            2: "High Risk Cluster – Critical indicators, immediate attention required"
        }[cluster_id]

        dmatrix = xgb.DMatrix(df)
        disease_risk = ["LOW", "MEDIUM", "HIGH"][int(np.argmax(xgb_model.predict(dmatrix)))]

        # ----- Output -----
        st.success(f"🕒 Predicted Length of Stay: {los_days} days")
        st.info(f"🧩 Patient Cluster: {cluster_info}")
        st.warning(f"⚠️ Disease Risk Level: {disease_risk}")

        st.subheader("📌 Medical Association Insights")
        st.json(association_rules[:3])

# =========================================================
# TAB 2 – X-RAY DETECTION (RESULT → RAG)
# =========================================================
with tabs[1]:
    st.subheader("🩻 X-ray Detection")

    img = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
    if img:
        image = Image.open(img).convert("RGB")
        st.image(image, width=300)

        arr = np.array(image.resize((224, 224))) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = cnn_model.predict(arr)[0][0]
        xray_result = "PNEUMONIA DETECTED" if pred > 0.5 else "NO PNEUMONIA DETECTED"

        st.success(f"X-ray Result: {xray_result}")

        rag_prompt = f"""
X-ray analysis result: {xray_result}.
Explain possible symptoms and give a basic medical diagnosis.
"""
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

    lang = st.selectbox("Select Language", LANGUAGES)
    q = st.text_input("Ask a medical question")

    if q:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Answer in {lang}. Medical question: {q}"
            }]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# TAB 4 – SENTIMENT (10 LANGUAGES)
# =========================================================
with tabs[3]:
    st.subheader("🧠 Clinical Sentiment Analysis")

    lang = st.selectbox("Sentiment Language", LANGUAGES)
    txt = st.text_area("Clinical Notes")

    if txt:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Analyze sentiment in {lang}: {txt}"
            }]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# TAB 5 – TRANSLATOR (10 LANGUAGES)
# =========================================================
with tabs[4]:
    st.subheader("🌍 Medical Translator")

    text = st.text_input("Text")
    tgt = st.selectbox("Translate to", LANGUAGES)

    if text:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Translate to {tgt}: {text}"
            }]
        )
        st.write(res.choices[0].message.content)

# =========================================================
# ABOUT
# =========================================================
with tabs[5]:
    st.write("""
**HealthAI – Professional Clinical AI System**
- Uses exact training features
- Disease clustering with explanation
- X-ray result → RAG reasoning
- Multilingual Medical RAG, Sentiment & Translation
- HuggingFace-hosted models
- Groq-powered intelligence
""")


