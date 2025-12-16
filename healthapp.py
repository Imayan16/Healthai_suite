# ==================================================
# HEALTHAI – FINAL STABLE MULTILINGUAL CLINICAL AI SYSTEM
# ==================================================

import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

# ==================================================
# ENV + PAGE
# ==================================================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI – Multilingual Clinical AI System")
st.caption("Clinical ML • Imaging AI • Medical RAG • Sentiment • Translator")

if not API_KEY:
    st.error("❌ GROQ_API_KEY missing")
    st.stop()

client = Groq(api_key=API_KEY)

# ==================================================
# PATHS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==================================================
# FEATURE ORDER (LOCKED 🔒)
# ==================================================
FEATURE_ORDER = [
    "age","gender","bmi","systolic_bp","diastolic_bp",
    "heart_rate","cholesterol","blood_sugar",
    "age_group","bmi_category","bp_category","metabolic_risk"
]

# ==================================================
# LOAD MEDICAL KNOWLEDGE
# ==================================================
@st.cache_resource
def load_medical_knowledge():
    with open(os.path.join(BASE_DIR, "medical_knowledge.txt"), "r", encoding="utf-8") as f:
        return f.read()

MEDICAL_TEXT = load_medical_knowledge()

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    los_model = joblib.load(os.path.join(MODEL_DIR, "los_model.pkl"))
    los_scaler = joblib.load(os.path.join(MODEL_DIR, "los_scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_cluster_model.pkl"))
    cluster_scaler = joblib.load(os.path.join(MODEL_DIR, "cluster_scaler_final.pkl"))
    cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "pneumonia_cnn_model.h5"))
    return los_model, los_scaler, kmeans, cluster_scaler, cnn

los_model, los_scaler, kmeans_model, cluster_scaler, cnn_model = load_models()

# ==================================================
# FEATURE ENGINEERING
# ==================================================
def engineer_features(age, gender, bmi, sbp, dbp, hr, chol, sugar):
    age_group = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    bmi_cat = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    bp_cat = 2 if sbp >= 140 or dbp >= 90 else 1 if sbp >= 130 or dbp >= 80 else 0
    metabolic = int(bmi_cat >= 2) + int(sugar >= 120) + int(chol >= 200)

    df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": hr,
        "cholesterol": chol,
        "blood_sugar": sugar,
        "age_group": age_group,
        "bmi_category": bmi_cat,
        "bp_category": bp_cat,
        "metabolic_risk": metabolic
    }])

    return df[FEATURE_ORDER]

# ==================================================
# EXPLANATIONS
# ==================================================
def risk_explanation(risk):
    return {
        "LOW": "🟢 Low risk patient with stable vitals.",
        "MEDIUM": "🟠 Moderate risk patient – monitoring recommended.",
        "HIGH": "🔴 High risk patient – clinical attention required."
    }[risk]

def cluster_explanation(cid):
    return {
        0: "Cluster 0: Low-risk stable patients.",
        1: "Cluster 1: Moderate-risk patients with BP/metabolic issues.",
        2: "Cluster 2: High-risk cardio-metabolic patients."
    }.get(cid, "Unknown cluster profile.")

# ==================================================
# MEDICAL RAG (10 LANGUAGES)
# ==================================================
def retrieve_context(query):
    sentences = MEDICAL_TEXT.split(".")
    words = query.lower().split()
    return ". ".join([s for s in sentences if sum(w in s.lower() for w in words) >= 2])

def rag_answer(question, language):
    context = retrieve_context(question)
    prompt = f"""
Answer STRICTLY in {language}.
Use medical knowledge context if available.
Do NOT diagnose or prescribe.
Add disclaimer.

Context:
{context}

Question:
{question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content.strip()

# ==================================================
# TRANSLATOR (STRICT)
# ==================================================
def translate_text(text, target_language):
    prompt = f"Translate EXACTLY to {target_language}. No explanation.\nText:\n{text}"
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content.strip()

# ==================================================
# SENTIMENT
# ==================================================
def sentiment_analysis(text):
    prompt = f"Classify sentiment as Positive, Neutral, or Negative:\n{text}"
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content.strip()

# ==================================================
# UI TABS
# ==================================================
tabs = st.tabs([
    "🧾 Clinical Assessment",
    "🩻 Imaging Diagnosis",
    "🧠 Medical RAG",
    "💬 Sentiment",
    "🌍 Translator"
])

# ==================================================
# TAB 1 – CLINICAL ASSESSMENT (DOCTOR DASHBOARD UI)
# ==================================================
with tabs[0]:
    with st.form("clinical_form"):
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age",1,120,50)
            gender = st.selectbox("Gender",["Male","Female"])
            bmi = st.number_input("BMI",10.0,60.0,25.0)
            hr = st.number_input("Heart Rate",40,150,80)
        with c2:
            sbp = st.number_input("Systolic BP",80,250,120)
            dbp = st.number_input("Diastolic BP",40,150,80)
            sugar = st.number_input("Blood Sugar",50,400,110)
            chol = st.number_input("Cholesterol",100,400,180)
        analyze = st.form_submit_button("Analyze")

    if analyze:
        X = engineer_features(age,1 if gender=="Male" else 0,bmi,sbp,dbp,hr,chol,sugar)

        los = max(1, los_model.predict(los_scaler.transform(X))[0])
        cluster = int(kmeans_model.predict(cluster_scaler.transform(X))[0])

        if sbp>=140 or sugar>=126:
            risk, color = "HIGH", "red"
        elif bmi>=25 or sbp>=130:
            risk, color = "MEDIUM", "orange"
        else:
            risk, color = "LOW", "green"

        st.markdown("---")
        st.subheader("📊 Clinical Assessment")

        # Risk Card
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:10px;
            background-color:{color};color:white;font-size:22px;">
            <b>Disease Risk: {risk}</b><br>
            {risk_explanation(risk)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        c1,c2 = st.columns(2)

        with c1:
            st.metric("🏥 Length of Stay", f"{los:.1f} days")
            st.caption("Estimated hospital stay duration")

        with c2:
            st.metric("👥 Patient Cluster", f"Cluster {cluster}")
            st.caption(cluster_explanation(cluster))

        report = f"""
HEALTHAI CLINICAL REPORT
-----------------------
Age: {age}
Gender: {gender}
Risk: {risk}
LOS: {los:.1f} days
Cluster: {cluster}

DISCLAIMER:
Decision-support only. Not a diagnosis.
"""
        st.download_button("📄 Download Clinical Report", report, "clinical_report.txt")

# ==================================================
# TAB 2 – IMAGING
# ==================================================
with tabs[1]:
    img = st.file_uploader("Upload Chest X-ray",["jpg","png","jpeg"])
    if img:
        im = Image.open(img).convert("RGB")
        st.image(im,width=300)
        arr = np.expand_dims(np.array(im.resize((224,224)))/255.0,0)
        prob = cnn_model.predict(arr)[0][0]
        label = "Pneumonia" if prob>0.5 else "Normal"
        st.success(f"{label} (Confidence {prob:.2f})")

# ==================================================
# TAB 3 – MEDICAL RAG
# ==================================================
with tabs[2]:
    lang = st.selectbox("Choose language",[
        "English","Tamil","Hindi","Telugu","Malayalam",
        "Kannada","Spanish","French","German","Arabic"
    ])
    q = st.text_input("Ask a medical question")
    if q:
        ans = rag_answer(q,lang)
        st.write(ans)
        st.download_button("⬇️ Download Answer", ans, "rag_answer.txt")

# ==================================================
# TAB 4 – SENTIMENT
# ==================================================
with tabs[3]:
    s = st.text_input("Enter text for sentiment analysis")
    if s:
        st.success(sentiment_analysis(s))

# ==================================================
# TAB 5 – TRANSLATOR
# ==================================================
with tabs[4]:
    t = st.text_input("Text to translate")
    tl = st.selectbox("Translate to",[
        "Tamil","Hindi","Telugu","Malayalam","Kannada",
        "Spanish","French","German","Arabic","English"
    ])
    if t:
        st.write(translate_text(t,tl))

st.caption("HealthAI | Final Stable Multilingual Clinical AI System 🚀")

