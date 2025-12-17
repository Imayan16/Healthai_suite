# ==================================================
# HEALTHAI – FINAL MULTILINGUAL CLINICAL AI SYSTEM
# ==================================================

import streamlit as st
import numpy as np
import os
import json
import joblib
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from groq import Groq

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI – Multilingual Clinical AI System")
st.caption("Clinical ML • Clustering • Imaging • Medical RAG • Sentiment • Translator")

# ==================================================
# ENV VARIABLES
# ==================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not HF_TOKEN or not GROQ_API_KEY:
    st.error("HF_TOKEN or GROQ_API_KEY missing in Streamlit Secrets")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# CONSTANTS
# ==================================================
LANGUAGES = [
    "English","Tamil","Hindi","Telugu","Malayalam",
    "Kannada","Spanish","French","German","Arabic"
]

CLUSTER_INFO = {
    0: ("Low Risk Group", "Stable vitals, routine monitoring sufficient."),
    1: ("Medium Risk Group", "Moderate vitals, observation and follow-up advised."),
    2: ("High Risk Group", "Critical vitals, immediate medical attention required.")
}

HF_REPO = "Imayan16/healthai-models"

# ==================================================
# LOAD MODELS FROM HUGGING FACE
# ==================================================
@st.cache_resource
def load_models():
    def hf(file):
        return hf_hub_download(
            repo_id=HF_REPO,
            filename=file,
            token=HF_TOKEN
        )

    los_model = joblib.load(hf("los_model.pkl"))
    los_scaler = joblib.load(hf("los_scaler.pkl"))

    kmeans = joblib.load(hf("kmeans_cluster_model.pkl"))
    cluster_scaler = joblib.load(hf("cluster_scaler_final.pkl"))

    with open(hf("association_rules.json"), "r") as f:
        assoc_rules = json.load(f)

    cnn = tf.keras.models.load_model(
        hf("pneumonia_cnn_model.h5"),
        compile=False
    )

    return los_model, los_scaler, kmeans, cluster_scaler, assoc_rules, cnn

los_model, los_scaler, kmeans_model, cluster_scaler, assoc_rules, cnn_model = load_models()

# ==================================================
# UI TABS
# ==================================================
tabs = st.tabs([
    "Clinical Assessment",
    "Imaging Diagnosis",
    "Medical RAG Chat",
    "Sentiment Analysis",
    "Translator",
    "About"
])

# ==================================================
# TAB 1 – CLINICAL ASSESSMENT
# ==================================================
with tabs[0]:
    st.subheader("🩺 Patient Clinical Assessment")

    with st.form("clinical_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 45)
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
            hr = st.number_input("Heart Rate", 40, 150, 80)
        with c2:
            sbp = st.number_input("Systolic BP", 80, 250, 120)
            dbp = st.number_input("Diastolic BP", 40, 150, 80)
            sugar = st.number_input("Blood Sugar", 50, 400, 110)

        submit = st.form_submit_button("Analyze")

    if submit:
        X = np.array([[age, bmi, hr, sbp, dbp, sugar]])

        los = los_model.predict(los_scaler.transform(X))[0]
        cluster = int(kmeans_model.predict(cluster_scaler.transform(X))[0])
        cname, cdesc = CLUSTER_INFO[cluster]

        st.success(f"🏥 Estimated Length of Stay: **{los:.1f} days**")
        st.info(f"🧩 Cluster: **{cname}**")
        st.write(cdesc)

        st.subheader("🔗 Key Association Rules")
        for r in assoc_rules[:3]:
            st.write(
                f"• **{r['antecedents']} → {r['consequents']}** "
                f"(Confidence: {r['confidence']:.2f}, Lift: {r['lift']:.2f})"
            )

        notes = f"""
Age: {age}
BMI: {bmi}
Heart Rate: {hr}
Blood Pressure: {sbp}/{dbp}
Blood Sugar: {sugar}
Length of Stay: {los:.1f} days
Cluster: {cname}
"""

        st.download_button(
            "⬇️ Download Clinical Notes",
            notes,
            file_name="clinical_notes.txt"
        )

# ==================================================
# TAB 2 – IMAGING DIAGNOSIS
# ==================================================
with tabs[1]:
    st.subheader("🫁 Chest X-ray Diagnosis")

    img_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

    if img_file:
        img = Image.open(img_file).convert("RGB").resize((224,224))
        st.image(img, width=300)

        arr = np.array(img) / 255.0
        pred = cnn_model.predict(arr[np.newaxis,...])[0][0]

        label = "Pneumonia" if pred > 0.5 else "Normal"
        st.error("🚨 Pneumonia Detected") if label == "Pneumonia" else st.success("✅ Normal Chest")

        prompt = f"""
        Diagnosis: {label}
        Explain basic symptoms, causes and general care guidance.
        """
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        st.write(res.choices[0].message.content)

# ==================================================
# TAB 3 – MEDICAL RAG CHAT
# ==================================================
with tabs[2]:
    st.subheader("💬 Medical Knowledge Assistant")

    lang = st.selectbox("Response Language", LANGUAGES)
    question = st.text_input("Ask a medical question")

    if question:
        prompt = f"Answer the following medical question in {lang}:\n{question}"
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        st.write(res.choices[0].message.content)

# ==================================================
# TAB 4 – SENTIMENT ANALYSIS
# ==================================================
with tabs[3]:
    st.subheader("🧠 Clinical Sentiment Analysis")

    s_lang = st.selectbox("Output Language", LANGUAGES)
    text = st.text_area("Enter patient notes or statements")

    if text:
        prompt = f"""
        Analyze the sentiment of the following clinical text.
        Classify as Positive, Neutral, or Negative.
        Give short clinical interpretation.
        Respond in {s_lang}.

        Text:
        {text}
        """
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        st.write(res.choices[0].message.content)

# ==================================================
# TAB 5 – TRANSLATOR
# ==================================================
with tabs[4]:
    st.subheader("🌍 Medical Translator")

    src = st.text_area("Enter text to translate")
    tgt = st.selectbox("Target Language", LANGUAGES)

    if src:
        prompt = f"Translate the following medical text to {tgt}:\n{src}"
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        st.write(res.choices[0].message.content)

# ==================================================
# TAB 6 – ABOUT
# ==================================================
with tabs[5]:
    st.write("""
### HealthAI – Final Version
✔ LOS Prediction  
✔ Risk Clustering with Explanation  
✔ Association Rules  
✔ Image Diagnosis (CNN)  
✔ Multilingual Medical RAG  
✔ Multilingual Sentiment Analysis  
✔ Multilingual Translator  

Built using Hugging Face + Groq APIs
""")
