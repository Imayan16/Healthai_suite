import os
import json
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from groq import Groq

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="HealthAI Suite", layout="wide")
st.title("🧠 HealthAI Suite – Intelligent Clinical Assistant")
st.caption("ML • DL • Association Rules • Multilingual Medical RAG (Groq)")

# ==================================================
# PATHS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MEDICAL_FILE = os.path.join(BASE_DIR, "medical_knowledge.txt")
ASSOC_FILE = os.path.join(MODEL_DIR, "association_rules.json")

# ==================================================
# GROQ CLIENT
# ==================================================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ==================================================
# SAFE LOAD HELPERS (NO CRASH)
# ==================================================
def safe_joblib(path):
    return joblib.load(path) if os.path.exists(path) else None

def safe_tf(path):
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

def safe_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ==================================================
# LOAD MODELS (UNCHANGED LOGIC, SAFE)
# ==================================================
@st.cache_resource
def load_models():
    return (
        safe_joblib(os.path.join(MODEL_DIR, "los_model.pkl")),
        safe_joblib(os.path.join(MODEL_DIR, "los_scaler.pkl")),
        safe_joblib(os.path.join(MODEL_DIR, "kmeans_cluster_model.pkl")),
        safe_joblib(os.path.join(MODEL_DIR, "cluster_scaler_final.pkl")),
        safe_tf(os.path.join(MODEL_DIR, "pneumonia_cnn_model.h5")),
        safe_json(ASSOC_FILE)
    )

los_model, los_scaler, kmeans_model, cluster_scaler, cnn_model, assoc_rules = load_models()

# ==================================================
# LOAD MEDICAL KNOWLEDGE
# ==================================================
@st.cache_data
def load_medical_text():
    if os.path.exists(MEDICAL_FILE):
        with open(MEDICAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

medical_text = load_medical_text()

# ==================================================
# GROQ-BASED NLP FUNCTIONS
# ==================================================
def groq_medical_answer(question, knowledge):
    prompt = f"""
You are a medical assistant.
Use the following medical knowledge to answer the question.
Answer in the SAME language as the question.

Medical Knowledge:
{knowledge}

Question:
{question}
"""
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def groq_sentiment(text):
    prompt = f"""
Classify the sentiment of this patient text as
Positive, Neutral, or Negative.

Text:
{text}
"""
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def groq_translate(text, target_lang):
    prompt = f"Translate the following text to {target_lang}:\n{text}"
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Disease Prediction",
    "🧪 Imaging Diagnosis",
    "📚 Multilingual Medical Knowledge",
    "🧠 Sentiment & Translation"
])

# ==================================================
# TAB 1 – DISEASE (LOS + CLUSTER + ASSOCIATION)
# ==================================================
with tab1:
    st.subheader("Disease Outcome Analysis")

    age = st.number_input("Age", 1, 100, 45)
    bp = st.number_input("Blood Pressure", 60, 200, 120)
    bmi = st.number_input("BMI", 10.0, 45.0, 23.5)

    if st.button("Analyze Patient"):
        X = np.array([[age, bp, bmi]])

        # LOS
        if los_model is not None and los_scaler is not None:
            Xs = los_scaler.transform(X)
            los = max(1, int(los_model.predict(Xs)[0]))
            st.success(f"🏥 Predicted Length of Stay: {los} days")
        else:
            st.info("ℹ️ LOS prediction available in local setup.")

        # Cluster
        if kmeans_model is not None and cluster_scaler is not None:
            Xc = cluster_scaler.transform(X)
            cluster = int(kmeans_model.predict(Xc)[0])
            st.success(f"🧩 Patient Cluster: {cluster}")
        else:
            st.info("ℹ️ Patient clustering available in local setup.")

        # Association Rules
        if assoc_rules:
            st.subheader("🔗 Disease Risk Associations")
            for rule in assoc_rules[:3]:
                st.write(
                    f"• **{rule['antecedents']} → {rule['consequents']}** "
                    f"(Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})"
                )
        else:
            st.info("ℹ️ Association rules available in local setup.")

# ==================================================
# TAB 2 – CNN IMAGING
# ==================================================
with tab2:
    st.subheader("Pneumonia Detection from X-ray")

    uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        if cnn_model is None:
            st.info("ℹ️ Imaging diagnosis available in local execution.")
        else:
            img = img.resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            prob = cnn_model.predict(arr)[0][0]
            label = "Pneumonia" if prob > 0.5 else "Normal"
            st.success(f"🫁 {label} (Confidence: {prob:.2f})")

# ==================================================
# TAB 3 – MULTILINGUAL MEDICAL RAG (GROQ)
# ==================================================
with tab3:
    st.subheader("Multilingual Medical Knowledge Assistant")

    query = st.text_input("Ask a medical question (Any Language)")

    if query:
        if medical_text:
            answer = groq_medical_answer(query, medical_text)
            st.write("📖 **Answer:**")
            st.write(answer)
        else:
            st.info("Medical knowledge base not available.")

# ==================================================
# TAB 4 – SENTIMENT + TRANSLATION (GROQ)
# ==================================================
with tab4:
    st.subheader("Sentiment Analysis & Translation")

    text = st.text_area("Enter patient description / symptoms")

    if text:
        sent = groq_sentiment(text)
        st.success(f"🧠 Sentiment: {sent}")

        target_lang = st.selectbox(
            "Translate to",
            ["English", "Tamil", "Hindi", "French"]
        )

        translated = groq_translate(text, target_lang)
        st.write("🌍 **Translated Text:**")
        st.write(translated)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption(
    "⚠️ Note: Predictive models are fully functional in local execution. "
    "In cloud deployment, the application demonstrates intelligent workflows "
    "with graceful feature availability."
)
