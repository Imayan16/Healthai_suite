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
# CONSTANTS
# =========================================================
REPO_ID = "imayan/healthai_models"

FEATURE_ORDER = [
    "age","gender","bmi","systolic_bp","diastolic_bp",
    "heart_rate","cholesterol","blood_sugar",
    "age_group","bmi_category","bp_category","metabolic_risk"
]

LANGUAGES = [
    "English","Tamil","Hindi","Telugu","Malayalam",
    "Kannada","Bengali","Spanish","French","German"
]
# =========================================================
# MEDICAL KNOWLEDGE BASE (RAG)
# =========================================================
@st.cache_resource
def load_medical_knowledge():
    try:
        with open("medical.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

MEDICAL_KB = load_medical_knowledge()


def is_rag_applicable(question, medical_text, threshold=2):
    q_words = question.lower().split()
    text = medical_text.lower()
    hits = sum(1 for w in q_words if w in text)
    return hits >= threshold

# =========================================================
# LOAD MODELS (HF)
# =========================================================
@st.cache_resource
def load_models():
    los_model = joblib.load(hf_hub_download(REPO_ID,"los_model.pkl",token=HF_TOKEN))
    los_scaler = joblib.load(hf_hub_download(REPO_ID,"los_scaler.pkl",token=HF_TOKEN))

    kmeans = joblib.load(hf_hub_download(REPO_ID,"kmeans_cluster_model.pkl",token=HF_TOKEN))
    cluster_scaler = joblib.load(hf_hub_download(REPO_ID,"cluster_scaler_final.pkl",token=HF_TOKEN))

    xgb_model = xgb.Booster()
    xgb_model.load_model(hf_hub_download(REPO_ID,"xgboost_disease_model.json",token=HF_TOKEN))

    with open(hf_hub_download(REPO_ID,"association_rules.json",token=HF_TOKEN)) as f:
        association_rules = json.load(f)

    cnn_model = tf.keras.models.load_model(
        hf_hub_download(REPO_ID,"pneumonia_cnn_model.h5",token=HF_TOKEN)
    )

    return los_model, los_scaler, kmeans, cluster_scaler, xgb_model, association_rules, cnn_model


los_model, los_scaler, kmeans, cluster_scaler, xgb_model, association_rules, cnn_model = load_models()

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
# TAB 1 – DISEASE ASSESSMENT
# =========================================================
with tabs[0]:
    st.subheader("🩺 Disease Assessment")

    with st.form("clinical_form"):
        age = st.number_input("Age",1,120,45)
        gender = st.selectbox("Gender",["Male","Female"])
        bmi = st.number_input("BMI",10.0,60.0,24.5)
        sbp = st.number_input("Systolic BP",80,240,120)
        dbp = st.number_input("Diastolic BP",40,150,80)
        hr = st.number_input("Heart Rate",40,160,78)
        chol = st.number_input("Cholesterol",100,400,180)
        sugar = st.number_input("Blood Sugar",60,400,110)
        submit = st.form_submit_button("Analyze")

    if submit:
        age_group = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        bp_category = 0 if (sbp < 120 and dbp < 80) else 1 if (sbp < 140 or dbp < 90) else 2
        metabolic_risk = 1 if (bmi >= 30 or sugar >= 126 or chol >= 240) else 0

        df = pd.DataFrame([{
            "age": age,
            "gender": 1 if gender=="Male" else 0,
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

        X = df[FEATURE_ORDER].values

        los_days = round(float(los_model.predict(los_scaler.transform(X))[0]),2)

        cluster_id = int(kmeans.predict(cluster_scaler.transform(X))[0])
        cluster_text = {
            0:"Low Risk Cluster – Stable vitals",
            1:"Medium Risk Cluster – Monitoring required",
            2:"High Risk Cluster – Critical condition"
        }[cluster_id]

        probs = xgb_model.predict(xgb.DMatrix(X))[0]
        ml_risk = ["LOW","MEDIUM","HIGH"][int(np.argmax(probs))]

        if metabolic_risk==1 and bp_category==2 and age_group>=2:
            disease_risk = "HIGH"
        elif metabolic_risk==1 or bp_category==2:
            disease_risk = "MEDIUM"
        else:
            disease_risk = ml_risk

        st.success(f"🕒 Predicted LOS: {los_days} days")
        st.info(f"🧩 Patient Cluster: {cluster_text}")
        st.warning(f"⚠️ Disease Risk Level: {disease_risk}")

        st.subheader("📌 Patient-Specific Association Insights")

        filtered = []
        for r in association_rules:
            ants = " ".join(r["antecedents"]).lower()
            if metabolic_risk==1 and ("diabetes" in ants or "obesity" in ants):
                filtered.append(r)
            elif bp_category==2 and ("hypertension" in ants or "bp" in ants):
                filtered.append(r)
            elif chol>=240 and "cholesterol" in ants:
                filtered.append(r)

        if not filtered:
            filtered = association_rules[:5]

        assoc_df = pd.DataFrame(filtered)
        assoc_df["antecedents"] = assoc_df["antecedents"].apply(lambda x:", ".join(x))
        assoc_df["consequents"] = assoc_df["consequents"].apply(lambda x:", ".join(x))

        st.dataframe(
            assoc_df[["antecedents","consequents","support","confidence","lift"]].head(5),
            use_container_width=True
        )

        report = df.copy()
        report["LOS_days"] = los_days
        report["Cluster"] = cluster_text
        report["Disease_Risk"] = disease_risk

        st.download_button(
            "⬇️ Download Clinical Report",
            report.to_csv(index=False),
            "clinical_report.csv",
            "text/csv"
        )

# =========================================================
# TAB 2 – X-RAY DETECTION
# =========================================================
with tabs[1]:
    st.subheader("🩻 X-ray Detection")

    img = st.file_uploader("Upload Chest X-ray",["jpg","png","jpeg"])
    if img:
        image = Image.open(img).convert("RGB")
        st.image(image,width=300)

        arr = np.expand_dims(np.array(image.resize((224,224)))/255.0,0)
        result = "PNEUMONIA DETECTED" if cnn_model.predict(arr)[0][0]>0.5 else "NO PNEUMONIA DETECTED"

        st.success(result)

        prompt = f"X-ray result: {result}. Explain symptoms and basic diagnosis."
        ans = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        st.write(ans.choices[0].message.content)

# =========================================================
# =========================================================
# TAB 3 – MEDICAL RAG (KB → LLM FALLBACK)
# =========================================================
with tabs[2]:
    st.subheader("💬 Medical RAG Chat")

    lang = st.selectbox("Language", LANGUAGES, key="rag_lang")
    q = st.text_input("Ask a medical question")

    if q:
        use_rag = is_rag_applicable(q, MEDICAL_KB)

        if use_rag and MEDICAL_KB.strip() != "":
            prompt = f"""
You are a clinical assistant.
Answer STRICTLY using the medical knowledge below.
If not found, say: "Not available in knowledge base".

Answer language: {lang}

MEDICAL KNOWLEDGE:
{MEDICAL_KB}

QUESTION:
{q}
"""
        else:
            prompt = f"""
You are a medical AI assistant.
Answer safely and clearly.

Answer language: {lang}

QUESTION:
{q}
"""

        with st.spinner("Thinking..."):
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = res.choices[0].message.content

        # ===== SOURCE INDICATOR UI =====
        if use_rag and MEDICAL_KB.strip() != "":
            st.success("✅ Answer Source: Medical Knowledge Base (RAG)")
        else:
            st.info("🤖 Answer Source: LLM Generated Response")

        st.markdown("### 🩺 Medical Answer")
        st.write(answer)


# =========================================================
# TAB 4 – SENTIMENT
# =========================================================
with tabs[3]:
    lang = st.selectbox("Language",LANGUAGES,key="sentiment_lang")
    txt = st.text_area("Clinical Notes")
    if txt:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":f"Analyze sentiment in {lang}: {txt}"}]
        )
        st.write(r.choices[0].message.content)

# =========================================================
# TAB 5 – TRANSLATOR
# =========================================================
with tabs[4]:
    text = st.text_input("Text")
    tgt = st.selectbox("Translate to",LANGUAGES,key="translator_lang")
    if text:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":f"Translate to {tgt}: {text}"}]
        )
        st.write(r.choices[0].message.content)

# =========================================================
# ABOUT
# =========================================================
with tabs[5]:
    st.write("""
**HealthAI – Professional Clinical AI System**

✔ Input-dependent Disease Risk  
✔ Patient-specific Association Rules  
✔ Exact training features  
✔ X-ray → RAG reasoning  
✔ Multilingual medical intelligence  
✔ Deployment ready
""")






