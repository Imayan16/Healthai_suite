HealthAI Suite - Multilingual Clinical AI System
A comprehensive Streamlit-based clinical AI system for healthcare applications.

Features
1. Clinical Assessment
Patient risk prediction (LOW/MEDIUM/HIGH)
Length of Stay (LOS) estimation
Patient clustering and segmentation
2. Imaging Diagnosis
CNN-based pneumonia detection from chest X-ray images
Confidence scoring
3. Medical RAG (Retrieval-Augmented Generation)
Medical question answering
Support for 10 languages: English, Tamil, Hindi, Telugu, Malayalam, Kannada, Spanish, French, German, Arabic
4. Sentiment Analysis
Text sentiment classification (Positive, Neutral, Negative)
5. Multilingual Translator
Translate text across 10 supported languages
Tech Stack
Streamlit (>=1.38.0) - Frontend
TensorFlow (2.20.0) - Deep Learning
Pandas, NumPy - Data Processing
scikit-learn, XGBoost - ML Models
PIL, OpenCV - Image Processing
Groq API - LLM Integration
python-dotenv - Environment Management
Installation
Clone repository:
git clone https://github.com/akashBv6680/HealthAI-Suite-Replica.git
cd HealthAI-Suite-Replica
Create virtual environment:
python -m venv venv
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Set environment variables:
echo "GROQ_API_KEY=your_groq_api_key" > .env
Usage
Run the app:

streamlit run healthapp.py
Project Files
healthapp.py - Main Streamlit application
heathai_suite.ipynb - Data preprocessing notebook
requirements.txt - Python dependencies
Disclaimer
This system is for decision-support only. Always consult qualified healthcare professionals.

Last Updated: December 17, 2025
