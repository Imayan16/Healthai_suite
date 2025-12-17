Model Files Information
This directory contains the pre-trained machine learning models for HealthAI Suite.

Required Files
1. los_model.pkl (Length of Stay Prediction Model)
Size: ~1 KB
Type: Scikit-learn Random Forest Regressor
Purpose: Predicts hospital stay duration based on patient vitals
Input Features: 12 clinical features
Output: Float (days)
2. los_scaler.pkl (LOS Feature Scaler)
Size: ~1 KB
Type: Scikit-learn StandardScaler
Purpose: Scales features for LOS model prediction
3. kmeans_cluster_model.pkl (Patient Clustering Model)
Size: ~392 KB
Type: Scikit-learn K-Means Clustering
Clusters: 3 patient risk groups
Purpose: Segments patients into risk categories
4. cluster_scaler_final.pkl (Clustering Feature Scaler)
Size: ~1 KB
Type: Scikit-learn StandardScaler
Purpose: Scales features for clustering model
5. xgboost_disease_model.json (Disease Risk Model)
Size: ~1 MB
Type: XGBoost model in JSON format
Purpose: Predicts disease risk classification
Classes: LOW, MEDIUM, HIGH
6. association_rules.json (Medical Association Rules)
Size: ~21 KB
Type: JSON-encoded association rules
Purpose: Medical symptom-disease associations
7. pneumonia_cnn_model.h5 (Pneumonia Detection CNN)
Size: Large (typically 20-50 MB)
Type: TensorFlow/Keras CNN model
Purpose: Detects pneumonia from chest X-ray images
Input: 224x224 RGB images
Output: Binary classification (Normal/Pneumonia)
File Placement
All files should be placed in the /models/ directory:

HealthAI-Suite-Replica/
├── models/
│   ├── los_model.pkl
│   ├── los_scaler.pkl
│   ├── kmeans_cluster_model.pkl
│   ├── cluster_scaler_final.pkl
│   ├── xgboost_disease_model.json
│   ├── association_rules.json
│   └── pneumonia_cnn_model.h5
├── healthapp.py
├── requirements.txt
└── README.md
How to Obtain These Files
From WhatsApp/Shared Links: The models are shared via Google Drive links
Download: Download each file and place in the /models/ directory
Alternative: Use the provided download script (if available)
Loading Models in Python
import joblib
import json
import tensorflow as tf
import xgboost as xgb

# Load LOS model
los_model = joblib.load('models/los_model.pkl')
los_scaler = joblib.load('models/los_scaler.pkl')

# Load Clustering model
kmeans = joblib.load('models/kmeans_cluster_model.pkl')
cluster_scaler = joblib.load('models/cluster_scaler_final.pkl')

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_disease_model.json')

# Load association rules
with open('models/association_rules.json', 'r') as f:
    rules = json.load(f)

# Load CNN model
cnn_model = tf.keras.models.load_model('models/pneumonia_cnn_model.h5')
Notes
Binary .pkl files preserve Python object serialization
JSON files are human-readable and language-agnostic
H5 format is specific to TensorFlow/Keras
Ensure TensorFlow version compatibility (2.20.0+)
License & Attribution
These models are trained specifically for HealthAI Suite.
