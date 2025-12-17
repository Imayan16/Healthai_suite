#!/usr/bin/env python3
"""
HealthAI Suite - Model Download & Generation Script
Generates dummy models for Streamlit Cloud deployment
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras

def create_models():
    """Generate and save all required models"""
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("🏥 Creating HealthAI Models...")
    
    # 1. Generate dummy training data
    X = np.random.randn(100, 12)  # 100 samples, 12 features
    y_los = np.random.uniform(1, 10, 100)  # Length of stay
    
    # 2. Create LOS Model (Length of Stay Prediction)
    print("  📊 Creating LOS Model...")
    los_model = LinearRegression()
    los_model.fit(X, y_los)
    joblib.dump(los_model, os.path.join(MODEL_DIR, "los_model.pkl"))
    print("  ✓ los_model.pkl created")
    
    # 3. Create LOS Scaler
    print("  📊 Creating LOS Scaler...")
    los_scaler = StandardScaler()
    los_scaler.fit(X)
    joblib.dump(los_scaler, os.path.join(MODEL_DIR, "los_scaler.pkl"))
    print("  ✓ los_scaler.pkl created")
    
    # 4. Create KMeans Clustering Model
    print("  👥 Creating KMeans Cluster Model...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_cluster_model.pkl"))
    print("  ✓ kmeans_cluster_model.pkl created")
    
    # 5. Create Cluster Scaler
    print("  📊 Creating Cluster Scaler...")
    cluster_scaler = StandardScaler()
    cluster_scaler.fit(X)
    joblib.dump(cluster_scaler, os.path.join(MODEL_DIR, "cluster_scaler_final.pkl"))
    print("  ✓ cluster_scaler_final.pkl created")
    
    # 6. Create Simple Pneumonia CNN Model
    print("  🫁 Creating Pneumonia CNN Model...")
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save(os.path.join(MODEL_DIR, "pneumonia_cnn_model.h5"))
    print("  ✓ pneumonia_cnn_model.h5 created")
    
    print("\n✅ All models created successfully!")
    print(f"📁 Models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    create_models()
