import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def build_and_train():
    # 1. Load Dataset
    print("Loading dataset...")
    df = pd.read_csv('cybersecurity_intrusion_data.csv')
    
    # Use less data as requested
    df = df.head(2000)
    
    # 2. Preprocessing
    print("Preprocessing data...")
    # Drop session_id (not useful for training)
    if 'session_id' in df.columns:
        df = df.drop('session_id', axis=1)
    
    # Label encoding for categorical variables
    categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # We want to reconstruct everything except maybe the label if it's for anomaly detection
    # But usually autoencoders reconstruct all features.
    # Let's keep 'attack_detected' out of the input to make it more like an anomaly detector
    # Or just include it if the user wants to reconstruct the whole row.
    # I'll exclude 'attack_detected' from training to make it a proper feature-based autoencoder.
    target = 'attack_detected'
    features = [col for col in df.columns if col != target]
    
    X = df[features].values
    
    # Scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (just for evaluation)
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # 3. Build Simple Autoencoder
    print("Building model...")
    input_dim = X_train.shape[1]
    
    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu') # Bottleneck
    ])
    
    decoder = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    autoencoder = models.Sequential([encoder, decoder])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 4. Train with less epochs
    print("Training model...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, X_test),
        verbose=1
    )
    
    # 5. Save Model and Preprocessing objects
    print("Saving model and artifacts...")
    autoencoder.save('autoencoder_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print("Success! Model trained and saved.")

if __name__ == "__main__":
    build_and_train()
