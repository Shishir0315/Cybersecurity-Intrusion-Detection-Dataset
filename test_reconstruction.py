import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def test_model():
    print("Loading model and artifacts...")
    model = tf.keras.models.load_model('autoencoder_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    
    # Load original data to get a sample
    df = pd.read_csv('cybersecurity_intrusion_data.csv')
    df_sample = df.head(5).copy()
    
    # Preprocess sample
    print("\nOriginal Sample (first 5 rows):")
    print(df_sample.drop('session_id', axis=1))
    
    sample_ids = df_sample['session_id'].values
    df_proc = df_sample.drop(['session_id', 'attack_detected'], axis=1)
    
    # Encode categorical
    for col, le in label_encoders.items():
        df_proc[col] = le.transform(df_proc[col].astype(str))
        
    # Scale
    X_sample = scaler.transform(df_proc.values)
    
    # Predict (Reconstruct)
    X_recon = model.predict(X_sample)
    
    # Calculate Reconstruction Error (MSE per sample)
    mse = np.mean(np.power(X_sample - X_recon, 2), axis=1)
    
    print("\nReconstruction Results:")
    for i in range(len(sample_ids)):
        print(f"Session: {sample_ids[i]} | Reconstruction Error (MSE): {mse[i]:.6f}")
        if mse[i] > 0.1: # Threshold could be tuned
            print("  -> Potential Anomaly Detected!")
        else:
            print("  -> Traffic looks Normal.")

if __name__ == "__main__":
    test_model()
