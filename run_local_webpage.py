from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load model and artifacts
print("Loading model and artifacts for local webpage...")
try:
    model = tf.keras.models.load_model('autoencoder_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to .h5 if it exists and .keras fails
    if os.path.exists('autoencoder_model.h5'):
        print("Attempting to load legacy .h5 model...")
        model = tf.keras.models.load_model('autoencoder_model.h5')

@app.route('/')
def index():
    # Provide labels for dropdowns
    dropdowns = {
        'protocol_type': list(label_encoders['protocol_type'].classes_),
        'encryption_used': list(label_encoders['encryption_used'].classes_),
        'browser_type': list(label_encoders['browser_type'].classes_)
    }
    return render_template('index.html', dropdowns=dropdowns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        
        # Ensure correct column order
        cols = [
            'network_packet_size', 'protocol_type', 'login_attempts', 
            'session_duration', 'encryption_used', 'ip_reputation_score', 
            'failed_logins', 'browser_type', 'unusual_time_access'
        ]
        df_input = df_input[cols]
        
        # Encode
        for col, le in label_encoders.items():
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                df_input[col] = le.transform([le.classes_[0]])
        
        # Scale
        X_scaled = scaler.transform(df_input.values)
        
        # Predict
        X_recon = model.predict(X_scaled)
        
        # MSE
        mse = float(np.mean(np.power(X_scaled - X_recon, 2)))
        
        # Threshold
        threshold = 0.05
        is_anomaly = mse > threshold
        
        return jsonify({
            'mse': round(mse, 6),
            'is_anomaly': is_anomaly,
            'confidence': round((1 - min(mse, 1)) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
