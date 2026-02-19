from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Cybersecurity Autoencoder Deployment")

# Load model and artifacts
print("Loading model and artifacts for deployment...")
model = tf.keras.models.load_model('autoencoder_model.keras')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

class NetworkTraffic(BaseModel):
    network_packet_size: float
    protocol_type: str
    login_attempts: int
    session_duration: float
    encryption_used: str
    ip_reputation_score: float
    failed_logins: int
    browser_type: str
    unusual_time_access: int

@app.get("/")
def home():
    return {"message": "Autoencoder API is running!"}

@app.post("/predict")
def predict(data: NetworkTraffic):
    try:
        # Convert input to DataFrame
        data_dict = data.dict()
        df_input = pd.DataFrame([data_dict])
        
        # Encode categorical features
        for col, le in label_encoders.items():
            if col in df_input.columns:
                # Handle unseen labels by mapping to a default if necessary (simple version)
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except ValueError:
                    # If label is unknown, use the first label in the encoder classes
                    df_input[col] = le.transform([le.classes_[0]])

        # Scale numerical features
        X_scaled = scaler.transform(df_input.values)
        
        # Predict (Reconstruct)
        X_recon = model.predict(X_scaled)
        
        # Calculate MSE
        mse = np.mean(np.power(X_scaled - X_recon, 2))
        
        # Set a threshold for anomaly detection (can be tweaked)
        is_anomaly = float(mse) > 0.05
        
        return {
            "reconstruction_error": float(mse),
            "is_anomaly": is_anomaly,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
