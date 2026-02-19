import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load model and artifacts
print("Loading model and artifacts...")
model = tf.keras.models.load_model('autoencoder_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

def predict_anomaly(
    network_packet_size, 
    protocol_type, 
    login_attempts, 
    session_duration, 
    encryption_used, 
    ip_reputation_score, 
    failed_logins, 
    browser_type, 
    unusual_time_access
):
    try:
        # Create input dict
        data = {
            "network_packet_size": network_packet_size,
            "protocol_type": protocol_type,
            "login_attempts": login_attempts,
            "session_duration": session_duration,
            "encryption_used": encryption_used,
            "ip_reputation_score": ip_reputation_score,
            "failed_logins": failed_logins,
            "browser_type": browser_type,
            "unusual_time_access": unusual_time_access
        }
        
        df_input = pd.DataFrame([data])
        
        # Encode categorical
        for col, le in label_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except ValueError:
                    df_input[col] = le.transform([le.classes_[0]])

        # Scale
        X_scaled = scaler.transform(df_input.values)
        
        # Reconstruct
        X_recon = model.predict(X_scaled)
        
        # Calculate MSE
        mse = float(np.mean(np.power(X_scaled - X_recon, 2)))
        
        # Result formatting
        is_anomaly = mse > 0.05
        status = "⚠️ Anomaly Detected" if is_anomaly else "✅ Normal Traffic"
        
        return {
            "Status": status,
            "Reconstruction Error (MSE)": round(mse, 6),
            "Details": f"The model reconstructed the input with {round((1-mse)*100, 2)}% confidence."
        }
    except Exception as e:
        return {"Error": str(e)}

# Gradio Interface
iface = gr.Interface(
    fn=predict_anomaly,
    inputs=[
        gr.Number(label="Network Packet Size"),
        gr.Dropdown(choices=list(label_encoders['protocol_type'].classes_), label="Protocol Type"),
        gr.Number(label="Login Attempts"),
        gr.Number(label="Session Duration"),
        gr.Dropdown(choices=list(label_encoders['encryption_used'].classes_), label="Encryption Used"),
        gr.Slider(0, 1, label="IP Reputation Score"),
        gr.Number(label="Failed Logins"),
        gr.Dropdown(choices=list(label_encoders['browser_type'].classes_), label="Browser Type"),
        gr.Radio([0, 1], label="Unusual Time Access")
    ],
    outputs="json",
    title="Cybersecurity Intrusion Autoencoder",
    description="This model detects anomalies in network traffic using an Autoencoder. Higher reconstruction error indicates potential intrusion."
)

if __name__ == "__main__":
    # share=True creates a public URL valid for 72 hours
    iface.launch(share=True)
