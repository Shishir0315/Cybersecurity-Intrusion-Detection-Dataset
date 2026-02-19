---
title: Cybersecurity Intrusion Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Autoencoder for Cybersecurity Intrusion Detection

This project implements an end-to-end Autoencoder model designed for cybersecurity data.

## Project Structure
- `cybersecurity_intrusion_data.csv`: The dataset.
- `train_autoencoder.py`: Script to preprocess data and train the model.
- `test_reconstruction.py`: Script to test the model's reconstruction capability.
- `autoencoder_model.h5`: The trained Keras model.
- `scaler.pkl`: Saved MinMaxScaler for feature scaling.
- `label_encoders.pkl`: Saved LabelEncoders for categorical features.

## Model Details
- **Architecture**: Simple bottleneck autoencoder (Input -> 8 -> 4 -> 8 -> Output).
- **Training**: Uses a subset of 2000 records and 20 epochs for fast and efficient training.
- **Application**: Useful for anomaly detection by monitoring reconstruction error.

## How to use
### 1. Local Dashboard (Webpage)
Run the dedicated Flask server to see the premium webpage dashboard:
```powershell
py run_local_webpage.py
```
Then open: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 2. Training (Optional)
Run the training script to retrain the autoencoder:
```powershell
py train_autoencoder.py
```

### 3. Deployment
- **Hugging Face**: Automatically deployed from this repository.
- **GitHub**: Source code is synchronized.
