---
title: Cybersecurity Intrusion Detection
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 5.16.0
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
1. Run `py train_autoencoder.py` to retrain (optional).
2. Run `py test_reconstruction.py` to see the model in action.
