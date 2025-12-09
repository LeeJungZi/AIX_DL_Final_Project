import os
import csv
import librosa
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

# Path Configuration
# Assumes MUSDB18-HQ is downloaded and located here.
MUSDB_ROOT_PATH = r"D:\download\DL\musdb18hq" 

OUTPUT_CSV = 'musdb_data.csv'

# Model Paths
GENRE_MODEL_PATH = os.path.join("model", "genre_model.h5")
SCALER_PATH = os.path.join("model", "genre_scaler.pkl")
ENCODER_PATH = os.path.join("model", "genre_encoder.pkl")

def create_musdb_dataset():
    # Load Model Artifacts
    if not (os.path.exists(GENRE_MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("Error: Model files not found.")
        return

    print("Loading model...")
    # [DL Model Loading] Load pre-trained DNN model
    model = tf.keras.models.load_model(GENRE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # CSV Header setup
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(header)
    file.close()

    print("Starting MUSDB18 analysis and genre prediction...")
    
    count = 0
    # Walk through directory
    for root, dirs, files in os.walk(MUSDB_ROOT_PATH):
        for filename in files:
            if filename == "mixture.wav" or filename.endswith(".mp4") or filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                
                song_name = os.path.basename(root) 
                if song_name == "test" or song_name == "train": 
                    song_name = filename

                try:
                    # (1) Feature Extraction
                    y, sr = librosa.load(file_path, mono=True, duration=30)
                    
                    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                    rmse = np.mean(librosa.feature.rms(y=y))
                    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    
                    features = [chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr]
                    for e in mfcc:
                        features.append(np.mean(e))

                    # (2) Genre Prediction (Pseudo-labeling)
                    # [DL Inference] Reshape and Scale input vector
                    input_data = np.array(features).reshape(1, -1)
                    input_scaled = scaler.transform(input_data)
                    
                    # Predict class probability and decode label
                    prediction = model.predict(input_scaled, verbose=0)
                    predicted_index = np.argmax(prediction, axis=1)[0]
                    predicted_genre = encoder.inverse_transform([predicted_index])[0]

                    # (3) Save to CSV
                    row_data = [song_name] + features + [predicted_genre]
                    
                    file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                    file.close()

                    count += 1
                    print(f"[{count}] Processed: {song_name} -> {predicted_genre}")

                except Exception as e:
                    print(f"Failed ({song_name}): {e}")

    print(f"Complete. Saved {count} tracks to {OUTPUT_CSV}.")

if __name__ == '__main__':
    create_musdb_dataset()
