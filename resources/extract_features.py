# extract_features.py
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# [Configuration]
DATASET_PATH = './data/genres_original' # Folder containing genre subfolders
OUTPUT_CSV = 'data.csv'

# Header for CSV file
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

# Create CSV file and write header
file = open(OUTPUT_CSV, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

print("[Info] Starting feature extraction...")

for g in genres:
    genre_path = os.path.join(DATASET_PATH, g)
    if not os.path.exists(genre_path):
        print(f"[Warning] Folder not found: {genre_path}")
        continue
        
    for filename in os.listdir(genre_path):
        songname = os.path.join(genre_path, filename)
        
        try:
            y, sr = librosa.load(songname, mono=True, duration=30)
            
            # Extract features (Based on your screenshot)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            
            # Prepare row data (mean values)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            
            # Write to CSV
            file = open(OUTPUT_CSV, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

print(f"[Success] Feature extraction complete. Saved to {OUTPUT_CSV}")