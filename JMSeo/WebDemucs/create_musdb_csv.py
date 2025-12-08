import os
import csv
import librosa
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

# [설정] MUSDB18 데이터셋이 있는 폴더 경로 (사용자 환경에 맞게 수정 필수)
# 예: D:\download\DL\musdb18hq
MUSDB_ROOT_PATH = r"D:\download\DL\musdb18hq" 

# 출력될 CSV 파일명
OUTPUT_CSV = 'musdb_data.csv'

# 모델 경로
GENRE_MODEL_PATH = os.path.join("model", "genre_model.h5")
SCALER_PATH = os.path.join("model", "genre_scaler.pkl")
ENCODER_PATH = os.path.join("model", "genre_encoder.pkl")

def create_musdb_dataset():
    # 1. 모델 로드
    if not (os.path.exists(GENRE_MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("오류: 모델 파일이 없습니다. train_genre.py를 먼저 실행하세요.")
        return

    print("모델 로드 중...")
    model = tf.keras.models.load_model(GENRE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # 2. CSV 헤더 작성 (학습 데이터와 동일한 구조)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(header)
    file.close()

    print("MUSDB18 데이터 분석 및 장르 예측 시작...")
    
    count = 0
    # 3. 폴더 순회 (train, test 폴더 포함)
    for root, dirs, files in os.walk(MUSDB_ROOT_PATH):
        for filename in files:
            # MUSDB18은 보통 mixture.wav 파일이 원본입니다.
            if filename == "mixture.wav" or filename.endswith(".mp4") or filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                
                # 폴더명을 곡 이름으로 사용 (MUSDB 구조: Artist - Song / mixture.wav)
                song_name = os.path.basename(root) 
                if song_name == "test" or song_name == "train": # 루트 폴더 제외
                    song_name = filename

                try:
                    # (1) 특징 추출
                    y, sr = librosa.load(file_path, mono=True, duration=30)
                    
                    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                    rmse = np.mean(librosa.feature.rms(y=y))
                    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    
                    # 피처 리스트 생성
                    features = [chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr]
                    for e in mfcc:
                        features.append(np.mean(e))

                    # (2) 장르 예측 (Labeling)
                    # 스케일링을 위해 2차원 배열로 변환
                    input_data = np.array(features).reshape(1, -1)
                    input_scaled = scaler.transform(input_data)
                    
                    prediction = model.predict(input_scaled, verbose=0)
                    predicted_index = np.argmax(prediction, axis=1)[0]
                    predicted_genre = encoder.inverse_transform([predicted_index])[0]

                    # (3) CSV 저장
                    row_data = [song_name] + features + [predicted_genre]
                    
                    file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                    file.close()

                    count += 1
                    print(f"[{count}] 처리 완료: {song_name} -> {predicted_genre}")

                except Exception as e:
                    print(f"실패 ({song_name}): {e}")

    print(f"완료. 총 {count}곡의 데이터가 {OUTPUT_CSV}에 저장되었습니다.")

if __name__ == '__main__':
    create_musdb_dataset()