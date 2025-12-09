import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

# 파일 경로 설정
CSV_PATH = 'data.csv'
MODEL_SAVE_PATH = 'genre_model.h5'      # 학습된 모델 (뇌)
SCALER_SAVE_PATH = 'genre_scaler.pkl'   # 데이터 규격화 도구
ENCODER_SAVE_PATH = 'genre_encoder.pkl' # 장르 이름표 (0=blues, 1=classical...)

def train_model():
    print("[Info] Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"[Error] {CSV_PATH} not found. Run extract_features.py first.")
        return

    # 1. 데이터 로드 및 전처리
    data = pd.read_csv(CSV_PATH)
    
    # 파일명은 학습에 필요 없으니 제거
    data = data.drop(['filename'], axis=1)

    # 정답(Label) 분리: 'blues', 'jazz' 같은 문자열
    genre_list = data.iloc[:, -1]
    
    # 문자열 라벨을 숫자(0, 1, 2...)로 변환
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    # 입력 데이터(Feature) 분리 및 정규화(Standard Scaling)
    # 데이터 단위를 통일해야 학습이 잘 됩니다.
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    # 학습용(Train)과 검증용(Test) 데이터 나누기 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 딥러닝 모델 설계 (Keras Sequential)
    print("[Info] Building Model...")
    model = keras.Sequential([
        # 입력층 & 은닉층 1
        layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),

        # 은닉층 2
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),

        # 은닉층 3
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        # 은닉층 4
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        # 출력층 (장르 10개 중 하나를 선택하는 확률 출력)
        layers.Dense(10, activation='softmax')
    ])

    # 3. 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. 모델 학습
    print("[Info] Starting Training...")
    # epochs=50: 데이터를 50번 반복해서 공부함
    history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

    # 5. 성능 평가
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n[Result] Test accuracy: {test_acc:.4f}") # 정확도 출력

    # 6. 결과물 저장 (중요!)
    # 이 파일들이 있어야 웹사이트에서 써먹을 수 있습니다.
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    
    print(f"[Success] Model saved to {MODEL_SAVE_PATH}")
    print(f"[Success] Tools saved to {SCALER_SAVE_PATH}, {ENCODER_SAVE_PATH}")

if __name__ == '__main__':
    train_model()