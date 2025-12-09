import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

# Configuration
CSV_PATH = 'data.csv'
MODEL_SAVE_PATH = 'genre_model.h5'      
SCALER_SAVE_PATH = 'genre_scaler.pkl'   
ENCODER_SAVE_PATH = 'genre_encoder.pkl' 

def train_model():
    print("[Info] Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"[Error] {CSV_PATH} not found. Run extract_features.py first.")
        return

    # Data Loading & Preprocessing
    data = pd.read_csv(CSV_PATH)
    
    data = data.drop(['filename'], axis=1)

    genre_list = data.iloc[:, -1]
    
    # Encode string labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    # [DL Preprocessing] Standardization
    # Normalize features to have mean 0 and variance 1 for stable DNN training
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    # Split dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # [DL Architecture] Build DNN Model using Keras Sequential API
    print("[Info] Building Model...")
    model = keras.Sequential([
        # Input Layer & Hidden Layer 1: 512 neurons, ReLU activation
        layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        # Regularization: Dropout to prevent overfitting
        layers.Dropout(0.2),

        # Hidden Layer 2
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),

        # Hidden Layer 3
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        # Hidden Layer 4
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        # Output Layer: Softmax for multi-class classification (10 genres)
        layers.Dense(10, activation='softmax')
    ])

    # [Model Compilation] Optimizer: Adam, Loss: Sparse Categorical Crossentropy
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # [Training Phase] Fit model on training data
    print("[Info] Starting Training...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n[Result] Test accuracy: {test_acc:.4f}") 

    # Save Model Artifacts
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    
    print(f"[Success] Model saved to {MODEL_SAVE_PATH}")
    print(f"[Success] Tools saved to {SCALER_SAVE_PATH}, {ENCODER_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
