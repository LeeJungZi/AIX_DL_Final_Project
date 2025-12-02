import os
import shutil
import subprocess
import uuid
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote

# [Added libraries for Genre Classification]
import numpy as np
import librosa
import joblib
import tensorflow as tf
from pydantic import BaseModel

# =================================================================
# Configuration & Paths
# =================================================================
DEMUCS_MODEL_PATH = os.path.join("model", "checkpoint.th")
EQ_MODEL_PATH = os.path.join("model", "model2_mastering.pth")

# Paths for Genre Model
GENRE_MODEL_PATH = os.path.join("model", "genre_model.h5")
SCALER_PATH = os.path.join("model", "genre_scaler.pkl")
ENCODER_PATH = os.path.join("model", "genre_encoder.pkl")

RESULT_DIR = "static_results"
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# =================================================================
# [Model 1] MasteringAI (EQ Recommendation)
# =================================================================
class MasteringAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_spec = T.MelSpectrogram(sample_rate=22050, n_mels=64)
        self.to_db = T.AmplitudeToDB()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 3)
        )
    def forward(self, x):
        spec = self.to_db(self.to_spec(x))
        return self.head(self.cnn(spec.unsqueeze(1)))

# Load MasteringAI Model
ai_engineer = None
if os.path.exists(EQ_MODEL_PATH):
    try:
        ai_engineer = MasteringAI().to(device)
        ai_engineer.load_state_dict(torch.load(EQ_MODEL_PATH, map_location=device))
        ai_engineer.eval()
        print(f"✅ MasteringAI Model Loaded: {EQ_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load MasteringAI: {e}")
else:
    print(f"⚠️ MasteringAI model not found at {EQ_MODEL_PATH}")

# =================================================================
# [Model 2] Genre Classifier (TensorFlow/Keras)
# =================================================================
genre_model = None
genre_scaler = None
genre_encoder = None

# Load Genre Model Artifacts
if os.path.exists(GENRE_MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        genre_model = tf.keras.models.load_model(GENRE_MODEL_PATH)
        genre_scaler = joblib.load(SCALER_PATH)
        genre_encoder = joblib.load(ENCODER_PATH)
        print(f"✅ Genre Classifier Loaded: {GENRE_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load Genre Classifier: {e}")
else:
    print("⚠️ Genre model files missing. Genre detection disabled.")

# =================================================================
# Helper Functions
# =================================================================

# 오디오 특성 추출 및 정규화 함수
def extract_audio_features_for_chart(file_path):
    try:
        # 1. 로드 (앞부분 60초만 빠르게 분석)
        y, sr = librosa.load(file_path, mono=True, duration=60)
        
        # 2. 특성 추출
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        # 3. 시각화를 위한 단순 정규화 (대략적인 0~100 스케일로 변환)
        norm_features = {
            "BPM (Tempo)": min(tempo[0] if isinstance(tempo, np.ndarray) else tempo, 200) / 2, 
            "Brightness (Centroid)": min(cent, 5000) / 50, 
            "Sharpness (Rolloff)": min(rolloff, 10000) / 100, 
            "Noisiness (ZCR)": min(zcr * 500, 100), 
            "Energy (RMS)": min(rms * 400, 100) 
        }
        
        return {k: round(v, 1) for k, v in norm_features.items()}

    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None

def predict_eq_values(file_path):
    """Predicts EQ settings (Low, Mid, High) for a given audio file."""
    if ai_engineer is None: return None
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != 22050: waveform = T.Resample(sr, 22050)(waveform)
        
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        duration = 3 * 22050
        if waveform.shape[1] > duration: input_tensor = waveform[:, :duration]
        else: input_tensor = torch.nn.functional.pad(waveform, (0, duration - waveform.shape[1]))
        
        with torch.no_grad():
            prediction = ai_engineer(input_tensor.to(device))
        
        vals = prediction[0].cpu().numpy()
        return {
            "low": round(float(vals[0]), 1), 
            "mid": round(float(vals[1]), 1), 
            "high": round(float(vals[2]), 1)
        }
    except Exception as e:
        print(f"❌ EQ Prediction Failed: {e}")
        return None

def predict_genre(file_path):
    if not genre_model or not genre_scaler: return "Unknown"
    
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        features = [
            np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), 
            np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
        ]
        for e in mfcc:
            features.append(np.mean(e))
            
        input_data = np.array(features).reshape(1, -1)
        input_scaled = genre_scaler.transform(input_data)
        
        prediction = genre_model.predict(input_scaled)
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        predicted_genre = genre_encoder.inverse_transform([predicted_index])[0]
        return predicted_genre.upper()
        
    except Exception as e:
        print(f"❌ Genre Prediction Failed: {e}")
        return "Error"

# =================================================================
# Routes
# =================================================================
@app.get("/")
def read_root(): return FileResponse('static/index.html')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), use_ai_eq: bool = Form(True)):
    # Save Uploaded File
    task_id = str(uuid.uuid4())
    save_folder = os.path.join(RESULT_DIR, task_id) 
    os.makedirs(save_folder, exist_ok=True)
    input_path = os.path.join(save_folder, file.filename)
    
    with open(input_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

    track_name = os.path.splitext(file.filename)[0]

    # 0. 오디오 피처 분석 (원본 파일 대상)
    print("⏳ Analyzing Audio Features...")
    chart_features = extract_audio_features_for_chart(input_path)

    # 1. Predict Genre
    print("⏳ Analyzing Genre...")
    detected_genre = predict_genre(input_path)
    print(f"🎵 Detected Genre: {detected_genre}")

    # 2. Run Demucs
    print("⏳ Running Demucs...")
    # [수정] 옵션을 제거하고 기본 모델(htdemucs) 사용 (환경에 따라 tasnet 옵션 추가 가능)
    cmd = [
        "python", "-m", "demucs.separate", 
        "-o", save_folder,
        "-j", "4",
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print("❌ Demucs Failed:", result.stderr)
    else:
        print("✅ Demucs Succeeded")

    # Locate Demucs Output
    target_dir = os.path.join(save_folder, "htdemucs", track_name)
    if not os.path.exists(target_dir):
        for entry in os.scandir(save_folder):
            if entry.is_dir():
                sub_entries = [f.path for f in os.scandir(entry.path) if f.is_dir()]
                if sub_entries: target_dir = sub_entries[0]

    stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    eq_results = {}
    
    # 3. Predict EQ Values
    if use_ai_eq and ai_engineer is not None:
        for stem in stems:
            stem_path = os.path.join(target_dir, stem)
            if os.path.exists(stem_path):
                res = predict_eq_values(stem_path)
                if res: eq_results[stem.replace(".wav", "")] = res

    # 4. Construct Response (중복 제거 및 통합)
    response_data = {
        "status": "success", 
        "track_name": track_name,
        "genre": detected_genre,
        "features": chart_features, # 분석된 피처 데이터 포함
        "urls": {}, 
        "eq_info": eq_results
    }
    
    abs_result_dir = os.path.abspath(RESULT_DIR)
    for root, dirs, files in os.walk(save_folder):
        for stem in stems:
            if stem in files:
                full_path = os.path.join(root, stem)
                rel_path = os.path.relpath(full_path, abs_result_dir).replace("\\", "/")
                web_path = quote(f"/results/{rel_path}").replace("%2Fresults%2F", "/results/")
                response_data["urls"][stem.replace(".wav", "")] = web_path

    return JSONResponse(content=response_data)

# --- [Function] Mix and Download Endpoint ---
class MixRequest(BaseModel):
    task_id: str
    track_name: str
    gains: dict

@app.post("/mix_download")
async def mix_and_download(req: MixRequest):
    try:
        base_dir = os.path.join(RESULT_DIR, req.task_id)
        
        target_dir = None
        for root, dirs, files in os.walk(base_dir):
            if "vocals.wav" in files:
                target_dir = root
                break
        
        if not target_dir: return {"status": "error", "message": "Files not found"}

        stems = ["vocals", "drums", "bass", "other"]
        mixed_signal = None
        sr = 44100
        output_dir = os.path.join(base_dir, "mixed_output")
        os.makedirs(output_dir, exist_ok=True)
        result_urls = {}

        for stem in stems:
            filepath = os.path.join(target_dir, f"{stem}.wav")
            waveform, sr = torchaudio.load(filepath)
            
            # Apply Gain
            vol = req.gains.get(stem, 1.0)
            adjusted = waveform * vol
            
            # Save Individual Edited Stem
            out_name = f"{stem}_edited.wav"
            out_path = os.path.join(output_dir, out_name)
            torchaudio.save(out_path, adjusted, sr)
            
            rel_path = os.path.relpath(out_path, os.path.abspath(RESULT_DIR)).replace("\\", "/")
            result_urls[f"{stem}_url"] = quote(f"/results/{rel_path}").replace("%2Fresults%2F", "/results/")

            # Mix
            if mixed_signal is None: mixed_signal = adjusted
            else:
                if mixed_signal.shape[1] > adjusted.shape[1]: mixed_signal[:, :adjusted.shape[1]] += adjusted
                else: mixed_signal += adjusted[:, :mixed_signal.shape[1]]

        # Save Full Mix
        mix_path = os.path.join(output_dir, "full_mix.wav")
        max_val = torch.max(torch.abs(mixed_signal))
        if max_val > 1.0: mixed_signal = mixed_signal / max_val
            
        torchaudio.save(mix_path, mixed_signal, sr)
        
        rel_mix = os.path.relpath(mix_path, os.path.abspath(RESULT_DIR)).replace("\\", "/")
        result_urls["mix_url"] = quote(f"/results/{rel_mix}").replace("%2Fresults%2F", "/results/")

        return {"status": "success", "urls": result_urls}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Listen on 127.0.0.1 for local development
    uvicorn.run(app, host="127.0.0.1", port=8000)