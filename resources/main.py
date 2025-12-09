import os
os.environ["TORCHAUDIO_USE_CODEC"] = "False"
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

# Genre model libraries
import numpy as np
import librosa
import joblib
import tensorflow as tf
from pydantic import BaseModel

# ================================================================
# Paths
# ================================================================
DEMUCS_MODEL = "htdemucs"
RESULT_DIR = "static_results"
MODEL_DIR = "model"

os.makedirs(RESULT_DIR, exist_ok=True)

GENRE_MODEL_PATH = os.path.join(MODEL_DIR, "genre_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "genre_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "genre_encoder.pkl")
EQ_MODEL_PATH = os.path.join(MODEL_DIR, "model2_mastering.pth")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

device = "cpu"
print("Running device:", device)

# ================================================================
# EQ MODEL
# ================================================================
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

ai_engineer = None
try:
    ai_engineer = MasteringAI().to(device)
    ai_engineer.load_state_dict(torch.load(EQ_MODEL_PATH, map_location=device))
    ai_engineer.eval()
    print("Loaded EQ model.")
except Exception as e:
    print("EQ model load failed:", e)

# ================================================================
# GENRE MODEL
# ================================================================
genre_model = None
genre_scaler = None
genre_encoder = None

try:
    genre_model = tf.keras.models.load_model(GENRE_MODEL_PATH)
    genre_scaler = joblib.load(SCALER_PATH)
    genre_encoder = joblib.load(ENCODER_PATH)
    print("Genre model loaded.")
except Exception as e:
    print("Genre model failed:", e)

# ================================================================
# Helper Functions
# ================================================================

def extract_audio_features(path):
    try:
        y, sr = librosa.load(path, mono=True, duration=60)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        features = {
            "BPM": round(float(tempo), 1),
            "Centroid": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 1),
            "Rolloff": round(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 1),
            "ZCR": round(float(np.mean(librosa.feature.zero_crossing_rate(y))), 3),
            "RMS": round(float(np.mean(librosa.feature.rms(y=y))), 3),
        }
        return features
    except Exception as e:
        print("Feature extraction error:", e)
        return None


def predict_genre(path):
    try:
        y, sr = librosa.load(path, mono=True, duration=30)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        features = [
            np.mean(chroma),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y))
        ]

        for m in mfcc:
            features.append(np.mean(m))

        arr = np.array(features).reshape(1, -1)
        arr = genre_scaler.transform(arr)
        pred = genre_model.predict(arr)
        idx = np.argmax(pred)
        return genre_encoder.inverse_transform([idx])[0].upper()

    except Exception as e:
        print("Genre error:", e)
        return "UNKNOWN"


def predict_eq(path):
    try:
        waveform, sr = torchaudio.load(path)

        if sr != 22050:
            waveform = T.Resample(sr, 22050)(waveform)

        waveform = torch.mean(waveform, dim=0, keepdim=True)

        D = 3 * 22050
        if waveform.shape[1] > D:
            waveform = waveform[:, :D]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, D - waveform.shape[1]))

        with torch.no_grad():
            out = ai_engineer(waveform.to(device))

        out = out[0].cpu().numpy()

        return {
            "low": round(float(out[0]), 2),
            "mid": round(float(out[1]), 2),
            "high": round(float(out[2]), 2),
        }
    except:
        return None


def find_demucs_output(folder):
    for root, dirs, files in os.walk(folder):
        if "vocals.wav" in files:
            return root
    return None


# ================================================================
# ROUTES
# ================================================================
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...), use_ai_eq: bool = Form(True)):

    task_id = str(uuid.uuid4())
    folder = os.path.join(RESULT_DIR, task_id)
    os.makedirs(folder, exist_ok=True)

    input_path = os.path.join(folder, file.filename)

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    track = os.path.splitext(file.filename)[0]

    # Features + Genre
    features = extract_audio_features(input_path)
    genre = predict_genre(input_path)

    # Run Demucs without TorchCodec
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", DEMUCS_MODEL,
        "-d", "cpu",
        "-j", "2",
        "-o", folder,
        input_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ, "TORCHAUDIO_USE_CODEC": "False"})

    target = find_demucs_output(folder)

    if target is None:
        return JSONResponse({"status": "error", "message": "Demucs output not found"})

    stems = ["vocals", "drums", "bass", "other"]
    urls = {}
    eq_info = {}

    for s in stems:
        p = os.path.join(target, f"{s}.wav")
        if os.path.exists(p):
            rel = os.path.relpath(p, RESULT_DIR).replace("\\", "/")
            urls[s] = f"/results/{quote(rel)}"

            if use_ai_eq:
                eq_info[s] = predict_eq(p)

    return JSONResponse({
        "status": "success",
        "track": track,
        "genre": genre,
        "features": features,
        "urls": urls,
        "eq": eq_info
    })


class MixRequest(BaseModel):
    task_id: str
    track: str
    gains: dict


@app.post("/mix_download")
async def mix(req: MixRequest):
    base = os.path.join(RESULT_DIR, req.task_id)

    target = find_demucs_output(base)
    if target is None:
        return {"status": "error", "message": "Stems not found"}

    stems = ["vocals", "drums", "bass", "other"]
    mix = None
    sr = 44100

    out_dir = os.path.join(base, "mixed")
    os.makedirs(out_dir, exist_ok=True)

    urls = {}

    for s in stems:
        path = os.path.join(target, f"{s}.wav")
        if not os.path.exists(path):
            continue

        wav, sr = torchaudio.load(path)
        gain = req.gains.get(s, 1.0)
        wav = wav * gain

        out_path = os.path.join(out_dir, f"{s}_edited.wav")
        torchaudio.save(out_path, wav, sr)

        rel = os.path.relpath(out_path, RESULT_DIR).replace("\\", "/")
        urls[f"{s}_url"] = f"/results/{rel}"

        if mix is None:
            mix = wav
        else:
            L = min(mix.shape[1], wav.shape[1])
            mix[:, :L] += wav[:, :L]

    final = os.path.join(out_dir, "final_mix.wav")
    torchaudio.save(final, mix, sr)

    rel = os.path.relpath(final, RESULT_DIR).replace("\\", "/")
    urls["mix"] = f"/results/{rel}"

    return {"status": "success", "urls": urls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
