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

# =================================================================
# [설정] 경로
# =================================================================
DEMUCS_MODEL_PATH = os.path.join("model", "checkpoint.th") 
EQ_MODEL_PATH = os.path.join("model", "model2_mastering.pth")
RESULT_DIR = "static_results"
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================
# [모델] MasteringAI 정의
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

ai_engineer = None
if os.path.exists(EQ_MODEL_PATH):
    try:
        ai_engineer = MasteringAI().to(device)
        ai_engineer.load_state_dict(torch.load(EQ_MODEL_PATH, map_location=device))
        ai_engineer.eval()
        print("✅ AI 모델 로드 완료")
    except:
        print("⚠️ 모델 로드 실패")

# =================================================================
# [함수] EQ 값만 예측해서 반환 (FFmpeg 적용 안 함!)
# =================================================================
def predict_eq_values(file_path):
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
        # [요청하신 기능 1] 소수점 첫째 자리 반올림
        return {
            "low": round(float(vals[0]), 1), 
            "mid": round(float(vals[1]), 1), 
            "high": round(float(vals[2]), 1)
        }
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        return None

# =================================================================
# [라우터]
# =================================================================
@app.get("/")
def read_root(): return FileResponse('static/index.html')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), use_ai_eq: bool = Form(True)):
    task_id = str(uuid.uuid4())
    save_folder = os.path.join(RESULT_DIR, task_id) 
    os.makedirs(save_folder, exist_ok=True)
    input_path = os.path.join(save_folder, file.filename)
    
    with open(input_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

    # Demucs 실행
    subprocess.run(["python", "-m", "demucs.separate", "-n", "tasnet", "--model", DEMUCS_MODEL_PATH, "-o", save_folder, input_path], capture_output=True)

    track_name = os.path.splitext(file.filename)[0]
    target_dir = os.path.join(save_folder, "tasnet", track_name)
    if not os.path.exists(target_dir):
        subfolders = [f.path for f in os.scandir(os.path.join(save_folder, "tasnet")) if f.is_dir()]
        if subfolders: target_dir = subfolders[0]

    stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    eq_results = {}
    
    # AI EQ 값 예측 (파일 변환 X, 값만 추출 O)
    if use_ai_eq and ai_engineer is not None:
        for stem in stems:
            stem_path = os.path.join(target_dir, stem)
            if os.path.exists(stem_path):
                res = predict_eq_values(stem_path)
                if res: eq_results[stem.replace(".wav", "")] = res

    # URL 생성
    response_data = {"status": "success", "track_name": track_name, "urls": {}, "eq_info": eq_results}
    abs_result_dir = os.path.abspath(RESULT_DIR)
    
    for root, dirs, files in os.walk(save_folder):
        for stem in stems:
            if stem in files:
                full_path = os.path.join(root, stem)
                rel_path = os.path.relpath(full_path, abs_result_dir).replace("\\", "/")
                web_path = quote(f"/results/{rel_path}").replace("%2Fresults%2F", "/results/")
                response_data["urls"][stem.replace(".wav", "")] = web_path

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)