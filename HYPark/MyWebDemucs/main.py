import os
import shutil
import subprocess
import uuid
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import asyncio
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

# =================================================================
# [설정] 경로 및 환경 변수
# =================================================================
# 모델 파일 경로 (셀1에서 복사한 경로와 일치해야 함)
EQ_MODEL_PATH = os.path.join("model", "model2_mastering.pth")
RESULT_DIR = "static_results"
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================
# [중요] 모델 클래스는 반드시 함수 밖(Global Scope)에 있어야 pickle 에러가 안 남
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

# 전역 모델 변수
ai_engineer = None

@app.on_event("startup")
def load_models():
    global ai_engineer
    print("=== 서버 시작: 모델 로딩 ===")

    # MasteringAI (EQ 예측 모델) 로드
    if os.path.exists(EQ_MODEL_PATH):
        try:
            ai_engineer = MasteringAI().to(device)
            # map_location을 사용하여 GPU/CPU 호환성 확보
            ai_engineer.load_state_dict(torch.load(EQ_MODEL_PATH, map_location=device))
            ai_engineer.eval()
            print("✅ AI EQ 예측 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ EQ 모델 로드 실패: {e}")
    else:
        print(f"⚠️ 경고: {EQ_MODEL_PATH} 파일을 찾을 수 없습니다.")

# =================================================================
# [유틸 함수] 파일명 안전하게 만들기 (werkzeug 대체)
# =================================================================
def secure_filename(filename):
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename

# =================================================================
# [함수] EQ 값만 예측해서 반환
# =================================================================
def predict_eq_values(file_path):
    if ai_engineer is None: return None
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != 22050:
            waveform = T.Resample(sr, 22050, dtype=waveform.dtype)(waveform)

        waveform = torch.mean(waveform, dim=0, keepdim=True)
        duration_samples = 3 * 22050

        if waveform.shape[1] > duration_samples:
            input_tensor = waveform[:, :duration_samples]
        else:
            input_tensor = torch.nn.functional.pad(waveform, (0, duration_samples - waveform.shape[1]))

        with torch.no_grad():
            prediction = ai_engineer(input_tensor.to(device))

        vals = prediction[0].cpu().numpy()
        return {
            "low": round(float(vals[0]), 1),
            "mid": round(float(vals[1]), 1),
            "high": round(float(vals[2]), 1)
        }
    except Exception as e:
        print(f"❌ EQ 예측 실패 ({file_path}): {e}")
        return None

# =================================================================
# [함수] Demucs 실행 (subprocess 사용이 가장 안정적임)
# =================================================================
def run_demucs_separation(input_path, save_folder, track_name):
    # Demucs 라이브러리는 CLI 명령어로 실행하는 것이 의존성 문제 없이 가장 확실함
    command = [
        "demucs",
        "-n", "htdemucs",  # 또는 'htdemucs_ft' (속도/품질에 따라 선택, 기본값 추천)
        "-o", save_folder,
        input_path
    ]

    # GPU가 없으면 CPU 강제 옵션 추가
    if device == "cpu":
        command.extend(["-d", "cpu"])

    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Demucs Error:\n{result.stderr}")
        raise RuntimeError(f"음원 분리 실패: {result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'}")

# =================================================================
# [라우터]
# =================================================================
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), use_ai_eq: bool = Form(True)):
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())

    # 폴더 구조: static_results / task_id / 원본파일
    save_folder = os.path.join(RESULT_DIR, task_id)
    os.makedirs(save_folder, exist_ok=True)
    input_path = os.path.join(save_folder, filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류: {e}")

    track_name_no_ext = os.path.splitext(filename)[0]

    print(f"[{track_name_no_ext}] 음원 분리 작업 시작...")
    try:
        # 비동기 실행
        await asyncio.to_thread(run_demucs_separation, input_path, save_folder, track_name_no_ext)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path) # 원본 삭제하여 용량 확보

    # Demucs 출력 폴더 찾기 (htdemucs 모델 기준)
    # 기본 구조: output_dir / htdemucs / track_name / stems...
    target_dir = os.path.join(save_folder, "htdemucs", track_name_no_ext)

    # 폴더명을 못 찾을 경우(특수문자 등)를 대비해 검색
    if not os.path.exists(target_dir):
        base_model_dir = os.path.join(save_folder, "htdemucs")
        if os.path.exists(base_model_dir):
            subfolders = [f.path for f in os.scandir(base_model_dir) if f.is_dir()]
            if subfolders: target_dir = subfolders[0]

    if not os.path.exists(target_dir):
        raise HTTPException(status_code=500, detail="음원 분리 결과 폴더를 찾을 수 없습니다.")

    stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    eq_results = {}

    # AI EQ 예측
    if use_ai_eq and ai_engineer is not None:
        for stem in stems:
            stem_path = os.path.join(target_dir, stem)
            if os.path.exists(stem_path):
                res = predict_eq_values(stem_path)
                if res:
                    eq_results[stem.replace(".wav", "")] = res

    # URL 반환
    response_data = {"status": "success", "track_name": track_name_no_ext, "urls": {}, "eq_info": eq_results, "task_id": task_id}
    abs_result_dir = os.path.abspath(RESULT_DIR)

    for stem in stems:
        full_path = os.path.join(target_dir, stem)
        if os.path.exists(full_path):
            # 절대 경로를 상대 경로로 변환 후 URL 인코딩
            rel_path = os.path.relpath(full_path, abs_result_dir).replace("\\", "/")
            # URL에 안전하게 인코딩 (공백 등 처리)
            web_path = f"/results/{quote(rel_path)}"
            response_data["urls"][stem.replace(".wav", "")] = web_path

    print(f"[{track_name_no_ext}] 완료.")
    return JSONResponse(content=response_data)