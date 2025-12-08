1. Settings

(1) FFmpeg
Windows (PowerShell): winget install Gyan.FFmpeg
macOS (Terminal): brew install ffmpeg
Linux: sudo apt install ffmpeg

(2) Conda Environment
  conda create -n demucs python=3.10
  conda activate demucs

(2) Libraries  
  pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
  pip install fastapi uvicorn python-multipart aiofiles
  pip install librosa soundfile numpy pandas scikit-learn joblib
  pip install tensorflow==2.13.1
  pip install demucs


2. Model Setup

(1) Pre-trained Models
model folder 안에 checkpoint.th download
https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

then, in terminal:
  conda env create -f environment-cuda.yml
  conda env create -f environment-cpu.yml

(2) Dataset (for genre analysis)
프로젝트 루트에 data/genres_original 폴더를 생성

GTZAN 데이터셋 다운로드 링크(Kaggle)에서 데이터를 다운 (1.2GB)
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

download as zip file and extract to data folder

오디오 특징 추출 및 모델 학습
run in terminal:
  python extract_features.py
  python train_genre.py

생성된 아래 3개 파일을 model/ 폴더로 이동
genre_model.h5
genre_scaler.pkl
genre_encoder.pkl


3. Check Final Structure of Folder

Project_Root/
│
├── main.py                 # 메인 실행 파일
├── extract_features.py     # 장르 특징 추출 코드
├── train_genre.py          # 장르 모델 학습 코드
├── data.csv     # 장르 분석 보조 데이터
│
├── model/                  # [필수] 모델 파일 5개가 여기 있어야 함
│   ├── checkpoint.th
│   ├── model2_mastering.pth
│   ├── genre_model.h5
│   ├── genre_scaler.pkl
│   └── genre_encoder.pkl
│
├── data/                   # 장르 학습용 데이터
│   └── genres_original/ ...
│
├── static/                 # 웹페이지 파일
│   └── index.html
│
└── static_results/         # (자동 생성됨) 결과물 저장소


4. Run

python main.py

브라우저 주소창에 http://127.0.0.1:8000 입력하여 접속
