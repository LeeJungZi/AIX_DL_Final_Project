1. Settings

(1) FFmpeg
Windows (PowerShell): winget install Gyan.FFmpeg
macOS (Terminal): brew install ffmpeg
Linux: sudo apt install ffmpeg

(2) Libraries
pip install fastapi uvicorn python-multipart aiofiles
pip install torch torchaudio
pip install tensorflow scikit-learn pandas numpy
pip install librosa soundfile musdb stempeg mir_eval
pip install demucs
pip install seaborn


2. Model Setup

(1) Pre-trained Models
model folder 안에 checkpoint.th downlaoad
https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

conda env create -f environment-cuda.yml
conda env create -f environment-cpu.yml

(2) Dataset (for genre analysis)
프로젝트 루트에 data/genres_original 폴더를 생성

GTZAN 데이터셋 다운로드 링크(Kaggle)에서 데이터를 다운 (1.2GB)
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
압축 풀고 장르별 폴더(blues, classical 등)가 data/genres_original/ 안에 들어가도록 배치

오디오 특징 추출 및 모델 학습
python extract_features.py
python train_genre.py

생성된 아래 3개 파일을 model/ 폴더로 이동
genre_model.h5
genre_scaler.pkl
genre_encoder.pkl

(3) MUSDB Data Preparation (Optional)
MUSDB18 데이터셋 분석을 위한 CSV 생성 (장르 모델 학습 후 실행)
python create_musdb_csv.py


3. Check Final Structure of Folder

Project_Root/
│
├── main.py                 # 메인 실행 파일
├── extract_features.py     # 장르 특징 추출 코드
├── train_genre.py          # 장르 모델 학습 코드
├── create_musdb_csv.py     # MUSDB 데이터셋 분석 코드
├── analyze_visualization.py # 분석 결과 시각화 코드
├── viz_test_pca.py         # PCA 분석 시각화 코드
├── data.csv                # GTZAN 특징 데이터
├── musdb_data.csv          # (생성됨) MUSDB 특징 데이터
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

(1) Web Service
python main.py
브라우저 주소창에 http://127.0.0.1:8000 입력하여 접속

(2) Visualization Tools
장르 분포 및 특성 차트 생성 (musdb_data.csv 필요):
python analyze_visualization.py

GTZAN Test Set PCA 산점도 생성:
python viz_test_pca.py
