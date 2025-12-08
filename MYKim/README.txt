Download all files provided in github repository
a. Create Folder: Project
b. Create Folder: Model
c. Create Folder: Static

Add index.html → Static
Add model2_mastering.pth → Model

1. Settings

(1) FFmpeg for Mac M1
  brew install ffmpeg

(2) Conda Environment (NO YAML FILES)
  conda create -n demucs python=3.10
  conda activate demucs

(3) Required Libraries (Final working versions)
PyTorch CPU builds for Mac M1

  pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

Web server
  pip install fastapi uvicorn python-multipart aiofiles

Audio + ML dependencies
  pip install librosa soundfile numpy pandas scikit-learn joblib

TensorFlow (only version that works on M1)
  pip install tensorflow==2.13.1

Demucs (CPU mode only)
  pip install demucs

NOTE:
You DO NOT use environment-cuda.yml or environment-cpu.yml.
M1 cannot use CUDA and the YAML breaks your environment.


2. Model Setup

(1) Pre-trained Models

Download checkpoint.th:
https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

Place it in:
Project/model/

NO conda env create -f environment-cuda.yml
NO conda env create -f environment-cpu.yml

(Removed because they break M1.)

(2) Dataset (For Genre Model Training — optional)

Create folder:
data/genres_original

Download GTZAN dataset (1.2GB):
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Unzip into:
data/genres_original/

Extract features:
  python extract_features.py

Train genre model:
  python train_genre.py

Move generated 3 files to /model:
genre_model.h5
genre_scaler.pkl
genre_encoder.pkl


3. Folder Structure (Final)
Project_Root/
│
├── main.py
├── extract_features.py
├── train_genre.py
├── data.csv
│
├── model/
│   ├── checkpoint.th
│   ├── model2_mastering.pth
│   ├── genre_model.h5
│   ├── genre_scaler.pkl
│   └── genre_encoder.pkl
│
├── data/
│   └── genres_original/ ...
│
├── static/
│   └── index.html
│
└── static_results/        # auto-generated at runtime

4. Run
  python main.py


Open browser:
http://127.0.0.1:8000
