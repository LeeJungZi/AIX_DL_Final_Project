# Windows

## 1. 파일 준비

(1) 전체 repository downdload 후 압축 풀기

GTZAN dataset (1.41 GB) download:

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

-> 압축 해제 후 Data/를 resources/로 이동

<br/>

(2) 모델 download

MUSDB18-HQ의 큰 용량과 긴 학습시간으로 인해 Source Separation model과 Auto Equalizer model은 저희가 직접 학습시킨 pre-trained model을 제공합니다.

Source Separation model (checkpoint.th):

https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

-> resources/model에 저장

Auto Equalizer model (model2_mastering.pth): resources/model/에 저장됨

	.../resources/
		│
		├── main.py
		├── extract_features.py
		├── train_genre.py
		├── environment-cpu.yml
		│
		├── model/
		│   ├── checkpoint.th
		│   └── model2_mastering.pth
		│
		├── Data/
		│   └── genres_original/ ...
		│
		└── static/
		    └── index.html

<br/>

## 2. env setting

```bash
cd “resources PATH”
```
```bash
conda env create -f environment-cpu.yml
```
```bash
conda activate demucs
```

<br/>

## 3. Genre model training

(1) Extract features
```bash
cd “resources PATH”
```
```bash
python extract_features.py
```

[Error] Failed to process jazz.00054.wav: 학습에 문제되지 않음

(2) Train genre model
```bash
python train_genre.py
```

(3) Move models to model/

genre_model.h5

genre_scaler.pkl

genre_encoder.pkl

-> resources/model/

	.../resources/
		│
		├── main.py
		├── extract_features.py
		├── train_genre.py
		├── environment-cpu.yml
		├── data.csv
		│
		├── model/
		│   ├── checkpoint.th
		│   ├── model2_mastering.pth
		│   ├── genre_model.h5
		│   ├── genre_scaler.pkl
		│   └── genre_encoder.pkl
		│
		├── Data/
		│   └── genres_original/ ...
		│
		├── static/
		│   └── index.html
		│
		└── static_results/        # auto-generated at runtime

<br/>

## 4. Run

```bash
python main.py
```

Open browser:

http://127.0.0.1:8000


(한 곡 분석에 약 5분 소요)

<br/>
<br/>

# Mac

## Download all files provided in github repository

a. Create Folder: Project

b. Create Folder: Model

c. Create Folder: Static

<br/>

Add index.html → Static

Add model2_mastering.pth → Model

## 1. Settings

(1) FFmpeg for Mac M1
```bash
brew install ffmpeg
```
(2) Conda Environment (NO YAML FILES)
```bash
conda create -n demucs python=3.10
```
```bash
conda activate demucs
```
(3) Required Libraries (Final working versions)

PyTorch CPU builds for Mac M1
```bash
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```
Demucs (CPU mode only)
```bash
pip install demucs
```
Web server
```bash
pip install fastapi==0.109.2 uvicorn==0.29.0 python-multipart aiofiles
```
```bash
pip install pydantic==2.6.0 pydantic-core==2.16.3
```
```bash
pip install typing_extensions==4.15.0
```

Audio + ML dependencies
```bash
pip install librosa==0.10.1
```
```bash
pip install soundfile numpy pandas scikit-learn joblib
```

<br/>

NOTE:
You DO NOT use environment-cuda.yml or environment-cpu.yml.

M1 cannot use CUDA and the YAML breaks your environment.


## 2. Model Setup

(1) Pre-trained Models

Download checkpoint.th:

https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

Place it in:

Project/model/

<br/>

NO conda env create -f environment-cuda.yml

NO conda env create -f environment-cpu.yml

(Removed because they break M1.)

(2) Dataset (For Genre Model Training — optional)

Create folder:

data/genres_original

Download GTZAN dataset (1.41 GB):

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Unzip into:

data/genres_original/

Extract features:
```bash
python extract_features.py
```

Train genre model:
```bash
python train_genre.py
```

Move generated 3 files to /model:
genre_model.h5
genre_scaler.pkl
genre_encoder.pkl


## 3. Folder Structure (Final)
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

## 4. Run
```bash
python main.py
```

Open browser:

http://127.0.0.1:8000







