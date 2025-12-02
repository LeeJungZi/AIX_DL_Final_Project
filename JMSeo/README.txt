1. Settings

(1) FFmpeg
Windows (PowerShell): winget install Gyan.FFmpeg
macOS (Terminal): brew install ffmpeg
Linux: sudo apt install ffmpeg

(2) Libraries
pip install fastapi uvicorn python-multipart
pip install torch torchaudio
pip install tensorflow scikit-learn pandas numpy
pip install librosa soundfile musdb stempeg mir_eval
pip install demucs


2. Model Setup

(1) Pre-trained Models
model folder 안에 checkpoint.th downlaoad
https://drive.google.com/file/d/1Vl-ho7_D4SKmqaCp8b8I7XisicKblUPA/view?usp=sharing

conda env create -f environment-cuda.yml
conda env create -f environment-cpu.yml
