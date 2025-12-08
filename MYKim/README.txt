1. Settings

(1) FFmpeg
음원 디코딩 및 Demucs 처리에 필요합니다.

macOS (Terminal):
  brew install ffmpeg

2) Conda Environment
M1에서 TorchCodec 충돌 및 TensorFlow 호환 문제를 피하기 위해 custom ARM64 환경을 사용합니다.

① 새 환경 생성
  conda create -n demucs python=3.10
  conda activate demucs

