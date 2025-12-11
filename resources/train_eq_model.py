import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# [설정] 데이터 경로를 본인 환경에 맞게 수정하세요!
# ==========================================
MUSDB_PATH = r".\musdb18hq"  # WAV 파일이 있는 폴더
SAMPLE_RATE = 22050
DURATION = 3  # 학습할 길이 (3초)
BATCH_SIZE = 16
EPOCHS = 20

# ---------------------------------------------------------
# 1. AI 모델 정의 (가벼운 CNN 구조)
# ---------------------------------------------------------
class MasteringAI(nn.Module):
    def __init__(self):
        super().__init__()
        # 소리를 그림(Mel-Spectrogram)으로 변환
        self.to_spec = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
        self.to_db = T.AmplitudeToDB()
        
        # 그림을 보고 특징을 찾는 눈 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)) # 어떤 길이의 오디오가 와도 크기 고정
        )
        
        # 최종 판단 (3개의 EQ 값 예측)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # [Low, Mid, High]
        )

    def forward(self, x):
        spec = self.to_db(self.to_spec(x))
        return self.head(self.cnn(spec.unsqueeze(1)))

# ---------------------------------------------------------
# 2. 데이터셋 정의 (핵심: 소리를 고의로 망가뜨리는 로직)
# ---------------------------------------------------------
class CorruptedAudioDataset(Dataset):
    def __init__(self, root_dir):
        # musdb 폴더 내의 모든 wav 파일을 찾습니다 (train/test 구분 없이 다 씀)
        self.files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        # 너무 짧은 파일이나 이상한 파일 제외
        self.files = [f for f in self.files if "mixture" not in f] # mixture는 제외하고 개별 스템만 사용

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. 파일 로드
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)
        
        # 2. 샘플링 레이트 맞추기
        if sr != SAMPLE_RATE:
            waveform = F.resample(waveform, sr, SAMPLE_RATE)
        
        # 3. 모노로 변환 및 길이 자르기 (3초)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        num_samples = SAMPLE_RATE * DURATION
        
        if waveform.shape[1] > num_samples:
            start = random.randint(0, waveform.shape[1] - num_samples)
            waveform = waveform[:, start:start+num_samples]
        else:
            # 짧으면 패딩
            waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.shape[1]))

        # 4. [핵심] 랜덤 EQ 적용 (소리 망가뜨리기)
        # Low Shelf (100Hz), Peaking (1000Hz), High Shelf (10000Hz)
        low_gain = random.uniform(-10.0, 10.0)
        mid_gain = random.uniform(-10.0, 10.0)
        high_gain = random.uniform(-10.0, 10.0)

        # Torchaudio 필터 적용
        # (주의: 실제 학습 속도를 위해 여기선 단순화했습니다. 실제론 GPU에서 돌리는게 빠릅니다)
        augmented = F.lowpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=100, Q=0.707) # 단순화를 위해 필터 하나만 예시로 적용하거나
        # 여기서는 "망가진 오디오"를 흉내내기 위해 EQ값을 적용한 파형을 만듭니다.
        # *실제 구현 팁*: 파이썬 루프에서 필터를 거는건 느리므로, 
        # 학습때는 "망가뜨렸다 치고" 정답(Target)만 반대로 주는 방식을 쓰기도 합니다.
        # 하지만 정확성을 위해 실제로 필터를 겁니다.
        
        corrupted = F.equalizer_biquad(waveform, SAMPLE_RATE, center_freq=100, gain=low_gain, Q=0.707)
        corrupted = F.equalizer_biquad(corrupted, SAMPLE_RATE, center_freq=1000, gain=mid_gain, Q=0.707)
        corrupted = F.equalizer_biquad(corrupted, SAMPLE_RATE, center_freq=10000, gain=high_gain, Q=0.707)

        # 5. 정답 라벨 생성 (복구하려면 반대로 해야 함)
        # 예: 저음을 5dB 올렸으면, 복구하려면 -5dB 해야 함
        target = torch.tensor([-low_gain, -mid_gain, -high_gain], dtype=torch.float32)

        return corrupted.squeeze(), target

# ---------------------------------------------------------
# 3. 학습 루프
# ---------------------------------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 학습 장치: {device}")

    # 데이터셋 준비
    dataset = CorruptedAudioDataset(MUSDB_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 윈도우라 workers=0
    
    print(f"🎵 데이터 개수: {len(dataset)}개")

    model = MasteringAI().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # 정답 숫자와 예측 숫자의 차이 계산

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (audio, target) in enumerate(dataloader):
            audio, target = audio.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(audio) # AI의 예측값
            loss = criterion(output, target) # 정답(복구값)과의 차이
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} 완료! 평균 Loss: {total_loss / len(dataloader):.4f}")
        
        # 모델 저장
        torch.save(model.state_dict(), "model2_mastering.pth")

if __name__ == "__main__":
    train()