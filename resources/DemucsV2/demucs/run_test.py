import torch
import torch.nn as nn
from pathlib import Path
import musdb
import museval
from dataclasses import dataclass, field

from .tasnet import ConvTasNet
from .test import evaluate

@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None

def run_evaluation():
    # -------------------------------------------------
    # 1. 설정 (본인 환경에 맞게 수정!)
    # -------------------------------------------------
    MUSDB_PATH = r"C:\Users\jwlee\AIXDL\00_Demucs2\musdb18hq"  # MUSDB 데이터셋 경로
    CHECKPOINT_PATH = r"C:\Users\jwlee\AIXDL\00_Demucs2\\checkpoints\\checkpoint.th"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------------------------------
    # 2. 모델 뼈대 만들기 (학습할 때와 똑같은 옵션이어야 함!)
    # -------------------------------------------------
    # N, L, B, H, P, X, R 등의 숫자가 학습 코드와 다르면 에러가 납니다.
    print("🏗️ 모델 생성 중...")
    model = ConvTasNet(
    sources=["drums", "bass", "other", "vocals"], 
    N=256, L=20, B=256, H=512, P=3, X=8, R=4,
    norm_type="gLN", causal=False, mask_nonlinear='relu'
    ).to(DEVICE)

    # -------------------------------------------------
    # 3. 가중치(Checkpoint) 로드하기 (최종 수정됨)
    # -------------------------------------------------
    print(f"💾 가중치 로드 중: {CHECKPOINT_PATH}")
    
    # 1) 일단 로드
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = None

    # 2) SavedState 객체 내부 탐색
    if isinstance(checkpoint, SavedState):
        print("ℹ️ 체크포인트 구조: SavedState 객체")
        
        # [핵심 수정] best_state가 있다면 그걸 우선적으로 가져옵니다.
        if hasattr(checkpoint, "best_state"):
            print("🌟 'best_state' (최고 성능 모델)를 찾았습니다!")
            state_dict = checkpoint.best_state
        elif hasattr(checkpoint, "last_state"):
            print("⚠️ 'best_state'가 없어 'last_state' (마지막 모델)를 사용합니다.")
            state_dict = checkpoint.last_state
        else:
            # 혹시 몰라 기존 로직 유지
            if hasattr(checkpoint, "model"):
                 state_dict = checkpoint.model.state_dict()
    
    elif isinstance(checkpoint, dict):
        # 딕셔너리인 경우
        if 'best_state' in checkpoint:
            state_dict = checkpoint['best_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

    # 3) 가중치 유효성 검사
    if state_dict is None:
        print("❌ 가중치를 추출하는 데 실패했습니다.")
        return

    # 4) 모델에 적용 (DDP 접두사 처리 포함)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("⚠️ 로드 실패, 데이터 병렬(DDP) 흔적인 'module.' 접두사를 제거하고 재시도합니다...")
        # 키(Key) 이름 앞에 붙은 'module.'을 제거
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e2:
            print("❌ 최종 로드 실패. 모델 구조(Parameter)가 체크포인트와 다른지 확인하세요.")
            print(e2)
            return
        
    print("✅ 가중치 적용 완료! 평가를 시작합니다.")

    # -------------------------------------------------
    # 4. 평가 함수 실행
    # -------------------------------------------------
    print("🚀 평가 시작 (시간이 오래 걸릴 수 있습니다)...")
    
    # 평가 결과를 저장할 폴더
    eval_output_folder = Path("./eval_results")

    evaluate(
        model=model,
        musdb_path=MUSDB_PATH,
        eval_folder=eval_output_folder,
        
        workers=0,
        device=DEVICE,
        save=True,
        is_wav=True,
        
        split=True        # <--- False에서 True로 변경
    )

    print("✅ 모든 평가가 완료되었습니다!")

if __name__ == "__main__":
    run_evaluation()