import os
import json
import gzip
import glob
import numpy as np
import pandas as pd

# ==========================================
# 1. 평가 결과가 저장된 폴더 경로 (수정 필요!)
# run_eval.py를 돌리고 생성된 'results/test' 폴더 경로를 정확히 적어주세요.
# ==========================================
RESULTS_DIR = r"C:\Users\jwlee\AIXDL\00_Demucs2\eval_results\results\test"

def load_results():
    print(f"📂 결과 폴더 읽는 중: {RESULTS_DIR}")
    
    # .json.gz 파일 찾기
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json.gz"))
    
    if len(files) == 0:
        print("❌ 결과 파일이 하나도 없습니다! 경로를 확인하거나, 평가(run_eval.py)가 제대로 끝났는지 확인하세요.")
        return

    print(f"📄 총 {len(files)}개의 결과 파일을 발견했습니다.")

    # 모든 곡의 점수를 모을 리스트
    all_scores = []

    for file_path in files:
        try:
            # 압축된 json 파일 읽기
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # 각 악기(target)별로 점수 추출
            for target in data['targets']:
                name = target['name'] # drums, bass, vocals, other
                
                # 프레임별 점수 가져오기
                frames = target['frames']
                
                # 프레임이 비어있는 경우 방지
                if not frames:
                    continue

                sdrs = [frame['metrics']['SDR'] for frame in frames]
                sirs = [frame['metrics']['SIR'] for frame in frames]
                sars = [frame['metrics']['SAR'] for frame in frames]
                isrs = [frame['metrics']['ISR'] for frame in frames]

                # nan(결측치) 제외하고 중간값 계산
                track_score = {
                    'target': name,
                    'SDR': np.nanmedian(sdrs),
                    'SIR': np.nanmedian(sirs),
                    'SAR': np.nanmedian(sars),
                    'ISR': np.nanmedian(isrs)
                }
                all_scores.append(track_score)

        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 ({os.path.basename(file_path)}): {e}")
            continue

    if not all_scores:
        print("❌ 점수를 추출하지 못했습니다.")
        return

    # [수정된 부분] 데이터프레임 만들기 (복잡한 변환 제거)
    df = pd.DataFrame(all_scores)
    
    # 혹시 모를 에러 방지를 위해 확실하게 숫자 컬럼만 선택
    numeric_cols = ['SDR', 'SIR', 'SAR', 'ISR']
    
    # 최종 결과 집계 (악기별 Median)
    print("\n" + "="*50)
    print("           🎵 최종 성적표 (Global Median)")
    print("="*50)
    
    # 악기별로 그룹화해서 중간값 출력
    summary = df.groupby('target')[numeric_cols].median()
    print(summary.round(2)) # 소수점 2자리까지
    print("="*50)
    
    # CSV 파일로도 저장해두면 보고서 쓸 때 편합니다
    summary.to_csv("final_scores.csv")
    print("💾 결과가 'final_scores.csv' 파일로 저장되었습니다.")
    
    return summary

if __name__ == "__main__":
    load_results()