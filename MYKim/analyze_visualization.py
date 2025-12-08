import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 데이터 로드 (분석할 CSV 파일명 확인)
CSV_FILE = 'musdb_data.csv' 
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ {CSV_FILE} 로드 성공! (총 {len(df)}곡)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {CSV_FILE}")
    print("create_musdb_csv.py를 먼저 실행했는지 확인해주세요.")
    exit()

# 불필요한 컬럼 제거
if 'filename' in df.columns:
    df_features = df.drop(['filename', 'label'], axis=1)
else:
    df_features = df.drop(['label'], axis=1)

labels = df['label']

# 2. 직관적인 이름으로 컬럼 변경 (시각화용)
feature_rename_map = {
    'chroma_stft': 'Harmony',
    'rmse': 'Energy',
    'spectral_centroid': 'Brightness',
    'spectral_bandwidth': 'Richness',
    'rolloff': 'Sharpness',
    'zero_crossing_rate': 'Noisiness'
}

# 3. 데이터 정규화 (스케일링)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# ==========================================
# [시각화 1] 장르 분포 (Genre Distribution)
# ==========================================
plt.figure(figsize=(10, 6))
sns.countplot(y=labels, palette="viridis", order=labels.value_counts().index)
plt.title(f'Genre Distribution in {CSV_FILE}', fontsize=15, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# [시각화 2] 피처별 평균값 비교 (Feature Bar Chart)
# ==========================================
# 이름 변경 적용
df_renamed = df_features.rename(columns=feature_rename_map)

# 변경된 이름으로 스케일링 다시 수행 (데이터프레임 생성용)
scaled_features_renamed = scaler.fit_transform(df_renamed)
df_scaled = pd.DataFrame(scaled_features_renamed, columns=df_renamed.columns)
df_scaled['label'] = labels

# 장르별 평균 계산
feature_means = df_scaled.groupby('label').mean()

# 보고 싶은 주요 피처만 선택 (직관적인 이름들)
selected_features = list(feature_rename_map.values())

# 그래프 그리기
ax = feature_means[selected_features].plot(kind='bar', figsize=(14, 8), width=0.8, colormap='viridis')

plt.title('Average Audio Characteristics by Genre', fontsize=16, fontweight='bold')
plt.ylabel('Normalized Score (Relative)', fontsize=12)
plt.xlabel('Genre', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Audio Features', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# [시각화 3] PCA 분석 (2D Projection)
# ==========================================
# 2차원으로 차원 축소
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Genre'] = labels.values

# 설명력 계산
var_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Genre', 
    data=pca_df, 
    palette='deep', 
    s=60, 
    alpha=0.7
)

plt.title(f'PCA Analysis of {CSV_FILE} (2D Projection)', fontsize=15, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({var_ratio[0]:.1%} Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]:.1%} Variance)', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()