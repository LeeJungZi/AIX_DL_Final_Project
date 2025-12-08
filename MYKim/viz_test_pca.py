import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 데이터 로드
CSV_FILE = 'data.csv'
df = pd.read_csv(CSV_FILE)

# 파일명 제거
df_features = df.drop(['filename'], axis=1)

# X(특성), y(장르) 분리
X = df_features.iloc[:, :-1]
y = df_features.iloc[:, -1]

# 2. 데이터 분할 (학습 때와 동일한 random_state=42 필수!)
# 그래야 학습 때 안 쓴 '진짜 테스트 데이터'만 골라낼 수 있습니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"전체 데이터: {len(X)}개")
print(f"테스트 데이터(PCA 대상): {len(X_test)}개")

# 3. 정규화 (StandardScaler)
scaler = StandardScaler()
# 학습 데이터 기준으로 스케일러를 맞추고, 테스트 데이터를 변환하는 것이 정석입니다.
scaler.fit(X_train) 
X_test_scaled = scaler.transform(X_test)

# 4. PCA 차원 축소 (2D)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_test_scaled)

# 시각화를 위한 데이터프레임 생성
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Genre'] = y_test.values  # 인덱스 초기화된 y_test 값 주입

# 설명력(Variance Ratio) 계산
var_ratio = pca.explained_variance_ratio_

# 5. 시각화 및 저장
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Genre', 
    data=pca_df, 
    palette='bright', # 색상 팔레트 (bright, deep, viridis 등 변경 가능)
    s=80,             # 점 크기
    alpha=0.7         # 투명도
)

plt.title('PCA Projection of GTZAN Test Set (Model Validation)', fontsize=16, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({var_ratio[0]:.1%} Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]:.1%} Variance)', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 이미지 파일로 저장
save_path = 'pca_testset.png'
plt.savefig(save_path, dpi=300)
print(f"✅ 그래프 저장 완료: {save_path}")

plt.show()