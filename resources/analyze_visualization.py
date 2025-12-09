import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Data
CSV_FILE = 'musdb_data.csv' 
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {CSV_FILE} (Total {len(df)} tracks)")
except FileNotFoundError:
    print(f"❌ File not found: {CSV_FILE}")
    print("Run create_musdb_csv.py first.")
    exit()

# Drop unnecessary columns
if 'filename' in df.columns:
    df_features = df.drop(['filename', 'label'], axis=1)
else:
    df_features = df.drop(['label'], axis=1)

labels = df['label']

# 2. Rename columns for intuitive visualization
feature_rename_map = {
    'chroma_stft': 'Harmony',
    'rmse': 'Energy',
    'spectral_centroid': 'Brightness',
    'spectral_bandwidth': 'Richness',
    'rolloff': 'Sharpness',
    'zero_crossing_rate': 'Noisiness'
}

# 3. Data Scaling
# [Preprocessing] Standardize features for PCA and Comparison
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# ==========================================
# [Visualization 1] Genre Distribution
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
# [Visualization 2] Feature Bar Chart
# ==========================================
df_renamed = df_features.rename(columns=feature_rename_map)

scaled_features_renamed = scaler.fit_transform(df_renamed)
df_scaled = pd.DataFrame(scaled_features_renamed, columns=df_renamed.columns)
df_scaled['label'] = labels

feature_means = df_scaled.groupby('label').mean()

selected_features = list(feature_rename_map.values())

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
# [Visualization 3] PCA Analysis (2D Projection)
# ==========================================
# [Dimensionality Reduction] PCA to 2 components for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Genre'] = labels.values

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
