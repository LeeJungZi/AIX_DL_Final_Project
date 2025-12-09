import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Data
CSV_FILE = 'data.csv'
df = pd.read_csv(CSV_FILE)

# Drop filename
df_features = df.drop(['filename'], axis=1)

# Separate Features (X) and Target (y)
X = df_features.iloc[:, :-1]
y = df_features.iloc[:, -1]

# 2. Data Split
# Split data to identify the unseen Test set (Validation set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Data: {len(X)}")
print(f"Test Data (PCA Target): {len(X_test)}")

# 3. Standardization
# [Preprocessing] Fit scaler on Train set only, then transform Test set
scaler = StandardScaler()
scaler.fit(X_train) 
X_test_scaled = scaler.transform(X_test)

# 4. PCA Dimensionality Reduction
# [Analysis] Project high-dimensional features into 2D space
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_test_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Genre'] = y_test.values

# Calculate Explained Variance
var_ratio = pca.explained_variance_ratio_

# 5. Visualization and Save
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Genre', 
    data=pca_df, 
    palette='bright', 
    s=80,             
    alpha=0.7         
)

plt.title('PCA Projection of GTZAN Test Set (Model Validation)', fontsize=16, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({var_ratio[0]:.1%} Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]:.1%} Variance)', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save image
save_path = 'pca_testset.png'
plt.savefig(save_path, dpi=300)
print(f"✅ Graph saved: {save_path}")

plt.show()
