from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data():
    df = pd.read_csv("seeds.csv")
    feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
    target_col = 'V8'
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

X, y = prepare_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2d = PCA(n_components=2)
X_reduced_2D = pca_2d.fit_transform(X_scaled)
explained_variance_ratio_2D = pca_2d.explained_variance_ratio_
loss_variance_2D = 1 - explained_variance_ratio_2D.sum()

plt.figure(figsize=(8, 6))
colors = {1: 'yellow', 2: 'purple', 3: '#10C999'}
class_names = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
for class_value in set(y):
    plt.scatter(X_reduced_2D[y == class_value, 0], X_reduced_2D[y == class_value, 1],
                c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)
plt.title('PCA - 2D Projection (seeds dataset)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

pca_3d = PCA(n_components=3)
X_reduced_3D = pca_3d.fit_transform(X_scaled)
explained_variance_ratio_3D = pca_3d.explained_variance_ratio_
loss_variance_3D = 1 - explained_variance_ratio_3D.sum()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for class_value in set(y):
    ax.scatter(X_reduced_3D[y == class_value, 0], X_reduced_3D[y == class_value, 1], X_reduced_3D[y == class_value, 2],
               c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)
plt.title('PCA - 3D Projection (seeds dataset)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.legend()
plt.show()

print(f'Loss Variance (2D PCA): {loss_variance_2D}')
print(f'Loss Variance (3D PCA): {loss_variance_3D}')
