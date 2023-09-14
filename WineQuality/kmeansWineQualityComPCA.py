import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("WineQT.csv")


X = df.drop(columns=['Id']).values
y = df['quality'].values 


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

np.random.seed(5)

estimators = [
    ("k_means_wine_8", KMeans(n_clusters=8, n_init="auto")),
    ("k_means_wine_3", KMeans(n_clusters=3, n_init="auto")),
    ("k_means_wine_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fig = plt.figure(figsize=(10, 8))
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
    est.fit(X_pca)
    labels = est.labels_

    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels.astype(float), edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel('Principal Component ')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(title)

# Adicionar o Ground Truth
ax = fig.add_subplot(2, 2, 4, projection="3d", elev=48, azim=134)

sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, edgecolor="k", cmap=plt.cm.get_cmap('viridis', 5))

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title("Ground Truth") #ground truth é o conjunto de dados de referência que é usado para comparar os resultados do modelo

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()


