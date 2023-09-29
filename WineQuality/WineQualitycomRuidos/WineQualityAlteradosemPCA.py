# Aplicando K-means para o dataframe com os 4 melhores e enchendo 16 colunas de ruído

#SEM PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("WineQT.csv")

X = df.drop(columns=['Id']).values

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
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(float), edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    ax.set_title(title)

y = df['quality'].values
ax = fig.add_subplot(2, 2, 4, projection="3d", elev=48, azim=134)


sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolor="k", cmap=plt.cm.get_cmap('viridis', 5))


ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
ax.set_zlabel(df.columns[2])
ax.set_title("Ground Truth")

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()
