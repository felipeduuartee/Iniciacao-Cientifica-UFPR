import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import ast
from google.colab import files

uploaded = files.upload()


df = pd.read_csv("touch_events.csv", dtype={"doc_created_utc_milli": str})

df["event"] = df["event"].apply(lambda x: np.array(ast.literal_eval(x)).astype(float))

#Aplicação do PCA
pca = PCA(n_components=3)
event_matrix_reduced = np.vstack(df["event"].apply(lambda x: pca.fit_transform(x)).values)


pd.DataFrame(event_matrix_reduced, columns=['events_x', 'events_y', 'events_z']).to_csv("event_matrix_reduced.csv", index=False) # index false para nao salvar os indices, apenas os dados e nomes das colunas
files.download("event_matrix_reduced.csv")

np.random.seed(5)

estimators = [
    ("k_means_iris_8", KMeans(n_clusters=8, n_init="auto")),
    ("k_means_iris_3", KMeans(n_clusters=3, n_init="auto")),
    ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fig = plt.figure(figsize=(10, 8))
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
    est.fit(event_matrix_reduced)
    labels = est.labels_

    ax.scatter3D(
        event_matrix_reduced[:, 0],
        event_matrix_reduced[:, 1],
        c=labels.astype(float),
        edgecolor="k",
    )

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel("eixo x")
    ax.set_ylabel("eixo y")
    ax.set_zlabel("eixo z")
    ax.set_title(title)

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()