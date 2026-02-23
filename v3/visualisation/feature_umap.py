import numpy as np
import umap
import matplotlib.pyplot as plt

EMBED_PATH = "data/corpora/russian2/feature_embeddings.npy"

print("Loading feature embeddings...")
X = np.load(EMBED_PATH)

print("Running UMAP...")
reducer = umap.UMAP(
    n_neighbors=25,
    min_dist=0.2,
    metric="cosine",
    random_state=42
)

embedding_2d = reducer.fit_transform(X)

plt.figure(figsize=(12, 9))
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    s=12,
    alpha=0.7
)

plt.title("Feature-Based Phonetic Space (Mean-Pooled PanPhon Features)", fontsize=14)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()
