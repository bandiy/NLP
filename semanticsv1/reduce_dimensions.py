import numpy as np
import umap
from pathlib import Path

PCA_PATH = Path("data/corpora/russian2/semantic_pca.npy")
OUT_UMAP_PATH = Path("data/corpora/russian2/semantic_umap.npy")

def main():
    reduced = np.load(PCA_PATH)
    print("Shape:", reduced.shape)


    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embedding_2d = reducer.fit_transform(reduced)

    print("UMAP output shape:", embedding_2d.shape)

    np.save(OUT_UMAP_PATH, embedding_2d)
    print("UMAP projection saved.")

if __name__ == "__main__":
    main()
