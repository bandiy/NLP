import numpy as np
import umap
import matplotlib.pyplot as plt

#config

EMBED_PATH = "data/corpora/russian2/feature_embeddings.npy"
LENGTH_PATH = "data/corpora/russian2/phoneme_lengths.npy"

# panphon feature indices - test.py
SYLLABIC_INDEX = 0   # 'syl'
VOICE_INDEX = 8      # 'voi'

#load

print("Loading feature embeddings...")
X = np.load(EMBED_PATH)

print("Loading phoneme lengths...")
lengths = np.load(LENGTH_PATH)

print("Embedding shape:", X.shape)

feature_dim = X.shape[1] // 2
mean_features = X[:, :feature_dim]
std_features = X[:, feature_dim:]

#umap

print("Running UMAP...")
reducer = umap.UMAP(
    metric="cosine",
    n_neighbors=25,
    min_dist=0.2,
    random_state=42
)

embedding = reducer.fit_transform(X)

print("UMAP complete.")

#linguistic metrics

vowel_prop = mean_features[:, SYLLABIC_INDEX]
voiced_prop = mean_features[:, VOICE_INDEX]
variance_mag = std_features.mean(axis=1)

#plot

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plots = [
    (vowel_prop, "Vowel Proportion (syl feature)"),
    (voiced_prop, "Voiced Proportion (voi feature)"),
    (lengths, "Word Length (Number of Phonemes)"),
    (variance_mag, "Average Feature Variance")
]

for ax, (values, title) in zip(axes.flatten(), plots):

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap="plasma",
        s=10,
        alpha=0.85
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=15)

fig.suptitle(
    "Feature-Based Phonetic Space (Mean + Std Pooling)\nLinguistic Dimension Analysis",
    fontsize=16
)

plt.tight_layout()
plt.show()
