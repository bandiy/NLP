import ast
import numpy as np
from pathlib import Path

#config

FEATURE_PATH = Path("data/corpora/russian2/lemma_features.tsv")

OUT_EMBED_PATH = Path("data/corpora/russian2/feature_embeddings.npy")
OUT_LEMMA_PATH = Path("data/corpora/russian2/feature_embedding_lemmas.txt")
OUT_LENGTH_PATH = Path("data/corpora/russian2/phoneme_lengths.npy")

symbol_map = {
    '+': 1.0,
    '-': -1.0,
    '0': 0.0
}

#embeddings
embeddings = []
lemmas = []
lengths = []

print("Loading feature lexicon...")

with FEATURE_PATH.open(encoding="utf-8") as f:
    for line in f:

        parts = line.strip().split("\t")

        if len(parts) != 3:
            continue

        lemma, ipa, feature_str = parts

        try:
            features = ast.literal_eval(feature_str)
        except Exception:
            continue

        if len(features) == 0:
            continue

        numeric_features = []

        for phoneme_vec in features:
            try:
                numeric_vec = [symbol_map[val] for val in phoneme_vec]
            except KeyError:
                continue

            numeric_features.append(numeric_vec)

        if len(numeric_features) == 0:
            continue

        numeric_features = np.array(numeric_features, dtype=np.float32)

        # phoneme length
        lengths.append(len(numeric_features))

        # std and mean pooling
        mean_vec = numeric_features.mean(axis=0)
        std_vec = numeric_features.std(axis=0)

        pooled = np.concatenate([mean_vec, std_vec])

        embeddings.append(pooled)
        lemmas.append(lemma)

# matrix
embeddings = np.vstack(embeddings)

print("Final embedding shape:", embeddings.shape)

# save

np.save(OUT_EMBED_PATH, embeddings)
np.save(OUT_LENGTH_PATH, np.array(lengths))

with OUT_LEMMA_PATH.open("w", encoding="utf-8") as f:
    for lemma in lemmas:
        f.write(lemma + "\n")

print("Saved:")
print(" - Feature embeddings:", OUT_EMBED_PATH)
print(" - Lemma order:", OUT_LEMMA_PATH)
print(" - Phoneme lengths:", OUT_LENGTH_PATH)
