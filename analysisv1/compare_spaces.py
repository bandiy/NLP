import numpy as np
import json
import random
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


PHON_KNN_PATH = Path("data/corpora/russian2/phonetic_knn_graph.json")
PHON_IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

SEM_EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
SEM_LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")

TOP_K = 5

def load_phonetic():
    with open(PHON_KNN_PATH, encoding="utf-8") as f:
        knn = json.load(f)

    phon_lemmas = []

    with open(PHON_IPA_PATH, encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            phon_lemmas.append(lemma)

    return knn, phon_lemmas


def load_semantic():
    embeddings = np.load(SEM_EMB_PATH)

    with open(SEM_LEMMA_PATH, encoding="utf-8") as f:
        lemmas = [line.strip() for line in f]

    return embeddings, lemmas


def build_semantic_knn(embeddings, lemmas):

    sim_matrix = cosine_similarity(embeddings)

    knn = {}

    for i, lemma in enumerate(lemmas):
        nearest = np.argsort(-sim_matrix[i])[1:TOP_K+1]
        knn[lemma] = [lemmas[j] for j in nearest]

    return knn

def main():

    phon_knn, phon_lemmas = load_phonetic()
    sem_embeddings, sem_lemmas = load_semantic()

    sem_knn = build_semantic_knn(sem_embeddings, sem_lemmas)

    intersection = list(set(phon_lemmas) & set(sem_lemmas))
    print("Intersection size:", len(intersection))

    overlaps = []
    random_overlaps = []

    aligned_examples = []
    divergent_examples = []

    for word in intersection:

        phon_neighbors = set(
            [n["lemma"] for n in phon_knn[word][:TOP_K]]
        )

        sem_neighbors = set(sem_knn[word])

        shared = phon_neighbors & sem_neighbors
        overlap = len(shared)

        overlaps.append(overlap)

        # collect aligned examples + shared neighbours
        if overlap >= 1:
            aligned_examples.append({
                "word": word,
                "shared": list(shared)
            })

        # collect divergent examples
        if overlap == 0:
            divergent_examples.append(word)

        # random baseline
        random_neighbors = set(random.sample(intersection, TOP_K))
        random_overlaps.append(
            len(random_neighbors & sem_neighbors)
        )

    print("\n--- RESULTS ---")
    print("Average overlap:", np.mean(overlaps))
    print("Median overlap:", np.median(overlaps))
    print("Percent with ≥1 overlap:",
          np.mean([o > 0 for o in overlaps]) * 100)

    print("Random baseline average overlap:",
          np.mean(random_overlaps))

    print("\n--- ALIGNED EXAMPLES ---")
    for item in aligned_examples[:15]:
        print(f"{item['word']} -> shared: {item['shared']}")

    print("\n--- DIVERGENT EXAMPLES ---")
    print(divergent_examples[:20])


if __name__ == "__main__":
    main()
