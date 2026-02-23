import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance

# =========================================
# CONFIG
# =========================================

ALPHA = 0.65
TOP_SEM = 200
TOP_FINAL = 10
SEMANTIC_THRESHOLD = 0.15

TEST_WORDS = ["брат", "мать", "ночь", "имя", "новый"]

RU_EMB_PATH = Path("data/corpora/russian_french/processed/russian_semantic_embeddings.npy")
FR_EMB_PATH = Path("data/corpora/russian_french/processed/french_semantic_embeddings.npy")

RU_LEMMA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
FR_LEMMA_PATH = Path("data/corpora/russian_french/processed/french_lemma_frequencies.tsv")

RU_IPA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_ipa.tsv")
FR_IPA_PATH = Path("data/corpora/russian_french/processed/french_lemma_ipa.tsv")

# load

def load_lemmas(path):
    with path.open(encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f]

def load_ipa(path):
    ipa_map = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            ipa_map[lemma] = ipa
    return ipa_map

# util
def minmax(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def phonetic_similarity(dist, ipa1, ipa2):
    raw = dist.weighted_feature_edit_distance(ipa1, ipa2)
    norm = raw / max(len(ipa1), len(ipa2), 1)
    return 1 / (1 + norm)

# score

def compute_semantic_only(ru_idx, sim_matrix, fr_lemmas):
    sem_scores = sim_matrix[ru_idx]
    ranked = np.argsort(-sem_scores)
    return [(fr_lemmas[i], sem_scores[i]) for i in ranked[:TOP_FINAL]]

def compute_hybrid(word,
                   ru_idx,
                   sim_matrix,
                   ru_ipa,
                   fr_ipa,
                   fr_lemmas,
                   dist):

    sem_scores_full = sim_matrix[ru_idx]
    top_indices = np.argsort(-sem_scores_full)[:TOP_SEM]

    candidates = []

    for i in top_indices:
        fr_word = fr_lemmas[i]
        sem_score = sem_scores_full[i]

        if sem_score < SEMANTIC_THRESHOLD:
            continue

        fr_sound = fr_ipa.get(fr_word, "")
        if not fr_sound:
            continue

        phon_sim = phonetic_similarity(
            dist,
            ru_ipa[word],
            fr_sound
        )

        candidates.append((fr_word, sem_score, phon_sim))

    if not candidates:
        return []

    fr_words, sem_scores, phon_scores = zip(*candidates)

    sem_scores = np.array(sem_scores)
    phon_scores = np.array(phon_scores)

    sem_norm = minmax(sem_scores)
    phon_norm = minmax(phon_scores)

    combined = ALPHA * sem_norm + (1 - ALPHA) * phon_norm

    ranked = np.argsort(-combined)

    return [(fr_words[i], combined[i]) for i in ranked[:TOP_FINAL]]

# main

def main():

    ru_emb = np.load(RU_EMB_PATH)
    fr_emb = np.load(FR_EMB_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    ru_ipa = load_ipa(RU_IPA_PATH)
    fr_ipa = load_ipa(FR_IPA_PATH)

    sim_matrix = cosine_similarity(ru_emb, fr_emb)

    dist = panphon.distance.Distance()

    print("\n=== SEMANTIC vs HYBRID COMPARISON ===\n")

    for word in TEST_WORDS:

        if word not in ru_lemmas:
            print(f"{word} not found.\n")
            continue

        ru_idx = ru_lemmas.index(word)

        sem_results = compute_semantic_only(ru_idx, sim_matrix, fr_lemmas)
        hyb_results = compute_hybrid(word,
                                     ru_idx,
                                     sim_matrix,
                                     ru_ipa,
                                     fr_ipa,
                                     fr_lemmas,
                                     dist)

        print(f"\n===== Russian: {word} =====\n")

        print("Semantic Top-10:")
        for w, s in sem_results:
            print(f"  {w:15} | {s:.3f}")

        print("\nHybrid Top-10 (alpha=0.65):")
        for w, s in hyb_results:
            print(f"  {w:15} | {s:.3f}")

        print("\n" + "-"*40)

if __name__ == "__main__":
    main()
