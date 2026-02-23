from pathlib import Path
import panphon

# config

IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")
OUT_PATH = Path("data/corpora/russian2/lemma_features.tsv")

# initialise panphon

ft = panphon.FeatureTable()

# main

def build_feature_lexicon():
    print("Loading IPA lexicon...")

    entries = []

    with IPA_PATH.open(encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            entries.append((lemma, ipa))

    print(f"Loaded {len(entries)} entries")

    print("Converting IPA to feature vectors...")

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for lemma, ipa in entries:
            # convert IPA to feature vectors
            features = ft.word_to_vector_list(ipa)

            # serilise features
            feature_str = str(features)

            out.write(f"{lemma}\t{ipa}\t{feature_str}\n")

    print("Done.")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    build_feature_lexicon()
