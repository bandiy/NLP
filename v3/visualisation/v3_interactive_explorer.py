import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path

#paths

EMBED_PATH = "data/corpora/russian2/feature_embeddings.npy"
LEMMA_PATH = "data/corpora/russian2/feature_embedding_lemmas.txt"
LENGTH_PATH = "data/corpora/russian2/phoneme_lengths.npy"
IPA_PATH = "data/corpora/russian2/lemma_ipa.tsv"

# load
print("Loading embeddings...")
X = np.load(EMBED_PATH)

print("Loading lemmas...")
with open(LEMMA_PATH, encoding="utf-8") as f:
    lemmas = [line.strip() for line in f]

print("Loading phoneme lengths...")
lengths = np.load(LENGTH_PATH)

print("Loading IPA...")
ipa_map = {}
with open(IPA_PATH, encoding="utf-8") as f:
    for line in f:
        lemma, ipa = line.strip().split("\t")
        ipa_map[lemma] = ipa

ipas = [ipa_map.get(lemma, "") for lemma in lemmas]

feature_dim = X.shape[1] // 2
mean_features = X[:, :feature_dim]
std_features = X[:, feature_dim:]

# feature indices - test.py
SYLLABIC_INDEX = 0
VOICE_INDEX = 8

vowel_prop = mean_features[:, SYLLABIC_INDEX]
voiced_prop = mean_features[:, VOICE_INDEX]
variance_mag = std_features.mean(axis=1)

# umap

print("Running UMAP...")
reducer = umap.UMAP(metric="cosine", random_state=42)
embedding = reducer.fit_transform(X)

# hover

hover_text = []
for i in range(len(lemmas)):
    text = (
        f"<b>Lemma:</b> {lemmas[i]}<br>"
        f"<b>IPA:</b> {ipas[i]}<br>"
        f"<b>Vowel proportion:</b> {vowel_prop[i]:.2f}<br>"
        f"<b>Voiced proportion:</b> {voiced_prop[i]:.2f}<br>"
        f"<b>Phoneme length:</b> {lengths[i]}<br>"
        f"<b>Feature variance:</b> {variance_mag[i]:.3f}"
    )
    hover_text.append(text)

# figure

fig = go.Figure()

color_options = {
    "Vowel Proportion": vowel_prop,
    "Voiced Proportion": voiced_prop,
    "Word Length": lengths,
    "Feature Variance": variance_mag
}

initial_metric = "Voiced Proportion"

fig.add_trace(go.Scattergl(
    x=embedding[:, 0],
    y=embedding[:, 1],
    mode="markers",
    marker=dict(
        size=6,
        color=color_options[initial_metric],
        colorscale="Turbo",
        showscale=True,
        colorbar=dict(title=initial_metric)
    ),
    text=hover_text,
    hoverinfo="text"
))

# dropdown

buttons = []

for metric_name, values in color_options.items():
    buttons.append(
        dict(
            label=metric_name,
            method="update",
            args=[
                {
                    "marker.color": [values],
                    "marker.colorbar.title": metric_name
                }
            ]
        )
    )

fig.update_layout(
    title="Interactive Feature-Based Phonetic Space (Mean + Std Pooling)",
    template="plotly_white",
    height=900,
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02,
            y=1.05
        )
    ]
)

# save show

output_file = "v3_feature_interactive.html"
fig.write_html(output_file, include_plotlyjs=True)

print(f"Saved interactive explorer to: {output_file}")

fig.show()
