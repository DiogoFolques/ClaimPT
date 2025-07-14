import os
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import yake

# === Settings ===
input_folder = "."  # folder with .txt files
model_name = "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"
num_clusters = 10  # set this to the number of desired clusters

# === Load Articles ===
print("Loading articles...")
articles = []
filenames = []
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
            articles.append(text)
            filenames.append(filename)

# === Generate Sentence Embeddings ===
print("📐 Generating embeddings with Serafim...")
model = SentenceTransformer(model_name)
embeddings = model.encode(articles, show_progress_bar=True)

# === Apply K-Means Clustering ===
print("Running K-Means clustering...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
kmeans.fit(embeddings)
labels = kmeans.labels_

# === Group Articles by Cluster ===
cluster_dict = {i: [] for i in range(num_clusters)}
for label, article in zip(labels, articles):
    cluster_dict[label].append(article)

# === Set Up YAKE ===
print("Extracting keywords with YAKE...")
custom_stopwords = {"lusa"}  # lowercase is safest
kw_extractor = yake.KeywordExtractor(lan="pt", n=1, top=10, stopwords=custom_stopwords)


# === Extract and Print Keywords per Cluster ===
for cluster_id, texts in cluster_dict.items():
    print(f"\n🔹 Cluster {cluster_id}:")
    full_text = " ".join(texts)
    keywords = kw_extractor.extract_keywords(full_text)
    for kw, score in keywords:
        print(f"   - {kw} ({round(score, 4)})")

print("\nClustering and keyword extraction complete.")
