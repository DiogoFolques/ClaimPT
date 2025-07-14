import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import yake

# === Settings ===
input_folder = "."  # current directory
model_name = "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"

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

# === Load Serafim Model ===
print("Generating embeddings with Serafim...")
embedding_model = SentenceTransformer(model_name)

# === Fit BERTopic Model ===
print("Running BERTopic...")
topic_model = BERTopic(embedding_model=embedding_model, language="portuguese")
topics, probs = topic_model.fit_transform(articles)

# === Organize Articles by Topic ===
topic_dict = {}
for topic_id, text in zip(topics, articles):
    if topic_id == -1:
        continue  # skip outliers
    topic_dict.setdefault(topic_id, []).append(text)

# === Set Up YAKE for Keyword Extraction ===
print("Extracting keywords with YAKE...")
kw_extractor = yake.KeywordExtractor(lan="pt", n=1, top=10)

# === Print Keywords per Topic ===
for topic_id, texts in topic_dict.items():
    print(f"\n Topic {topic_id}:")
    full_text = " ".join(texts)
    keywords = kw_extractor.extract_keywords(full_text)
    for kw, score in keywords:
        print(f"   - {kw} ({round(score, 4)})")

print("\n BERTopic clustering and keyword extraction complete.")
