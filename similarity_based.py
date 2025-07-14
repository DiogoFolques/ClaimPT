import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Sentence-BERT Models
# -----------------------------
similarity_models = {
    "MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "MPNet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "Serafim": "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"

}

# -----------------------------
# Candidate Labels & Mapping
# -----------------------------
candidate_labels = [
    "Política", "Sociedade", "Economia", "Internacional", "Desporto",
    "Saúde", "Cultura", "Ciência e Ambiente", "Tecnologia", "Notícias Locais"
]
abbreviations = [label.split(":")[0].strip() for label in candidate_labels]
label_mapping = {
    "Política": "politics",
    "Sociedade": "society",
    "Economia": "economy",
    "Internacional": "international",
    "Desporto": "sports",
    "Saúde": "health",
    "Cultura": "culture",
    "Ciência e Ambiente": "science and environment",
    "Tecnologia": "technology",
    "Notícias Locais": "local"
}

# -----------------------------
# Load Ground Truth
# -----------------------------
df_gt = pd.read_csv("ground_truth.csv").head(50)
df_gt["Document"] = df_gt["Document"].astype(str)
df_gt["Curator"] = df_gt["Curator"].str.strip().str.lower()
ground_truth_dict = dict(zip(df_gt["Document"], df_gt["Curator"]))

# -----------------------------
# Load Models and Precompute Label Embeddings
# -----------------------------
models = {}
label_embeddings = {}
for name, path in similarity_models.items():
    model = SentenceTransformer(path)
    models[name] = model
    label_embeddings[name] = model.encode(candidate_labels, convert_to_tensor=True)

# -----------------------------
# Classify Articles
# -----------------------------
predictions = {model: [] for model in similarity_models}
true_labels = []

for doc_name, true_label in ground_truth_dict.items():
    if not os.path.exists(doc_name):
        print(f"Skipping missing file: {doc_name}")
        continue
    print(f"\nClassifying: {doc_name}") 
    with open(doc_name, "r", encoding="utf-8") as f:
        article_text = f.read()
    true_labels.append(true_label)

    for model_name, model in models.items():
        article_embedding = model.encode(article_text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(article_embedding, label_embeddings[model_name])[0]
        best_idx = int(similarities.argmax())
        pred_abbr = abbreviations[best_idx]
        pred_label = label_mapping.get(pred_abbr, pred_abbr.lower())
        predictions[model_name].append(pred_label)
        print(f"{model_name} → {doc_name}: predicted '{pred_label}' vs ground truth '{true_label}'")

# -----------------------------
# Evaluation
# -----------------------------
results = {}
for model_name, preds in predictions.items():
    results[model_name] = {
        "Accuracy": accuracy_score(true_labels, preds),
        "Precision": precision_score(true_labels, preds, average="weighted", zero_division=0),
        "Recall": recall_score(true_labels, preds, average="weighted", zero_division=0),
        "F1-Score": f1_score(true_labels, preds, average="weighted", zero_division=0)
    }

df = pd.DataFrame(results).T
df.to_csv("similarity_results.csv")
print(df)

# -----------------------------
# Plot Results
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))
df.plot(kind='bar', ax=ax)
plt.title("Similarity-Based Classification Performance")
plt.ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
