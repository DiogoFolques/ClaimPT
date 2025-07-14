import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Zero-Shot Models
# -----------------------------
zero_shot_models = {

    "mDeBERTa-v3-base-mnli-xnli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "mDeBERTa-v3-base-2mil7": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "DeBERTa-v3-large-fever-anli": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "xlm-roberta-large-xnli": "joeddav/xlm-roberta-large-xnli"
    #"bart-large-mnli": "facebook/bart-large-mnli",
    #"faquad-nli": "ruanchaves/bert-base-portuguese-cased-faquad-nli",
    #"legal-nli-sts": "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1",
    #"assin2-nli": "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"
}

# -----------------------------
# Candidate Labels & Mapping
# -----------------------------
candidate_labels = [
    "Política: Discussões sobre governo, eleições, leis, líderes políticos e políticas públicas",
    "Sociedade: Questões sociais, comportamento, tendências culturais e mudanças na opinião pública",
    "Economia: Informações sobre mercado financeiro, investimentos, PIB, inflação, emprego e negócios",
    "Internacional: Notícias sobre relações diplomáticas, conflitos globais, tratados e geopolítica",
    "Desporto: Notícias sobre eventos desportivos, equipas, campeonatos, atletas e resultados de competições",
    "Saúde: Atualizações sobre doenças, medicina, bem-estar, serviços de saúde e políticas sanitárias",
    "Cultura: Cobertura de artes, cinema, música, literatura, entretenimento e manifestações culturais",
    "Ciência e Ambiente: Descobertas científicas, avanços em pesquisa, estudos acadêmicos, novas tecnologias e questões ambientais",
    "Tecnologia: Inovações digitais, inteligência artificial, cibersegurança, gadgets e tendências tecnológicas",
    "Notícias Locais: Acontecimentos regionais, serviços públicos, infraestrutura e comunidades locais"
]
abbreviations = [label.split(":")[0].strip() for label in candidate_labels]
hypothesis_template = "Este artigo discute '{}'?"
formatted_labels = [hypothesis_template.format(label) for label in candidate_labels]

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
# Load Ground Truth from CSV
# -----------------------------
df_gt = pd.read_csv("ground_truth.csv").head(50)
df_gt["Document"] = df_gt["Document"].astype(str)
df_gt["Curator"] = df_gt["Curator"].str.strip().str.lower()
ground_truth_dict = dict(zip(df_gt["Document"], df_gt["Curator"]))

# -----------------------------
# Classify Files with Matching Ground Truth
# -----------------------------
predictions = {model: [] for model in zero_shot_models}
true_labels = []

print("Loading models...")
zero_shot_pipelines = {}
for name, path in zero_shot_models.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    zero_shot_pipelines[name] = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

print("Classifying...")
for doc_name, true_label in ground_truth_dict.items():
    if not os.path.exists(doc_name):
        print(f"Missing file: {doc_name}")
        continue

    print(f"\nClassifying: {doc_name}") 

    with open(doc_name, "r", encoding="utf-8") as f:
        article_text = f.read()

    true_labels.append(true_label)

    for model_name, classifier in zero_shot_pipelines.items():
        print(f"  → Using model: {model_name}") 
        result = classifier(article_text, formatted_labels)
        best_label = result["labels"][0]
        idx = formatted_labels.index(best_label)
        pred_abbr = abbreviations[idx]
        pred_label = label_mapping.get(pred_abbr, pred_abbr.lower())
        predictions[model_name].append(pred_label)
        print(f"  → Ground truth: {true_label}")
        print(f"  → Predicted: {pred_label}")


# -----------------------------
# Compute and Display Metrics
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
print(df)
df.to_csv("zero_shot_results.csv", index=True)


# -----------------------------
# Plot Results
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))
df.plot(kind='bar', ax=ax)
plt.title("Zero-Shot Classification Performance")
plt.ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
