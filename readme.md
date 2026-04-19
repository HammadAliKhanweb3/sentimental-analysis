# 🧠 Sentiment Analysis — NLP Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.7%2B-green?style=for-the-badge)
![TextBlob](https://img.shields.io/badge/TextBlob-0.17%2B-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A complete end-to-end NLP pipeline for text sentiment classification using classical machine learning.**  
Bag-of-Words · TF-IDF · Naïve Bayes · Logistic Regression

[Overview](#-overview) • [Pipeline](#-pipeline) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Structure](#-project-structure) • [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

This project implements a full sentiment analysis pipeline — from raw text ingestion to trained classifier — using classical NLP techniques. It benchmarks three model configurations and identifies the best-performing combination for production use.

| | |
|---|---|
| **Task** | Sentiment Classification (Positive / Negative / Neutral) |
| **Approach** | Classical ML — BoW & TF-IDF vectorization |
| **Best Accuracy** | **86.56%** — Logistic Regression + TF-IDF |
| **Models Tested** | Naïve Bayes (BoW), Naïve Bayes (TF-IDF), Logistic Regression (TF-IDF) |
| **Language** | Python 3.8+ |

---

## 🔁 Pipeline

The project is structured into three major phases:

```
Raw Text Dataset
      │
      ▼
┌─────────────────────────────────────────────┐
│              TEXT CLEANING                  │
│  1. Lowercase  2. Remove Punctuation        │
│  3. Remove Numbers  4. Remove Emojis        │
│  5. Remove Stopwords  6. Spell Correction   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            DATA PREPROCESSING               │
│  1. Split into text/label columns           │
│  2. Drop null values                        │
│  3. Encode labels to numeric                │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           FEATURE ENGINEERING               │
│         Train / Test Split (80/20)          │
│    ┌──────────────┬──────────────────┐      │
│    │     BoW      │     TF-IDF       │      │
│    └──────┬───────┴────────┬─────────┘      │
│           │                │                │
│      Naïve Bayes     Naïve Bayes            │
│                      Logistic Regression    │
└─────────────────────────────────────────────┘
                      │
                      ▼
              Accuracy Evaluation
```

---

## 📊 Results

| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| Naïve Bayes | Bag-of-Words | 77.50% |
| Naïve Bayes | TF-IDF | 73.16% |
| **Logistic Regression** | **TF-IDF** | **86.56% 🏆** |

> **Why does NB perform worse with TF-IDF?**  
> Naïve Bayes assumes feature independence and is optimized for raw counts. TF-IDF introduces continuous weighted values that violate this assumption, hurting performance. Logistic Regression handles weighted features natively — hence the 11.8 pp gain.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK assets
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

**`requirements.txt`**
```
pandas>=1.4
numpy>=1.21
scikit-learn>=1.1
nltk>=3.7
textblob>=0.17
emoji>=2.0
matplotlib>=3.5
```

---

## 🚀 Usage

### Run the full pipeline

```bash
python src/main.py --data data/raw/dataset.csv
```

### Step-by-step

```python
from src.cleaning import clean_text
from src.preprocessing import preprocess
from src.features import vectorize
from src.models import train_evaluate

import pandas as pd

# Load
df = pd.read_csv("data/raw/dataset.csv")

# Clean
df["cleaned_text"] = df["text"].apply(clean_text)

# Preprocess
df = preprocess(df)

# Vectorize + Train + Evaluate
results = train_evaluate(df)
print(results)
```

### Expected output

```
[Naïve Bayes + BoW]         Accuracy: 0.7750
[Naïve Bayes + TF-IDF]      Accuracy: 0.7316
[Logistic Regression + TF-IDF]  Accuracy: 0.8656  ✅ Best
```

---

## 🧹 Text Cleaning — Details

Each text sample passes through a 6-step cleaning function:

| Step | Operation | Method |
|------|-----------|--------|
| 1 | Lowercase conversion | `str.lower()` |
| 2 | Punctuation removal | `str.translate` + `string.punctuation` |
| 3 | Number removal | `re.sub(r'\d+', '', text)` |
| 4 | Emoji removal | `re.sub(r'[^\x00-\x7F]+', '', text)` |
| 5 | Stopword removal | `nltk.corpus.stopwords` |
| 6 | Spell correction | `TextBlob(text).correct()` |

```python
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = str(TextBlob(text).correct())
    return text
```

> ⚠️ **Performance Note:** Spell correction is slow on large datasets. For >100K rows, consider [SymSpell](https://github.com/wolfgarbe/SymSpell) as a faster alternative.

---

## 📁 Project Structure

```
sentiment-analysis-nlp/
├── data/
│   ├── raw/                  # Original dataset (CSV)
│   └── processed/            # Cleaned & encoded data
├── notebooks/
│   └── exploration.ipynb     # EDA and prototyping
├── src/
│   ├── cleaning.py           # 6-step text cleaning
│   ├── preprocessing.py      # Null removal, label encoding
│   ├── features.py           # BoW & TF-IDF vectorization
│   ├── models.py             # Train, evaluate, compare
│   └── main.py               # End-to-end runner
├── models/
│   ├── nb_bow.pkl            # Saved NB + BoW model
│   ├── nb_tfidf.pkl          # Saved NB + TF-IDF model
│   └── lr_tfidf.pkl          # Saved LR + TF-IDF model
├── reports/
│   └── results.csv           # Accuracy comparison
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [x] 6-step text cleaning pipeline
- [x] BoW and TF-IDF vectorization
- [x] Naïve Bayes baseline
- [x] Logistic Regression — best model
- [ ] Cross-validation (k-fold) for robust evaluation
- [ ] Precision / Recall / F1 per class
- [ ] BERT / RoBERTa transformer fine-tuning
- [ ] REST API with FastAPI for real-time inference
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork → Clone → Branch → Commit → PR
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/HammadAliKhanWeb3">HammadAliKhanWeb3</a>
</div>
