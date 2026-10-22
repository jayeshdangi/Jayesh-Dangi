# SmartNLP: Dual-Engine Text Classification System 🚀

A modular machine learning and NLP toolkit designed for text preprocessing, feature extraction, and dual-engine classification (Spam Detection & Customer Sentiment Analysis).

---

## 📌 Features

- **Advanced Text Preprocessing (`TextCleaner`)**:
  - Regex-based entity masking for URLs, currency values, phone numbers, and emails.
  - Unicode character normalization (`NFKD`).
  - Punctuation removal and case folding.
  - Custom English stopword removal with **sentiment-aware negation preservation** (e.g., retains `not`, `no`, `never`, `cannot`).
- **Dual-Engine ML Classifiers**:
  - **Spam vs. Ham Engine**: Multinomial Naive Bayes classifier powered by TF-IDF n-grams (`ngram_range=(1, 2)`).
  - **Sentiment Analysis Engine**: Logistic Regression model trained for polarity detection (Positive vs. Negative).
- **Clean Architecture**: Decoupled modules with clean import structure and minimal external overhead.

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py          # Package initialization and metadata
│   └── text_cleaner.py      # Core text normalization and cleaning pipeline
├── ml.py                    # End-to-end ML pipeline demo & inference
├── requirements.txt         # Core dependencies (scikit-learn, pandas, numpy, etc.)
└── .gitignore               # Standard Python ignores (caches, venvs, IDE configs)
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline Demo
```bash
python ml.py
```

### 4. Use `TextCleaner` in Your Own Code
```python
from src.text_cleaner import TextCleaner

cleaner = TextCleaner(remove_stopwords=True, preserve_negations=True)
raw_text = "Call 9876543210 or visit https://example.com! We won't let you down."
cleaned_text = cleaner.clean(raw_text)

print(cleaned_text)
# Output: call phone_number visit url_link wont let
```

---

## 👤 Author

- **Jayesh Dangi**
  - GitHub: [@jayeshdangi](https://github.com/jayeshdangi)
