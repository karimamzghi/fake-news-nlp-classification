# FakeBusters
### AI-Powered Fake News Detection

> **Can AI distinguish between real and fake news?**
>
> FakeBusters investigates multiple Natural Language Processing (NLP) techniques and Machine Learning models to identify misinformation in news articles. The project compares traditional text vectorization methods, word embeddings, deep learning, and transformer models to determine the most effective approach.

---

# Team FakeBusters

- Karima Mzoughi
- Kriti Amin

---

# Project Objective

The goal of this project is to build an AI system capable of classifying news articles as either: Real News (1) OR Fake News (0)

Rather than relying on a single model, this project explores multiple NLP pipelines and compares their performance to identify the best-performing solution.

---

# Dataset

The dataset contains the following columns:

| Column | Description |
|----------|-------------|
| label | 0 = Fake, 1 = Real |
| title | News headline |
| text | Full news article |
| subject | Topic/category |
| date | Publication date |

After training, the best model will predict the labels for `validation_data.csv`.

---

# Project Structure

```text
fakebusters-fake-news-classification/
│
├── dataset/
│   ├── data.csv
│   └── validation_data.csv
│
├── deployment/
│   └── app.py
│
├── models/
│   ├── bow_models/
│   ├── tfidf_models/
│   ├── embedding_models/
│   ├── lstm/
│   └── distilbert/
│
├── experiments/
│   ├── 01_baseline_logistic_regression_bow.ipynb
│   ├── 02_naive_bayes_bow.ipynb
│   ├── 03_random_forest_bow.ipynb
│   ├── 04_svm_bow.ipynb
│   │
│   ├── 05_logistic_regression_tfidf.ipynb
│   ├── 06_naive_bayes_tfidf.ipynb
│   ├── 07_random_forest_tfidf.ipynb
│   ├── 08_svm_tfidf.ipynb
│   │
│   ├── 09_logistic_regression_embeddings.ipynb
│   ├── 10_random_forest_embeddings.ipynb
│   ├── 11_svm_embeddings.ipynb
│   │
│   ├── 12_bilstm.ipynb
│   ├── 13_distilbert.ipynb
│   └── 14_final_predictions.ipynb
│
├── results/
│   ├── confusion_matrices/
│   ├── plots/
│   ├── predictions/
│   └── model_tracking.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── evaluator.py
│   ├── predictor.py
│   └── experiment_tracker.py
│
├── README.md
└── .gitignore
```

---

# Methodology

Our workflow follows a complete NLP pipeline.

```
Dataset

↓

Exploratory Data Analysis (EDA)

↓

Train/Test Split (80/20)

↓

Text Preprocessing

↓

Feature Extraction

↓

Model Training

↓

Evaluation

↓

Model Comparison

↓

Best Model Selection

↓

Prediction on Validation Dataset

↓

Deployment
```

---

#  Text Preprocessing

The following preprocessing techniques will be evaluated throughout the experiments:

- Tokenization
- Lowercasing
- Removing punctuation
- Stopword removal
- Lemmatization
- Text cleaning

---

# Feature Extraction Techniques

Three feature extraction approaches will be compared.

## 1. Bag of Words (BoW)

Converts each document into word occurrence counts.

---

## 2. TF-IDF + n-grams

Measures word importance while capturing sequences of words.

---

## 3. Word Embeddings

Dense semantic vector representations of words.

---

# Machine Learning Experiments

## Bag of Words

| Experiment | Model |
|------------|-------------------------|
| Exp 01 | Logistic Regression (Baseline) |
| Exp 02 | Naïve Bayes |
| Exp 03 | Random Forest |
| Exp 04 | Support Vector Machine |

---

## TF-IDF + n-grams

| Experiment | Model |
|------------|-------------------------|
| Exp 05 | Logistic Regression |
| Exp 06 | Naïve Bayes |
| Exp 07 | Random Forest |
| Exp 08 | Support Vector Machine |

---

## Word Embeddings

| Experiment | Model |
|------------|-------------------------|
| Exp 09 | Logistic Regression |
| Exp 10 | Random Forest |
| Exp 11 | Support Vector Machine |

> Note: Naïve Bayes is intentionally excluded because it is designed for non-negative count-based features (e.g., BoW or TF-IDF) and is not suitable for dense word embeddings.

---

## Deep Learning

| Experiment | Model |
|------------|----------------|
| Exp 12 | Bidirectional LSTM |

---

## Transfer Learning

| Experiment | Model |
|------------|----------------|
| Exp 13 | DistilBERT Fine-tuning |

---

# 📊 Model Evaluation

Each experiment will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model will be selected based primarily on the **F1-score**, as it balances precision and recall, making it well-suited for fake news classification.

---

# Experiment Tracking

Every experiment is automatically logged in:

```
results/model_tracking.csv
```

### Tracking Fields

| Column |
|----------|
| model_id |
| model_name |
| features |
| preprocessing |
| algorithm |
| train_accuracy |
| test_accuracy |
| precision |
| recall |
| f1_score |
| notes |
| model_path |

---

# Deployment

The final selected model will be deployed using **Hugging Face Spaces**, allowing users to enter a news headline or article and receive a prediction indicating whether the content is likely to be fake or real.

---

# Team Responsibilities

## Karima

- Design project architecture
- Build project structure
- Train six experiments
- Experiment tracking
- Deployment

---

## Kriti

- Exploratory Data Analysis (EDA)
- Train six experiments
- Data analysis
- Performance comparison

---

## Together

- Compare all models
- Select best-performing model
- Generate validation predictions
- Build deployment
- Final presentation
- Documentation

---

# 👥 Authors

| Name | Role |
|------|------|
| **Karima Mzoughi** | Project Architecture, Machine Learning, Deployment |
| **Kriti B.** | Exploratory Data Analysis (EDA), Machine Learning |
