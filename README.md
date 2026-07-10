# FakeBusters: Fake News Detection using Natural Language Processing

## Project Overview

FakeBusters is an end-to-end Natural Language Processing (NLP) project that automatically classifies news articles as **REAL** or **FAKE**.

The objective of this project was not only to build an accurate fake news classifier, but also to systematically compare multiple Natural Language Processing techniques, feature engineering strategies and machine learning algorithms in order to understand the trade-offs between predictive performance and production complexity.

A total of **16 experiments** were implemented, ranging from classical Machine Learning models to Deep Learning architectures.

---

# Objectives

The main objectives of this project were:

- Build a reusable NLP pipeline.
- Compare different text representations.
- Compare traditional Machine Learning models.
- Compare Deep Learning against classical Machine Learning.
- Evaluate models using multiple metrics.
- Track every experiment.
- Recommend a production-ready model based on engineering trade-offs rather than accuracy alone.

---

# Dataset

The project uses a Fake News dataset containing:

- News title
- News article
- Binary label
    - REAL
    - FAKE

The dataset is divided into:

- Training dataset
- Test dataset
- Validation dataset

The validation dataset was intentionally kept unseen during model development and was only used after selecting the final model.

---

# Project Structure

```
fake-news-nlp-classification/
│
├── dataset/
│   ├── data.csv
│   └── validation.csv
│
├── experiments/
│   │
│   ├── bow/
│   │   ├── 01_bow_logistic_regression.ipynb
│   │   ├── 02_bow_naive_bayes.ipynb
│   │   ├── 03_bow_random_forest.ipynb
│   │   └── 04_bow_svm.ipynb
│   │
│   ├── tfidf/
│   │   ├── 05_tfidf_logistic_regression.ipynb
│   │   ├── 06_tfidf_naive_bayes.ipynb
│   │   ├── 07_tfidf_random_forest.ipynb
│   │   └── 08_tfidf_svm.ipynb
│   │
│   ├── tfidf_bigrams/
│   │   ├── 09_tfidf_bigrams_logistic_regression.ipynb
│   │   ├── 10_tfidf_bigrams_naive_bayes.ipynb
│   │   ├── 11_tfidf_bigrams_random_forest.ipynb
│   │   └── 12_tfidf_bigrams_svm.ipynb
│   │
│   ├── word_embeddings/
│   │   ├── 13_embeddings_logistic_regression.ipynb
│   │   └── 14_embeddings_random_forest.ipynb
│   │
│   ├── bilstm/
│   │   └── 16_bilstm.ipynb
│   │
│   ├── transformers/
│   │   └── 17_distilbert.ipynb
│   │
│   └── final_evaluation/
│       ├── 18_model_comparison.ipynb
│       └── 19_validation_predictions.ipynb
│
├── models/
│   ├── trained_models/
│   ├── vectorizers/
│   ├── tokenizers/
│   └── embeddings/
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
│   ├── bilstm.py
│   ├── evaluator.py
│   ├── experiment_tracker.py
│   ├── model_manager.py
│
├── README.md
└── .gitignore
```

---

# Data Preprocessing Pipeline

Every model uses the same preprocessing pipeline to ensure a fair comparison.

The preprocessing pipeline consists of:

1. Merge title and article
2. Remove HTML tags
3. Remove JavaScript
4. Remove CSS
5. Remove comments
6. Remove URLs
7. Remove punctuation
8. Remove numbers
9. Convert text to lowercase
10. Remove stopwords
11. Lemmatize words
12. Create the final `combined_text`

Pipeline:

```
Title
        \
         \
Article -----> Merge

↓

Remove HTML

↓

Remove URLs

↓

Remove punctuation

↓

Remove numbers

↓

Lowercase

↓

Remove stopwords

↓

Lemmatization

↓

combined_text
```

---

# Why combine the title and article?

Several records contained missing or very short article text.

Instead of discarding those samples, the project combines the cleaned title and cleaned article into a single field called `combined_text`.

This approach:

- preserves as much information as possible;
- reduces information loss caused by missing values;
- provides additional context to the classifier.

---

# Why lowercase the text?

Although lowercasing is not always mandatory, especially for modern Transformer models, it was intentionally included to create a robust preprocessing pipeline for classical NLP techniques.

Lowercasing:

- reduces vocabulary size;
- treats "News", "NEWS" and "news" as the same token;
- improves feature consistency.

---

# Feature Engineering

Several feature engineering techniques were evaluated.

## Bag of Words

Represents each document as the frequency of its words.

Advantages:

- very fast
- simple
- interpretable

---

## TF-IDF

Improves Bag of Words by giving more importance to informative words while reducing the influence of very common words.

---

## TF-IDF + Bigrams

Captures two-word expressions in addition to individual words.

Example:

```
fake news

machine learning

white house
```

This allows the model to preserve additional context.

---

## TF-IDF Weighted GloVe Embeddings

Instead of using sparse vectors, this approach represents each article using pretrained GloVe word embeddings.

Each word embedding is weighted using its TF-IDF importance before averaging.

This combines semantic meaning with statistical importance.

---

## Bi-LSTM

The final Deep Learning experiment uses a Bidirectional Long Short-Term Memory network.

Unlike classical Machine Learning, the Bi-LSTM learns sequential relationships between words and captures contextual dependencies within the article.

---

# Models Evaluated

## Bag of Words

- Logistic Regression
- Naive Bayes
- Random Forest
- Linear SVM

---

## TF-IDF

- Logistic Regression
- Naive Bayes
- Random Forest
- Linear SVM

---

## TF-IDF + Bigrams

- Logistic Regression
- Naive Bayes
- Random Forest
- Linear SVM

---

## Word Embeddings

- Logistic Regression
- Random Forest

---

## Deep Learning

- Bi-LSTM

---

## Future Work

- DistilBERT
- RoBERTa

---

# Experiment Tracking

Every experiment stores:

- model
- feature representation
- train accuracy
- test accuracy
- precision
- recall
- F1-score
- accuracy gap
- confusion matrix
- trained model
- vectorizer (when applicable)

All experiments are automatically recorded inside:

```
results/model_tracking.csv
```

---

# Experimental Results

The best experimental model was:

**Bi-LSTM**

Performance:

- Test Accuracy ≈ 99.91%
- Precision ≈ 99.90%
- Recall ≈ 99.93%
- F1-score ≈ 99.91%

---

Other top-performing models included:

- Random Forest + TF-IDF
- Random Forest + TF-IDF + Bigrams
- Random Forest + Word Embeddings

---

# Production Recommendation

The objective of this project was not only to identify the model with the highest predictive performance, but also to evaluate the engineering trade-offs required for deploying a real-world NLP system.

## Experimental Winner

The best-performing model during experimentation was the **Bi-LSTM**.

Performance:

| Metric | Value |
|---------|------:|
| Test Accuracy | **99.9124%** |
| Precision | **99.9000%** |
| Recall | **99.9250%** |
| F1-score | **99.9125%** |
| Train-Test Accuracy Gap | **0.000545** |

The Bi-LSTM achieved the highest overall predictive performance and demonstrated excellent generalization with an extremely small train-test gap.

---

## Production Recommendation

Although the Bi-LSTM achieved the best results, the final production recommendation is:

**Logistic Regression + Bag of Words**

Performance:

| Metric | Value |
|---------|------:|
| Test Accuracy | **99.4868%** |
| Precision | **99.5494%** |
| Recall | **99.4250%** |
| F1-score | **99.4872%** |
| Train-Test Accuracy Gap | **0.005101** |

Although the Bi-LSTM outperformed Logistic Regression, the improvement in F1-score was only:

**0.999125 − 0.994872 = 0.004253**

This represents less than half of one percentage point while requiring substantially greater computational resources.

The Logistic Regression model offers several engineering advantages:

- significantly faster training;
- faster inference;
- lower computational cost;
- lower memory consumption;
- simpler deployment;
- easier maintenance;
- greater explainability;
- easier debugging.

For many real-world applications, these operational advantages outweigh the relatively small improvement in predictive performance provided by the Bi-LSTM.

---

## Final Decision

This project therefore distinguishes between two different objectives.

### Experimental Winner

**Bi-LSTM**

Selected because it achieved the highest predictive performance.

---

### Production Recommendation

**Logistic Regression + Bag of Words**

Selected because it provides the best balance between:

- predictive performance;
- computational efficiency;
- scalability;
- explainability;
- deployment simplicity;
- operational cost.

This decision demonstrates that model selection should consider both predictive performance and engineering constraints rather than relying solely on benchmark metrics.

---

# Team Responsibilities

## Karima

- Design project architecture
- Exploratory Data Analysis (EDA
- Build project structure
- Train  experiments
- Experiment tracking
- Data analysis
- Deployment

---

## Kriti

- Exploratory Data Analysis (EDA
- Build project structure
- Train  experiments
- Experiment tracking
- Data analysis
- Deployment

---

## Together

- Compare all models
- Select best-performing model
- Generate validation predictions
- Build deployment
- Final presentation
- Documentation

---

# Authors

| Name | Role |
|------|------|
| **Karima Mzoughi** | Exploratory Data Analysis (EDA), Project Architecture | Machine Learning, Deployment |
| **Kriti Amin.** | Exploratory Data Analysis (EDA), Machine Learning, Deployment |

