from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
import numpy as np


def create_bow_features(X_train, X_test, column="combined_text"):
    vectorizer = CountVectorizer()

    X_train_bow = vectorizer.fit_transform(X_train[column])
    X_test_bow = vectorizer.transform(X_test[column])

    return X_train_bow, X_test_bow, vectorizer



def load_glove_model():
    # Load pre-trained GloVe embeddings from gensim.
    return api.load("glove-wiki-gigaword-100")


def create_glove_tfidf_weighted_embedding_features(
    X_train,
    X_test,
    glove_model,
    column="combined_text",
    vector_size=100,
    max_features=10000
):
    # Fast TF-IDF weighted GloVe document embeddings.
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[column])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[column])

    feature_names = tfidf_vectorizer.get_feature_names_out()

    embedding_matrix = np.zeros((len(feature_names), vector_size))

    for index, word in enumerate(feature_names):
        if word in glove_model:
            embedding_matrix[index] = glove_model[word]

    X_train_embeddings = X_train_tfidf.dot(embedding_matrix)
    X_test_embeddings = X_test_tfidf.dot(embedding_matrix)

    return X_train_embeddings, X_test_embeddings, tfidf_vectorizer

def create_tfidf_features(
    X_train,
    X_test,
    column="combined_text",
    max_features=10000
):
    # The vectorizer learns its vocabulary only from the training data.
    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train[column])
    X_test_tfidf = vectorizer.transform(X_test[column])

    return X_train_tfidf, X_test_tfidf, vectorizer


def create_tfidf_bigram_features(
    X_train,
    X_test,
    column="combined_text",
    max_features=10000
):
    #  ngram_range=(1, 2) means: individual words (unigrams) , two-word combinations (bigrams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train[column])
    X_test_tfidf = vectorizer.transform(X_test[column])

    return X_train_tfidf, X_test_tfidf, vectorizer
