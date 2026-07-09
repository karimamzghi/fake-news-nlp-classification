from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_bow_features(X_train, X_test, column="combined_text"):
    vectorizer = CountVectorizer()

    X_train_bow = vectorizer.fit_transform(X_train[column])
    X_test_bow = vectorizer.transform(X_test[column])

    return X_train_bow, X_test_bow, vectorizer
