"""
Simple reusable helpers for the FakeBusters Bi-LSTM experiment.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def create_lstm_features(
    X_train,
    X_test,
    column="combined_text",
    max_words=20000,
    max_length=300
):
    """
    Learn a vocabulary from training text and convert documents to
    padded integer sequences.
    """
    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train[column])

    X_train_sequences = tokenizer.texts_to_sequences(X_train[column])
    X_test_sequences = tokenizer.texts_to_sequences(X_test[column])

    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    return X_train_padded, X_test_padded, tokenizer


def build_bilstm_model(
    max_words=20000,
    embedding_dim=100,
    max_length=300
):
    """
    Build a small binary-classification Bi-LSTM model.
    """
    model = Sequential([
        Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            input_length=max_length
        ),
        Bidirectional(
            LSTM(64)
        ),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
