import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, DROP_COLUMNS


def load_data(path=DATA_PATH):
    data = pd.read_csv(path)
    return data


def prepare_features_and_labels(data):
    data = data.copy()

    data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    X = data.drop(columns="label")
    y = data["label"]

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
