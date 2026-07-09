import os
import joblib

from src.config import MODELS_DIR


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def save_vectorizer(vectorizer, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    vectorizer_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    return vectorizer_path