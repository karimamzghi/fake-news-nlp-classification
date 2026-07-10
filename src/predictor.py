from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import preprocess_data


def predict_validation_data(
    validation_path,
    model_path,
    vectorizer_path,
    output_path="results/predictions/validation_predictions.csv"
):
    validation_data = pd.read_csv(validation_path)

    validation_clean = preprocess_data(validation_data)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    validation_features = vectorizer.transform(
        validation_clean["combined_text"]
    )

    predictions = model.predict(validation_features)

    validation_predictions = validation_data.copy()
    validation_predictions["label"] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(
        output_path,
        index=False
    )

    return validation_predictions
