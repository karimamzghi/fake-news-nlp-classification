import os
import pandas as pd

from src.config import MODEL_TRACKING_DIR

def save_experiment_result(
    model_id,
    model_name,
    features,
    preprocessing,
    algorithm,
    train_accuracy,
    test_accuracy,
    precision,
    recall,
    f1_score,
    notes,
    model_path,
    csv_path= MODEL_TRACKING_DIR
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    experiment = {
        "model_id": model_id,
        "model_name": model_name,
        "features": features,
        "preprocessing": preprocessing,
        "algorithm": algorithm,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "notes": notes,
        "model_path": model_path
    }

    if os.path.exists(csv_path):
        tracking_df = pd.read_csv(csv_path)
        tracking_df = pd.concat(
            [tracking_df, pd.DataFrame([experiment])],
            ignore_index=True
        )
    else:
        tracking_df = pd.DataFrame([experiment])

    tracking_df.to_csv(csv_path, index=False)

    return tracking_df
