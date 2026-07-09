import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOT_DIR, CONFUSION_MAT_DIR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }

    print(classification_report(y_test, predictions))

    return metrics, predictions


def save_metrics_plot(metrics, experiment_name, save_dir=PLOT_DIR):

    os.makedirs(save_dir, exist_ok=True)

    metric_names = [
        "test_accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    metric_values = [
        metrics["test_accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"]
    ]

    plt.figure(figsize=(7,5))

    plt.bar(metric_names, metric_values)

    plt.ylim(0,1)

    plt.ylabel("Score")

    plt.title(experiment_name)

    for i, value in enumerate(metric_values):
        plt.text(
            i,
            value + 0.01,
            f"{value:.3f}",
            ha="center"
        )

    file_path = f"{save_dir}/{experiment_name}_metrics.png"

    plt.savefig(file_path, bbox_inches="tight")

    plt.show()
    
    return file_path


def save_confusion_matrix(y_test, predictions, model_id, save_dir=CONFUSION_MAT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )

    plt.title(f"Confusion Matrix - {model_id}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    file_path = f"{save_dir}/{model_id}_confusion_matrix.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.show()

    return file_path
