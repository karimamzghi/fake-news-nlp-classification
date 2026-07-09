from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "data.csv"
VALIDATION_DATA_PATH = BASE_DIR / "dataset" / "validation_data.csv"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
PLOT_DIR = RESULTS_DIR / "plots"
CONFUSION_MAT_DIR = RESULTS_DIR / "confusion_matrices"
MODEL_TRACKING_DIR = RESULTS_DIR / "model_tracking.csv"

TEXT_COLUMNS = ["title", "text"]
DROP_COLUMNS = ["subject", "date"]
