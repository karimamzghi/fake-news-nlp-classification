from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "data.csv"
VALIDATION_DATA_PATH = BASE_DIR / "dataset" / "validation_data.csv"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

TEXT_COLUMNS = ["title", "text"]
DROP_COLUMNS = ["subject", "date"]
