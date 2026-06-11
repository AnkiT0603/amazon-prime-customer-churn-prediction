from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "amazon_prime_churn_preprocessed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "amazon_prime_churn_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATABASE_PATH = PROJECT_ROOT / "data" / "amazon_prime_churn.db"

TARGET_COLUMN = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5
