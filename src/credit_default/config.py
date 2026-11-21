from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FINDINGS_DIR = PROJECT_ROOT / "reports" / "findings"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "default_of_credit_card_clients.xls"
PROCESSED_FILE = PROCESSED_DIR / "credit_default_clean.csv"