from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "Combined_News_DJIA.csv"
DB_PATH = ROOT / "data" / "marketmind.db"
REPORTS = ROOT / "reports"
DATA_URL = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Combined_News_DJIA.csv"
SEED = 7
