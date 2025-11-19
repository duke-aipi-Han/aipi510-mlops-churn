from pathlib import Path

import pandas as pd

from src.config import load_config
from src.utils.azure_utils import upload_file_to_blob
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def clean_telco_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning for Telco Customer Churn dataset."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df = df.dropna()
    return df


def main() -> Path:
    config = load_config()
    raw_path = Path(config["data"]["local_raw_path"])
    clean_path = Path(config["data"]["local_clean_path"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Download it from Kaggle and place it there."
        )

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    df_clean = clean_telco_dataset(df)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    logger.info("Saved cleaned data to %s", clean_path)

    upload_file_to_blob(
        container=config["data"]["azure_container"],
        blob_name=config["data"]["azure_clean_blob_name"],
        local_path=clean_path,
    )
    logger.info("Uploaded cleaned data to Azure Blob")
    return clean_path


if __name__ == "__main__":
    main()
