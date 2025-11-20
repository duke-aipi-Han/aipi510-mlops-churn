from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.config import load_config
from src.preprocessing import apply_preprocessor, TARGET_COL
from src.utils.azure_utils import download_blob_to_file
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _prepare(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not present for evaluation")
    y = df[TARGET_COL].apply(lambda x: 1 if str(x).lower() in {"yes", "true", "1"} else 0)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def evaluate(model_path: Path, preprocessor_path: Path, df: pd.DataFrame) -> Dict[str, float]:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    X, y = _prepare(df)
    X_proc = apply_preprocessor(preprocessor, X)
    proba = model.predict_proba(X_proc)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": log_loss(y, proba),
    }
    return metrics


def main() -> Dict[str, float]:
    config = load_config()
    clean_path = Path(config["data"]["local_clean_path"])
    if not clean_path.exists():
        download_blob_to_file(
            container=config["data"]["azure_container"],
            blob_name=config["data"]["azure_clean_blob_name"],
            local_path=clean_path,
        )

    model_path = Path("models") / config["azure"]["model_blob_name"]
    preprocessor_path = Path("models") / config["azure"]["preprocessor_blob_name"]

    if not model_path.exists():
        download_blob_to_file(config["azure"]["model_container"], model_path.name, model_path)
    if not preprocessor_path.exists():
        download_blob_to_file(
            config["azure"]["model_container"], preprocessor_path.name, preprocessor_path
        )

    df = pd.read_csv(clean_path)
    metrics = evaluate(model_path, preprocessor_path, df)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
