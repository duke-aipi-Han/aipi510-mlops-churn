from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import load_config
from src.preprocessing import build_preprocessor, apply_preprocessor, TARGET_COL
from src.utils.azure_utils import download_blob_to_file, upload_file_to_blob
from src.utils.logging_utils import get_logger
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics

logger = get_logger(__name__)


def _prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")
    # Convert churn labels to binary 0/1
    y = df[TARGET_COL].apply(lambda x: 1 if str(x).lower() in {"yes", "true", "1"} else 0)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def _split_data(X: pd.DataFrame, y: pd.Series, split_cfg: Dict) -> tuple:
    test_size = split_cfg["test_fraction"]
    val_fraction = split_cfg["val_fraction"]
    random_state = split_cfg.get("random_state", 42)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_fraction / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _evaluate(model: XGBClassifier, X, y) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
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
        logger.info("Clean data %s not found; downloading from Azure", clean_path)
        download_blob_to_file(
            container=config["data"]["azure_container"],
            blob_name=config["data"]["azure_clean_blob_name"],
            local_path=clean_path,
        )

    df = pd.read_csv(clean_path)
    X, y = _prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y, config["split"])

    preprocessor = build_preprocessor(pd.concat([X_train, y_train], axis=1))
    preprocessor.fit(X_train)

    X_train_processed = apply_preprocessor(preprocessor, X_train)
    X_val_processed = apply_preprocessor(preprocessor, X_val)
    X_test_processed = apply_preprocessor(preprocessor, X_test)

    model_params = config["model"]["params"]
    model = XGBClassifier(**model_params, n_jobs=4)
    model.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)

    metrics = {
        "train": _evaluate(model, X_train_processed, y_train),
        "val": _evaluate(model, X_val_processed, y_val),
        "test": _evaluate(model, X_test_processed, y_test),
    }

    run = init_wandb(config)
    flat_metrics = {f"{split}_{k}": v for split, res in metrics.items() for k, v in res.items()}
    log_metrics(flat_metrics)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / config["azure"]["model_blob_name"]
    preprocessor_path = models_dir / config["azure"]["preprocessor_blob_name"]

    model.save_model(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Saved model to %s and preprocessor to %s", model_path, preprocessor_path)

    upload_file_to_blob(config["azure"]["model_container"], model_path.name, model_path)
    upload_file_to_blob(config["azure"]["model_container"], preprocessor_path.name, preprocessor_path)
    logger.info("Uploaded artifacts to Azure Blob Storage")

    finish_wandb()
    return flat_metrics


if __name__ == "__main__":
    results = main()
    logger.info("Training complete. Metrics: %s", results)
