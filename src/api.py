from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBClassifier

from src.config import load_config
from src.schemas import ChurnRequest, ChurnResponse
from src.preprocessing import apply_preprocessor
from src.utils.azure_utils import download_blob_to_file
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Telco Churn Prediction API")

model: Optional[XGBClassifier] = None
preprocessor = None
config = load_config()


def _load_model_artifacts() -> None:
    global model, preprocessor
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / config["azure"]["model_blob_name"]
    preprocessor_path = models_dir / config["azure"]["preprocessor_blob_name"]

    if not model_path.exists():
        logger.info("Downloading model from Azure")
        download_blob_to_file(config["azure"]["model_container"], model_path.name, model_path)
    if not preprocessor_path.exists():
        logger.info("Downloading preprocessor from Azure")
        download_blob_to_file(
            config["azure"]["model_container"], preprocessor_path.name, preprocessor_path
        )

    xgb_model = XGBClassifier()
    xgb_model.load_model(model_path)
    model = xgb_model
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Artifacts loaded into memory")


@app.on_event("startup")
def startup_event():
    _load_model_artifacts()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict_churn(payload: ChurnRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    data = pd.DataFrame([payload.model_dump()])

    try:
        transformed = apply_preprocessor(preprocessor, data)
        proba = model.predict_proba(transformed)[:, 1][0]
        label = bool(proba >= 0.5)
    except Exception as exc:  # pragma: no cover - minimal starter
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return ChurnResponse(churn_probability=float(proba), churn_label=label)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
