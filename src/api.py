from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier

from src.config import load_config
from src.schemas import ChurnRequest, ChurnResponse
from src.preprocessing import apply_preprocessor
from src.utils.azure_utils import download_blob_to_file
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Telco Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[RandomForestClassifier] = None
preprocessor = None
config = load_config()

REQUEST_TO_TRAINING_COLUMNS = {
    "senior_citizen": "seniorcitizen",
    "phone_service": "phoneservice",
    "multiple_lines": "multiplelines",
    "internet_service": "internetservice",
    "online_security": "onlinesecurity",
    "online_backup": "onlinebackup",
    "device_protection": "deviceprotection",
    "tech_support": "techsupport",
    "streaming_tv": "streamingtv",
    "streaming_movies": "streamingmovies",
    "paperless_billing": "paperlessbilling",
    "payment_method": "paymentmethod",
    "monthly_charges": "monthlycharges",
    "total_charges": "totalcharges",
}

CUSTOMER_ID_PLACEHOLDER = "local-client"

"""
Load model and preprocessor from Azure Blob Storage if not present locally.
"""
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

    model = joblib.load(model_path)
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
    # Align request columns with the training/preprocessor feature names.
    data = data.rename(columns=REQUEST_TO_TRAINING_COLUMNS)
    if "customerid" not in data.columns:
        # Training pipeline retained customerid, so supply a placeholder that will be ignored
        # by the OneHotEncoder (handle_unknown="ignore").
        data["customerid"] = CUSTOMER_ID_PLACEHOLDER

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
