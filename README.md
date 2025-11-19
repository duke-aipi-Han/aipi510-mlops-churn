# AIPI510 MLOps Churn Starter

Predict Telco customer churn with an XGBoost model, track experiments in Weights & Biases, serve a FastAPI REST API in Docker/Azure Container Apps, and ship a React frontend to Azure Static Web Apps. Use this starter as a teaching scaffold to explore the full ML lifecycle end-to-end.

## Architecture

```
+--------------------+          +-------------------+
|  Local / Notebooks |          |   Weights & Biases|
|  data cleaning     |          |   metrics & runs  |
+---------+----------+          +---------+---------+
          |                               ^
          v                               |
+---------+----------+           +--------+--------+
| Azure Blob Storage |<----------+  Train pipeline |
| raw/clean/model    |           |  XGBoost + SKL  |
+---------+----------+           +-----------------+
          |                                   |
          v                                   v
   +------+-------+                    +------+-------+
   | FastAPI API  |  <--- React POST --|  Frontend    |
   | (Container   |                    | (Static Web) |
   |  App)        |                    +--------------+
   +--------------+
```

## Dataset

Telco Customer Churn (Kaggle). Download the CSV from Kaggle and place it at `data/raw/telco_churn_raw.csv`. The cleaning script standardizes column names, coerces `TotalCharges` to numeric, drops missing rows, and writes `data/clean/telco_churn_clean.csv` (also uploaded to Azure Blob Storage).

## Quickstart (local)

1. **Setup env**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   cp .env.example .env
   # fill AZURE_BLOB_CONNECTION_STRING and WANDB_API_KEY
   ```
2. **Clean data**
   ```bash
   python src/data_ingestion.py
   ```
3. **Train model**
   ```bash
   python src/train.py
   ```
4. **Run API locally**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```
5. **Use frontend**
   ```bash
   cd frontend && npm install && npm run dev
   ```

## Configuration

See `config.yaml` for paths, split fractions, model hyperparameters, W&B, Azure blob targets, and API host/port. Environment variables (`.env`) carry secrets (`AZURE_BLOB_CONNECTION_STRING`, `WANDB_API_KEY`, `API_ENV`).

## Data pipeline

- Place raw Kaggle CSV at `data/raw/telco_churn_raw.csv`.
- Run `src/data_ingestion.py` to clean and upload `data/clean/telco_churn_clean.csv` to Azure Blob (`ml-data` container by default).

## Training

`src/train.py` loads config, downloads clean data if missing, splits train/val/test based on config fractions, builds a ColumnTransformer (one-hot + scaling), trains an XGBoost classifier, evaluates, logs metrics to W&B, saves artifacts to `models/`, and uploads model/preprocessor to Azure Blob Storage.

## Serving (FastAPI)

`src/api.py` loads artifacts from Azure on startup and exposes:
- `GET /` health check
- `POST /predict` accepting a JSON body defined in `src/schemas.py`, returning churn probability and label.

## Docker

Build and run locally:
```bash
docker build -t aipi510-mlops-churn .
docker run --env-file .env -p 8000:8000 aipi510-mlops-churn
```

## Deployment (Azure)

### Azure Blob Storage
- Create storage account + container (`ml-data` for data, `ml-models` for artifacts).
- Set connection string in `AZURE_BLOB_CONNECTION_STRING`.
- `src/utils/azure_utils.py` handles uploads/downloads.

### Azure Container Registry (ACR)
```bash
az acr login --name <registry>
docker tag aipi510-mlops-churn <registry>.azurecr.io/aipi510-mlops-churn:latest
docker push <registry>.azurecr.io/aipi510-mlops-churn:latest
```

### Azure Container Apps
- Create a Container App pointing to the ACR image.
- Set environment variables (connection string, WANDB key, API_ENV).
- Expose port 8000.

### Azure Static Web Apps
- Deploy `frontend/` build output (`dist/`).
- Build command: `npm run build`. Output folder: `dist`.
- Configure `VITE_API_BASE_URL` to point to the Container App URL.

## W&B usage

- Toggle via `wandb.use_wandb` in `config.yaml`.
- Set `WANDB_API_KEY` in `.env`.
- Training logs metrics (`train/val/test` accuracy, ROC-AUC, log-loss) and config automatically.

## Ethical considerations

Churn models can reinforce unfair treatment (e.g., aggressive retention offers only to certain demographics). Monitor feature importance, audit bias across groups, and avoid using protected attributes or proxies in decisioning without policy review. Communicate limitations and ensure opt-out mechanisms where appropriate.

## Branching / PR guide

- Create feature branches from `main`.
- Add focused commits; keep PRs small.
- Include brief description, testing notes, and any config changes.
- Avoid committing credentials or data; keep `.env` local.

## Project layout

- `src/`: ML pipeline, FastAPI service, utilities
- `models/`: saved model + preprocessor (after training)
- `data/raw`, `data/clean`: local data storage
- `notebooks/`: exploratory notebooks
- `frontend/`: Vite React client
- `Dockerfile`, `config.yaml`, `.env.example`, `requirements.txt`
