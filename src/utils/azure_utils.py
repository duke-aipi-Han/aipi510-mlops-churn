import os
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient

from src.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_blob_service_client() -> BlobServiceClient:
    config = load_config()
    env_var = config["azure"]["blob_connection_string_env_var"]
    connection_string = os.getenv(env_var)
    if not connection_string:
        raise EnvironmentError(f"Azure blob connection string not found in env var {env_var}")
    return BlobServiceClient.from_connection_string(connection_string)


def download_blob_to_file(container: str, blob_name: str, local_path: str | Path) -> Path:
    """Download a blob to a local file path."""
    local_path = Path(local_path)
    client = _get_blob_service_client()
    blob_client = client.get_container_client(container).get_blob_client(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())
    logger.info("Downloaded %s to %s", blob_name, local_path)
    return local_path


def upload_file_to_blob(container: str, blob_name: str, local_path: str | Path) -> None:
    """Upload a local file to Azure Blob Storage."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path {local_path} not found for upload")

    client = _get_blob_service_client()
    blob_client = client.get_container_client(container).get_blob_client(blob_name)
    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    logger.info("Uploaded %s to container %s", local_path, container)
