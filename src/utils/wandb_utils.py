import os
from typing import Any, Dict, Optional

import wandb

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def init_wandb(config_dict: Dict[str, Any]) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize a Weights & Biases run if enabled in config."""
    if not config_dict.get("wandb", {}).get("use_wandb", False):
        logger.info("W&B disabled in config")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise EnvironmentError("WANDB_API_KEY not found in environment")

    project = config_dict["wandb"]["project_name"]
    entity = config_dict["wandb"].get("entity")
    run = wandb.init(project=project, entity=entity, config=config_dict)
    logger.info("Initialized W&B run: %s", run.name if run else "none")
    return run


def log_metrics(metrics_dict: Dict[str, Any]) -> None:
    if wandb.run is not None:
        wandb.log(metrics_dict)
        logger.info("Logged metrics to W&B: %s", metrics_dict)


def finish_wandb() -> None:
    if wandb.run is not None:
        wandb.finish()
        logger.info("Finished W&B run")
