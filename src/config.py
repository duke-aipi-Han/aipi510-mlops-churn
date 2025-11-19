import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}
