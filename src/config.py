import os
import yaml


def load_config(path: str | None = None) -> dict:
    """Load YAML config. Resolution order: explicit path > MULTILQA_CONFIG env var > config.yaml."""
    if path is None:
        path = os.environ.get("MULTILQA_CONFIG", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
