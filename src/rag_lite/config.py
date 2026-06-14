from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_dirs(config: dict) -> None:
    paths = config.get("paths", {})

    for key in ("dense_index_dir", "bm25_index", "logs_dir", "outputs_dir"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    data = config.get("data", {})
    for key in ("processed_path",):
        if key in data:
            Path(data[key]).parent.mkdir(parents=True, exist_ok=True)
