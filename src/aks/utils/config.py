"""Load and cache YAML config files."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml


import os

_env_home = os.environ.get("AKS_HOME")
# config/ lives at the project root (4 levels up from this file)
_project_root = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = Path(_env_home) / "config" if _env_home else _project_root / "config"
DATA_DIR = Path(_env_home) if _env_home else Path.home() / ".local" / "share" / "aks"


@functools.lru_cache(maxsize=None)
def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def system_config() -> dict:
    return load_yaml(CONFIG_DIR / "system.yaml")["aks"]


def models_config() -> dict:
    return load_yaml(CONFIG_DIR / "models.yaml")["models"]


def get_fallback_chain() -> list[dict]:
    return load_yaml(CONFIG_DIR / "models.yaml").get("fallback_chain", [])


def get_provider() -> str:
    return load_yaml(CONFIG_DIR / "models.yaml").get("provider", "gemini")


def agent_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / "agents" / f"{name}.yaml")
