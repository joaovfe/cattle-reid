from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_repo_dotenv() -> None:
    """
    Carrega `<repo>/.env` para os.environ.

    O Hugging Face Hub e outras libs leem HF_TOKEN só do ambiente do processo;
    ter o token no .env não basta sem este passo (a não ser export manual no shell).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = repo_root() / ".env"
    if path.is_file():
        # overwrite=True para o `.env` do repositório vencer variáveis antigas exportadas na shell
        # (ex.: MINIO_PORT=9000 de Docker Compose global vs MINIO_PORT=6300 para S3 na máquina host).
        load_dotenv(path, override=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Override wins on scalar conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _detect_profile() -> str:
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_profile_overlay(profile: str) -> dict[str, Any]:
    path = repo_root() / "configs" / "profiles" / f"{profile}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """
    Load YAML config and merge hardware profile on top.

    Profile selection (in priority order):
      1. NOVUS_PROFILE env var ('gpu' or 'cpu')
      2. Auto-detect via torch.cuda.is_available()

    Profile files live in configs/profiles/{profile}.yaml and only contain
    hardware overrides (device, half, imgsz, batch_size, frame_skip, model_name).
    All functional params (reid thresholds, tracking, detection) stay in default.yaml.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            base = yaml.safe_load(f) or {}
    except Exception:
        return {}

    profile_name = os.environ.get("NOVUS_PROFILE", "").strip().lower()
    if not profile_name:
        profile_name = _detect_profile()

    overlay = _load_profile_overlay(profile_name)
    if overlay:
        base = _deep_merge(base, overlay)
        print(f"[novus] profile={profile_name}", flush=True)

    return base


def resolve_path(maybe_path: str | None) -> str | None:
    """Resolve a (possibly relative) path from repository root."""
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)

