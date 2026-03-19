from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """
    Load YAML config.

    - If config_path is relative, it is resolved from repository root.
    - Returns {} on missing/parse errors (keeps API resilient).
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve_path(maybe_path: str | None) -> str | None:
    """Resolve a (possibly relative) path from repository root."""
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)

