from __future__ import annotations

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

