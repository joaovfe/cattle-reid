"""Configuração via variáveis de ambiente (MinIO, etc.)."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.config import repo_root

_REPO_DOTENV = repo_root() / ".env"
_settings_kwargs: dict[str, Any] = {
    "env_file_encoding": "utf-8",
    "extra": "ignore",
    "populate_by_name": True,
}
if _REPO_DOTENV.is_file():
    _settings_kwargs["env_file"] = str(_REPO_DOTENV)


class MinioSettings(BaseSettings):
    """Cliente S3-compatible apontando para o MinIO da API."""

    model_config = SettingsConfigDict(**_settings_kwargs)

    minio_enabled: bool = Field(default=True, alias="MINIO_ENABLED")
    minio_endpoint: str = Field(default="localhost", alias="MINIO_ENDPOINT")
    minio_port: int = Field(default=6300, alias="MINIO_PORT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin123", alias="MINIO_SECRET_KEY")
    minio_use_ssl: bool = Field(default=False, alias="MINIO_USE_SSL")
    minio_bucket: str = Field(default="images", alias="MINIO_BUCKET")
    # Opcional: mesmo esquema do cattle-api (MINIO_BUCKET_VIDEOS, etc.); quando vazio usa MINIO_BUCKET
    minio_bucket_videos: str | None = Field(default=None, alias="MINIO_BUCKET_VIDEOS")
    minio_bucket_thumbnails: str | None = Field(default=None, alias="MINIO_BUCKET_THUMBNAILS")
    minio_bucket_results: str | None = Field(default=None, alias="MINIO_BUCKET_RESULTS")
    minio_bucket_reports: str | None = Field(default=None, alias="MINIO_BUCKET_REPORTS")
    minio_bucket_crops: str | None = Field(default=None, alias="MINIO_BUCKET_CROPS")
    # URL pública para links (mapear porta do host, ex.: http://localhost:6300 — não use 9000 no host)
    minio_base_url: str = Field(default="http://localhost:6300", alias="MINIO_BASE_URL")

    def primary_bucket(self) -> str:
        return self.minio_bucket.strip()

    def videos_bucket(self) -> str:
        b = (self.minio_bucket_videos or "").strip()
        return b or self.primary_bucket()

    def crops_bucket(self) -> str:
        b = (self.minio_bucket_crops or "").strip()
        return b or self.primary_bucket()

    def results_bucket(self) -> str:
        b = (self.minio_bucket_results or "").strip()
        return b or self.primary_bucket()
