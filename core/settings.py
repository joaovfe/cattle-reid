"""Configuração via variáveis de ambiente (MinIO, etc.)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinioSettings(BaseSettings):
    """Cliente S3-compatible apontando para o MinIO da API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    minio_enabled: bool = Field(default=True, alias="MINIO_ENABLED")
    minio_endpoint: str = Field(default="localhost", alias="MINIO_ENDPOINT")
    minio_port: int = Field(default=6300, alias="MINIO_PORT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin123", alias="MINIO_SECRET_KEY")
    minio_use_ssl: bool = Field(default=False, alias="MINIO_USE_SSL")
    minio_bucket: str = Field(default="images", alias="MINIO_BUCKET")
    # URL pública para links (mapear porta do host, ex.: http://localhost:6300 — não use 9000 no host)
    minio_base_url: str = Field(default="http://localhost:6300", alias="MINIO_BASE_URL")
