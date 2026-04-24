"""Cliente MinIO: upload e URLs para persistência no banco."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minio import Minio

from core.settings import MinioSettings


def _endpoint_host_port(settings: MinioSettings) -> str:
    host = settings.minio_endpoint.strip()
    if ":" in host:
        return host
    return f"{host}:{settings.minio_port}"


class MinioStorage:
    def __init__(self, client: Minio, settings: MinioSettings) -> None:
        self._client = client
        self._settings = settings

    @property
    def bucket(self) -> str:
        return self._settings.minio_bucket

    @property
    def settings(self) -> MinioSettings:
        return self._settings

    @classmethod
    def from_settings(cls, settings: MinioSettings | None = None) -> MinioStorage | None:
        if settings is None:
            settings = MinioSettings()
        if not settings.minio_enabled or not settings.minio_bucket:
            return None
        from minio import Minio

        client = Minio(
            _endpoint_host_port(settings),
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )
        return cls(client, settings)

    def ensure_bucket(self) -> None:
        from minio.error import S3Error

        name = self._settings.minio_bucket
        try:
            if not self._client.bucket_exists(name):
                self._client.make_bucket(name)
        except S3Error:
            raise

    def put_bytes(self, object_key: str, data: bytes, content_type: str) -> str:
        """Envia objeto e devolve URL HTTP (path-style) para gravar em `source_path`."""
        self._client.put_object(
            self._settings.minio_bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return self.public_url(object_key)

    def public_url(self, object_key: str) -> str:
        base = self._settings.minio_base_url.rstrip("/")
        b = self._settings.minio_bucket
        key = object_key.lstrip("/")
        return f"{base}/{b}/{key}"


def get_minio_storage() -> MinioStorage | None:
    """Instância a partir do `.env` (ou defaults). Retorna None se MinIO desligado."""
    return MinioStorage.from_settings(MinioSettings())
