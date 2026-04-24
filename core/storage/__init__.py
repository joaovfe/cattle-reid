"""Armazenamento de objetos (MinIO / S3)."""

from core.storage.minio_storage import MinioStorage, get_minio_storage

__all__ = ["MinioStorage", "get_minio_storage"]
