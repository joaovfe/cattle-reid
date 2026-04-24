"""Enrollment: POST /animals, POST /animals/{id}/enroll, GET /animals/{id}."""
from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class AnimalCreate(BaseModel):
    external_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] | None = None


class AnimalResponse(BaseModel):
    id: int
    external_id: str | None
    name: str | None
    metadata: dict[str, Any] | None


@router.post("", response_model=AnimalResponse)
async def create_animal(body: AnimalCreate, request: Request):
    from core.database.session import async_session_factory
    from core.database.models import Animal

    async with async_session_factory() as session:
        animal = Animal(external_id=body.external_id, name=body.name, metadata_=body.metadata)
        session.add(animal)
        await session.commit()
        await session.refresh(animal)
        return AnimalResponse(id=animal.id, external_id=animal.external_id, name=animal.name, metadata=animal.metadata_)


@router.post("/{animal_id}/enroll")
async def enroll_animal(animal_id: int, request: Request, files: list[UploadFile] = File(...)):
    store = request.app.state.faiss_store
    minio_storage = getattr(request.app.state, "minio", None)
    enroll_uid = str(uuid.uuid4())
    crops: list[np.ndarray] = []
    for idx, f in enumerate(files):
        content = await f.read()
        try:
            import cv2
            arr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                crops.append(img)
                if minio_storage is not None:
                    key = f"enrollment/{animal_id}/{enroll_uid}_{idx}.jpg"
                    await asyncio.to_thread(minio_storage.put_bytes, key, content, "image/jpeg")
        except Exception:
            try:
                from PIL import Image
                arr_rgb = np.array(Image.open(io.BytesIO(content)))
                crops.append(arr_rgb[:, :, ::-1])
                if minio_storage is not None:
                    key = f"enrollment/{animal_id}/{enroll_uid}_{idx}.jpg"
                    await asyncio.to_thread(minio_storage.put_bytes, key, content, "image/jpeg")
            except Exception:
                pass
    if not crops:
        raise HTTPException(status_code=400, detail="No valid images")
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder
    from src.ai.oriented_crop.cropper import OrientedCropper
    cropper = OrientedCropper(output_size=224, padding=0.1)
    encoder = CattleEmbeddingEncoder()
    embeddings_list = [encoder.encode_one(cropper.crop(img, [0, 0, img.shape[1], img.shape[0]])) for img in crops]
    embeddings = np.stack(embeddings_list, axis=0)
    store.add_batch([animal_id] * len(embeddings), embeddings)
    store.save()
    from core.database.session import async_session_factory
    from core.database.models import EmbeddingRef
    async with async_session_factory() as session:
        for _ in embeddings_list:
            session.add(EmbeddingRef(animal_id=animal_id))
        await session.commit()
    return {"enrolled": len(embeddings_list), "animal_id": animal_id}


@router.get("/{animal_id}", response_model=AnimalResponse)
async def get_animal(animal_id: int, request: Request):
    from core.database.session import async_session_factory
    from core.database.models import Animal
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(select(Animal).where(Animal.id == animal_id))
        animal = result.scalar_one_or_none()
        if animal is None:
            raise HTTPException(status_code=404, detail="Animal not found")
        return AnimalResponse(id=animal.id, external_id=animal.external_id, name=animal.name, metadata=animal.metadata_)
