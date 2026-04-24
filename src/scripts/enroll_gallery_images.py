"""Enroll gallery images into DB + FAISS using DINO embeddings."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _list_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(images)


def _read_image(path: Path):  # noqa: ANN201
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image

        arr = np.array(Image.open(path))
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3][:, :, ::-1]
    except Exception:
        pass
    return None


async def _enroll_gallery(
    gallery_dir: Path,
    config: dict,
    faiss_path: str | None,
    dry_run: bool,
) -> dict:
    from sqlalchemy import select

    from core.config import resolve_path
    from core.database.models import Animal, EmbeddingRef
    from core.database.session import async_session_factory
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV3_SMALL
    from src.ai.reid.faiss_store import FAISSStore

    emb_cfg = (config.get("models", {}) or {}).get("embedding", {}) or {}
    model_name = str(emb_cfg.get("model_name", DINOV3_SMALL))
    device = emb_cfg.get("device")
    half = bool(emb_cfg.get("half", True))
    encoder = CattleEmbeddingEncoder(model_name=model_name, device=device, half=half)

    resolved_faiss = faiss_path or str((Path(__file__).resolve().parent.parent.parent / "core" / "data" / "faiss_index"))
    store = FAISSStore(index_path=resolved_faiss)
    if Path(resolved_faiss).exists():
        store.load()

    total_animals = 0
    total_images = 0
    total_embeddings = 0

    folders = [d for d in sorted(gallery_dir.iterdir()) if d.is_dir()]
    if not folders:
        return {"animals": 0, "images": 0, "embeddings": 0, "skipped": True}

    async with async_session_factory() as session:
        for animal_dir in folders:
            external_id = animal_dir.name.strip()
            if not external_id:
                continue
            image_paths = _list_images(animal_dir)
            if not image_paths:
                continue

            result = await session.execute(select(Animal).where(Animal.external_id == external_id))
            animal = result.scalar_one_or_none()
            if animal is None and not dry_run:
                animal = Animal(
                    external_id=external_id,
                    name=external_id,
                    metadata_={"enrollment": "gallery_images", "source_dir": str(animal_dir)},
                )
                session.add(animal)
                await session.flush()
            elif animal is None and dry_run:
                # Simula ID sem persistir.
                class _DryAnimal:
                    id = -1

                animal = _DryAnimal()

            crops = []
            for image_path in image_paths:
                img = _read_image(image_path)
                if img is not None:
                    crops.append(img)

            if not crops:
                continue

            embeddings = encoder.encode_batch(crops)
            if embeddings.size == 0:
                continue

            if not dry_run:
                animal_ids = [int(animal.id)] * len(embeddings)
                store.add_batch(animal_ids, embeddings)
                for _ in animal_ids:
                    session.add(EmbeddingRef(animal_id=int(animal.id)))

            total_animals += 1
            total_images += len(crops)
            total_embeddings += int(embeddings.shape[0])

        if not dry_run:
            await session.commit()
            store.save()

    return {
        "animals": total_animals,
        "images": total_images,
        "embeddings": total_embeddings,
        "skipped": False,
        "model_name": model_name,
    }


def main() -> None:
    from core.config import load_repo_dotenv

    load_repo_dotenv()

    parser = argparse.ArgumentParser(description="Enroll gallery images for cattle Re-ID")
    parser.add_argument("--gallery_dir", type=str, default="gallery_images")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--faiss", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery_dir)
    if not gallery_dir.exists():
        print(f"Galeria nao encontrada: {gallery_dir}")
        return

    config = _load_config(args.config)
    stats = asyncio.run(_enroll_gallery(gallery_dir=gallery_dir, config=config, faiss_path=args.faiss, dry_run=args.dry_run))
    if stats.get("skipped"):
        print(f"Nenhuma subpasta encontrada em {gallery_dir}.")
        return
    print(
        "Enrollment concluido: "
        f"animais={stats['animals']}, imagens={stats['images']}, embeddings={stats['embeddings']}, "
        f"modelo={stats.get('model_name')}"
    )


if __name__ == "__main__":
    main()
