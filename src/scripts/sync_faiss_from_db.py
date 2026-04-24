"""Rebuild FAISS gallery from animal images stored in database."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np


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


def _read_image(path: Path):  # noqa: ANN201
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return None


async def _sync_from_db(config: dict, faiss_path: str | None, limit_per_animal: int) -> dict:
    from sqlalchemy import select

    from core.database.models import AnimalCrop
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

    valid_images = 0
    per_animal_count: dict[int, int] = {}
    animal_ids: list[int] = []
    embeddings: list[np.ndarray] = []

    async with async_session_factory() as session:
        rows = await session.execute(
            select(AnimalCrop.animal_id, AnimalCrop.source_path, AnimalCrop.created_at).order_by(AnimalCrop.created_at.desc())
        )
        for animal_id, source_path, _created_at in rows.all():
            animal_id = int(animal_id)
            if per_animal_count.get(animal_id, 0) >= limit_per_animal:
                continue
            if not source_path:
                continue
            path = Path(str(source_path))
            if not path.exists():
                continue
            img = _read_image(path)
            if img is None:
                continue
            emb = encoder.encode_one(img)
            if emb.size == 0:
                continue
            animal_ids.append(animal_id)
            embeddings.append(emb)
            per_animal_count[animal_id] = per_animal_count.get(animal_id, 0) + 1
            valid_images += 1

    if embeddings:
        emb_arr = np.stack(embeddings, axis=0)
        store.rebuild_index(animal_ids=animal_ids, embeddings=emb_arr)
        store.save()
    else:
        # cria diretório/meta vazio se necessário
        Path(resolved_faiss).mkdir(parents=True, exist_ok=True)

    return {
        "animals": len(per_animal_count),
        "images": valid_images,
        "embeddings": len(embeddings),
        "faiss_path": resolved_faiss,
        "model_name": model_name,
    }


def main() -> None:
    from core.config import load_repo_dotenv

    load_repo_dotenv()

    parser = argparse.ArgumentParser(description="Sync FAISS gallery from DB animal crops")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--faiss", type=str, default=None)
    parser.add_argument("--limit_per_animal", type=int, default=30)
    args = parser.parse_args()

    config = _load_config(args.config)
    stats = asyncio.run(_sync_from_db(config=config, faiss_path=args.faiss, limit_per_animal=max(1, args.limit_per_animal)))
    print(
        "FAISS sincronizado do banco: "
        f"animais={stats['animals']}, imagens={stats['images']}, embeddings={stats['embeddings']}, "
        f"modelo={stats['model_name']}, faiss={stats['faiss_path']}"
    )


if __name__ == "__main__":
    main()
