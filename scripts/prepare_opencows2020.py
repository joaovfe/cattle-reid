#!/usr/bin/env python3
"""
Prepara o dataset OpenCows2020 para treinamento de Re-ID.

O OpenCows2020 vem com a seguinte estrutura:
  OpenCows2020/
    identification/
      images/
        {cow_id}/
          *.jpg
      train.txt   ← lista de imagens de treino
      test.txt    ← lista de imagens de teste

Este script reorganiza para o formato ImageFolder esperado por train_reid.py:
  opencows2020_reid/
    train/
      {cow_id}/
        *.jpg
    val/
      {cow_id}/
        *.jpg

Uso:
  uv run python scripts/prepare_opencows2020.py \
      --src /caminho/OpenCows2020 \
      --dst data/opencows2020_reid
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara OpenCows2020 para treino Re-ID")
    p.add_argument(
        "--src",
        default="MetricLearningIdentification/datasets/OpenCows2020",
        help="Raiz do OpenCows2020 (default: já presente no projeto)",
    )
    p.add_argument("--dst", default="data/opencows2020_reid", help="Destino do dataset organizado")
    p.add_argument("--val_split", type=float, default=0.2, help="Fração de imagens para validação (0–1)")
    p.add_argument("--min_images", type=int, default=4, help="Mínimo de imagens por identidade para incluir")
    return p.parse_args()


def find_images_dir(src: Path) -> Path:
    """Localiza a pasta de imagens independente da estrutura do zip."""
    candidates = [
        src / "identification" / "images",  # OpenCows2020 já baixado no projeto
        src / "images",
        src,
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if subdirs:
                return c
    raise FileNotFoundError(
        f"Não encontrei pasta de identidades em {src}. "
        "Verifique se o dataset foi extraído corretamente."
    )


def collect_by_identity(images_dir: Path) -> dict[str, list[Path]]:
    """Agrupa arquivos de imagem por subdiretório (= identidade do animal)."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    by_id: dict[str, list[Path]] = defaultdict(list)
    for cow_dir in sorted(images_dir.iterdir()):
        if not cow_dir.is_dir():
            continue
        imgs = sorted([f for f in cow_dir.iterdir() if f.suffix.lower() in EXTS])
        if imgs:
            by_id[cow_dir.name] = imgs
    return dict(by_id)


def split_and_copy(
    by_id: dict[str, list[Path]],
    dst: Path,
    val_split: float,
    min_images: int,
) -> tuple[int, int, int]:
    """Copia imagens para dst/train e dst/val mantendo estrutura por identidade."""
    train_count = val_count = skipped = 0

    for cow_id, imgs in by_id.items():
        if len(imgs) < min_images:
            skipped += 1
            continue

        n_val = max(1, int(len(imgs) * val_split))
        # últimas imagens → val (mais recentes temporalmente)
        train_imgs = imgs[:-n_val]
        val_imgs = imgs[-n_val:]

        for img in train_imgs:
            dest = dst / "train" / cow_id / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            train_count += 1

        for img in val_imgs:
            dest = dst / "val" / cow_id / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            val_count += 1

    return train_count, val_count, skipped


def print_stats(dst: Path) -> None:
    train_ids = sorted((dst / "train").iterdir()) if (dst / "train").exists() else []
    val_ids = sorted((dst / "val").iterdir()) if (dst / "val").exists() else []
    train_imgs = sum(len(list(d.iterdir())) for d in train_ids)
    val_imgs = sum(len(list(d.iterdir())) for d in val_ids)

    print("\n" + "=" * 50)
    print("  OpenCows2020 — Dataset preparado")
    print("=" * 50)
    print(f"  Identidades no treino : {len(train_ids)}")
    print(f"  Identidades na val    : {len(val_ids)}")
    print(f"  Imagens treino        : {train_imgs}")
    print(f"  Imagens val           : {val_imgs}")
    print(f"  Total                 : {train_imgs + val_imgs}")
    print(f"  Destino               : {dst.resolve()}")
    print("=" * 50)
    print("\nPróximo passo:")
    print(f"  uv run python -m src.training.train_reid --data_root {dst.resolve()} --epochs 30\n")


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.exists():
        raise SystemExit(f"Diretório de origem não encontrado: {src}")

    if dst.exists():
        resp = input(f"Destino {dst} já existe. Sobrescrever? [s/N] ").strip().lower()
        if resp != "s":
            raise SystemExit("Abortado.")
        shutil.rmtree(dst)

    print(f"Localizando imagens em {src}...")
    images_dir = find_images_dir(src)
    print(f"Encontrado: {images_dir}")

    by_id = collect_by_identity(images_dir)
    print(f"Identidades encontradas: {len(by_id)}")
    print(f"Total de imagens: {sum(len(v) for v in by_id.values())}")

    print(f"\nOrganizando para {dst} (val_split={args.val_split}, min_images={args.min_images})...")
    train_n, val_n, skipped = split_and_copy(by_id, dst, args.val_split, args.min_images)

    print(f"Identidades ignoradas (< {args.min_images} imagens): {skipped}")
    print_stats(dst)


if __name__ == "__main__":
    main()
