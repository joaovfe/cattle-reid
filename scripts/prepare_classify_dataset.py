#!/usr/bin/env python3
"""
Prepara dataset de classificação de postura a partir dos crops gerados pelo pipeline.

Organiza imagens em pastas por classe para treinamento do classificador YOLOv8-cls.

Uso:
  1. Organize seus crops em pastas por classe manualmente:
       data/classify_raw/em_pe/*.jpg
       data/classify_raw/deitado/*.jpg
       data/classify_raw/caido/*.jpg

  2. Execute este script para gerar o split train/val:
       uv run python scripts/prepare_classify_dataset.py \
           --src data/classify_raw \
           --dst data/classify \
           --val_split 0.2

  3. Treine o classificador:
       uv run python -m src.training.train_classify \
           --data data/classify \
           --epochs 50
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara dataset de classificação de postura")
    p.add_argument("--src", required=True,
                   help="Pasta com subpastas por classe: em_pe/, deitado/, caido/")
    p.add_argument("--dst", default="data/classify",
                   help="Destino com split train/val por classe")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fração de imagens para val (ex: 0.2 = 20%%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_images", type=int, default=10,
                   help="Mínimo de imagens por classe para incluir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    random.seed(args.seed)

    if not src.exists():
        raise SystemExit(f"Pasta de origem não encontrada: {src}")

    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"Nenhuma subpasta de classe encontrada em {src}")

    print(f"Classes encontradas: {classes}")

    if dst.exists():
        resp = input(f"Destino {dst} já existe. Sobrescrever? [s/N] ").strip().lower()
        if resp != "s":
            raise SystemExit("Abortado.")
        shutil.rmtree(dst)

    total_train = total_val = 0

    for cls in classes:
        imgs = sorted([f for f in (src / cls).iterdir() if f.suffix.lower() in EXTS])
        if len(imgs) < args.min_images:
            print(f"  ⚠  {cls}: apenas {len(imgs)} imagens (mínimo {args.min_images}) — ignorado")
            continue

        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.val_split))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for img in train_imgs:
            dest = dst / "train" / cls / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            total_train += 1

        for img in val_imgs:
            dest = dst / "val" / cls / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            total_val += 1

        print(f"  {cls:15s} → train: {len(train_imgs):4d}  val: {len(val_imgs):4d}")

    print("\n" + "=" * 50)
    print(f"  Total treino : {total_train} imagens")
    print(f"  Total val    : {total_val} imagens")
    print(f"  Destino      : {dst}")
    print("=" * 50)
    print(f"\nPróximo passo:")
    print(f"  uv run python -m src.training.train_classify --data {dst} --epochs 50\n")


if __name__ == "__main__":
    main()
