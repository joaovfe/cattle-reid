#!/usr/bin/env python3
"""
Prepara a parte de detecção do OpenCows2020 para treino YOLO.

O OpenCows2020 já tem labels no formato darknet/YOLO em:
  detection_and_localisation/
    images/        ← 396 imagens
    labels-darknet/ ← 7.043 labels (396 com imagem correspondente)

Este script:
  1. Copia apenas os pares imagem+label válidos
  2. Divide em train (80%) e val (20%)
  3. Gera data.yaml pronto para train_yolo.py

Uso:
  uv run python scripts/prepare_opencows2020_detection.py
  uv run python scripts/prepare_opencows2020_detection.py \
      --src MetricLearningIdentification/datasets/OpenCows2020 \
      --dst data/opencows2020_detection \
      --val_split 0.2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara OpenCows2020 detecção para YOLO")
    p.add_argument(
        "--src",
        default="MetricLearningIdentification/datasets/OpenCows2020",
        help="Raiz do OpenCows2020",
    )
    p.add_argument("--dst", default="data/opencows2020_detection")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    random.seed(args.seed)

    imgs_dir = src / "detection_and_localisation" / "images"
    lbls_dir = src / "detection_and_localisation" / "labels-darknet"

    if not imgs_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {imgs_dir}")
    if not lbls_dir.exists():
        raise FileNotFoundError(f"Pasta de labels não encontrada: {lbls_dir}")

    # Coleta apenas pares imagem+label válidos
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbls_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    print(f"Pares imagem+label encontrados: {len(pairs)}")

    if dst.exists():
        resp = input(f"Destino {dst} já existe. Sobrescrever? [s/N] ").strip().lower()
        if resp != "s":
            raise SystemExit("Abortado.")
        shutil.rmtree(dst)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def copy_pairs(pair_list: list[tuple[Path, Path]], split: str) -> None:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in pair_list:
            shutil.copy2(img, dst / split / "images" / img.name)
            shutil.copy2(lbl, dst / split / "labels" / lbl.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    # Gera data.yaml
    data_yaml = {
        "path": str(dst),
        "train": str(dst / "train" / "images"),
        "val": str(dst / "val" / "images"),
        "nc": 1,
        "names": ["cow"],
    }
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    print("\n" + "=" * 55)
    print("  OpenCows2020 — Detecção pronta para YOLO")
    print("=" * 55)
    print(f"  Imagens treino : {len(train_pairs)}")
    print(f"  Imagens val    : {len(val_pairs)}")
    print(f"  data.yaml      : {yaml_path}")
    print("=" * 55)
    print("\nPróximo passo — Treinamento YOLO26:")
    print(f"  uv run python -m src.training.train_yolo \\")
    print(f"      --model yolo26m.pt \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --epochs 300 --imgsz 1280\n")


if __name__ == "__main__":
    main()
