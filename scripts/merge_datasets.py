#!/usr/bin/env python3
"""
Combina múltiplos datasets YOLO em um único dataset para treinamento.

Fontes combinadas:
  - agro-vision/datasets/aerial_cows_10k.yolo26  (2.338 imgs, vista aérea)
  - data/opencows2020_detection                   (396 imgs, vista lateral/mista)

Normaliza nomes de classe para "cow" e gera data.yaml final.

Uso:
  uv run python scripts/merge_datasets.py
  uv run python scripts/merge_datasets.py --dst data/merged_detection
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge datasets YOLO para detecção de bovinos")
    p.add_argument("--dst", default="data/merged_detection",
                   help="Destino do dataset combinado")
    p.add_argument("--agrovision",
                   default="/home/joao/projects/agro-vision/datasets/aerial_cows_10k.yolo26",
                   help="Dataset agro-vision (aerial_cows_10k.yolo26)")
    p.add_argument("--opencows",
                   default="data/opencows2020_detection",
                   help="Dataset OpenCows2020 preparado")
    return p.parse_args()


def load_yaml_dataset(data_yaml: Path) -> dict:
    with open(data_yaml) as f:
        return yaml.safe_load(f) or {}


def resolve_split_dir(cfg: dict, data_yaml: Path, split: str) -> Path | None:
    """Resolve caminho absoluto de uma split a partir do data.yaml."""
    raw = cfg.get(split)
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # relativo ao path: do yaml
    base = Path(cfg.get("path", data_yaml.parent))
    candidate = base / raw
    if candidate.exists():
        return candidate
    # relativo ao diretório do yaml
    candidate2 = data_yaml.parent / raw
    if candidate2.exists():
        return candidate2
    return None


def file_hash(path: Path) -> str:
    """MD5 dos primeiros 4KB para dedup rápido."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(4096))
    return h.hexdigest()


def copy_split(
    img_dir: Path,
    lbl_dir: Path,
    dst_img: Path,
    dst_lbl: Path,
    prefix: str,
    seen_hashes: set[str],
    class_remap: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Copia imagens+labels para dst, renomeia com prefix, deduplica por hash."""
    copied = skipped = 0
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue

        h = file_hash(img)
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        new_stem = f"{prefix}_{img.stem}"
        shutil.copy2(img, dst_img / f"{new_stem}{img.suffix.lower()}")

        # Remap classes se necessário
        if class_remap:
            lines = lbl.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                new_cls = class_remap.get(cls, cls)
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
            (dst_lbl / f"{new_stem}.txt").write_text("\n".join(new_lines))
        else:
            shutil.copy2(lbl, dst_lbl / f"{new_stem}.txt")

        copied += 1

    return copied, skipped


def process_dataset(
    data_yaml: Path,
    dst: Path,
    prefix: str,
    seen: dict[str, set[str]],
    target_class: str = "cow",
) -> dict[str, int]:
    """Processa um dataset e copia para dst/{train,val,test}."""
    cfg = load_yaml_dataset(data_yaml)
    src_names: list[str] = cfg.get("names", ["cow"])

    # Monta remap: índice original → 0 (única classe: cow)
    class_remap = {i: 0 for i in range(len(src_names))}

    stats = {}
    for split, dst_split in [("train", "train"), ("val", "val"), ("test", "val")]:
        img_dir = resolve_split_dir(cfg, data_yaml, split)
        if not img_dir or not img_dir.exists():
            continue
        lbl_dir = img_dir.parent.parent / split / "labels"
        if not lbl_dir.exists():
            lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            # tenta labels irmão de images
            lbl_dir = img_dir.parent.parent / "labels"
        if not lbl_dir.exists():
            print(f"  ⚠  Labels não encontrados para {split} em {img_dir}")
            continue

        dst_img = dst / dst_split / "images"
        dst_lbl = dst / dst_split / "labels"
        copied, skipped = copy_split(
            img_dir, lbl_dir, dst_img, dst_lbl,
            f"{prefix}_{split}", seen[dst_split], class_remap
        )
        stats[split] = copied
        if skipped:
            print(f"    {split}: {copied} copiadas, {skipped} duplicatas ignoradas")
        else:
            print(f"    {split}: {copied} imagens copiadas")

    return stats


def main() -> None:
    args = parse_args()
    dst = Path(args.dst).resolve()
    agrovision_yaml = Path(args.agrovision) / "data.yaml"
    opencows_yaml = Path(args.opencows) / "data.yaml"

    print("\n" + "=" * 60)
    print("  MERGE DE DATASETS — Detecção de Bovinos")
    print("=" * 60)

    missing = []
    if not agrovision_yaml.exists():
        missing.append(str(agrovision_yaml))
    if not opencows_yaml.exists():
        missing.append(str(opencows_yaml))

    if missing:
        print("\n⚠  Datasets não preparados:")
        for m in missing:
            print(f"   {m}")
        if str(opencows_yaml) in missing:
            print("\n   Execute primeiro:")
            print("   uv run python scripts/prepare_opencows2020_detection.py")
        return

    if dst.exists():
        resp = input(f"\nDestino {dst} já existe. Sobrescrever? [s/N] ").strip().lower()
        if resp != "s":
            raise SystemExit("Abortado.")
        shutil.rmtree(dst)

    seen: dict[str, set[str]] = {"train": set(), "val": set()}

    print(f"\n[1/2] agro-vision (aerial_cows_10k.yolo26)")
    stats_agro = process_dataset(agrovision_yaml, dst, "agro", seen)

    print(f"\n[2/2] OpenCows2020 detection")
    stats_opencows = process_dataset(opencows_yaml, dst, "ocows", seen)

    # Totais
    train_total = len(list((dst / "train" / "images").glob("*"))) if (dst / "train" / "images").exists() else 0
    val_total   = len(list((dst / "val"   / "images").glob("*"))) if (dst / "val"   / "images").exists() else 0

    # Gera data.yaml final
    data_yaml_out = dst / "data.yaml"
    data_yaml_out.write_text(
        yaml.safe_dump({
            "path": str(dst),
            "train": str(dst / "train" / "images"),
            "val":   str(dst / "val"   / "images"),
            "nc": 1,
            "names": ["cow"],
        }, sort_keys=False, allow_unicode=True)
    )

    print("\n" + "=" * 60)
    print("  DATASET COMBINADO")
    print("=" * 60)
    print(f"  Total treino : {train_total} imagens")
    print(f"  Total val    : {val_total} imagens")
    print(f"  Total geral  : {train_total + val_total} imagens")
    print(f"  Classes      : 1 (cow)")
    print(f"  data.yaml    : {data_yaml_out}")
    print("=" * 60)
    print(f"\nPróximo passo:")
    print(f"  uv run python -m src.training.train_yolo \\")
    print(f"      --model yolo26m.pt \\")
    print(f"      --data {data_yaml_out} \\")
    print(f"      --epochs 300 --imgsz 1280 --batch 4\n")


if __name__ == "__main__":
    main()
