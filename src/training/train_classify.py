#!/usr/bin/env python3
"""
Treinamento do classificador de postura/comportamento (YOLOv8-cls).

Espera dataset organizado por classe:
  data/classify/
    train/
      em_pe/     ← bovino em pé visto de cima
        *.jpg
      deitado/   ← bovino deitado visto de cima
        *.jpg
      caido/     ← bovino prostrado / comportamento anormal
        *.jpg
    val/
      em_pe/
      deitado/
      caido/

Como gerar o dataset de classificação a partir dos seus crops:
  uv run python scripts/prepare_classify_dataset.py \
      --crops_dir core/data/crops \
      --dst data/classify

Treinamento:
  uv run python -m src.training.train_classify \
      --data data/classify \
      --model yolov8s-cls.pt \
      --epochs 50

Fine-tuning sobre modelo existente:
  uv run python -m src.training.train_classify \
      --data data/classify \
      --model models/cow_pose_classifier.pt \
      --epochs 30 --finetune
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina classificador de postura de bovinos")
    p.add_argument("--data", required=True,
                   help="Pasta raiz com train/{classe}/ e val/{classe}/")
    p.add_argument("--model", default="yolov8s-cls.pt",
                   help="Backbone: yolov8n-cls.pt | yolov8s-cls.pt | yolov8m-cls.pt, ou checkpoint existente")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224,
                   help="Tamanho de entrada — 224 alinhado com o crop padrão do pipeline")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/classify")
    p.add_argument("--name", default="novus_pose_v1")
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tuning: congela backbone, ajusta só o head")
    p.add_argument("--device", default="", help="cuda | cpu | '' auto")
    return p.parse_args()


def check_dataset(data_dir: Path) -> list[str]:
    """Verifica estrutura e retorna lista de classes encontradas."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Pasta train/ não encontrada em {data_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Pasta val/ não encontrada em {data_dir}")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em {train_dir}")

    print("\n  Classes encontradas:", classes)
    for cls in classes:
        t = len(list((train_dir / cls).glob("*.*")))
        v = len(list((val_dir / cls).glob("*.*"))) if (val_dir / cls).exists() else 0
        print(f"    {cls:15s} → train: {t:4d}  val: {v:4d}")
    print()
    return classes


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Instale ultralytics: uv add ultralytics")

    args = parse_args()
    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_dir}")

    classes = check_dataset(data_dir)

    # Resolve modelo
    model_path = Path(args.model)
    if not model_path.exists():
        project_root = Path(__file__).resolve().parents[2]
        for candidate in [project_root / args.model, project_root / "models" / args.model]:
            if candidate.exists():
                model_path = candidate
                break

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    batch = args.batch if vram_gb >= 8 else max(8, args.batch // 2)

    model = YOLO(str(model_path))

    common = dict(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        val=True,
        plots=True,
        amp=True,
        cos_lr=True,
        device=args.device or None,
    )

    if args.finetune:
        print(f"[Fine-tuning] modelo: {model_path}")
        model.train(
            **common,
            freeze=len(model.model.model) - 2,  # congela tudo exceto o head
            optimizer="AdamW",
            lr0=5e-5,
            lrf=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            warmup_epochs=2,
            patience=15,
            # Augmentação leve para fine-tuning
            degrees=180,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.01,
            hsv_s=0.4,
            hsv_v=0.3,
            erasing=0.2,
        )
    else:
        print(f"[Treinamento base] modelo: {model_path}, classes: {classes}")
        model.train(
            **common,
            optimizer="AdamW",
            lr0=1e-3,
            lrf=0.01,
            momentum=0.937,
            weight_decay=5e-4,
            warmup_epochs=3,
            patience=20,
            # Augmentação robusta para crops dorsais
            degrees=180,        # vista aérea: sem orientação padrão
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.3,
            mixup=0.1,
            auto_augment="randaugment",
        )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✅ Classificador treinado. Melhor modelo: {best.resolve()}")
    print(f"   Copie para: cp {best.resolve()} models/cow_pose_classifier.pt")
    print(f"   Classes: {classes}")


if __name__ == "__main__":
    main()
