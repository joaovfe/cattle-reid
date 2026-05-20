#!/usr/bin/env python3
"""
Treinamento YOLO26 otimizado para detecção de bovinos em visão aérea (nadir).

Fase 1 — Treinamento base:
  uv run python -m src.training.train_yolo \
      --model yolo26m.pt \
      --data data/detection/data.yaml \
      --epochs 300 --batch 8 --imgsz 1280

Fase 2 — Fine-tuning com hard negative mining:
  uv run python -m src.training.train_yolo \
      --model runs/detect/fase1/weights/best.pt \
      --data data/detection/data.yaml \
      --epochs 100 --batch 8 --imgsz 1280 \
      --finetune --freeze 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treinamento YOLO26 para bovinos aéreos")
    p.add_argument("--model", default="yolo26m.pt",
                   help="Checkpoint base (yolo26m.pt, yolo26n.pt, yolov8n.pt, ou best.pt)")
    p.add_argument("--data", required=True, help="Caminho para data.yaml do dataset")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="cattle_yolo26_v1")
    p.add_argument("--finetune", action="store_true",
                   help="Modo fine-tuning: lr menor, menos augmentação, backbone congelado")
    p.add_argument("--freeze", type=int, default=10,
                   help="Número de camadas do backbone a congelar no fine-tuning")
    p.add_argument("--device", default="", help="cuda, cpu, ou '' para auto-detectar")
    p.add_argument("--cache", default="disk",
                   help="Cache de imagens: 'disk' (seguro), 'ram' ou False")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping: épocas sem melhora antes de parar")
    return p.parse_args()


def resolve_model(model_arg: str) -> str:
    """Resolve nome curto para path, tenta CWD → raiz do projeto."""
    path = Path(model_arg)
    if path.exists():
        return str(path)
    # tenta na raiz do projeto (dois níveis acima de src/training/)
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / model_arg
    if candidate.exists():
        return str(candidate)
    # tenta models/
    candidate2 = project_root / "models" / model_arg
    if candidate2.exists():
        return str(candidate2)
    return model_arg  # deixa o ultralytics tentar baixar


def auto_batch_imgsz(batch: int, imgsz: int) -> tuple[int, int]:
    """Reduz batch/imgsz automaticamente se GPU < 16 GB."""
    if not torch.cuda.is_available():
        print("⚠  CUDA não disponível — usando CPU com batch=2 e imgsz=640.")
        return 2, 640
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb < 8:
        print(f"GPU {vram_gb:.1f}GB detectada — ajustando para batch=2, imgsz=960.")
        return 2, 960
    if vram_gb < 16:
        print(f"GPU {vram_gb:.1f}GB detectada — ajustando para batch=4, imgsz=1280.")
        return 4, imgsz
    return batch, imgsz


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Instale ultralytics: uv add ultralytics")

    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml}")

    model_path = resolve_model(args.model)
    batch, imgsz = auto_batch_imgsz(args.batch, args.imgsz)

    model = YOLO(model_path)

    # ─── Hiperparâmetros comuns ────────────────────────────────────────────────
    cache_val = args.cache if args.cache in ("disk", "ram") else False
    common = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        val=True,
        plots=True,
        save_period=10,
        amp=True,       # FP16
        cache=cache_val,
        cos_lr=True,    # cosine annealing
        device=args.device or None,
    )

    if args.finetune:
        # ── Fase 2: Fine-tuning ──────────────────────────────────────────────
        print(f"\n[Fase 2] Fine-tuning a partir de {model_path}")
        print(f"  freeze={args.freeze} camadas, lr0=0.0001, augmentação reduzida\n")
        model.train(
            **common,
            freeze=args.freeze,
            optimizer="AdamW",
            lr0=0.0001,         # lr baixo para não destruir o que aprendeu
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=args.patience,
            # Augmentação moderada (não agressiva como no treino base)
            degrees=180,
            translate=0.05,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1,
            copy_paste=0.1,
            erasing=0.2,
            close_mosaic=10,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
    else:
        # ── Fase 1: Treinamento base ─────────────────────────────────────────
        print(f"\n[Fase 1] Treinamento base com {model_path}")
        print(f"  epochs={args.epochs}, imgsz={imgsz}, batch={batch}\n")
        model.train(
            **common,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            patience=args.patience,
            # Augmentação agressiva para visão aérea
            degrees=180,        # rotação completa: sem orientação definida no nadir
            translate=0.1,
            scale=0.9,
            flipud=0.5,         # flip vertical válido em vista aérea
            fliplr=0.5,
            mosaic=1.0,         # combina 4 imagens por patch
            mixup=0.2,
            copy_paste=0.3,
            erasing=0.4,
            auto_augment="randaugment",
            close_mosaic=15,    # desativa mosaic nas últimas 15 épocas
            box=7.5,            # peso da loss de regressão de bbox
            cls=0.5,
            dfl=1.5,
            rect=False,
        )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✅ Treinamento concluído. Melhor modelo: {best.resolve()}")
    print(f"   Para fine-tuning: adicione --finetune --model {best.resolve()}")


if __name__ == "__main__":
    main()
