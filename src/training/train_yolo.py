"""Train YOLO (Ultralytics) for cattle detection. Expects dataset in YOLO format."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO for cattle detection")
    parser.add_argument("--config", type=str, default="configs/train_yolo.yaml")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics: uv add ultralytics") from e

    data_yaml = args.data
    if not data_yaml and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        data_yaml = cfg.get("data") or "data.yaml"
    if not data_yaml or not Path(data_yaml).exists():
        print("No data.yaml found. Example: uv run src.training.train_yolo --data path/to/data.yaml --epochs 50")
        return

    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
