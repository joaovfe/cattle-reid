"""Train Re-ID encoder (DINOv2/DINOv3 backbone + head) on BECA-D / OpenCows2020 style data."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Re-ID encoder")
    parser.add_argument("--config", type=str, default="configs/train_reid.yaml")
    parser.add_argument("--data_root", type=str, help="Root with train/ and val/ class folders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=str, default="core/data/checkpoints/reid")
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        data_root = cfg.get("data_root") or (cfg.get("data") or {}).get("root")
    if not data_root or not Path(data_root).exists():
        print("Re-ID training expects --data_root with train/ and val/ (class subdirs).")
        print("Example: uv run src.training.train_reid --data_root path/to/BECA-D --epochs 30")
        return
    print(f"Re-ID training placeholder: data_root={data_root}, epochs={args.epochs}")


if __name__ == "__main__":
    main()
