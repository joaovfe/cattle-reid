"""Placeholder loaders for BECA, OpenCows2020, and generic YOLO format."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_yolo_dataset(data_yaml: str | Path) -> Iterator[tuple[str, list[tuple[int, list[float]]]]]:
    """Iterate YOLO dataset: (image_path, [(class_id, xywhn), ...])."""
    import yaml
    path = Path(data_yaml)
    if not path.exists():
        return
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", path.parent))
    for split in ("train", "val"):
        images_dir = root / cfg.get(split, split) / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists():
            continue
        for img_path in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")):
            label_path = labels_dir / (img_path.stem + ".txt")
            labels = []
            if label_path.exists():
                for line in label_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        labels.append((int(parts[0]), [float(x) for x in parts[1:5]]))
            yield str(img_path), labels
