"""
YOLO-based cattle detector. Frame BGR in -> list of detections (bbox, score, class).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2]
    score: float
    class_name: str = "cow"
    class_id: int = 0


class YOLOCattleDetector:
    """Wrapper for Ultralytics YOLO for cattle detection (dorsal/top-down view)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.55,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        device: str | None = None,
        half: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self._model_path = str(model_path) if model_path else None
        self._device = device
        self._half = half
        self._model: Any = None

    def _get_model(self):  # noqa: ANN201
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics: uv add ultralytics") from e
        default = "yolo11n.pt"
        if self._model_path and Path(self._model_path).exists():
            self._model = YOLO(self._model_path)
        else:
            self._model = YOLO(self._model_path or default)
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a single frame (BGR). Returns list of Detection."""
        model = self._get_model()
        results = model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
            half=self._half,
            device=self._device,
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[i].cpu().numpy()
                conf = float(r.boxes.conf[i].cpu().numpy())
                cls_id = int(r.boxes.cls[i].cpu().numpy())
                out.append(
                    Detection(
                        bbox=xyxy.tolist(),
                        score=conf,
                        class_name=r.names.get(cls_id, "cow"),
                        class_id=cls_id,
                    )
                )
        return out
