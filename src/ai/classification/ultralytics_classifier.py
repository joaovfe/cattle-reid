from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Classification:
    label: str
    score: float


class UltralyticsImageClassifier:
    """
    Wrapper for Ultralytics YOLO classification models.

    Accepts BGR crops (numpy HxWx3) and returns (label, score).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        half: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.half = half
        self._model: Any = None

    def _get_model(self):  # noqa: ANN201
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics: uv add ultralytics") from e
        self._model = YOLO(self.model_path)
        return self._model

    def predict_one(self, crop_bgr: np.ndarray) -> Classification:
        model = self._get_model()
        res = model.predict(
            crop_bgr,
            verbose=False,
            device=self.device,
            half=self.half,
        )
        # Ultralytics returns a list of Results.
        if not res:
            return Classification(label="unknown", score=0.0)
        r0 = res[0]
        probs = getattr(r0, "probs", None)
        if probs is None:
            return Classification(label="unknown", score=0.0)
        top1 = int(getattr(probs, "top1", -1))
        top1conf = float(getattr(probs, "top1conf", 0.0))
        names = getattr(r0, "names", {}) or {}
        label = str(names.get(top1, "unknown"))
        return Classification(label=label, score=top1conf)

    def predict_batch(self, crops_bgr: list[np.ndarray]) -> list[Classification]:
        if not crops_bgr:
            return []
        model = self._get_model()
        res_list = model.predict(
            crops_bgr,
            verbose=False,
            device=self.device,
            half=self.half,
        )
        out: list[Classification] = []
        for r in res_list or []:
            probs = getattr(r, "probs", None)
            if probs is None:
                out.append(Classification(label="unknown", score=0.0))
                continue
            top1 = int(getattr(probs, "top1", -1))
            top1conf = float(getattr(probs, "top1conf", 0.0))
            names = getattr(r, "names", {}) or {}
            out.append(Classification(label=str(names.get(top1, "unknown")), score=top1conf))
        return out

