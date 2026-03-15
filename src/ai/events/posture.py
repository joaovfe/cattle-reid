"""
Posture classification: standing / lying (and optional fallen). Assigns event_type per detection/track.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class PostureLabel(str, Enum):
    STANDING = "standing"
    LYING = "lying"
    FALLEN = "caido"
    UNKNOWN = "unknown"


class PostureClassifier:
    """
    Placeholder classifier: uses aspect ratio heuristic (lying = elongated bbox).
    Can be replaced by a trained head on DINO or CNN.
    """

    def __init__(self, aspect_ratio_threshold: float = 1.8) -> None:
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def predict(self, crop: np.ndarray, bbox: list[float] | None = None) -> tuple[PostureLabel, float]:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if h > 0:
                ar = w / h
                if ar > self.aspect_ratio_threshold:
                    return PostureLabel.LYING, 0.7
                return PostureLabel.STANDING, 0.7
        return PostureLabel.UNKNOWN, 0.0

    def predict_batch(
        self,
        crops: list[np.ndarray],
        bboxes: list[list[float]] | None = None,
    ) -> list[tuple[PostureLabel, float]]:
        if bboxes is None:
            bboxes = [None] * len(crops)
        return [self.predict(c, b) for c, b in zip(crops, bboxes)]
