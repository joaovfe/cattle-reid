"""
Oriented crop: frame + bbox -> aligned crop (e.g. 224x224). Minimal version: axis-aligned bbox with padding.
"""
from __future__ import annotations

import numpy as np


def crop_bbox(
    frame: np.ndarray,
    bbox: list[float],
    output_size: int = 224,
    padding: float = 0.1,
) -> np.ndarray:
    """
    Crop frame using axis-aligned bbox with optional padding. Resize to output_size x output_size.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_w = w * padding
    pad_h = h * padding
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(frame.shape[1], x2 + pad_w)
    y2 = min(frame.shape[0], y2 + pad_h)
    crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
    if crop.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=frame.dtype)
    try:
        import cv2
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        pil = Image.fromarray(crop[:, :, ::-1] if crop.shape[-1] == 3 else crop)
        pil = pil.resize((output_size, output_size), Image.BILINEAR)
        return np.array(pil)[:, :, ::-1]


class OrientedCropper:
    """Crops bovine regions from frame; minimal implementation with bbox + padding."""

    def __init__(self, output_size: int = 224, padding: float = 0.1) -> None:
        self.output_size = output_size
        self.padding = padding

    def crop(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        return crop_bbox(
            frame,
            bbox,
            output_size=self.output_size,
            padding=self.padding,
        )

    def crop_many(
        self,
        frame: np.ndarray,
        bboxes: list[list[float]],
    ) -> list[np.ndarray]:
        return [self.crop(frame, bbox) for bbox in bboxes]
