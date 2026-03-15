"""
ByteTrack-style tracker: detections in -> detections out with stable track_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackerConfig:
    max_age: int = 30
    iou_threshold: float = 0.3


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    last_seen_frame: int


class ByteTrackTracker:
    """IoU-based multi-object tracker (ByteTrack-style). Assigns stable track_id to detections."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._frame_index = 0

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Update tracker with new detections. Each detection must have 'bbox' and 'score'.
        Adds 'track_id' to each detection in-place.
        """
        self._frame_index += 1
        if not detections:
            self._purge_old()
            return detections

        bboxes = np.array([d["bbox"] for d in detections], dtype=float)
        assigned: dict[int, int] = {}

        for track_id, track in list(self._tracks.items()):
            best_idx: int | None = None
            best_iou = 0.0
            for idx in range(len(bboxes)):
                if idx in assigned.values():
                    continue
                iou = _iou(track.bbox, bboxes[idx].tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= self.config.iou_threshold:
                assigned[track_id] = best_idx
                track.bbox = bboxes[best_idx].tolist()
                track.last_seen_frame = self._frame_index

        for idx in range(len(bboxes)):
            if idx in assigned.values():
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=bboxes[idx].tolist(),
                last_seen_frame=self._frame_index,
            )
            assigned[tid] = idx

        self._purge_old()
        det_idx_to_track: dict[int, int] = {v: k for k, v in assigned.items()}
        for idx, det in enumerate(detections):
            det["track_id"] = det_idx_to_track.get(idx)
        return detections

    def _purge_old(self) -> None:
        to_del = [
            tid
            for tid, t in self._tracks.items()
            if (self._frame_index - t.last_seen_frame) > self.config.max_age
        ]
        for tid in to_del:
            self._tracks.pop(tid, None)

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self._frame_index = 0
