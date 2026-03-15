"""
Métricas de detecção: precision, recall, F1, mAP (quando há ground truth).
Útil para comparar runs e garantir acurácia > 95%.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DetectionMetrics:
    """Métricas de detecção para comparação entre runs."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    num_detections: int
    num_ground_truth: int
    iou_threshold: float


def _iou_box(box_a: List[float], box_b: List[float]) -> float:
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


def compute_detection_metrics(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """
    Calcula precision, recall e F1 entre caixas preditas e ground truth.
    pred_boxes / gt_boxes: listas de [x1, y1, x2, y2].
    """
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_gt == 0 and num_pred == 0:
        return DetectionMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            tp=0,
            fp=0,
            fn=0,
            num_detections=0,
            num_ground_truth=0,
            iou_threshold=iou_threshold,
        )
    if num_gt == 0:
        return DetectionMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tp=0,
            fp=num_pred,
            fn=0,
            num_detections=num_pred,
            num_ground_truth=0,
            iou_threshold=iou_threshold,
        )
    if num_pred == 0:
        return DetectionMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tp=0,
            fp=0,
            fn=num_gt,
            num_detections=0,
            num_ground_truth=num_gt,
            iou_threshold=iou_threshold,
        )

    iou_matrix = np.zeros((num_pred, num_gt))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = _iou_box(pb, gb)

    gt_matched = set()
    tp = 0
    for i in range(num_pred):
        best_j = int(np.argmax(iou_matrix[i]))
        if iou_matrix[i, best_j] >= iou_threshold and best_j not in gt_matched:
            gt_matched.add(best_j)
            tp += 1
    fp = num_pred - tp
    fn = num_gt - len(gt_matched)

    precision = tp / num_pred if num_pred else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        num_detections=num_pred,
        num_ground_truth=num_gt,
        iou_threshold=iou_threshold,
    )
