"""
Métricas para detecção, contagem e Re-ID — comparações e avaliação.
"""
from src.ai.metrics.detection_metrics import (
    compute_detection_metrics,
    DetectionMetrics,
)
from src.ai.metrics.count_metrics import (
    compute_count_metrics,
    CountMetrics,
)
from src.ai.metrics.tracklet_evaluation import (
    compute_tracklet_evaluation_metrics,
    TrackletEvaluationMetrics,
)

__all__ = [
    "compute_detection_metrics",
    "DetectionMetrics",
    "compute_count_metrics",
    "CountMetrics",
    "compute_tracklet_evaluation_metrics",
    "TrackletEvaluationMetrics",
]
