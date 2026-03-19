from pathlib import Path

import pytest

from src.ai.detection.yolo_detector import YOLOCattleDetector
from src.scripts.run_inference_video import _aggregate_classifications, _require_model_weights


def test_yolo_detector_requires_existing_weights_file():
    detector = YOLOCattleDetector(model_path="models/does-not-exist.pt")

    with pytest.raises(FileNotFoundError):
        detector._get_model()


def test_require_model_weights_rejects_wrong_filename(tmp_path: Path):
    wrong_file = tmp_path / "other_model.pt"
    wrong_file.write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit):
        _require_model_weights(str(wrong_file), model_label="YOLO detector", expected_filename="best_cow.pt")


def test_aggregate_classifications_returns_best_label_summary():
    aggregated = _aggregate_classifications(
        {
            7: [
                {"label": "em_pe", "score": 0.80},
                {"label": "em_pe", "score": 0.92},
                {"label": "deitado", "score": 0.95},
            ]
        }
    )

    assert 7 in aggregated
    assert aggregated[7]["label"] == "em_pe"
    assert aggregated[7]["num_frames"] == 3
