"""CLI: run full pipeline on a video file. Optionally save video, DB and metrics."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def _draw_detections(frame, frame_dets, track_id_to_animal, frame_idx):
    """Draw bbox, track_id and animal_id on frame. frame_dets = [(bbox, track_id), ...]."""
    import cv2
    for bbox, track_id in frame_dets:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        animal_id = track_id_to_animal.get(track_id)
        label = f"T{track_id}"
        if animal_id is not None:
            label += f" ID:{animal_id}"
        else:
            label += " (unknown)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cattle Re-ID pipeline on video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--faiss", type=str, default=None, help="Path to FAISS index (e.g. core/data/faiss_index)")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--output", type=str, default=None, help="JSON results path")
    parser.add_argument("--output_video", type=str, default=None, help="Save video with bboxes and IDs drawn")
    parser.add_argument("--save_db", action="store_true", help="Persist tracklets and count event to PostgreSQL")
    parser.add_argument("--gt_count", type=int, default=None, help="Ground truth count (for metrics comparison)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python required: uv add opencv-python") from None

    config = _load_config(args.config)
    models_cfg = config.get("models", {})
    yolo_cfg = models_cfg.get("yolo", {})
    emb_cfg = models_cfg.get("embedding", {})
    track_cfg = config.get("tracking", {})
    reid_cfg = config.get("reid", {})
    crop_cfg = config.get("oriented_crop", {})
    batch_size = config.get("inference", {}).get("batch_size", 16)

    from src.ai.detection.yolo_detector import YOLOCattleDetector
    from src.ai.tracking.bytetrack_tracker import ByteTrackTracker, TrackerConfig
    from src.ai.oriented_crop.cropper import OrientedCropper
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV2_SMALL
    from src.ai.reid.faiss_store import FAISSStore
    from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings
    from src.ai.metrics.count_metrics import compute_count_metrics

    detector = YOLOCattleDetector(
        model_path=yolo_cfg.get("weights"),
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.55)),
        iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
        max_det=yolo_cfg.get("max_det", 300),
        half=yolo_cfg.get("half", True),
    )
    tracker = ByteTrackTracker(TrackerConfig(
        max_age=track_cfg.get("max_age", 30),
        iou_threshold=track_cfg.get("iou_threshold", 0.3),
    ))
    cropper = OrientedCropper(
        crop_cfg.get("output_size", 224),
        crop_cfg.get("padding", 0.1),
    )
    encoder = CattleEmbeddingEncoder(
        model_name=emb_cfg.get("model_name", DINOV2_SMALL),
        half=emb_cfg.get("half", True),
    )
    decision = IdentityDecision(
        similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.25)),
        min_embeddings_per_tracklet=int(reid_cfg.get("min_embeddings_per_tracklet", 4)),
    )
    faiss_path = args.faiss or str(Path(__file__).resolve().parent.parent.parent / "core" / "data" / "faiss_index")
    store = FAISSStore(index_path=faiss_path)
    if Path(faiss_path).exists():
        store.load()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracklet_embeddings = defaultdict(list)
    frame_detections = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue
        detections = detector.detect(frame)
        det_list = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked = tracker.update(det_list)
        frame_detections[frame_idx] = [(det["bbox"], det["track_id"]) for det in tracked if det.get("track_id")]
        # Batch embedding por frame (mais rápido)
        batch: list[tuple[int, np.ndarray]] = []
        for det in tracked:
            tid = det.get("track_id")
            if tid is None:
                continue
            crop = cropper.crop(frame, det["bbox"])
            batch.append((tid, crop))
        if batch:
            tids, crops = zip(*batch)
            embs = encoder.encode_batch(list(crops))
            for i, tid in enumerate(tids):
                tracklet_embeddings[tid].append(embs[i])
        frame_idx += 1

    cap.release()

    results = []
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        animal_id, score = decision.decide(agg, store, top_k=5)
        results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    track_id_to_animal = {r["track_id"]: r["animal_id"] for r in results}

    unique_identified = sum(1 for r in results if r.get("animal_id") is not None)
    count_metrics = compute_count_metrics(
        unique_tracklets=len(results),
        unique_identified=unique_identified,
        ground_truth_count=args.gt_count,
    )

    print(f"Frames processados: {frame_idx}")
    print(f"Contagem: {count_metrics.predicted_count} indivíduos (tracklets únicos)")
    print(f"Identificados (Re-ID): {count_metrics.unique_identified}")
    if count_metrics.ground_truth_count is not None:
        print(f"Ground truth: {count_metrics.ground_truth_count} | Erro absoluto: {count_metrics.absolute_error}")
    for r in results:
        print(r)

    if args.save_db:
        async def _persist():
            from core.database.session import async_session_factory
            from core.database.inference_persistence import save_inference_results
            metrics_dict = {
                "count": count_metrics.predicted_count,
                "unique_identified": count_metrics.unique_identified,
                "ground_truth_count": count_metrics.ground_truth_count,
                "absolute_error": count_metrics.absolute_error,
            }
            async with async_session_factory() as session:
                await save_inference_results(
                    session,
                    str(video_path.resolve()),
                    results,
                    frame_detections,
                    metrics=metrics_dict,
                )
                await session.commit()
        asyncio.run(_persist())
        print("Resultados salvos no banco (tracklets + evento count_summary).")

    if args.output:
        out_data = {
            "frames_processed": frame_idx,
            "tracklets": results,
            "count": count_metrics.predicted_count,
            "unique_identified": count_metrics.unique_identified,
            "metrics": {
                "ground_truth_count": count_metrics.ground_truth_count,
                "absolute_error": count_metrics.absolute_error,
            },
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)

    if args.output_video:
        cap = cv2.VideoCapture(str(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_detections:
                frame = _draw_detections(frame, frame_detections[idx], track_id_to_animal, idx)
            else:
                cv2.putText(frame, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
            idx += 1
        cap.release()
        out.release()
        print(f"Video salvo: {args.output_video}")


if __name__ == "__main__":
    main()
