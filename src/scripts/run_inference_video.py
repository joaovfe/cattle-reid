"""CLI: run full pipeline on a video file. Optionally save video, DB and metrics."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np


def _draw_detections(
    frame,
    frame_dets,
    track_id_to_animal,
    track_id_to_classification,
    frame_idx,
    selected_track_id=None,
):
    """Draw bbox with track_id, animal_id and pose classification."""
    import cv2
    for bbox, track_id in frame_dets:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        animal_id = track_id_to_animal.get(track_id)
        cls_info = track_id_to_classification.get(track_id)
        label = f"T{track_id}"
        if animal_id is not None:
            label += f" ID:{animal_id}"
        else:
            label += " (unknown)"
        if cls_info:
            label += f" {cls_info['label']} {cls_info['score']:.2f}"
        color = (0, 255, 0)
        thickness = 2
        if selected_track_id is not None and int(track_id) == int(selected_track_id):
            color = (0, 165, 255)
            thickness = 3
            cv2.putText(
                frame,
                "SELECTED",
                (x1, max(y1 - 28, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if selected_track_id is not None:
        cv2.putText(
            frame,
            f"Selected Track: T{selected_track_id}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
        )
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


def _require_model_weights(
    maybe_path: str | None,
    model_label: str,
    expected_filename: str | None = None,
) -> str:
    if not maybe_path:
        raise SystemExit(f"{model_label} weights are required.")
    path = Path(maybe_path)
    if not path.exists():
        raise SystemExit(f"{model_label} weights not found: {path}")
    if expected_filename and path.name != expected_filename:
        raise SystemExit(f"{model_label} must use {expected_filename}. Received: {path.name}")
    return str(path)


def _aggregate_classifications(
    tracklet_classifications: dict[int, list[dict[str, float | str]]],
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for tid, preds in tracklet_classifications.items():
        if not preds:
            continue
        by_label: dict[str, list[float]] = {}
        for p in preds:
            lbl = str(p.get("label", "unknown"))
            sc = float(p.get("score", 0.0))
            by_label.setdefault(lbl, []).append(sc)
        best_label = max(by_label.items(), key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))))[0]
        best_score = float(np.mean(by_label[best_label])) if by_label[best_label] else 0.0
        out[int(tid)] = {"label": best_label, "score": best_score, "num_frames": len(preds)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cattle Re-ID pipeline on video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--faiss", type=str, default=None, help="Path to FAISS index (e.g. core/data/faiss_index)")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--output", type=str, default=None, help="JSON results path")
    parser.add_argument("--output_video", type=str, default=None, help="Save video with bboxes and IDs drawn")
    parser.add_argument("--selected_track_id", type=int, default=None, help="Highlight one tracked bovine in overlay")
    parser.add_argument("--show_video", action="store_true", help="Show live video while processing")
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
    cls_cfg = models_cfg.get("classifier", {})
    emb_cfg = models_cfg.get("embedding", {})
    track_cfg = config.get("tracking", {})
    reid_cfg = config.get("reid", {})
    crop_cfg = config.get("oriented_crop", {})
    batch_size = config.get("inference", {}).get("batch_size", 16)
    crops_per_tracklet = int(config.get("inference", {}).get("crops_per_tracklet", 3))
    require_classifier = bool(config.get("inference", {}).get("require_classifier", True))

    from src.ai.detection.yolo_detector import YOLOCattleDetector
    from src.ai.tracking.bytetrack_tracker import ByteTrackTracker, TrackerConfig
    from src.ai.oriented_crop.cropper import OrientedCropper
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV2_SMALL
    from src.ai.reid.faiss_store import FAISSStore
    from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings
    from src.ai.metrics.count_metrics import compute_count_metrics
    from src.ai.classification.ultralytics_classifier import UltralyticsImageClassifier
    from core.config import resolve_path, repo_root

    yolo_weights = _require_model_weights(
        resolve_path(yolo_cfg.get("weights")),
        model_label="YOLO detector",
        expected_filename="best_cow.pt",
    )
    detector = YOLOCattleDetector(
        model_path=yolo_weights,
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.55)),
        iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
        max_det=yolo_cfg.get("max_det", 300),
        allowed_class_ids=yolo_cfg.get("allowed_class_ids"),
        allowed_class_names=yolo_cfg.get("allowed_class_names"),
        half=yolo_cfg.get("half", True),
    )
    cls_weights = None
    if cls_cfg.get("weights"):
        cls_weights = _require_model_weights(
            resolve_path(cls_cfg.get("weights")) or str(cls_cfg.get("weights")),
            model_label="Pose classifier",
            expected_filename="cow_pose_classifier.pt",
        )
    if require_classifier and not cls_weights:
        raise SystemExit("Classifier weights are required for CLI inference.")
    classifier = (
        UltralyticsImageClassifier(
            model_path=cls_weights,
            device=cls_cfg.get("device"),
            half=bool(cls_cfg.get("half", True)),
        )
        if cls_weights
        else None
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
        top1_top2_margin=float(reid_cfg.get("top1_top2_margin", 0.08)),
    )
    faiss_path = args.faiss or str(Path(__file__).resolve().parent.parent.parent / "core" / "data" / "faiss_index")
    store = FAISSStore(index_path=faiss_path)
    if Path(faiss_path).exists():
        store.load()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

    tracklet_embeddings = defaultdict(list)
    tracklet_classifications: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    tracklet_crops: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_detections = {}
    track_id_to_animal_live: dict[int, int | None] = {}
    track_id_to_classification_live: dict[int, dict[str, Any]] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        draw_frame = frame.copy()
        if frame_idx % args.frame_skip != 0:
            if out is not None:
                cv2.putText(draw_frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(draw_frame)
            if args.show_video:
                cv2.imshow("Cattle ReID - Analise em tempo real", draw_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_idx += 1
                    break
            frame_idx += 1
            continue
        detections = detector.detect(frame)
        det_list = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked = tracker.update(det_list)
        frame_detections[frame_idx] = [(det["bbox"], det["track_id"]) for det in tracked if det.get("track_id")]
        # Batch embedding por frame (mais rápido)
        batch: list[tuple[int, np.ndarray]] = []
        cls_batch: list[tuple[int, np.ndarray]] = []
        for det in tracked:
            tid = det.get("track_id")
            if tid is None:
                continue
            crop = cropper.crop(frame, det["bbox"])
            batch.append((tid, crop))
            if classifier is not None:
                cls_batch.append((tid, crop))
            if crops_per_tracklet > 0 and len(tracklet_crops[int(tid)]) < crops_per_tracklet:
                tracklet_crops[int(tid)].append(
                    {"crop_bgr": crop, "frame_index": int(frame_idx), "bbox": det.get("bbox")}
                )
        if batch:
            tids, crops = zip(*batch)
            embs = encoder.encode_batch(list(crops))
            for i, tid in enumerate(tids):
                tracklet_embeddings[tid].append(embs[i])
                if len(tracklet_embeddings[tid]) >= decision.min_embeddings_per_tracklet:
                    agg_live = aggregate_embeddings(tracklet_embeddings[tid])
                    animal_id_live, _score_live = decision.decide(agg_live, store, top_k=5)
                    track_id_to_animal_live[int(tid)] = animal_id_live
                else:
                    track_id_to_animal_live.setdefault(int(tid), None)
        if classifier is not None and cls_batch:
            cls_tids, cls_crops = zip(*cls_batch)
            preds = classifier.predict_batch(list(cls_crops))
            for tid, p in zip(cls_tids, preds):
                tracklet_classifications[tid].append({"label": p.label, "score": float(p.score)})
            track_id_to_classification_live = _aggregate_classifications(tracklet_classifications)

        if out is not None or args.show_video:
            draw_frame = _draw_detections(
                draw_frame,
                frame_detections[frame_idx],
                track_id_to_animal_live,
                track_id_to_classification_live,
                frame_idx,
                selected_track_id=args.selected_track_id,
            )
            if out is not None:
                out.write(draw_frame)
            if args.show_video:
                cv2.imshow("Cattle ReID - Analise em tempo real", draw_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_idx += 1
                    break
        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    if args.show_video:
        cv2.destroyAllWindows()

    results = []
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        animal_id, score = decision.decide(agg, store, top_k=5)
        results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    track_id_to_animal = {r["track_id"]: r["animal_id"] for r in results}
    track_id_to_classification = _aggregate_classifications(tracklet_classifications)

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
        info = track_id_to_classification.get(int(r["track_id"]))
        if info:
            print({**r, "classification": info})
        else:
            print(r)

    if args.save_db:
        async def _persist():
            from core.database.session import async_session_factory
            from core.database.inference_persistence import save_inference_results
            from core.database.auto_enroll import auto_enroll_unknown_tracklets
            from core.database.models import Tracklet
            from sqlalchemy import select

            metrics_dict = {
                "count": count_metrics.predicted_count,
                "unique_identified": count_metrics.unique_identified,
                "ground_truth_count": count_metrics.ground_truth_count,
                "absolute_error": count_metrics.absolute_error,
                "classification": {"enabled": classifier is not None},
            }
            async with async_session_factory() as session:
                aggregated_by_tid: dict[int, np.ndarray] = {}
                for r in results:
                    tid = int(r["track_id"])
                    embs = tracklet_embeddings.get(tid, [])
                    if len(embs) < decision.min_embeddings_per_tracklet:
                        continue
                    aggregated_by_tid[tid] = aggregate_embeddings(embs)
                await auto_enroll_unknown_tracklets(
                    session=session,
                    store=store,
                    tracklet_results=results,
                    tracklet_aggregated_embeddings=aggregated_by_tid,
                    video_source=str(video_path.resolve()),
                )
                await save_inference_results(
                    session,
                    str(video_path.resolve()),
                    results,
                    frame_detections,
                    metrics=metrics_dict,
                )

                rows = await session.execute(
                    select(Tracklet.id, Tracklet.track_id).where(Tracklet.video_source == str(video_path.resolve()))
                )
                tracklet_id_by_track: dict[int, int] = {int(tid): int(tid_db) for tid_db, tid in rows.all()}

                events_payload: list[dict[str, object]] = []
                for tid, info in track_id_to_classification.items():
                    aid = track_id_to_animal.get(int(tid))
                    events_payload.append(
                        {
                            "event_type": "classification_summary",
                            "animal_id": aid,
                            "tracklet_id": tracklet_id_by_track.get(int(tid)),
                            "payload": info,
                        }
                    )

                crops_payload: list[dict[str, object]] = []
                crops_dir = repo_root() / "core" / "data" / "crops"
                crops_dir.mkdir(parents=True, exist_ok=True)
                for r in results:
                    tid = r.get("track_id")
                    aid = r.get("animal_id")
                    if tid is None or aid is None:
                        continue
                    for j, item in enumerate(tracklet_crops.get(int(tid), [])):
                        crop_bgr = item["crop_bgr"]
                        frame_index = int(item.get("frame_index") or 0)
                        bbox = item.get("bbox")
                        out_dir = Path(crops_dir) / f"animal_{int(aid)}"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{video_path.stem}_t{int(tid)}_f{frame_index}_{j}.jpg"
                        cv2.imwrite(str(out_path), crop_bgr)
                        crops_payload.append(
                            {
                                "animal_id": int(aid),
                                "tracklet_id": tracklet_id_by_track.get(int(tid)),
                                "source_path": str(out_path),
                                "frame_index": frame_index,
                                "bbox": bbox,
                                "metadata": {"video_source": str(video_path.resolve()), "track_id": int(tid)},
                            }
                        )
                await save_inference_results(
                    session,
                    str(video_path.resolve()),
                    [],
                    {},
                    metrics=None,
                    animal_crops=crops_payload,
                    extra_events=events_payload,
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
            "classification_enabled": classifier is not None,
            "metrics": {
                "ground_truth_count": count_metrics.ground_truth_count,
                "absolute_error": count_metrics.absolute_error,
            },
            "classifications": track_id_to_classification,
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)

    if args.output_video:
        print(f"Video salvo: {args.output_video}")


if __name__ == "__main__":
    main()
