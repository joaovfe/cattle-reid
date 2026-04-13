"""
POST /inference/video and /inference/frames: run pipeline (YOLOv11 + DINOv3), return tracklets, count and metrics.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query

router = APIRouter()


def _require_model_weights(
    model_path: str | None,
    model_label: str,
    expected_filename: str | None = None,
) -> str:
    if not model_path:
        raise HTTPException(status_code=500, detail=f"{model_label} weights are required.")
    path = Path(model_path)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"{model_label} weights not found: {path}")
    if expected_filename and path.name != expected_filename:
        raise HTTPException(
            status_code=500,
            detail=f"{model_label} must use {expected_filename}. Received: {path.name}",
        )
    return str(path)


def _get_inference_components(request: Request):
    config = getattr(request.app.state, "config", {}) or {}
    models_cfg = config.get("models", {})
    yolo_cfg = models_cfg.get("yolo", {}) or {}
    cls_cfg = models_cfg.get("classifier", {}) or {}
    emb_cfg = models_cfg.get("embedding", {}) or {}
    track_cfg = config.get("tracking", {}) or {}
    reid_cfg = config.get("reid", {}) or {}
    crop_cfg = config.get("oriented_crop", {}) or {}

    from src.ai.detection.yolo_detector import YOLOCattleDetector
    from src.ai.tracking.bytetrack_tracker import ByteTrackTracker, TrackerConfig
    from src.ai.oriented_crop.cropper import OrientedCropper
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV2_SMALL
    from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings

    from core.config import resolve_path

    yolo_weights = _require_model_weights(
        resolve_path(yolo_cfg.get("weights")),
        model_label="YOLO detector",
        expected_filename="best_cow.pt",
    )
    detector = YOLOCattleDetector(
        model_path=yolo_weights,
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.55)),
        iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
        max_det=int(yolo_cfg.get("max_det", 300)),
        allowed_class_ids=yolo_cfg.get("allowed_class_ids"),
        allowed_class_names=yolo_cfg.get("allowed_class_names"),
        device=yolo_cfg.get("device"),
        half=bool(yolo_cfg.get("half", True)),
    )
    classifier = None
    require_classifier = bool(config.get("inference", {}).get("require_classifier", True))
    if cls_cfg.get("weights"):
        backend = str(cls_cfg.get("backend", "ultralytics")).lower()
        if backend == "ultralytics":
            from src.ai.classification.ultralytics_classifier import UltralyticsImageClassifier

            cls_weights = _require_model_weights(
                resolve_path(cls_cfg.get("weights")) or str(cls_cfg.get("weights")),
                model_label="Pose classifier",
                expected_filename="cow_pose_classifier.pt",
            )
            classifier = UltralyticsImageClassifier(
                model_path=cls_weights,
                device=cls_cfg.get("device"),
                half=bool(cls_cfg.get("half", True)),
            )
    elif require_classifier:
        raise HTTPException(status_code=500, detail="Classifier weights are required for inference.")
    if require_classifier and classifier is None:
        raise HTTPException(status_code=500, detail="Classifier backend misconfigured or unavailable.")
    tracker = ByteTrackTracker(
        TrackerConfig(
            max_age=int(track_cfg.get("max_age", 30)),
            iou_threshold=float(track_cfg.get("iou_threshold", 0.3)),
        )
    )
    cropper = OrientedCropper(
        int(crop_cfg.get("output_size", 224)),
        float(crop_cfg.get("padding", 0.1)),
    )
    encoder = CattleEmbeddingEncoder(
        model_name=str(emb_cfg.get("model_name", DINOV2_SMALL)),
        device=emb_cfg.get("device"),
        half=bool(emb_cfg.get("half", True)),
    )
    decision = IdentityDecision(
        similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.25)),
        min_embeddings_per_tracklet=int(reid_cfg.get("min_embeddings_per_tracklet", 4)),
        top1_top2_margin=float(reid_cfg.get("top1_top2_margin", 0.08)),
    )
    return detector, classifier, tracker, cropper, encoder, decision


@router.post("/video")
async def inference_video(
    request: Request,
    file: UploadFile = File(...),
    save_db: bool = Query(False, description="Persist tracklets and count event to PostgreSQL"),
):
    """Upload video; detection (YOLOv11) -> tracking -> crop -> embed (DINOv3) -> Re-ID. Returns count and metrics."""
    content = await file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        import cv2
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}") from e

    detector, classifier, tracker, cropper, encoder, decision = _get_inference_components(request)
    store = request.app.state.faiss_store

    config = getattr(request.app.state, "config", {}) or {}
    crops_per_tracklet = int(config.get("inference", {}).get("crops_per_tracklet", 3))

    tracklet_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    tracklet_classifications: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    tracklet_crops: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_detections: dict[int, list[tuple[list[float], int]]] = {}
    frame_skip = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        detections = detector.detect(frame)
        det_list = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked = tracker.update(det_list)
        frame_detections[frame_idx] = [(det["bbox"], det["track_id"]) for det in tracked if det.get("track_id")]
        batch = []
        cls_batch = []
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
        if classifier is not None and cls_batch:
            cls_tids, cls_crops = zip(*cls_batch)
            preds = classifier.predict_batch(list(cls_crops))
            for tid, p in zip(cls_tids, preds):
                tracklet_classifications[tid].append({"label": p.label, "score": float(p.score)})
        frame_idx += 1

    cap.release()
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    results = []
    aggregated_by_tid: dict[int, np.ndarray] = {}
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        aggregated_by_tid[int(tid)] = agg
        animal_id, score = decision.decide(agg, store, top_k=5)
        results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    unique_identified = sum(1 for r in results if r.get("animal_id") is not None)
    count = len(results)

    if save_db:
        from core.database.session import async_session_factory
        from core.database.inference_persistence import save_inference_results
        from core.database.auto_enroll import auto_enroll_unknown_tracklets
        from core.config import repo_root

        async with async_session_factory() as session:
            auto_enrolled = await auto_enroll_unknown_tracklets(
                session=session,
                store=store,
                tracklet_results=results,
                tracklet_aggregated_embeddings=aggregated_by_tid,
                video_source=file.filename or "upload",
            )

            # Persist a few crops per tracklet/animal on disk + DB.
            crops_payload: list[dict[str, object]] = []
            events_payload: list[dict[str, object]] = []
            crops_dir = repo_root() / "core" / "data" / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            # build mapping track_id -> tracklet_id after save_inference_results (needs tracklets)
            await save_inference_results(
                session,
                file.filename or "upload",
                results,
                frame_detections,
                metrics={
                    "count": count,
                    "unique_identified": unique_identified,
                    "classification": {"enabled": classifier is not None},
                    "auto_enroll": {"created": len(auto_enrolled)},
                },
                animal_crops=[],
                extra_events=[],
            )

            # Tracklets are created above; fetch their ids for linking crops/events
            # (keep simple: map by track_id for this video_source)
            from sqlalchemy import select
            from core.database.models import Tracklet

            rows = await session.execute(
                select(Tracklet.id, Tracklet.track_id).where(Tracklet.video_source == (file.filename or "upload"))
            )
            tracklet_id_by_track: dict[int, int] = {int(tid): int(tid_db) for tid_db, tid in rows.all()}

            # classification summary per tracklet
            for r in results:
                tid = r.get("track_id")
                aid = r.get("animal_id")
                if tid is None:
                    continue
                preds = tracklet_classifications.get(int(tid), [])
                if not preds:
                    continue
                by_label: dict[str, list[float]] = {}
                for p in preds:
                    lbl = str(p.get("label", "unknown"))
                    sc = float(p.get("score", 0.0))
                    by_label.setdefault(lbl, []).append(sc)
                best_label = max(by_label.items(), key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))))[0]
                best_score = float(np.mean(by_label[best_label])) if by_label[best_label] else 0.0
                events_payload.append(
                    {
                        "event_type": "classification_summary",
                        "animal_id": aid,
                        "tracklet_id": tracklet_id_by_track.get(int(tid)),
                        "payload": {"label": best_label, "score": best_score, "num_frames": len(preds)},
                    }
                )

            # crops save
            try:
                import cv2
            except Exception:
                cv2 = None
            if cv2 is not None:
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
                        out_path = out_dir / f"{Path(file.filename or 'upload').stem}_t{int(tid)}_f{frame_index}_{j}.jpg"
                        cv2.imwrite(str(out_path), crop_bgr)
                        crops_payload.append(
                            {
                                "animal_id": int(aid),
                                "tracklet_id": tracklet_id_by_track.get(int(tid)),
                                "source_path": str(out_path),
                                "frame_index": frame_index,
                                "bbox": bbox,
                                "metadata": {"video_source": file.filename or "upload", "track_id": int(tid)},
                            }
                        )

            # Now persist crops + events in the same transaction.
            await save_inference_results(
                session,
                file.filename or "upload",
                [],
                {},
                metrics=None,
                animal_crops=crops_payload,
                extra_events=events_payload,
            )
            await session.commit()
            try:
                store.save()
            except Exception:
                pass

    from src.ai.metrics.tracklet_evaluation import compute_tracklet_evaluation_metrics

    evaluation_metrics = compute_tracklet_evaluation_metrics(
        tracklet_results=results,
        min_embeddings_threshold=10,
    )

    return {
        "frames_processed": frame_idx,
        "count": count,
        "unique_identified": unique_identified,
        "classification_enabled": classifier is not None,
        "auto_enrolled": sum(1 for r in results if r.get("enrolled")),
        "tracklets": [
            {"track_id": r["track_id"], "num_embeddings": r["num_embeddings"], "animal_id": r["animal_id"], "score": r["score"], "enrolled": r.get("enrolled", False)}
            for r in results
        ],
        "metrics": {
            "valid_animals": evaluation_metrics.valid_animals,
            "false_tracks": evaluation_metrics.false_tracks,
            "duplicate_count": evaluation_metrics.duplicate_count,
            "predicted_count": evaluation_metrics.predicted_count,
        },
    }


@router.post("/frames")
async def inference_frames(
    request: Request,
    files: list[UploadFile] = File(...),
    save_db: bool = Query(False, description="Persist tracklets and count to PostgreSQL"),
):
    """Upload list of frame images; same pipeline (YOLOv11 + DINOv3), batch embedding."""
    if not files:
        raise HTTPException(status_code=400, detail="No frames")
    try:
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="opencv-python required")
    frames = []
    for f in files:
        content = await f.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    if not frames:
        raise HTTPException(status_code=400, detail="No valid images")

    detector, classifier, tracker, cropper, encoder, decision = _get_inference_components(request)
    store = request.app.state.faiss_store
    tracklet_embeddings = defaultdict(list)
    tracklet_classifications: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    config = getattr(request.app.state, "config", {}) or {}
    crops_per_tracklet = int(config.get("inference", {}).get("crops_per_tracklet", 3))
    tracklet_crops: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_detections: dict[int, list[tuple[list[float], int]]] = {}

    for frame_idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        det_list = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked = tracker.update(det_list)
        frame_detections[frame_idx] = [(det["bbox"], det["track_id"]) for det in tracked if det.get("track_id")]
        batch = []
        cls_batch = []
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
        if classifier is not None and cls_batch:
            cls_tids, cls_crops = zip(*cls_batch)
            preds = classifier.predict_batch(list(cls_crops))
            for tid, p in zip(cls_tids, preds):
                tracklet_classifications[tid].append({"label": p.label, "score": float(p.score)})

    tracklet_results = []
    aggregated_by_tid: dict[int, np.ndarray] = {}
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            tracklet_results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        aggregated_by_tid[int(tid)] = agg
        animal_id, score = decision.decide(agg, store, top_k=5)
        tracklet_results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    unique_identified = sum(1 for r in tracklet_results if r.get("animal_id") is not None)
    count = len(tracklet_results)

    if save_db:
        from core.database.session import async_session_factory
        from core.database.inference_persistence import save_inference_results
        from core.database.auto_enroll import auto_enroll_unknown_tracklets
        from core.config import repo_root
        async with async_session_factory() as session:
            auto_enrolled = await auto_enroll_unknown_tracklets(
                session=session,
                store=store,
                tracklet_results=tracklet_results,
                tracklet_aggregated_embeddings=aggregated_by_tid,
                video_source="frames_upload",
            )
            await save_inference_results(
                session,
                "frames_upload",
                tracklet_results,
                frame_detections,
                metrics={
                    "count": count,
                    "unique_identified": unique_identified,
                    "classification": {"enabled": classifier is not None},
                    "auto_enroll": {"created": len(auto_enrolled)},
                },
                animal_crops=[],
                extra_events=[],
            )

            # link crops/events
            from sqlalchemy import select
            from core.database.models import Tracklet

            rows = await session.execute(select(Tracklet.id, Tracklet.track_id).where(Tracklet.video_source == "frames_upload"))
            tracklet_id_by_track: dict[int, int] = {int(tid): int(tid_db) for tid_db, tid in rows.all()}

            events_payload: list[dict[str, object]] = []
            for r in tracklet_results:
                tid = r.get("track_id")
                aid = r.get("animal_id")
                if tid is None:
                    continue
                preds = tracklet_classifications.get(int(tid), [])
                if not preds:
                    continue
                by_label: dict[str, list[float]] = {}
                for p in preds:
                    lbl = str(p.get("label", "unknown"))
                    sc = float(p.get("score", 0.0))
                    by_label.setdefault(lbl, []).append(sc)
                best_label = max(by_label.items(), key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))))[0]
                best_score = float(np.mean(by_label[best_label])) if by_label[best_label] else 0.0
                events_payload.append(
                    {
                        "event_type": "classification_summary",
                        "animal_id": aid,
                        "tracklet_id": tracklet_id_by_track.get(int(tid)),
                        "payload": {"label": best_label, "score": best_score, "num_frames": len(preds)},
                    }
                )

            crops_payload: list[dict[str, object]] = []
            crops_dir = repo_root() / "core" / "data" / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            for r in tracklet_results:
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
                    out_path = out_dir / f"frames_upload_t{int(tid)}_f{frame_index}_{j}.jpg"
                    cv2.imwrite(str(out_path), crop_bgr)
                    crops_payload.append(
                        {
                            "animal_id": int(aid),
                            "tracklet_id": tracklet_id_by_track.get(int(tid)),
                            "source_path": str(out_path),
                            "frame_index": frame_index,
                            "bbox": bbox,
                            "metadata": {"video_source": "frames_upload", "track_id": int(tid)},
                        }
                    )

            await save_inference_results(
                session,
                "frames_upload",
                [],
                {},
                metrics=None,
                animal_crops=crops_payload,
                extra_events=events_payload,
            )
            await session.commit()
            try:
                store.save()
            except Exception:
                pass

    from src.ai.metrics.tracklet_evaluation import compute_tracklet_evaluation_metrics

    evaluation_metrics = compute_tracklet_evaluation_metrics(
        tracklet_results=tracklet_results,
        min_embeddings_threshold=10,
    )

    return {
        "frames_processed": len(frames),
        "count": count,
        "unique_identified": unique_identified,
        "classification_enabled": classifier is not None,
        "auto_enrolled": sum(1 for r in tracklet_results if r.get("enrolled")),
        "tracklets": tracklet_results,
        "metrics": {
            "valid_animals": evaluation_metrics.valid_animals,
            "false_tracks": evaluation_metrics.false_tracks,
            "duplicate_count": evaluation_metrics.duplicate_count,
            "predicted_count": evaluation_metrics.predicted_count,
        },
    }
