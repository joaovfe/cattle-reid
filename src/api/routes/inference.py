"""
POST /inference/video and /inference/frames: run pipeline (YOLOv11 + DINOv3), return tracklets, count and metrics.
"""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query

router = APIRouter()


def _get_inference_components():
    from src.ai.detection.yolo_detector import YOLOCattleDetector
    from src.ai.tracking.bytetrack_tracker import ByteTrackTracker, TrackerConfig
    from src.ai.oriented_crop.cropper import OrientedCropper
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV2_SMALL
    from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings
    detector = YOLOCattleDetector(conf_threshold=0.55, iou_threshold=0.45, half=True)
    tracker = ByteTrackTracker(TrackerConfig(max_age=30, iou_threshold=0.3))
    cropper = OrientedCropper(224, 0.1)
    encoder = CattleEmbeddingEncoder(model_name=DINOV2_SMALL, half=True)
    decision = IdentityDecision(similarity_threshold=0.25, min_embeddings_per_tracklet=4)
    return detector, tracker, cropper, encoder, decision


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

    detector, tracker, cropper, encoder, decision = _get_inference_components()
    store = request.app.state.faiss_store

    tracklet_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
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
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    results = []
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        animal_id, score = decision.decide(agg, store, top_k=5)
        results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    unique_identified = sum(1 for r in results if r.get("animal_id") is not None)
    count = len(results)

    if save_db:
        from core.database.session import async_session_factory
        from core.database.inference_persistence import save_inference_results
        async with async_session_factory() as session:
            await save_inference_results(
                session,
                file.filename or "upload",
                results,
                frame_detections,
                metrics={"count": count, "unique_identified": unique_identified},
            )
            await session.commit()

    return {
        "frames_processed": frame_idx,
        "count": count,
        "unique_identified": unique_identified,
        "tracklets": [
            {"track_id": r["track_id"], "num_embeddings": r["num_embeddings"], "animal_id": r["animal_id"], "score": r["score"]}
            for r in results
        ],
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

    detector, tracker, cropper, encoder, decision = _get_inference_components()
    store = request.app.state.faiss_store
    tracklet_embeddings = defaultdict(list)
    frame_detections: dict[int, list[tuple[list[float], int]]] = {}

    for frame_idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        det_list = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked = tracker.update(det_list)
        frame_detections[frame_idx] = [(det["bbox"], det["track_id"]) for det in tracked if det.get("track_id")]
        batch = []
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

    tracklet_results = []
    for tid, embs in tracklet_embeddings.items():
        if len(embs) < decision.min_embeddings_per_tracklet:
            tracklet_results.append({"track_id": tid, "animal_id": None, "score": 0.0, "num_embeddings": len(embs)})
            continue
        agg = aggregate_embeddings(embs)
        animal_id, score = decision.decide(agg, store, top_k=5)
        tracklet_results.append({"track_id": tid, "animal_id": animal_id, "score": float(score), "num_embeddings": len(embs)})

    unique_identified = sum(1 for r in tracklet_results if r.get("animal_id") is not None)
    count = len(tracklet_results)

    if save_db:
        from core.database.session import async_session_factory
        from core.database.inference_persistence import save_inference_results
        async with async_session_factory() as session:
            await save_inference_results(
                session,
                "frames_upload",
                tracklet_results,
                frame_detections,
                metrics={"count": count, "unique_identified": unique_identified},
            )
            await session.commit()

    return {
        "frames_processed": len(frames),
        "count": count,
        "unique_identified": unique_identified,
        "tracklets": tracklet_results,
    }
