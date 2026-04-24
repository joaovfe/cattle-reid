"""
POST /inference/video and /inference/frames: run pipeline (YOLOv11 + DINOv3), return tracklets, count and metrics.
POST /inference/video/job: aceita vídeo e retorna 202 + job_id (evita timeout HTTP em vídeos longos).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings
from src.ai.metrics.count_metrics import compute_tracklet_evaluation_metrics

router = APIRouter()

_video_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def _minio_put_jpeg(storage, object_key: str, crop_bgr: np.ndarray, cv2) -> str:
    _, buf = cv2.imencode(".jpg", crop_bgr)
    return await asyncio.to_thread(storage.put_bytes, object_key, buf.tobytes(), "image/jpeg")


async def _minio_put_video_bytes(storage, object_key: str, content: bytes) -> str:
    return await asyncio.to_thread(storage.put_bytes, object_key, content, "video/mp4")


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


def _get_inference_components_from_config(config: dict):
    config = config or {}
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
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder, DINOV3_SMALL
    from core.config import resolve_path

    yolo_weights = _require_model_weights(
        resolve_path(yolo_cfg.get("weights")),
        model_label="YOLO detector",
        expected_filename="best_cow.pt",
    )
    detector = YOLOCattleDetector(
        model_path=yolo_weights,
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.45)),
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
        model_name=str(emb_cfg.get("model_name", DINOV3_SMALL)),
        device=emb_cfg.get("device"),
        half=bool(emb_cfg.get("half", True)),
    )
    decision = IdentityDecision(
        similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.25)),
        min_embeddings_per_tracklet=int(reid_cfg.get("min_embeddings_per_tracklet", 4)),
        top1_top2_margin=float(reid_cfg.get("top1_top2_margin", 0.08)),
    )
    return detector, classifier, tracker, cropper, encoder, decision


def _get_inference_components(request: Request):
    config = getattr(request.app.state, "config", {}) or {}
    return _get_inference_components_from_config(config)


def _sync_video_inference_core(
    config: dict,
    tmp_path: str,
    filename: str,
    upload_id: str,
    faiss_store,
) -> dict[str, Any]:
    """Pipeline CPU/GPU (OpenCV + YOLO + embeddings + FAISS decide). Executa fora do event loop."""
    import cv2

    detector, classifier, tracker, cropper, encoder, decision = _get_inference_components_from_config(config)
    store = faiss_store
    crops_per_tracklet = int(config.get("inference", {}).get("crops_per_tracklet", 3))
    video_stem = Path(filename or "upload").stem

    tracklet_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    tracklet_classifications: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    tracklet_crops: dict[int, list[dict[str, object]]] = defaultdict(list)
    frame_detections: dict[int, list[tuple[list[float], int]]] = {}
    frame_skip = 1
    frame_idx = 0

    cap = cv2.VideoCapture(tmp_path)
    try:
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
    finally:
        cap.release()

    results: list[dict[str, Any]] = []
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
    eval_metrics = compute_tracklet_evaluation_metrics(results)
    metrics = {
        "valid_animals": eval_metrics.valid_animals,
        "false_tracks": eval_metrics.false_tracks,
        "duplicate_animals": eval_metrics.duplicate_animals,
        "predicted_count": eval_metrics.predicted_count,
    }

    return {
        "results": results,
        "aggregated_by_tid": aggregated_by_tid,
        "tracklet_classifications": tracklet_classifications,
        "tracklet_crops": tracklet_crops,
        "frame_detections": frame_detections,
        "frame_idx": frame_idx,
        "count": count,
        "unique_identified": unique_identified,
        "metrics": metrics,
        "classifier_enabled": classifier is not None,
        "video_stem": video_stem,
        "upload_id": upload_id,
        "filename": filename or "upload",
    }


async def _persist_video_inference_to_db(
    *,
    minio_storage,
    faiss_store,
    content: bytes,
    core: dict[str, Any],
) -> str | None:
    from core.database.session import async_session_factory
    from core.database.inference_persistence import save_inference_results
    from core.database.auto_enroll import auto_enroll_unknown_tracklets
    from core.config import repo_root

    filename = core["filename"]
    video_stem = core["video_stem"]
    upload_id = core["upload_id"]
    results: list = core["results"]
    aggregated_by_tid = core["aggregated_by_tid"]
    tracklet_classifications = core["tracklet_classifications"]
    tracklet_crops = core["tracklet_crops"]
    frame_detections = core["frame_detections"]
    count = core["count"]
    unique_identified = core["unique_identified"]
    classification_enabled = core["classifier_enabled"]

    video_storage_url: str | None = None
    if minio_storage is not None:
        vkey = f"videos/{video_stem}_{upload_id}/source.mp4"
        video_storage_url = await _minio_put_video_bytes(minio_storage, vkey, content)

    async with async_session_factory() as session:
        auto_enrolled = await auto_enroll_unknown_tracklets(
            session=session,
            store=faiss_store,
            tracklet_results=results,
            tracklet_aggregated_embeddings=aggregated_by_tid,
            video_source=filename,
        )
        eval_metrics = compute_tracklet_evaluation_metrics(results)
        response_metrics = {
            "valid_animals": eval_metrics.valid_animals,
            "false_tracks": eval_metrics.false_tracks,
            "duplicate_animals": eval_metrics.duplicate_animals,
            "predicted_count": eval_metrics.predicted_count,
        }

        crops_payload: list[dict[str, object]] = []
        events_payload: list[dict[str, object]] = []
        crops_dir = repo_root() / "core" / "data" / "crops"
        if minio_storage is None:
            crops_dir.mkdir(parents=True, exist_ok=True)

        await save_inference_results(
            session,
            filename,
            results,
            frame_detections,
            metrics={
                "count": count,
                "unique_identified": unique_identified,
                **response_metrics,
                "classification": {"enabled": classification_enabled},
                "auto_enroll": {"created": len(auto_enrolled)},
            },
            animal_crops=[],
            extra_events=[],
        )

        from sqlalchemy import select
        from core.database.models import Tracklet

        rows = await session.execute(
            select(Tracklet.id, Tracklet.track_id).where(Tracklet.video_source == filename)
        )
        tracklet_id_by_track: dict[int, int] = {int(tid): int(tid_db) for tid_db, tid in rows.all()}

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
                    if minio_storage is not None:
                        okey = f"crops/animal_{int(aid)}/{video_stem}_t{int(tid)}_f{frame_index}_{j}.jpg"
                        stored_path = await _minio_put_jpeg(minio_storage, okey, crop_bgr, cv2)
                    else:
                        out_dir = Path(crops_dir) / f"animal_{int(aid)}"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{video_stem}_t{int(tid)}_f{frame_index}_{j}.jpg"
                        cv2.imwrite(str(out_path), crop_bgr)
                        stored_path = str(out_path)
                    crops_payload.append(
                        {
                            "animal_id": int(aid),
                            "tracklet_id": tracklet_id_by_track.get(int(tid)),
                            "source_path": stored_path,
                            "frame_index": frame_index,
                            "bbox": bbox,
                            "metadata": {
                                "video_source": filename,
                                "track_id": int(tid),
                                "upload_id": upload_id,
                                "storage": "minio" if minio_storage is not None else "local",
                            },
                        }
                    )

        await save_inference_results(
            session,
            filename,
            [],
            {},
            metrics=None,
            animal_crops=crops_payload,
            extra_events=events_payload,
        )
        await session.commit()
        try:
            faiss_store.save()
        except Exception:
            pass

    return video_storage_url


def _build_video_json_response(core: dict[str, Any], video_storage_url: str | None, save_db: bool) -> dict[str, object]:
    results = core["results"]
    eval_metrics = compute_tracklet_evaluation_metrics(results)
    out: dict[str, object] = {
        "frames_processed": core["frame_idx"],
        "count": core["count"],
        "unique_identified": core["unique_identified"],
        "classification_enabled": core["classifier_enabled"],
        "auto_enrolled": sum(1 for r in results if r.get("enrolled")),
        "metrics": {
            "valid_animals": eval_metrics.valid_animals,
            "false_tracks": eval_metrics.false_tracks,
            "duplicate_animals": eval_metrics.duplicate_animals,
            "predicted_count": eval_metrics.predicted_count,
        },
        "tracklets": [
            {"track_id": r["track_id"], "num_embeddings": r["num_embeddings"], "animal_id": r["animal_id"], "score": r["score"]}
            for r in results
        ],
    }
    if save_db and video_storage_url:
        out["video_storage_url"] = video_storage_url
    return out


@router.post("/video/job")
async def inference_video_submit_job(
    request: Request,
    file: UploadFile = File(...),
    save_db: bool = Query(False, description="Persist tracklets and count event to PostgreSQL"),
):
    """
    Enfileira inferência de vídeo e responde na hora com job_id.
    Use GET /inference/video/job/{job_id} para obter o resultado (evita timeout de cliente HTTP em vídeos longos).
    """
    content = await file.read()
    filename = file.filename or "upload"
    upload_id = str(uuid.uuid4())
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        import cv2

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        cap.release()
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}") from e

    job_id = str(uuid.uuid4())
    config = getattr(request.app.state, "config", {}) or {}
    store = request.app.state.faiss_store
    minio_storage = getattr(request.app.state, "minio", None)

    async with _jobs_lock:
        _video_jobs[job_id] = {"status": "pending", "filename": filename}

    path_for_worker = tmp_path

    async def _worker() -> None:
        try:
            async with _jobs_lock:
                _video_jobs[job_id]["status"] = "processing"
            core = await run_in_threadpool(
                _sync_video_inference_core,
                config,
                path_for_worker,
                filename,
                upload_id,
                store,
            )
            video_storage_url = None
            if save_db:
                video_storage_url = await _persist_video_inference_to_db(
                    minio_storage=minio_storage,
                    faiss_store=store,
                    content=content,
                    core=core,
                )
            out = _build_video_json_response(core, video_storage_url, save_db)
            async with _jobs_lock:
                _video_jobs[job_id] = {"status": "completed", "filename": filename, "result": out}
        except Exception as e:
            async with _jobs_lock:
                _video_jobs[job_id] = {"status": "failed", "filename": filename, "error": str(e)}
        finally:
            if path_for_worker:
                try:
                    os.unlink(path_for_worker)
                except Exception:
                    pass

    asyncio.create_task(_worker())

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "poll_url": f"/inference/video/job/{job_id}",
        },
    )


@router.get("/video/job/{job_id}")
async def inference_video_job_status(job_id: str):
    async with _jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/video")
async def inference_video(
    request: Request,
    file: UploadFile = File(...),
    save_db: bool = Query(False, description="Persist tracklets and count event to PostgreSQL"),
):
    """Upload video; detection (YOLOv11) -> tracking -> crop -> embed (DINOv3) -> Re-ID. Returns count and metrics."""
    content = await file.read()
    filename = file.filename or "upload"
    upload_id = str(uuid.uuid4())
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        import cv2

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        cap.release()
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}") from e

    config = getattr(request.app.state, "config", {}) or {}
    store = request.app.state.faiss_store

    try:
        core = await run_in_threadpool(
            _sync_video_inference_core,
            config,
            tmp_path,
            filename,
            upload_id,
            store,
        )
        video_storage_url = None
        if save_db:
            video_storage_url = await _persist_video_inference_to_db(
                minio_storage=getattr(request.app.state, "minio", None),
                faiss_store=store,
                content=content,
                core=core,
            )
        return _build_video_json_response(core, video_storage_url, save_db)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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

    frames_batch_id = str(uuid.uuid4())

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
    response_metrics = {}

    if save_db:
        from core.database.session import async_session_factory
        from core.database.inference_persistence import save_inference_results
        from core.database.auto_enroll import auto_enroll_unknown_tracklets
        from core.config import repo_root

        minio_storage = getattr(request.app.state, "minio", None)

        async with async_session_factory() as session:
            auto_enrolled = await auto_enroll_unknown_tracklets(
                session=session,
                store=store,
                tracklet_results=tracklet_results,
                tracklet_aggregated_embeddings=aggregated_by_tid,
                video_source="frames_upload",
            )
            eval_metrics = compute_tracklet_evaluation_metrics(tracklet_results)
            response_metrics = {
                "valid_animals": eval_metrics.valid_animals,
                "false_tracks": eval_metrics.false_tracks,
                "duplicate_animals": eval_metrics.duplicate_animals,
                "predicted_count": eval_metrics.predicted_count,
            }
            await save_inference_results(
                session,
                "frames_upload",
                tracklet_results,
                frame_detections,
                metrics={
                    "count": count,
                    "unique_identified": unique_identified,
                    **response_metrics,
                    "classification": {"enabled": classifier is not None},
                    "auto_enroll": {"created": len(auto_enrolled)},
                },
                animal_crops=[],
                extra_events=[],
            )

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
            if minio_storage is None:
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
                    if minio_storage is not None:
                        okey = f"crops/animal_{int(aid)}/frames_{frames_batch_id}_t{int(tid)}_f{frame_index}_{j}.jpg"
                        stored_path = await _minio_put_jpeg(minio_storage, okey, crop_bgr, cv2)
                    else:
                        out_dir = Path(crops_dir) / f"animal_{int(aid)}"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"frames_upload_t{int(tid)}_f{frame_index}_{j}.jpg"
                        cv2.imwrite(str(out_path), crop_bgr)
                        stored_path = str(out_path)
                    crops_payload.append(
                        {
                            "animal_id": int(aid),
                            "tracklet_id": tracklet_id_by_track.get(int(tid)),
                            "source_path": stored_path,
                            "frame_index": frame_index,
                            "bbox": bbox,
                            "metadata": {
                                "video_source": "frames_upload",
                                "track_id": int(tid),
                                "batch_id": frames_batch_id,
                                "storage": "minio" if minio_storage is not None else "local",
                            },
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
    else:
        eval_metrics = compute_tracklet_evaluation_metrics(tracklet_results)
        response_metrics = {
            "valid_animals": eval_metrics.valid_animals,
            "false_tracks": eval_metrics.false_tracks,
            "duplicate_animals": eval_metrics.duplicate_animals,
            "predicted_count": eval_metrics.predicted_count,
        }

    return {
        "frames_processed": len(frames),
        "count": count,
        "unique_identified": unique_identified,
        "classification_enabled": classifier is not None,
        "auto_enrolled": sum(1 for r in tracklet_results if r.get("enrolled")),
        "metrics": response_metrics,
        "tracklets": tracklet_results,
    }
