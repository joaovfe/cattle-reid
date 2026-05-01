"""
POST /inference/video and /inference/frames: run pipeline (YOLOv11 + DINOv3), return tracklets, count and metrics.
POST /inference/video/job: aceita vídeo e retorna 202 + job_id (evita timeout HTTP em vídeos longos).
"""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid
from collections import defaultdict
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings
from src.ai.metrics.count_metrics import compute_tracklet_evaluation_metrics

router = APIRouter()
_log = logging.getLogger(__name__)

_inference_stdout = os.environ.get("INFERENCE_LOG_STDOUT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "",
)


def _configure_inference_logger() -> None:
    """Garantir INFO no stderr (uvicorn não configura todos os named loggers)."""
    _log.setLevel(logging.INFO)
    if _log.handlers:
        return
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    _log.addHandler(h)
    _log.propagate = False


_configure_inference_logger()


def _speak_inference(msg: str) -> None:
    """Espelha no stdout para consola/`uv run`/WatchFiles sempre visível."""
    if _inference_stdout:
        print(msg, flush=True)


def _warn_missing_minio_for_annotated() -> None:
    msg = (
        "include_annotated_video está ativo mas MinIO não está ligado ao worker FastAPI "
        "(MINIO_ENABLED=false, erro em get_minio_storage ou credenciais). "
        "Não será enviado annotated_video_minio_key."
    )
    _log.warning(msg)
    _speak_inference(f"[inferência] WARN — {msg}")


_video_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def _minio_put_jpeg(storage, object_key: str, crop_bgr: np.ndarray, cv2) -> str:
    _, buf = cv2.imencode(".jpg", crop_bgr)
    b = storage.settings.crops_bucket()
    return await asyncio.to_thread(
        lambda: storage.put_bytes(object_key, buf.tobytes(), "image/jpeg", bucket=b)
    )


async def _minio_put_video_bytes(storage, object_key: str, content: bytes) -> str:
    b = storage.settings.videos_bucket()
    return await asyncio.to_thread(
        lambda: storage.put_bytes(object_key, content, "video/mp4", bucket=b)
    )


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
    inf_cfg = config.get("inference", {}) or {}
    imgsz_raw = inf_cfg.get("imgsz")
    try:
        yolo_imgsz = int(imgsz_raw) if imgsz_raw is not None else None
    except (TypeError, ValueError):
        yolo_imgsz = None
    if yolo_imgsz is not None and yolo_imgsz <= 0:
        yolo_imgsz = None

    detector = YOLOCattleDetector(
        model_path=yolo_weights,
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.45)),
        iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
        max_det=int(yolo_cfg.get("max_det", 300)),
        allowed_class_ids=yolo_cfg.get("allowed_class_ids"),
        allowed_class_names=yolo_cfg.get("allowed_class_names"),
        device=yolo_cfg.get("device"),
        half=bool(yolo_cfg.get("half", True)),
        imgsz=yolo_imgsz,
    )
    classifier = None
    require_classifier = bool(inf_cfg.get("require_classifier", True))
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
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        video_fps = float(fps_raw) if fps_raw and float(fps_raw) > 1e-6 else 0.0
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
        "video_fps": video_fps,
    }


async def _persist_video_inference_to_db(
    *,
    minio_storage,
    faiss_store,
    content: bytes,
    core: dict[str, Any],
) -> tuple[str | None, dict[int, str]]:
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

    thumbnail_minio_keys_by_track: dict[int, str] = {}

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
                        if j == 0:
                            thumbnail_minio_keys_by_track[int(tid)] = okey
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

    return video_storage_url, thumbnail_minio_keys_by_track


def _render_annotated_video_and_upload_sync(
    tmp_path: str,
    core: dict[str, Any],
    minio_storage,
) -> str | None:
    """Gera vídeo mp4 com caixas e IDs; envia ao bucket results do MinIO. Executa na threadpool."""
    import os
    import tempfile

    import cv2

    frame_detections_raw = core.get("frame_detections") or {}
    frame_detections: dict[int, list[tuple[list[float], int]]] = {}
    for k, v in frame_detections_raw.items():
        try:
            frame_detections[int(k)] = v  # type: ignore[assignment]
        except (TypeError, ValueError):
            continue

    results: list = core.get("results") or []
    tid_to_aid: dict[int, Any] = {}
    for r in results:
        tid = r.get("track_id")
        if tid is None:
            continue
        tid_to_aid[int(tid)] = r.get("animal_id")

    video_stem = str(core.get("video_stem") or "upload")
    upload_id = str(core.get("upload_id") or "")

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    writer: Any = None
    cap = cv2.VideoCapture(tmp_path)
    try:
        if not cap.isOpened():
            os.unlink(out_path)
            return None

        fps = float(core.get("video_fps") or 0)
        if fps <= 1e-6:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        w = max(1, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            os.unlink(out_path)
            return None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for item in frame_detections.get(frame_idx, []):
                if not item or len(item) < 2:
                    continue
                bbox = item[0]
                tid = item[1]
                if bbox is None or len(bbox) < 4:
                    continue
                try:
                    x1 = int(max(0.0, float(bbox[0])))
                    y1 = int(max(0.0, float(bbox[1])))
                    x2 = int(min(float(w - 1), float(bbox[2])))
                    y2 = int(min(float(h - 1), float(bbox[3])))
                except (TypeError, ValueError):
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                aid = tid_to_aid.get(int(tid))
                lab = f"T{int(tid)}"
                if aid is not None:
                    lab = f"{lab} ID:{aid}"
                cv2.putText(
                    frame,
                    lab,
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2,
                )
            writer.write(frame)
            frame_idx += 1
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        cap.release()

    try:
        with open(out_path, "rb") as f:
            data_b = f.read()
        os.unlink(out_path)
    except OSError:
        return None

    if not data_b:
        return None

    key = f"annotated/{video_stem}_{upload_id}/overlay.mp4"
    bkt = minio_storage.settings.results_bucket()
    try:
        minio_storage.put_bytes(key, data_b, "video/mp4", bucket=bkt)
    except Exception:
        _log.exception(
            "Falha ao enviar vídeo anotado ao MinIO bucket=%s key=%s (bytes=%s)",
            bkt,
            key,
            len(data_b),
        )
        return None
    _log.info("Vídeo anotado gravado em MinIO bucket=%s key=%s", bkt, key)
    _speak_inference(f"[inferência] vídeo anotado MinIO ok bucket={bkt} key={key}")
    return key


def _xyxy_list_to_bbox_dict(bbox: object) -> dict[str, float] | None:
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError):
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _build_video_json_response(
    core: dict[str, Any],
    video_storage_url: str | None,
    save_db: bool,
    *,
    thumbnail_minio_keys_by_track: dict[int, str],
    include_thumbnails: bool = False,
    max_thumbnail_tracklets: int = 50,
) -> dict[str, object]:
    import cv2

    results = core["results"]
    eval_metrics = compute_tracklet_evaluation_metrics(results)
    tracklet_crops: dict[int, list] = core.get("tracklet_crops") or {}
    fps = float(core.get("video_fps") or 0)

    tracklets_payload: list[dict[str, object]] = []
    thumb_budget = max(0, int(max_thumbnail_tracklets))
    for r in results:
        tid_raw = r.get("track_id")
        tid_int = int(tid_raw) if tid_raw is not None else None
        row: dict[str, object] = {
            "track_id": r["track_id"],
            "num_embeddings": r["num_embeddings"],
            "animal_id": r["animal_id"],
            "score": r["score"],
        }
        if tid_int is not None:
            first_items = tracklet_crops.get(tid_int) or []
            first = first_items[0] if first_items else None
            bbox_raw = first.get("bbox") if isinstance(first, dict) else None
            bbox_norm = _xyxy_list_to_bbox_dict(bbox_raw)
            fi = int(first.get("frame_index") or 0) if isinstance(first, dict) else None
            if bbox_norm:
                row["bounding_box"] = bbox_norm
            if fi is not None:
                row["frame_index"] = fi
                if fps > 1e-6:
                    row["frame_timestamp"] = float(fi) / fps

            tk = thumbnail_minio_keys_by_track.get(tid_int)
            if tk:
                row["thumbnail_minio_key"] = tk

            if include_thumbnails and thumb_budget > 0 and isinstance(first, dict):
                crop_bgr = first.get("crop_bgr")
                if crop_bgr is not None:
                    try:
                        _, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        row["crop_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
                        thumb_budget -= 1
                    except Exception:
                        pass

        tracklets_payload.append(row)

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
        "tracklets": tracklets_payload,
    }
    if save_db and video_storage_url:
        out["video_storage_url"] = video_storage_url
    return out


def _announce_pipeline_core_done(core: dict[str, Any], *, filename: str) -> None:
    fd = core.get("frame_detections") or {}
    if not isinstance(fd, dict):
        fd = {}
    frames_with_boxes = sum(1 for v in fd.values() if v)
    boxes = sum(len(v) for v in fd.values() if isinstance(v, list))
    results = core.get("results") or []
    n_res = len(results) if isinstance(results, list) else 0
    msg = (
        f"ficheiro={filename!r} frames_lidos={core.get('frame_idx')} "
        f"frames_com_bbox={frames_with_boxes} bbox_em_frames={boxes} "
        f"tracklets={n_res} count={core.get('count')} únicos_ReID={core.get('unique_identified')}"
    )
    _log.info("Pipeline vídeo termina — %s", msg)
    _speak_inference(f"[inferência pipeline] {msg}")


def _summarize_inference_out_for_log(out: dict[str, Any]) -> dict[str, Any]:
    raw = out.get("tracklets")
    if not isinstance(raw, list):
        return {"tracklets_len": 0}
    thumb_key = crop_b64 = 0
    sample: list[dict[str, Any]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        if t.get("thumbnail_minio_key"):
            thumb_key += 1
        crop_b64_val = t.get("crop_base64")
        if isinstance(crop_b64_val, str) and crop_b64_val.strip():
            crop_b64 += 1

    for t in raw[:10]:
        if not isinstance(t, dict):
            continue
        cb = t.get("crop_base64")
        sample.append(
            {
                "track_id": t.get("track_id"),
                "animal_id": t.get("animal_id"),
                "score": t.get("score"),
                "thumbnail_minio_key": bool(t.get("thumbnail_minio_key")),
                "crop_base64": isinstance(cb, str) and len(cb) > 0,
            },
        )
    return {
        "tracklets_len": len(raw),
        "tracklets_com_thumbnail_minio": thumb_key,
        "tracklets_com_crop_base64": crop_b64,
        "primeiros_tracklets": sample,
    }


def _log_inference_out_para_consumidor(
    out: dict[str, Any],
    *,
    tag: str,
    filename: str,
) -> None:
    resume = _summarize_inference_out_for_log(out)
    _log.info(
        "Inferência vídeo (%s): resposta cattle-api/downstream arquivo=%s count=%s unique=%s frames=%s "
        "classificação=%s auto_matriculas=%s metrics=%s annotated_minio=%s detail=%s",
        tag,
        filename,
        out.get("count"),
        out.get("unique_identified"),
        out.get("frames_processed"),
        out.get("classification_enabled"),
        out.get("auto_enrolled"),
        out.get("metrics"),
        bool(out.get("annotated_video_minio_key")),
        resume,
    )
    _speak_inference(
        f"[inferência → cattle-api] ({tag}) arquivo={filename!r} count={out.get('count')} "
        f"unique={out.get('unique_identified')} annotated_minio={bool(out.get('annotated_video_minio_key'))} "
        f"resumo={resume}",
    )


@router.post("/video/job")
async def inference_video_submit_job(
    request: Request,
    file: UploadFile = File(...),
    save_db: bool = Query(False, description="Persist tracklets and count event to PostgreSQL"),
    include_thumbnails: bool = Query(
        False,
        description="Incluir crop JPEG base64 nos tracklets (limite máx. por max_thumbnail_tracklets); aumenta payload",
    ),
    max_thumbnail_tracklets: int = Query(
        50,
        ge=0,
        le=500,
        description="Máximo de tracklets que recebem crop_base64 quando include_thumbnails=true",
    ),
    include_annotated_video: bool = Query(
        True,
        description="Gerar vídeo mp4 com overlays (MinIO bucket results); precisa MinIO ativo",
    ),
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
            _announce_pipeline_core_done(core, filename=filename)
            annotated_minio_key: str | None = None
            if include_annotated_video and minio_storage is None:
                _warn_missing_minio_for_annotated()
            if include_annotated_video and minio_storage is not None:
                annotated_minio_key = await run_in_threadpool(
                    _render_annotated_video_and_upload_sync,
                    path_for_worker,
                    core,
                    minio_storage,
                )
            persist_thumb_keys: dict[int, str] = {}
            video_storage_url: str | None = None
            if save_db:
                video_storage_url, persist_thumb_keys = await _persist_video_inference_to_db(
                    minio_storage=minio_storage,
                    faiss_store=store,
                    content=content,
                    core=core,
                )
            out = _build_video_json_response(
                core,
                video_storage_url,
                save_db,
                thumbnail_minio_keys_by_track=persist_thumb_keys,
                include_thumbnails=include_thumbnails,
                max_thumbnail_tracklets=max_thumbnail_tracklets,
            )
            if annotated_minio_key:
                out["annotated_video_minio_key"] = annotated_minio_key
            _log_inference_out_para_consumidor(
                out, tag=f"async_job:{job_id}", filename=filename
            )
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
    include_thumbnails: bool = Query(
        False,
        description="Incluir crop JPEG base64 por tracklet (limitado por max_thumbnail_tracklets)",
    ),
    max_thumbnail_tracklets: int = Query(50, ge=0, le=500),
    include_annotated_video: bool = Query(
        True,
        description="Gerar vídeo mp4 com overlays e enviar ao MinIO (bucket results)",
    ),
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
        _announce_pipeline_core_done(core, filename=filename)
        annotated_minio_key: str | None = None
        minio_st = getattr(request.app.state, "minio", None)
        if include_annotated_video and minio_st is None:
            _warn_missing_minio_for_annotated()
        if include_annotated_video and minio_st is not None:
            annotated_minio_key = await run_in_threadpool(
                _render_annotated_video_and_upload_sync,
                tmp_path,
                core,
                minio_st,
            )
        persist_thumb_keys: dict[int, str] = {}
        video_storage_url: str | None = None
        if save_db:
            video_storage_url, persist_thumb_keys = await _persist_video_inference_to_db(
                minio_storage=minio_st,
                faiss_store=store,
                content=content,
                core=core,
            )
        out = _build_video_json_response(
            core,
            video_storage_url,
            save_db,
            thumbnail_minio_keys_by_track=persist_thumb_keys,
            include_thumbnails=include_thumbnails,
            max_thumbnail_tracklets=max_thumbnail_tracklets,
        )
        if annotated_minio_key:
            out["annotated_video_minio_key"] = annotated_minio_key
        _log_inference_out_para_consumidor(out, tag="sync_post", filename=filename)
        return out
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
