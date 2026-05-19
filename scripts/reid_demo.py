#!/usr/bin/env python3
"""
Demo de re-identificação in-memory para TCC.
Auto-enrolla bovinos na galeria FAISS durante o processamento — sem DB.
Exporta vídeo anotado + screenshot + métricas JSON.

Uso:
  uv run python scripts/reid_demo.py \
      --video "videos 2/20m 1,5m_s.mp4" \
      --max_frames 600 \
      --frame_skip 2 \
      --gt_count 79
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",       required=True)
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--max_frames",  type=int, default=600, help="Frames máximos a processar")
    p.add_argument("--frame_skip",  type=int, default=2)
    p.add_argument("--gt_count",    type=int, default=None)
    p.add_argument("--output_video", default="data/output_reid_demo.mp4")
    p.add_argument("--screenshot",   default="data/screenshot_reid.jpg")
    p.add_argument("--output_json",  default="data/results_reid_demo.json")
    return p.parse_args()


def _draw(frame, frame_dets, track_to_animal, track_to_cls, frame_idx):
    import cv2
    colors = {}

    def get_color(aid):
        if aid not in colors:
            np.random.seed(aid % 1000)
            colors[aid] = tuple(np.random.randint(80, 255, 3).tolist())
        return colors[aid]

    for bbox, tid in frame_dets:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        aid = track_to_animal.get(tid)
        cls = track_to_cls.get(tid)
        label = f"ID:{aid}" if aid is not None else f"T{tid}"
        if cls:
            label += f" {cls['label'][:2].upper()}{cls['score']:.0%}"
        color = get_color(aid if aid is not None else tid + 1000)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(frame, f"Frame {frame_idx}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    enrolled = len([v for v in track_to_animal.values() if v is not None])
    cv2.putText(frame, f"IDs ativos: {enrolled}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    return frame


def main():
    args = parse_args()

    import yaml
    from src.ai.detection.yolo_detector import YOLOCattleDetector
    from src.ai.tracking.bytetrack_tracker import ByteTrackTracker
    from src.ai.oriented_crop.cropper import OrientedCropper
    from src.ai.embedding.dino_encoder import CattleEmbeddingEncoder
    from src.ai.reid.faiss_store import FAISSStore
    from src.ai.reid.identity_decision import IdentityDecision, aggregate_embeddings

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    yolo_cfg = cfg["models"]["yolo"]
    emb_cfg  = cfg["models"]["embedding"]
    reid_cfg = cfg["reid"]
    track_cfg = cfg["tracking"]
    crop_cfg  = cfg.get("oriented_crop", {})

    print("[1/5] Carregando modelos...")
    detector  = YOLOCattleDetector(
        weights=yolo_cfg["weights"],
        conf=yolo_cfg.get("conf_threshold", 0.43),
        iou=yolo_cfg.get("iou_threshold", 0.52),
    )
    tracker   = ByteTrackTracker(
        max_age=track_cfg.get("max_age", 35),
        iou_threshold=track_cfg.get("iou_threshold", 0.28),
    )
    cropper   = OrientedCropper(
        output_size=crop_cfg.get("output_size", 224),
        padding=crop_cfg.get("padding", 0.1),
    )
    encoder   = CattleEmbeddingEncoder(
        model_name=emb_cfg.get("model_name"),
        half=emb_cfg.get("half", True),
    )
    store     = FAISSStore(index_path="core/data/faiss_index_demo")
    decision  = IdentityDecision(
        similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.4530)),
        min_embeddings_per_tracklet=int(reid_cfg.get("min_embeddings_per_tracklet", 6)),
        top1_top2_margin=float(reid_cfg.get("top1_top2_margin", 0.12)),
    )

    print("[2/5] Abrindo vídeo...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {total} frames @ {fps:.1f}fps | {W}x{H}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (W, H))

    tracklet_embeddings: dict[int, list] = defaultdict(list)
    frame_detections: dict[int, list]   = {}
    track_to_animal: dict[int, int|None] = {}
    track_to_cls: dict[int, dict]        = {}
    next_animal_id = 1
    enrolled_tracks: set[int]            = set()
    screenshot_saved = False
    frame_idx = 0
    t0 = time.time()

    print("[3/5] Processando...")
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip != 0:
            out.write(frame)
            frame_idx += 1
            continue

        # Detecção + rastreamento
        detections = detector.detect(frame)
        det_list   = [{"bbox": d.bbox, "score": d.score} for d in detections]
        tracked    = tracker.update(det_list)
        frame_detections[frame_idx] = [(d["bbox"], d["track_id"]) for d in tracked if d.get("track_id")]

        # Embeddings
        batch = [(d["track_id"], cropper.crop(frame, d["bbox"])) for d in tracked if d.get("track_id")]
        if batch:
            tids, crops = zip(*batch)
            embs = encoder.encode_batch(list(crops))
            for tid, emb in zip(tids, embs):
                tracklet_embeddings[tid].append(emb)

                n_embs = len(tracklet_embeddings[tid])
                if n_embs < decision.min_embeddings_per_tracklet:
                    track_to_animal.setdefault(tid, None)
                    continue

                agg = aggregate_embeddings(tracklet_embeddings[tid])

                # Tenta identificar contra galeria
                if store.size > 0:
                    animal_id, score = decision.decide(agg, store, top_k=5)
                else:
                    animal_id, score = None, 0.0

                if animal_id is not None:
                    track_to_animal[tid] = animal_id
                elif tid not in enrolled_tracks:
                    # Auto-enroll: novo indivíduo
                    store.add(next_animal_id, agg)
                    track_to_animal[tid] = next_animal_id
                    enrolled_tracks.add(tid)
                    next_animal_id += 1
                else:
                    track_to_animal[tid] = track_to_animal.get(tid)

        draw = _draw(frame.copy(), frame_detections[frame_idx],
                     track_to_animal, track_to_cls, frame_idx)
        out.write(draw)

        # Screenshot quando tiver IDs suficientes
        n_ids = sum(1 for v in track_to_animal.values() if v is not None)
        if not screenshot_saved and n_ids >= 5:
            cv2.imwrite(args.screenshot, draw)
            screenshot_saved = True
            print(f"  Screenshot salvo em frame {frame_idx} ({n_ids} IDs visíveis)")

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{args.max_frames} | {next_animal_id-1} animais enrollados | {elapsed:.0f}s")

        frame_idx += 1

    cap.release()
    out.release()
    if not screenshot_saved:
        print("  AVISO: screenshot não gerado (poucos IDs visíveis)")

    # Métricas finais
    identified = sum(1 for v in track_to_animal.values() if v is not None)
    total_tracks = len(track_to_animal)
    enrolled_count = next_animal_id - 1

    print(f"\n[4/5] Métricas:")
    print(f"  Frames processados : {frame_idx}")
    print(f"  Tracks detectados  : {total_tracks}")
    print(f"  Animais enrollados : {enrolled_count}")
    print(f"  Tracks identificados: {identified}/{total_tracks} ({100*identified/max(total_tracks,1):.1f}%)")
    if args.gt_count:
        err = abs(enrolled_count - args.gt_count)
        acc = max(0, 100 - 100 * err / args.gt_count)
        print(f"  GT count={args.gt_count} | Erro={err} | Acurácia contagem={acc:.1f}%")

    results = {
        "frames_processed": frame_idx,
        "total_tracks": total_tracks,
        "enrolled_animals": enrolled_count,
        "identified_tracks": identified,
        "identification_rate_pct": round(100 * identified / max(total_tracks, 1), 2),
        "gt_count": args.gt_count,
        "count_error": abs(enrolled_count - args.gt_count) if args.gt_count else None,
        "count_accuracy_pct": round(max(0, 100 - 100 * abs(enrolled_count - args.gt_count) / args.gt_count), 2) if args.gt_count else None,
        "per_track": [{"track_id": int(k), "animal_id": v} for k, v in track_to_animal.items()],
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[5/5] Arquivos salvos:")
    print(f"  Vídeo : {args.output_video}")
    print(f"  Screenshot: {args.screenshot}")
    print(f"  JSON  : {args.output_json}")


if __name__ == "__main__":
    main()
