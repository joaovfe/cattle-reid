"""Extrai frames de vídeos para preparar dados de treino YOLO (Ultralytics / formato YOLO).

Uso rápido::

    python -m src.scripts.extract_video_frames \\
        --input-dir /data/videos/raw \\
        --output-dir /data/yolo_dataset \\
        --target-fps 2 \\
        --split video --val-ratio 0.2 --seed 42 \\
        --write-yaml

    python -m src.scripts.extract_video_frames \\
        --videos a.mp4 b.mp4 \\
        --output-dir ./frames_out \\
        --stride 30 \\
        --format png \\
        --resize 1280 720

Modos de amostragem (mutuamente exclusivos):

- ``--target-fps``: intervalo derivado do FPS do vídeo (~ ``stride = round(fps_video / target_fps)``).
- ``--stride``: guarda 1 frame em cada N frames do vídeo (N=1 = todos).

Splits opcionais:

- ``video``: cada vídeo vai inteiro para train ou val (útil para evitar leakage).
- ``frame``: cada frame extraído é atribuído aleatoriamente a train/val com ``--val-ratio``.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _collect_videos(input_dir: Path | None, extra: list[Path], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    if input_dir is not None:
        if not input_dir.is_dir():
            raise SystemExit(f"Diretório não encontrado: {input_dir}")
        globber = input_dir.rglob if recursive else input_dir.glob
        for p in globber("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(p.resolve())
    for p in extra:
        if not p.is_file():
            raise SystemExit(f"Vídeo não encontrado: {p}")
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Aviso: extensão não listada como vídeo comum: {p}", file=sys.stderr)
        paths.append(p.resolve())
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise SystemExit("Nenhum vídeo encontrado: use --input-dir e/ou --videos.")
    return sorted(unique, key=lambda x: str(x))


def _ensure_label_dirs(root: Path, split: str | None) -> None:
    if split is None:
        (root / "labels").mkdir(parents=True, exist_ok=True)
        return
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def _destination_dir(output_dir: Path, split_mode: str | None, bucket: str) -> Path:
    if split_mode is None:
        return output_dir / "images"
    return output_dir / "images" / bucket


def _write_yaml_stub(
    output_dir: Path,
    split_mode: str | None,
    dataset_name: str,
    nc: int,
    names: list[str],
) -> Path:
    """Gera um YAML mínimo compatível com Ultralytics (paths relativos ao ficheiro)."""
    if split_mode is None:
        train_rel = "images"
        val_rel = "images"
    else:
        train_rel = "images/train"
        val_rel = "images/val"

    payload = {
        "path": str(output_dir.resolve()),
        "train": train_rel,
        "val": val_rel,
        "names": {i: n for i, n in enumerate(names)},
        "nc": nc,
        # Metadados opcionais para referência humana
        "dataset": dataset_name,
    }
    out = output_dir / "dataset.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out


def _save_frame(
    frame: np.ndarray,
    path: Path,
    fmt: str,
    jpeg_quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpg":
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
    else:
        ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Falha ao gravar: {path}")


def _maybe_resize(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None and height is None:
        return frame
    if width is None or height is None:
        raise SystemExit("--resize requer largura e altura.")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai frames de vídeos para labeling YOLO.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Pasta com vídeos.")
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="*",
        default=[],
        help="Lista explícita de ficheiros de vídeo.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta de saída do dataset.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Com --input-dir, procura vídeos recursivamente.",
    )

    samp = parser.add_mutually_exclusive_group()
    samp.add_argument(
        "--target-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Amostragem aproximada: stride ≈ round(fps_do_vídeo / este valor).",
    )
    samp.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="N",
        help="Guardar 1 em cada N frames (predefinição 1 se não usar --target-fps).",
    )

    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        metavar="M",
        help="Máximo de frames extraídos por vídeo (ordem cronológica).",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="Formato de imagem.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        metavar="Q",
        help="Qualidade JPEG (1-100), apenas para .jpg.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Redimensionar para largura x altura antes de gravar.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente para splits aleatórios.")

    parser.add_argument(
        "--split",
        choices=("none", "video", "frame"),
        default="none",
        help="Split train/val: por vídeo inteiro ou por frame; 'none' = só images/.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proporção para val (0–1), usado com --split video ou frame.",
    )

    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Escrever dataset.yaml (stub Ultralytics) em --output-dir.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="novus_frames",
        help="Nome documental no YAML.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=["bovine"],
        help="Nomes de classes para o stub YAML (predefinição: bovine).",
    )

    args = parser.parse_args()

    if args.stride is not None and args.stride < 1:
        raise SystemExit("--stride deve ser >= 1.")
    if args.target_fps is not None and args.target_fps <= 0:
        raise SystemExit("--target-fps deve ser > 0.")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise SystemExit("--val-ratio deve estar entre 0 e 1.")

    # Sem --target-fps: usar --stride ou 1.
    stride_fixed = (
        args.stride
        if args.stride is not None
        else (1 if args.target_fps is None else None)
    )

    jpeg_quality = max(1, min(100, args.jpeg_quality))

    videos = _collect_videos(args.input_dir, list(args.videos), args.recursive)
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    split_mode = None if args.split == "none" else args.split
    ext = args.format
    if ext == "jpg":
        suffix = ".jpg"
    else:
        suffix = ".png"

    rng = random.Random(args.seed)

    # Atribuição train/val por vídeo (inteiro)
    val_video_paths: set[Path] = set()
    if split_mode == "video":
        shuffled = videos.copy()
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * args.val_ratio))
        n_val = min(n_val, len(shuffled))
        val_video_paths = set(shuffled[:n_val])

    _ensure_label_dirs(out_root, split_mode)

    total_written = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Aviso: não foi possível abrir {video_path}, a saltar.", file=sys.stderr)
            continue

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 1e-6:
            native_fps = 30.0

        if args.target_fps is not None:
            stride = max(1, int(round(native_fps / args.target_fps)))
        else:
            stride = stride_fixed if stride_fixed is not None else 1

        stem = video_path.stem
        extracted = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            frame = _maybe_resize(
                frame,
                args.resize[0] if args.resize else None,
                args.resize[1] if args.resize else None,
            )

            if split_mode == "frame":
                bucket = "val" if rng.random() < args.val_ratio else "train"
            elif split_mode == "video":
                bucket = "val" if video_path in val_video_paths else "train"
            else:
                bucket = "train"

            dest_root = _destination_dir(out_root, split_mode, bucket)
            out_path = dest_root / f"{stem}_f{frame_idx:06d}{suffix}"

            _save_frame(frame, out_path, ext, jpeg_quality)
            total_written += 1
            extracted += 1

            if args.max_frames_per_video is not None and extracted >= args.max_frames_per_video:
                break
            frame_idx += 1

        cap.release()

    if args.write_yaml:
        yaml_path = _write_yaml_stub(
            out_root,
            split_mode,
            args.dataset_name,
            len(args.classes),
            list(args.classes),
        )
        print(f"YAML: {yaml_path}")

    print(f"Frames gravados: {total_written} em {out_root}")


if __name__ == "__main__":
    main()
