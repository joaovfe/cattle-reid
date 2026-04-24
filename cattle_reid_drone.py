#!/usr/bin/env python3
"""
CLI simples (estilo Agro Vision) para o Cattle ReID Drone.
Fluxo único: analisar vídeo a partir da pasta videos/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.config import load_repo_dotenv


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _videos_dir() -> Path:
    path = _repo_root() / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _list_videos(videos_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(videos_dir.glob(f"*{ext}"))
    return sorted(videos)


def _choose_video(videos: list[Path]) -> Path | None:
    if not videos:
        print("\nNenhum vídeo encontrado em videos/.")
        print("Coloque seus arquivos em videos/ e rode novamente.")
        return None

    print("\nVideos disponiveis:")
    for i, video in enumerate(videos, start=1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {i}. {video.name} ({size_mb:.1f} MB)")

    choice = input("\nEscolha o numero do video para analisar: ").strip()
    try:
        idx = int(choice)
    except ValueError:
        print("Opcao invalida.")
        return None
    if idx < 1 or idx > len(videos):
        print("Opcao invalida.")
        return None
    return videos[idx - 1]


def _run_analysis(video_path: Path) -> int:
    repo_root = _repo_root()
    out_dir = repo_root / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir = repo_root / "core" / "data" / "faiss_index"

    output_video = out_dir / f"{video_path.stem}_anotado.mp4"
    output_json = out_dir / f"{video_path.stem}_resultado.json"

    print("Sincronizando FAISS a partir do banco (AnimalCrop + IDs dos bois)...")
    sync_command = [
        sys.executable,
        str(repo_root / "src" / "scripts" / "sync_faiss_from_db.py"),
        "--config",
        "configs/default.yaml",
        "--faiss",
        str(faiss_dir),
    ]
    sync_code = subprocess.call(sync_command, cwd=str(repo_root))
    if sync_code != 0:
        print(f"Falha ao sincronizar FAISS do banco (codigo {sync_code}).")
        return sync_code

    print("\nIniciando analise de video...")
    print(f"Video: {video_path}")
    print(f"Saida video: {output_video}")
    print(f"Saida json: {output_json}\n")

    command = [
        sys.executable,
        str(repo_root / "src" / "scripts" / "run_inference_video.py"),
        "--video",
        str(video_path),
        "--show_video",
        "--faiss",
        str(faiss_dir),
        "--output_video",
        str(output_video),
        "--output",
        str(output_json),
        "--save_db",
    ]
    return subprocess.call(command, cwd=str(repo_root))


def main() -> None:
    load_repo_dotenv()
    print("=" * 58)
    print(" CATTLE REID DRONE - CLI")
    print(" Opcao disponivel: analisar video")
    print("=" * 58)

    videos_dir = _videos_dir()
    print(f"\nPasta para analise: {videos_dir}")
    print("Coloque os videos nessa pasta e execute este comando novamente.\n")
    print("Base de identidade: banco de dados (imagens/AnimalCrop + IDs dos bois).")
    print("O FAISS e alimentado automaticamente a partir do banco antes da analise.\n")

    videos = _list_videos(videos_dir)
    selected = _choose_video(videos)
    if selected is None:
        raise SystemExit(1)

    exit_code = _run_analysis(selected)
    if exit_code == 0:
        print("\nAnalise finalizada com sucesso.")
    else:
        print(f"\nAnalise finalizada com erro (codigo {exit_code}).")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
        raise SystemExit(130)
