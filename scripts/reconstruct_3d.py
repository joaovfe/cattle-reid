#!/usr/bin/env python3
"""
Reconstrução 3D de cena de pasto via 3D Gaussian Splatting (gsplat + pycolmap).

Pipeline:
  1. Extrai frames do vídeo drone (a cada N frames)
  2. Roda COLMAP SfM para estimar poses de câmera e nuvem de pontos esparsa
  3. Treina modelo 3DGS via gsplat (30.000 iterações, ~8-15 min RTX 3060)
  4. Exporta cena como .ply (visualizável no MeshLab / Gaussian Viewer)
  5. Renderiza novel view (opcional) para uso no pipeline Re-ID

Uso:
  uv run python scripts/reconstruct_3d.py --video "videos 2/20m 1,5m_s.mp4"
  uv run python scripts/reconstruct_3d.py --video "videos 2/20m 1,5m_s.mp4" \
      --output data/3d_scene --n_frames 120 --iters 30000
"""
from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─── Extração de frames ───────────────────────────────────────────────────────

def extract_frames(video_path: Path, out_dir: Path, n_frames: int = 100) -> list[Path]:
    """Extrai N frames uniformemente distribuídos do vídeo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n_frames)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Vídeo: {total} frames @ {fps:.1f}fps | {w}x{h}")

    saved = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(saved) < n_frames:
            path = out_dir / f"frame_{idx:06d}.jpg"
            # Reduz resolução para 1280p max (COLMAP e 3DGS ficam mais rápidos)
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(path)
        idx += 1
    cap.release()
    print(f"  {len(saved)} frames extraídos → {out_dir}")
    return saved


# ─── COLMAP SfM ───────────────────────────────────────────────────────────────

def run_colmap(images_dir: Path, colmap_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roda COLMAP feature extraction + matching + mapping via pycolmap.
    Retorna: (points3D, cameras_c2w, intrinsics K)
    """
    import pycolmap

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print("  [COLMAP] Feature extraction...")
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.max_num_features = 4096
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_opts,
    )

    print("  [COLMAP] Feature matching (exhaustive para N < 200 frames)...")
    pycolmap.match_exhaustive(database_path=str(db_path))

    print("  [COLMAP] Incremental SfM...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )

    if not reconstructions:
        raise RuntimeError("COLMAP não conseguiu reconstruir a cena. Tente mais frames ou melhor sobreposição.")

    # incremental_mapping retorna dict {id: Reconstruction} no pycolmap 4.x
    rec = reconstructions[0] if isinstance(reconstructions, list) else reconstructions[0]
    n_imgs = len(rec.images)
    n_pts  = len(rec.points3D)
    print(f"  [COLMAP] OK: {n_imgs} câmeras, {n_pts} pontos 3D esparsos")

    # Extrai pontos 3D
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)  # (N, 3)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0  # (N, 3)

    # Extrai poses câmera → world (c2w) via cam_from_world (pycolmap 4.x API)
    c2w_list, K_list = [], []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        pose = img.cam_from_world        # Rigid3d: world→camera
        R_w2c = pose.rotation.matrix()  # (3, 3) world-to-camera
        t_w2c = pose.translation         # (3,)
        # Inverte para camera-to-world
        Rc2w = R_w2c.T
        tc2w = -R_w2c.T @ t_w2c
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = Rc2w
        c2w[:3, 3]  = tc2w
        c2w_list.append(c2w)
        cam = rec.cameras[img.camera_id]
        # Assume PINHOLE / SIMPLE_PINHOLE
        params = cam.params
        if cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]; cx = params[1]; cy = params[2]
        else:
            fx = params[0]; fy = params[1]; cx = params[2]; cy = params[3]
        K_list.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))

    c2w_arr = np.stack(c2w_list)  # (M, 4, 4)
    K_arr   = np.stack(K_list)    # (M, 3, 3)
    return pts, rgb, c2w_arr, K_arr


# ─── 3D Gaussian Splatting ────────────────────────────────────────────────────

def train_3dgs(
    points: np.ndarray,
    colors: np.ndarray,
    c2w: np.ndarray,
    Ks: np.ndarray,
    images_dir: Path,
    out_dir: Path,
    iters: int = 30_000,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Treina 3D Gaussian Splatting e exporta PLY."""
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [3DGS] Treinando {iters} iterações em {device}...")

    N = len(points)
    pts_t = torch.tensor(points, device=device, dtype=torch.float32)
    rgb_t = torch.tensor(colors, device=device, dtype=torch.float32)

    # Inicializa parâmetros gaussianos
    means   = torch.nn.Parameter(pts_t.clone())
    # Escala inicial: sqrt(distância média ao vizinho mais próximo)
    from torch import cdist
    dists = cdist(pts_t[:min(N, 10000)], pts_t[:min(N, 10000)]).topk(4, largest=False).values[:, 1:]
    scale_init = torch.log(dists.mean(dim=1).clamp(min=1e-6).sqrt()).mean()
    scales  = torch.nn.Parameter(scale_init.expand(N, 3).clone())
    quats   = torch.nn.Parameter(torch.zeros(N, 4, device=device))
    quats.data[:, 0] = 1.0  # identidade

    # Opacidades (sigmoid space)
    opacities = torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device)))

    # Cores SH grau 0 (RGB direto) + harmônicos superiores zerados
    sh0 = torch.nn.Parameter(rgb_to_sh_torch(rgb_t).unsqueeze(1))  # (N, 1, 3)
    shN = torch.nn.Parameter(torch.zeros(N, 15, 3, device=device))  # graus 1-3

    strategy = DefaultStrategy(verbose=False)
    state = strategy.initialize_state(scene_scale=1.0)

    opt_means   = torch.optim.Adam([means],     lr=1.6e-4, eps=1e-15)
    opt_scales  = torch.optim.Adam([scales],    lr=5e-3,   eps=1e-15)
    opt_quats   = torch.optim.Adam([quats],     lr=1e-3,   eps=1e-15)
    opt_opa     = torch.optim.Adam([opacities], lr=5e-2,   eps=1e-15)
    opt_sh0     = torch.optim.Adam([sh0],       lr=2.5e-3, eps=1e-15)
    opt_shN     = torch.optim.Adam([shN],       lr=2.5e-4 / 20, eps=1e-15)
    optimizers  = [opt_means, opt_scales, opt_quats, opt_opa, opt_sh0, opt_shN]

    M = len(c2w)
    # Carrega imagens de referência
    img_paths = sorted(images_dir.glob("frame_*.jpg"))
    ref_imgs: list[np.ndarray] = []
    for p in img_paths[:M]:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h_img, w_img = img.shape[:2]
        if w_img != width or h_img != height:
            img = cv2.resize(img, (width, height))
        ref_imgs.append(img)

    c2w_t  = torch.tensor(c2w[:len(ref_imgs)],  device=device, dtype=torch.float32)
    Ks_t   = torch.tensor(Ks[:len(ref_imgs)],   device=device, dtype=torch.float32)
    pixels = torch.tensor(np.stack(ref_imgs),    device=device, dtype=torch.float32)  # (M, H, W, 3)

    t0 = time.time()
    for step in range(1, iters + 1):
        # Amostra aleatória de câmeras
        cam_idx = torch.randint(0, len(ref_imgs), (1,)).item()
        c2w_i   = c2w_t[cam_idx:cam_idx+1]   # (1, 4, 4)
        K_i     = Ks_t[cam_idx:cam_idx+1]    # (1, 3, 3)
        gt      = pixels[cam_idx]             # (H, W, 3)

        for opt in optimizers:
            opt.zero_grad()

        sh_deg = min(3, step // 1000)
        renders, alphas, info = rasterization(
            means=means,
            quats=F.normalize(quats, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.cat([sh0, shN], dim=1),
            viewmats=torch.linalg.inv(c2w_i),
            Ks=K_i,
            width=width,
            height=height,
            sh_degree=sh_deg,
            packed=False,
            render_mode="RGB",
        )
        render = renders[0]  # (H, W, 3)

        l1   = F.l1_loss(render, gt)
        loss = l1
        loss.backward()

        strategy.step_pre_backward(
            params={"means": means, "scales": scales, "quats": quats,
                    "opacities": opacities, "sh0": sh0, "shN": shN},
            optimizers={"means": opt_means, "scales": opt_scales, "quats": opt_quats,
                        "opacities": opt_opa, "sh0": opt_sh0, "shN": opt_shN},
            state=state, step=step, info=info,
        )

        for opt in optimizers:
            opt.step()

        strategy.step_post_backward(
            params={"means": means, "scales": scales, "quats": quats,
                    "opacities": opacities, "sh0": sh0, "shN": shN},
            optimizers={"means": opt_means, "scales": opt_scales, "quats": opt_quats,
                        "opacities": opt_opa, "sh0": opt_sh0, "shN": opt_shN},
            state=state, step=step, info=info, lr=1.6e-4,
        )

        if step % 2000 == 0 or step == iters:
            elapsed = time.time() - t0
            n_gauss = means.shape[0]
            print(f"  iter {step:>6}/{iters}  loss={loss.item():.4f}  gaussians={n_gauss:>7,}  {elapsed:.0f}s")

    # Exporta PLY
    from gsplat import export_splats
    ply_path = out_dir / "scene.ply"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=means.detach(),
        scales=torch.exp(scales.detach()),
        quats=F.normalize(quats.detach(), dim=-1),
        opacities=torch.sigmoid(opacities.detach()),
        sh0=sh0.detach(),
        shN=shN.detach(),
        format="ply",
        save_to=str(ply_path),
    )
    print(f"  → PLY exportado: {ply_path} ({means.shape[0]:,} gaussianos)")
    return ply_path


def rgb_to_sh_torch(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] → coeficiente SH grau 0."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


# ─── Renderiza novel view ─────────────────────────────────────────────────────

def render_novel_view(ply_path: Path, out_path: Path, elevation_deg: float = -30.0) -> None:
    """Renderiza a cena de um novo ângulo usando os gaussianos salvos."""
    print(f"  [Render] Novel view a {elevation_deg}° de elevação...")
    # Placeholder — carrega PLY e renderiza (requer reconstrução completa)
    print(f"  Abra {ply_path} no SuperSplat Viewer:")
    print("  https://superspl.at/editor")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstrução 3D de cena de pasto via gsplat")
    p.add_argument("--video",    required=True, help="Caminho do vídeo drone")
    p.add_argument("--output",   default="data/3d_scene", help="Pasta de saída")
    p.add_argument("--n_frames", type=int, default=100,  help="Frames a extrair para SfM")
    p.add_argument("--iters",    type=int, default=30000, help="Iterações 3DGS")
    p.add_argument("--width",    type=int, default=1280)
    p.add_argument("--height",   type=int, default=720)
    p.add_argument("--skip_colmap", action="store_true", help="Pular SfM (usar poses existentes)")
    p.add_argument("--skip_3dgs",   action="store_true", help="Pular treinamento 3DGS (só extrai frames + COLMAP)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    out   = Path(args.output)

    print("\n" + "=" * 60)
    print("  RECONSTRUÇÃO 3D — 3D Gaussian Splatting (gsplat)")
    print("=" * 60)
    print(f"  Vídeo  : {video}")
    print(f"  Output : {out}")
    print(f"  Frames : {args.n_frames}  |  Iterações: {args.iters}")

    frames_dir = out / "frames"
    colmap_dir = out / "colmap"
    gs_dir     = out / "gaussians"

    # 1. Extração de frames
    print("\n[1/3] Extraindo frames...")
    frames = extract_frames(video, frames_dir, n_frames=args.n_frames)

    # 2. COLMAP SfM
    if not args.skip_colmap:
        print("\n[2/3] Rodando COLMAP SfM...")
        pts, rgb, c2w, Ks = run_colmap(frames_dir, colmap_dir)
    else:
        print("\n[2/3] COLMAP ignorado (--skip_colmap).")
        pts = rgb = c2w = Ks = None

    if args.skip_3dgs:
        print("\n[3/3] 3DGS ignorado (--skip_3dgs).")
        print("\n" + "=" * 60)
        print("  COLMAP concluído — pronto para 3DGS quando GPU estiver livre.")
        print(f"  Nuvem COLMAP: {colmap_dir}/sparse/")
        print("=" * 60 + "\n")
        return

    # 3. 3DGS
    print("\n[3/3] Treinando 3D Gaussian Splatting...")
    ply = train_3dgs(
        points=pts, colors=rgb,
        c2w=c2w, Ks=Ks,
        images_dir=frames_dir,
        out_dir=gs_dir,
        iters=args.iters,
        width=args.width,
        height=args.height,
    )

    print("\n" + "=" * 60)
    print("  CONCLUÍDO")
    print(f"  Cena 3D     : {ply}")
    print(f"  Nuvem COLMAP: {colmap_dir}/sparse/")
    print("\n  Visualizar no SuperSplat Viewer (browser):")
    print("  https://superspl.at/editor")
    print("\n  Ou via MeshLab (File → Import Mesh → scene.ply)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
