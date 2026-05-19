"""
Train Re-ID encoder (DINOv2 backbone + projection head) on OpenCows2020 identification data.

Pipeline:
  - Backbone: facebook/dinov2-small (384-D CLS token)
  - Head: Linear(384 → 256) + L2-norm  →  256-D embedding
  - Loss: BatchHard TripletLoss (online hard mining within PK batch)
  - Sampler: P identities × K images per batch  (default P=8, K=4 → batch=32)
  - Freeze backbone except last 2 transformer blocks
  - Evaluate: Recall@1 and mAP on test split

Uso:
  uv run python -m src.training.train_reid \
      --data_root MetricLearningIdentification/datasets/OpenCows2020/identification/images \
      --epochs 30 --output data/checkpoints/reid
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ReIDDataset(Dataset):
    """ImageFolder with identity labels — each subdir is one cow ID."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Path, transform=None) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        for idx, cls_dir in enumerate(sorted(root.iterdir())):
            if not cls_dir.is_dir():
                continue
            self.class_to_idx[cls_dir.name] = idx
            self.idx_to_class[idx] = cls_dir.name
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((img, idx))

        # index by label for PK sampler
        self.label_to_indices: Dict[int, List[int]] = {}
        for i, (_, label) in enumerate(self.samples):
            self.label_to_indices.setdefault(label, []).append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


# ─── PK Sampler ───────────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """Sample P random identities × K random images per identity per iteration."""

    def __init__(self, dataset: ReIDDataset, P: int = 8, K: int = 4) -> None:
        self.dataset = dataset
        self.P = P
        self.K = K
        self.n_iters = len(dataset) // (P * K)

    def __iter__(self):
        label_to_idx = {
            lbl: list(idxs)
            for lbl, idxs in self.dataset.label_to_indices.items()
            if len(idxs) >= 2
        }
        labels = list(label_to_idx.keys())
        for _ in range(self.n_iters):
            batch = []
            chosen = random.sample(labels, min(self.P, len(labels)))
            for lbl in chosen:
                pool = label_to_idx[lbl]
                chosen_imgs = random.choices(pool, k=self.K)
                batch.extend(chosen_imgs)
            yield batch

    def __len__(self) -> int:
        return self.n_iters * self.P * self.K


# ─── Model ────────────────────────────────────────────────────────────────────

class ReIDEncoder(nn.Module):
    """DINOv3-small backbone + trainable projection head."""

    BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        import os
        from transformers import AutoModel
        token = os.environ.get("HF_TOKEN")
        self.backbone = AutoModel.from_pretrained(self.BACKBONE, token=token)

        # Freeze all backbone params
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Unfreeze last 2 transformer blocks (DINOv3: backbone.layer is a ModuleList)
        encoder_layers = self.backbone.layer
        for block in encoder_layers[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)

        # Unfreeze final norm
        norm = getattr(self.backbone, "norm", None) or getattr(self.backbone, "layernorm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)

        backbone_dim = self.backbone.config.hidden_size  # 384 for dinov2-small
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]  # CLS token
        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)


# ─── Batch Hard Triplet Loss ──────────────────────────────────────────────────

def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Online hard mining: for each anchor pick hardest positive (max dist same class)
    and hardest negative (min dist different class).
    """
    dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B) same identity
    labels_ne = ~labels_eq

    # Hardest positive: max intra-class distance
    dist_pos = dist * labels_eq.float()
    hardest_pos, _ = dist_pos.max(dim=1)

    # Hardest negative: min inter-class distance (mask same-class with large value)
    dist_neg = dist + labels_eq.float() * 1e9
    hardest_neg, _ = dist_neg.min(dim=1)

    loss = F.relu(hardest_pos - hardest_neg + margin).mean()
    return loss


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Compute Recall@1 and mAP on the full split (gallery = entire split)."""
    model.eval()
    all_embs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        embs = model(imgs)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    embs = torch.cat(all_embs)            # (N, D)
    labels = torch.cat(all_labels)        # (N,)

    # Cosine similarity matrix (embeddings already L2-normalised → dot product = cosine)
    sim = embs @ embs.T                   # (N, N)
    sim.fill_diagonal_(-2.0)              # exclude self

    labels_np = labels.numpy()
    r1_hits, ap_list = 0, []
    for i in range(len(labels_np)):
        scores = sim[i].numpy()
        sorted_idx = np.argsort(-scores)
        gt = labels_np[i]
        # Recall@1
        if labels_np[sorted_idx[0]] == gt:
            r1_hits += 1
        # mAP
        hits, cum_prec = 0, []
        for rank, j in enumerate(sorted_idx):
            if labels_np[j] == gt:
                hits += 1
                cum_prec.append(hits / (rank + 1))
        ap_list.append(np.mean(cum_prec) if cum_prec else 0.0)

    n = len(labels_np)
    return {"recall@1": r1_hits / n, "mAP": float(np.mean(ap_list))}


# ─── Threshold computation ────────────────────────────────────────────────────

@torch.no_grad()
def compute_threshold(model: nn.Module, loader: DataLoader, device: str) -> float:
    """τ = midpoint between mean positive and mean negative cosine similarity."""
    model.eval()
    all_embs, all_labels = [], []
    for imgs, labels in loader:
        all_embs.append(model(imgs.to(device)).cpu())
        all_labels.append(labels)
    embs = torch.cat(all_embs)
    labels = torch.cat(all_labels).numpy()

    sim = (embs @ embs.T).numpy()
    np.fill_diagonal(sim, -2.0)
    pos_sims, neg_sims = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                pos_sims.append(sim[i, j])
            else:
                neg_sims.append(sim[i, j])
    tau = (float(np.mean(pos_sims)) + float(np.mean(neg_sims))) / 2.0
    print(f"  pos_sim μ={np.mean(pos_sims):.4f}  neg_sim μ={np.mean(neg_sims):.4f}  τ={tau:.4f}")
    return tau


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Re-ID encoder (DINOv2 + head, TripletLoss)")
    p.add_argument("--data_root", required=True,
                   help="Root with train/ and test/ (or val/) subdirs of class folders")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--P", type=int, default=8, help="Identities per batch")
    p.add_argument("--K", type=int, default=4, help="Images per identity per batch")
    p.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--output", default="data/checkpoints/reid")
    p.add_argument("--config", default="configs/train_reid.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    train_root = root / "train"
    val_root = root / "test" if (root / "test").exists() else root / "val"

    print("\n" + "=" * 60)
    print("  Re-ID Training — DINOv2-small + TripletLoss")
    print("=" * 60)
    print(f"  Data   : {root}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  P={args.P}  K={args.K}  margin={args.margin}")
    print(f"  Output : {out_dir}")

    # ── Transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ReIDDataset(train_root, transform=train_tf)
    val_ds   = ReIDDataset(val_root,   transform=val_tf)
    print(f"\n  Train  : {len(train_ds)} imgs, {train_ds.num_classes} identities")
    print(f"  Val    : {len(val_ds)} imgs, {val_ds.num_classes} identities")

    sampler = PKSampler(train_ds, P=args.P, K=args.K)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── Model
    print("\n  Loading DINOv2-small backbone...")
    model = ReIDEncoder(embed_dim=args.embed_dim).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params : {trainable:,} trainable / {total:,} total")

    # ── Optimizer: different LR for backbone vs head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop
    best_r1 = 0.0
    history = []
    print(f"\n{'Epoch':>5} {'Loss':>8} {'R@1':>6} {'mAP':>6} {'Time':>6}")
    print("-" * 40)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = model(imgs)
            loss = batch_hard_triplet_loss(embs, labels, margin=args.margin)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)

        metrics = evaluate(model, val_loader, device)
        r1  = metrics["recall@1"]
        mAP = metrics["mAP"]
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "recall@1": r1, "mAP": mAP})

        marker = " ★" if r1 > best_r1 else ""
        print(f"{epoch:>5} {avg_loss:>8.4f} {r1:>6.3f} {mAP:>6.3f} {elapsed:>5.0f}s{marker}")

        if r1 > best_r1:
            best_r1 = r1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "recall@1": r1,
                "mAP": mAP,
                "embed_dim": args.embed_dim,
            }, out_dir / "best_reid.pt")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "recall@1": r1,
            "mAP": mAP,
        }, out_dir / "last_reid.pt")

    # ── Final metrics
    print("\n" + "=" * 60)
    print(f"  Melhor Recall@1 : {best_r1:.4f}")
    best = max(history, key=lambda x: x["recall@1"])
    print(f"  Melhor época    : {best['epoch']} (loss={best['loss']:.4f}, mAP={best['mAP']:.4f})")
    print(f"  Checkpoint      : {out_dir / 'best_reid.pt'}")

    # ── Compute threshold τ
    print("\n  Calculando threshold τ de similaridade...")
    # Reload best
    ckpt = torch.load(out_dir / "best_reid.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tau = compute_threshold(model, val_loader, device)
    print(f"\n  → Adicione ao configs/default.yaml: similarity_threshold: {tau:.4f}")

    # ── Save CSV history
    import csv
    csv_path = out_dir / "reid_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "recall@1", "mAP"])
        writer.writeheader()
        writer.writerows(history)
    print(f"  Histórico salvo : {csv_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
