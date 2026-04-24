"""
Embedding encoder for cattle crops: DINOv3 or DINOv2 via HuggingFace.
Crops (BGR or RGB) -> L2-normalized embeddings.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


# DINOv3 (HuggingFace): requires transformers with DINOv3 support
DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
# DINOv2 (mais compatível por padrão)
DINOV2_SMALL = "facebook/dinov2-small"


class CattleEmbeddingEncoder:
    """
    Extract global embeddings from cattle crops using DINOv3 or DINOv2 (HuggingFace).
    Default: DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m). Para DINOv2 use model_name=
    facebook/dinov2-small.
    """

    def __init__(
        self,
        model_name: str = DINOV3_SMALL,
        device: str | None = None,
        half: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._half = half
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        if self._half and self.device.startswith("cuda"):
            self._model = self._model.half()
        self._model.eval()

    @torch.no_grad()
    def encode_one(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Single crop (H, W, 3) BGR -> embedding (D,) float32, L2-normalized."""
        self._load()
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self._half and self.device.startswith("cuda"):
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        out = self._model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state.mean(dim=1)
        emb = emb.squeeze(0).cpu().float().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @torch.no_grad()
    def encode_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """List of crops -> (N, D) float32, L2-normalized."""
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        rgb_list = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops]
        inputs = self._processor(images=rgb_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self._half and self.device.startswith("cuda"):
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        out = self._model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state.mean(dim=1)
        emb = emb.cpu().float().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        emb = emb / norms
        return emb
