"""
FAISS index for Novus embeddings: add, search, rebuild, persist to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class FAISSStore:
    """In-memory FAISS index + mapping to animal_id. Persists index and id mapping to disk."""

    def __init__(self, index_path: str | Path | None = None, metric: str = "cosine") -> None:
        self.index_path = Path(index_path) if index_path else None
        self.metric = metric
        self._index = None
        self._id_list: List[int] = []
        self._dim: int | None = None

    def _ensure_index(self, dim: int) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("Install faiss-cpu: uv add faiss-cpu") from e
        if self._index is not None and self._dim == dim:
            return
        self._dim = dim
        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)
        if self._id_list:
            self._id_list = []

    def add(self, animal_id: int, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if self.metric == "cosine" and np.abs(np.linalg.norm(emb) - 1.0) > 1e-5:
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
        self._ensure_index(emb.shape[0])
        self._index.add(emb.reshape(1, -1))
        self._id_list.append(animal_id)

    def add_batch(self, animal_ids: List[int], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            embeddings = embeddings / norms
        self._ensure_index(embeddings.shape[1])
        self._index.add(embeddings)
        self._id_list.extend(animal_ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or len(self._id_list) == 0:
            return np.array([]), np.array([], dtype=np.int64)
        q = np.asarray(query, dtype=np.float32).ravel().reshape(1, -1)
        if self.metric == "cosine":
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
        k = min(k, self._index.ntotal)
        if k <= 0:
            return np.array([]), np.array([], dtype=np.int64)
        scores, indices = self._index.search(q, k)
        return scores[0], indices[0]

    def search_animal_ids(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        scores, indices = self.search(query, k)
        out: List[Tuple[int, float]] = []
        for idx, sc in zip(indices, scores):
            if 0 <= idx < len(self._id_list):
                out.append((self._id_list[idx], float(sc)))
        return out

    def rebuild_index(self, animal_ids: List[int], embeddings: np.ndarray) -> None:
        self._index = None
        self._dim = None
        self._id_list = []
        self.add_batch(animal_ids, embeddings)

    def save(self, path: str | Path | None = None) -> None:
        path = Path(path or self.index_path) if (path or self.index_path) else None
        if path is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        try:
            import faiss
            index_file = path / "index.faiss"
            if self._index is not None:
                faiss.write_index(self._index, str(index_file))
            meta = path / "meta.json"
            with open(meta, "w") as f:
                json.dump({"id_list": self._id_list, "dim": self._dim, "metric": self.metric}, f)
        except Exception:
            pass

    def load(self, path: str | Path | None = None) -> None:
        path = Path(path or self.index_path) if (path or self.index_path) else None
        if path is None or not path.exists():
            return
        try:
            import faiss
            index_file = path / "index.faiss"
            if index_file.exists():
                self._index = faiss.read_index(str(index_file))
                self._dim = self._index.d
            meta = path / "meta.json"
            if meta.exists():
                with open(meta) as f:
                    data = json.load(f)
                    self._id_list = data.get("id_list", [])
                    self._dim = self._dim or data.get("dim")
                    self.metric = data.get("metric", self.metric)
        except Exception:
            self._index = None
            self._id_list = []
            self._dim = None

    def __len__(self) -> int:
        return len(self._id_list)
