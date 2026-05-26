"""On-disk face identity profile.

Stores N L2-normalized embeddings produced by `face_detect.landmarks_to_embedding`.
`matches()` returns the maximum cosine similarity across all stored
captures, plus a boolean against `MATCH_THRESHOLD`. Storage is a single
`.npz` file under data/face_profile.npz (gitignored).

The threshold is high because cosine similarity on translation+scale
normalized landmark vectors of the same identity tends to sit well above
0.9; cross-identity similarity drops sharply. Tune empirically in Task 11.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MATCH_THRESHOLD = 0.92
PROFILE_VERSION = 1
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "face_profile.npz"

_log = logging.getLogger(__name__)


class FaceProfile:
    def __init__(self, embeddings: np.ndarray, created_at: str = "", version: int = PROFILE_VERSION):
        # embeddings is (N, D), float32, each row L2-normalized.
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.created_at = created_at
        self.version = version

    @classmethod
    def create_empty(cls, embedding_dim: int) -> "FaceProfile":
        empty = np.zeros((0, embedding_dim), dtype=np.float32)
        return cls(empty, created_at="", version=PROFILE_VERSION)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def capture_count(self) -> int:
        return int(self.embeddings.shape[0])

    def add_capture(self, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding of shape ({self.embedding_dim},), "
                f"got {emb.shape}"
            )
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            raise ValueError("Cannot add a zero-norm embedding")
        emb_unit = emb / norm
        self.embeddings = np.vstack([self.embeddings, emb_unit[None, :]])

    def matches(self, embedding: np.ndarray) -> tuple[bool, float]:
        if self.capture_count == 0:
            return False, 0.0
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            return False, 0.0
        emb_unit = emb / norm
        # Stored embeddings are already unit; cosine sim is just dot product.
        sims = self.embeddings @ emb_unit
        max_sim = float(sims.max())
        return (max_sim >= MATCH_THRESHOLD), max_sim

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            created_at=np.array(
                self.created_at or datetime.now(timezone.utc).isoformat()
            ),
            version=np.array(self.version),
        )

    @classmethod
    def load(cls, path: Path) -> "FaceProfile | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                embeddings = data["embeddings"]
                created_at = str(data["created_at"]) if "created_at" in data else ""
                version = int(data["version"]) if "version" in data else PROFILE_VERSION
        except (OSError, ValueError, KeyError, EOFError) as exc:
            _log.warning("Failed to load face profile at %s: %s", path, exc)
            return None
        if embeddings.ndim != 2:
            _log.warning("Face profile at %s has unexpected shape %s", path, embeddings.shape)
            return None
        return cls(embeddings.astype(np.float32, copy=False), created_at=created_at, version=version)
