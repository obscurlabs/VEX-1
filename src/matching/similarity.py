"""Cosine similarity between normalized ArcFace embeddings.

A similarity is a distance measure, not a probability. Never present it as
"N% likely to be this person".
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG
from ..models import FaceEmbedding


def cosine_similarity(a: np.ndarray | FaceEmbedding, b: np.ndarray | FaceEmbedding) -> float:
    """Similarity in [-1, 1]. Inputs are expected to be L2-normalized."""
    va = a.vector if isinstance(a, FaceEmbedding) else a
    vb = b.vector if isinstance(b, FaceEmbedding) else b
    if va.shape != vb.shape:
        raise ValueError(f"dimension mismatch: {va.shape} vs {vb.shape}")
    # Divide by the norms anyway so the function is correct for raw vectors too.
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cannot compare a zero vector")
    return float(np.dot(va, vb) / (na * nb))


def best_match(target: FaceEmbedding, candidates: list[FaceEmbedding]) -> tuple[int, float]:
    """Highest-scoring candidate face. Returns (index, similarity)."""
    if not candidates:
        return -1, float("-inf")
    scores = [cosine_similarity(target, c) for c in candidates]
    i = int(np.argmax(scores))
    return i, scores[i]


def is_match(similarity: float, threshold: float | None = None) -> bool:
    return similarity >= (CONFIG.match.threshold if threshold is None else threshold)
