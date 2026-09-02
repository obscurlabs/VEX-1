"""Shared fixtures. The model pack loads once for the whole session."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.embedder import ArcFaceEmbedder

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def embedder() -> ArcFaceEmbedder:
    return ArcFaceEmbedder()


@pytest.fixture(scope="session")
def detector(embedder) -> object:
    return embedder.detector


@pytest.fixture(scope="session")
def group_image() -> np.ndarray:
    """A photograph containing several distinct people."""
    from insightface.data import get_image

    return get_image("t1")


@pytest.fixture(scope="session")
def portrait(corpus) -> np.ndarray:
    """A single-face photograph from the confirmed corpus."""
    identity = sorted(corpus)[0]
    return cv2.imread(str(corpus[identity][0]))


@pytest.fixture(scope="session")
def multi_image_identities(corpus) -> dict[str, list[Path]]:
    """Only the identities that have two or more confirmed photographs."""
    return {k: v for k, v in corpus.items() if len(v) >= 2}


@pytest.fixture(scope="session")
def corpus() -> dict[str, list[Path]]:
    """Confirmed identity -> image paths. Skips the suite if not built."""
    manifest = DATA / "confirmed.json"
    if not manifest.exists():
        pytest.skip("test corpus missing - run scripts/fetch_test_images.py")

    out: dict[str, list[Path]] = {}
    for slug in json.loads(manifest.read_text()):
        paths = sorted((DATA / slug).glob("*.jpg"))
        if paths:
            out[slug] = paths
    # Single-image identities still contribute different-person pairs; only
    # same-person pairs need an identity to have two or more.
    if sum(1 for v in out.values() if len(v) >= 2) < 1 or len(out) < 3:
        pytest.skip("corpus too small - run scripts/fetch_test_images.py")
    return out


@pytest.fixture
def no_face_image() -> np.ndarray:
    """A synthetic image that contains no face at all."""
    rng = np.random.default_rng(1234)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # A gradient sky over textured ground - structure, but nothing face-like.
    for y in range(480):
        img[y, :] = (200 - y // 4, 150 - y // 6, 90 + y // 8)
    img[300:, :] = (img[300:, :] * 0.5).astype(np.uint8)
    noise = rng.integers(0, 30, size=img.shape, dtype=np.uint8)
    return cv2.add(img, noise)
