"""Image validation and lightweight quality metrics.

These exist to decide whether an image is worth trusting, not to improve it.
Nothing here modifies pixel data used for embedding.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import CONFIG
from ..models import ImageQuality, ImageStatus


def decode_array(data: bytes) -> np.ndarray | None:
    """Decode image bytes to BGR, or None if no available decoder can read them.

    OpenCV first, because it is what the rest of the pipeline is built on and
    it handles the common formats. Pillow second, because this OpenCV build
    ships no AVIF decoder (``cv2.haveImageReader('x.avif')`` is False) and news
    publishers increasingly serve AVIF - 108 candidates across the stored
    evidence were downloaded intact and then discarded for exactly that reason.

    Pillow is already a pinned dependency, so this adds reach, not weight. The
    pixels are only ever converted, never resampled or enhanced: whatever the
    decoder produces is what gets embedded.
    """
    if not data:
        return None

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None and img.size:
        return img

    try:
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()
            rgb = handle.convert("RGB")
        converted = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    return converted if converted.size else None


def load_image(path: str | Path) -> tuple[ImageStatus, np.ndarray | None, str | None]:
    """Read an image from disk, returning a BGR array or an explicit status."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ImageStatus.NOT_FOUND, None, f"no such file: {p}"

    # Read the bytes ourselves rather than using imread: it handles non-ASCII
    # paths on Windows, and it routes through the same decoder chain as a
    # downloaded candidate, so an input is accepted on the same terms.
    try:
        img = decode_array(np.fromfile(str(p), dtype=np.uint8).tobytes())
    except Exception as exc:  # unreadable/corrupt file
        return ImageStatus.INVALID_IMAGE, None, f"decode failed: {exc}"

    if img is None or img.size == 0:
        return ImageStatus.INVALID_IMAGE, None, "not a decodable image"

    return validate_array(img)


def decode_bytes(data: bytes) -> tuple[ImageStatus, np.ndarray | None, str | None]:
    """Decode in-memory image bytes (used for downloaded candidates)."""
    if not data:
        return ImageStatus.INVALID_IMAGE, None, "empty response body"
    try:
        img = decode_array(data)
    except Exception as exc:
        return ImageStatus.INVALID_IMAGE, None, f"decode failed: {exc}"
    if img is None or img.size == 0:
        return ImageStatus.INVALID_IMAGE, None, "not a decodable image"
    return validate_array(img)


def validate_array(img: np.ndarray) -> tuple[ImageStatus, np.ndarray | None, str | None]:
    """Check a decoded array is large enough and shaped as 3-channel BGR."""
    if img.ndim != 3 or img.shape[2] != 3:
        return ImageStatus.INVALID_IMAGE, None, f"unexpected shape {img.shape}"

    h, w = img.shape[:2]
    min_dim = CONFIG.vision.min_image_dim
    if min(h, w) < min_dim:
        return ImageStatus.TOO_SMALL, None, f"{w}x{h} below minimum {min_dim}px"

    return ImageStatus.OK, img, None


def assess(img: np.ndarray) -> ImageQuality:
    """Compute blur/brightness/contrast for an image or a face crop."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Variance of the Laplacian: low variance means few sharp edges, i.e. blur.
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = img.shape[:2]
    return ImageQuality(
        width=w,
        height=h,
        blur_variance=blur,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        is_blurry=blur < CONFIG.vision.min_blur_variance,
    )


def crop_face(img: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.0) -> np.ndarray:
    """Return the face region, clamped to the image bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    if pad:
        dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]
