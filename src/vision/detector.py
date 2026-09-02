"""Face detection.

buffalo_l ships SCRFD (det_10g.onnx) as its detector - the successor to
RetinaFace from the same authors - so that is what we actually run. The
name is reported honestly everywhere it surfaces.
"""
from __future__ import annotations

import contextlib
import io
import threading

import numpy as np

from ..config import CONFIG
from ..models import DetectedFace
from . import quality

_LOCK = threading.Lock()
_APP = None
_META: dict[str, str] = {}


def _build_app(verbose: bool = False):
    from insightface.app import FaceAnalysis

    sink = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    with ctx:
        # Only the two modules the pipeline needs. Loading genderage and the
        # landmark models costs time and buys nothing here.
        app = FaceAnalysis(
            name=CONFIG.vision.model_name,
            allowed_modules=["detection", "recognition"],
            providers=list(CONFIG.vision.providers),
        )
        app.prepare(ctx_id=-1, det_size=(CONFIG.vision.det_size, CONFIG.vision.det_size))
    return app


def get_app(verbose: bool = False):
    """Load the model pack once per process and reuse it."""
    global _APP
    with _LOCK:
        if _APP is None:
            _APP = _build_app(verbose=verbose)
            det = _APP.models["detection"]
            rec = _APP.models["recognition"]
            _META.update(
                {
                    "pack": CONFIG.vision.model_name,
                    "detector": type(det).__name__,
                    "detector_file": getattr(det, "model_file", "?").split("\\")[-1].split("/")[-1],
                    "recognizer": type(rec).__name__,
                    "recognizer_file": getattr(rec, "model_file", "?").split("\\")[-1].split("/")[-1],
                    "providers": ",".join(CONFIG.vision.providers),
                }
            )
    return _APP


def model_info() -> dict[str, str]:
    get_app()
    return dict(_META)


class FaceDetector:
    """Locates faces and attaches per-face quality signals."""

    def __init__(self, verbose: bool = False):
        self.app = get_app(verbose=verbose)

    def detect(self, img: np.ndarray, assess_quality: bool = True) -> list[DetectedFace]:
        """Return every face above the configured confidence, largest first."""
        raw = self.app.get(img)

        faces: list[DetectedFace] = []
        for f in raw:
            if float(f.det_score) < CONFIG.vision.min_det_score:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            face = DetectedFace(
                index=0,  # assigned after sorting
                bbox=(x1, y1, x2, y2),
                det_score=float(f.det_score),
                keypoints=np.asarray(f.kps) if f.kps is not None else None,
                _raw=f,
            )
            if assess_quality:
                face.quality = quality.assess(quality.crop_face(img, face.bbox))
            faces.append(face)

        # Largest face first: a stable, defensible order for "primary" face.
        faces.sort(key=lambda f: f.area, reverse=True)
        for i, f in enumerate(faces):
            f.index = i
        return faces

    def is_usable(self, face: DetectedFace) -> tuple[bool, str | None]:
        """Whether a detection carries enough signal to embed."""
        if min(face.width, face.height) < CONFIG.vision.min_face_pixels:
            return False, f"face {face.width}x{face.height} below {CONFIG.vision.min_face_pixels}px"
        if face.quality and face.quality.is_blurry:
            return False, f"face blur variance {face.quality.blur_variance:.1f} below {CONFIG.vision.min_blur_variance}"
        return True, None
