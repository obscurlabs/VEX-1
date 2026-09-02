"""ArcFace embedding generation.

Runs the recognition model (w600k_r50) locally via ONNX Runtime and returns
L2-normalized 512-d vectors suitable for cosine similarity. Embeddings stay
on this machine - they are never uploaded and never written on-chain.
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG
from ..models import DetectedFace, FaceEmbedding, FaceStatus, ImageStatus, VisionResult
from . import quality
from .detector import FaceDetector

EMBEDDING_DIM = 512


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (v / n).astype(np.float32)


class ArcFaceEmbedder:
    """Turns a detected face into a normalized identity vector."""

    def __init__(self, detector: FaceDetector | None = None):
        self.detector = detector or FaceDetector()
        self.rec = self.detector.app.models["recognition"]
        self.model_name = f"{CONFIG.vision.model_name}/w600k_r50 (ArcFace)"

    def embed(self, img: np.ndarray, face: DetectedFace) -> FaceEmbedding:
        """Align via the detector's 5-point landmarks, then run ArcFace."""
        # rec.get() does the similarity-transform crop to 112x112 itself,
        # using the keypoints the detector already produced.
        vec = self.rec.get(img, face._raw)
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-d embedding, got {vec.shape[0]}")
        return FaceEmbedding(
            vector=l2_normalize(vec),
            face_index=face.index,
            model=self.model_name,
        )

    def process_image(self, img: np.ndarray, all_faces: bool = True) -> VisionResult:
        """Full local flow for a decoded image: detect -> filter -> embed."""
        result = VisionResult(image_status=ImageStatus.OK, quality=quality.assess(img))

        faces = self.detector.detect(img)
        result.faces = faces
        if not faces:
            result.face_status = FaceStatus.NO_FACE
            result.error = "no face detected"
            return result

        usable: list[DetectedFace] = []
        reasons: list[str] = []
        for f in faces:
            ok, why = self.detector.is_usable(f)
            if ok:
                usable.append(f)
            else:
                reasons.append(str(why))

        if not usable:
            result.face_status = FaceStatus.LOW_QUALITY
            result.error = "; ".join(str(r) for r in reasons)
            return result

        targets = usable if all_faces else usable[:1]
        result.embeddings = [self.embed(img, f) for f in targets]
        result.face_status = FaceStatus.OK
        return result

    def process_path(self, path, all_faces: bool = True) -> VisionResult:
        """Same as process_image but starting from a file on disk."""
        status, img, err = quality.load_image(path)
        if status is not ImageStatus.OK or img is None:
            return VisionResult(image_status=status, error=err)
        return self.process_image(img, all_faces=all_faces)
