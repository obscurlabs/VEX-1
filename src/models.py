"""Internal data types. Deliberately independent of any vendor's structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ImageStatus(str, Enum):
    """Outcome of validating/decoding an image."""

    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    INVALID_IMAGE = "INVALID_IMAGE"
    TOO_SMALL = "TOO_SMALL"


class CandidateStatus(str, Enum):
    """Lifecycle of a single search candidate.

    Every terminal state is explicit so one candidate's failure is recorded
    rather than raised - a 403 must never end the run.
    """

    PENDING = "PENDING"
    RETRIEVED = "RETRIEVED"        # image downloaded and decoded
    FETCH_FAILED = "FETCH_FAILED"  # connection/DNS/SSL error
    HTTP_403 = "HTTP_403"
    HTTP_404 = "HTTP_404"
    HTTP_ERROR = "HTTP_ERROR"      # any other non-200
    TIMEOUT = "TIMEOUT"
    TOO_LARGE = "TOO_LARGE"
    INVALID_IMAGE = "INVALID_IMAGE"
    NO_IMAGE_URL = "NO_IMAGE_URL"
    # Reserved for Phase 2 (matching); defined here so the state machine is
    # visible in one place.
    NO_FACE = "NO_FACE"
    LOW_QUALITY = "LOW_QUALITY"
    REJECTED = "REJECTED"
    MATCH = "MATCH"

    @property
    def is_failure(self) -> bool:
        return self not in (
            CandidateStatus.PENDING,
            CandidateStatus.RETRIEVED,
            CandidateStatus.MATCH,
            CandidateStatus.REJECTED,
        )


class FaceStatus(str, Enum):
    """Outcome of looking for a usable face in an image."""

    OK = "OK"
    NO_FACE = "NO_FACE"
    LOW_QUALITY = "LOW_QUALITY"


@dataclass
class ImageQuality:
    """Cheap, non-destructive quality signals for an image or a face crop."""

    width: int
    height: int
    blur_variance: float
    brightness: float
    contrast: float
    is_blurry: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "blur_variance": round(self.blur_variance, 4),
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "is_blurry": self.is_blurry,
        }


@dataclass
class DetectedFace:
    """One face located in an image."""

    index: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    keypoints: np.ndarray | None = None
    quality: ImageQuality | None = None
    # The upstream detector object, kept so the embedder can reuse the
    # landmarks for alignment. Never serialised.
    _raw: Any = field(default=None, repr=False, compare=False)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "det_score": round(float(self.det_score), 4),
            "width": self.width,
            "height": self.height,
            "quality": self.quality.to_dict() if self.quality else None,
        }


@dataclass
class FaceEmbedding:
    """A normalized ArcFace embedding. The vector never leaves the machine."""

    vector: np.ndarray  # shape (512,), L2-normalized
    face_index: int
    model: str

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))


@dataclass
class VisionResult:
    """Everything Phase 0 extracts from a single image."""

    image_status: ImageStatus
    face_status: FaceStatus | None = None
    quality: ImageQuality | None = None
    faces: list[DetectedFace] = field(default_factory=list)
    embeddings: list[FaceEmbedding] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.image_status is ImageStatus.OK and self.face_status is FaceStatus.OK

    @property
    def faces_detected(self) -> int:
        return len(self.faces)


@dataclass
class SearchCandidate:
    """One result from a reverse-image search, normalized.

    Nothing downstream may depend on a provider's raw JSON shape; that stays
    quarantined in raw_metadata.
    """

    url: str
    title: str
    source_domain: str
    image_url: str | None
    thumbnail_url: str | None
    position: int
    provider: str
    raw_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def best_image_url(self) -> str | None:
        """Full-resolution image if the provider gave one, else the thumbnail."""
        return self.image_url or self.thumbnail_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "url": self.url,
            "title": self.title,
            "source_domain": self.source_domain,
            "image_url": self.image_url,
            "thumbnail_url": self.thumbnail_url,
            "provider": self.provider,
        }


@dataclass
class CandidateResult:
    """What actually happened when we tried to retrieve a candidate."""

    candidate: SearchCandidate
    status: CandidateStatus = CandidateStatus.PENDING
    detail: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    bytes_downloaded: int = 0
    elapsed_ms: float = 0.0
    image_size: tuple[int, int] | None = None
    image: Any = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status is CandidateStatus.RETRIEVED

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "status": self.status.value,
            "detail": self.detail,
            "http_status": self.http_status,
            "content_type": self.content_type,
            "bytes_downloaded": self.bytes_downloaded,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "image_size": list(self.image_size) if self.image_size else None,
        }


@dataclass
class SearchResult:
    """A complete provider response plus the normalized candidates."""

    provider: str
    live: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    candidates: list[SearchCandidate] = field(default_factory=list)
    image_id: str | None = None
    search_id: str | None = None
    raw_result_count: int = 0
    elapsed_ms: float = 0.0
