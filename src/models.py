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
    # Phase 2 (matching) outcomes.
    NO_FACE = "NO_FACE"
    LOW_QUALITY = "LOW_QUALITY"
    REJECTED = "REJECTED"
    MATCH = "MATCH"
    # >=2 faces in the image and the best one cleared the threshold. Kept
    # distinct from MATCH so the record never reads as "this image is the
    # target" - it means "one face in this image matches the target".
    MULTIPLE_FACE_MATCH = "MULTIPLE_FACE_MATCH"

    @property
    def is_match(self) -> bool:
        return self in (CandidateStatus.MATCH, CandidateStatus.MULTIPLE_FACE_MATCH)

    @property
    def is_failure(self) -> bool:
        return self not in (
            CandidateStatus.PENDING,
            CandidateStatus.RETRIEVED,
            CandidateStatus.MATCH,
            CandidateStatus.MULTIPLE_FACE_MATCH,
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
    content_sha256: str | None = None
    bytes_downloaded: int = 0
    elapsed_ms: float = 0.0
    image_size: tuple[int, int] | None = None
    image: Any = field(default=None, repr=False, compare=False)
    # The exact bytes downloaded, kept so the evidence bundle can store the
    # candidate image verbatim rather than a re-encoded copy.
    content: bytes = field(default=b"", repr=False, compare=False)

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
            "content_sha256": self.content_sha256,
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
    # The provider's response exactly as it arrived on the wire. Evidence
    # preserves these bytes verbatim; re-serialising the parsed dict would
    # silently change key order, spacing and number formatting.
    raw_bytes: bytes = field(default=b"", repr=False)
    candidates: list[SearchCandidate] = field(default_factory=list)
    image_id: str | None = None
    search_id: str | None = None
    raw_result_count: int = 0
    elapsed_ms: float = 0.0


@dataclass
class FaceMatch:
    """One face inside a candidate image, scored against the target."""

    face_index: int
    similarity: float
    bbox: tuple[int, int, int, int]
    det_score: float
    face_px: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_index": self.face_index,
            "similarity": round(float(self.similarity), 6),
            "bbox": list(self.bbox),
            "det_score": round(float(self.det_score), 4),
            "face_px": list(self.face_px),
        }


@dataclass
class CandidateMatch:
    """Result of matching one candidate image against the target embedding.

    Retrieval failures are carried through unchanged so a single ranked list
    can show every candidate's fate.
    """

    candidate: SearchCandidate
    status: CandidateStatus
    faces_detected: int = 0
    faces_embedded: int = 0
    faces: list[FaceMatch] = field(default_factory=list)
    best_face_index: int | None = None
    best_similarity: float | None = None
    runner_up_similarity: float | None = None
    threshold: float | None = None
    detail: str | None = None
    elapsed_ms: float = 0.0
    image_size: tuple[int, int] | None = None
    # True when the retrieved bytes are the input image itself. Re-finding the
    # source file is a legitimate discovery result, but it is NOT independent
    # corroboration, so it is flagged rather than silently ranked as a match.
    identical_to_input: bool = False
    # Retrieval facts carried forward so the evidence manifest can describe
    # the candidate without re-fetching it.
    retrieval: "CandidateResult | None" = field(default=None, repr=False, compare=False)

    @property
    def is_match(self) -> bool:
        return self.status.is_match

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "status": self.status.value,
            "faces_detected": self.faces_detected,
            "faces_embedded": self.faces_embedded,
            "best_face_index": self.best_face_index,
            "best_similarity": (
                round(float(self.best_similarity), 6)
                if self.best_similarity is not None else None
            ),
            "runner_up_similarity": (
                round(float(self.runner_up_similarity), 6)
                if self.runner_up_similarity is not None else None
            ),
            "threshold": self.threshold,
            "face_similarities": [f.to_dict() for f in self.faces],
            "image_size": list(self.image_size) if self.image_size else None,
            "identical_to_input": self.identical_to_input,
            "detail": self.detail,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class MatchGroup:
    """One source, and the duplicate evidence that collapsed into it.

    Google Lens routinely returns several URLs backed by the same underlying
    source - three thumbnail sizes of one video, a page listed twice under two
    image URLs. Presenting those as separate corroboration would overstate how
    many independent sources were found, so they are grouped behind the
    highest-scoring member rather than discarded.
    """

    representative: "CandidateMatch"
    duplicates: list[tuple["CandidateMatch", str]] = field(default_factory=list)
    key: str = ""

    @property
    def size(self) -> int:
        return 1 + len(self.duplicates)

    @property
    def similarity(self) -> float | None:
        return self.representative.best_similarity

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "representative": self.representative.to_dict(),
            "duplicates": [
                {**m.to_dict(), "duplicate_reason": reason}
                for m, reason in self.duplicates
            ],
        }


@dataclass
class MatchCensus:
    """How many candidates survived each stage of the funnel.

    Every number is a count of real candidates, named so the stages cannot be
    confused with one another.
    """

    discovered: int = 0        # normalized from the provider response
    evaluated: int = 0         # retrieval was attempted
    retrieved: int = 0         # downloaded and decoded
    face_matched: int = 0      # a face was embedded and scored
    qualifying: int = 0        # cleared the similarity threshold
    independent: int = 0       # distinct sources, excluding the input itself

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "evaluated": self.evaluated,
            "retrieved": self.retrieved,
            "face_matched": self.face_matched,
            "qualifying": self.qualifying,
            "independent": self.independent,
        }
