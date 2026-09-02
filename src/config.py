"""Central configuration. Everything tunable lives here, sourced from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional at import time
    def load_dotenv(*_a, **_k):  # type: ignore
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _f(key: str, default: float) -> float:
    v = os.getenv(key)
    return float(v) if v not in (None, "") else default


def _i(key: str, default: int) -> int:
    v = os.getenv(key)
    return int(v) if v not in (None, "") else default


def _s(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class VisionConfig:
    """Face detection + embedding settings."""

    model_name: str = _s("FACE_MODEL_NAME", "buffalo_l")
    det_size: int = _i("FACE_DET_SIZE", 640)
    # Detections below this confidence are discarded outright.
    min_det_score: float = _f("MIN_DET_SCORE", 0.50)
    # An image smaller than this on either axis is not worth processing.
    min_image_dim: int = _i("MIN_IMAGE_DIM", 64)
    # A face box smaller than this (shorter side, px) has too little signal
    # for a trustworthy ArcFace embedding.
    min_face_pixels: int = _i("MIN_FACE_PIXELS", 40)
    # Variance of the Laplacian below this reads as "too blurry".
    min_blur_variance: float = _f("MIN_BLUR_VARIANCE", 10.0)
    providers: tuple[str, ...] = ("CPUExecutionProvider",)


@dataclass(frozen=True)
class MatchConfig:
    """Identity matching settings."""

    # PROVISIONAL - see README. Cosine similarity, not a probability.
    threshold: float = _f("FACE_MATCH_THRESHOLD", 0.30)


@dataclass(frozen=True)
class SearchConfig:
    """SerpAPI / Google Lens settings. Endpoints verified against the live API."""

    provider: str = "google_lens"
    upload_url: str = "https://serpapi.com/image"
    search_url: str = "https://serpapi.com/search"
    # SerpApi rejects uploads over 500 KB; stay under it with a safety margin.
    max_upload_bytes: int = _i("SERPAPI_MAX_UPLOAD_BYTES", 480_000)
    upload_timeout: float = _f("SERPAPI_UPLOAD_TIMEOUT", 60.0)
    search_timeout: float = _f("SERPAPI_SEARCH_TIMEOUT", 90.0)
    hl: str = _s("SERPAPI_HL", "en")
    country: str = _s("SERPAPI_COUNTRY", "us")
    # Formats the upload endpoint accepts.
    allowed_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class RetrievalConfig:
    """Fetching candidate images. Bounded so we do not hammer other people's
    servers, and capped so one huge file cannot stall the run."""

    max_candidates: int = _i("MAX_CANDIDATES", 25)
    concurrency: int = _i("RETRIEVAL_CONCURRENCY", 5)
    # requests applies a scalar timeout to connect AND read separately, so a
    # single value of N actually bounds a request at ~2N. Split them so the
    # worst case is predictable and stated.
    connect_timeout: float = _f("RETRIEVAL_CONNECT_TIMEOUT", 6.0)
    read_timeout: float = _f("RETRIEVAL_READ_TIMEOUT", 12.0)
    max_bytes: int = _i("RETRIEVAL_MAX_BYTES", 12_000_000)
    user_agent: str = _s(
        "RETRIEVAL_USER_AGENT",
        "Mozilla/5.0 (compatible; HHGoaTask3/1.0; face-evidence pipeline)",
    )

    @property
    def timeout(self) -> tuple[float, float]:
        return (self.connect_timeout, self.read_timeout)

    @property
    def worst_case_seconds(self) -> float:
        return self.connect_timeout + self.read_timeout


@dataclass(frozen=True)
class Config:
    vision: VisionConfig = VisionConfig()
    match: MatchConfig = MatchConfig()
    search: SearchConfig = SearchConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    pipeline_version: str = _s("PIPELINE_VERSION", "1.0.0")
    serpapi_key: str = _s("SERPAPI_KEY")
    polygon_rpc_url: str = _s("POLYGON_RPC_URL")
    contract_address: str = _s("CONTRACT_ADDRESS")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def evidence_dir(self) -> Path:
        return PROJECT_ROOT / "evidence"

    @property
    def cache_dir(self) -> Path:
        return PROJECT_ROOT / "cache"


CONFIG = Config()
