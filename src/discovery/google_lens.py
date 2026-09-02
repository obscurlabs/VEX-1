"""SerpApi -> Google Lens reverse image search (live).

The documented local-image workflow, verified against the live API:

    local image -> POST https://serpapi.com/image  (multipart, field "image",
                   max 500 KB) -> temporary image_id
                -> GET  https://serpapi.com/search?engine=google_lens
                   &image_id=... -> visual matches

There is no cached path in this module. If the provider fails, it raises.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import requests

from ..config import CONFIG
from ..models import SearchCandidate, SearchResult
from . import normalizer
from .base import (
    SearchAuthError,
    SearchProvider,
    SearchProviderError,
    SearchRateLimitError,
    SearchTransportError,
    SearchUploadError,
)


class GoogleLensProvider(SearchProvider):
    """Live Google Lens search via SerpApi."""

    name = "google_lens"

    def __init__(self, api_key: str | None = None, logger=None):
        self.api_key = api_key if api_key is not None else CONFIG.serpapi_key
        if not self.api_key:
            raise SearchAuthError("SERPAPI_KEY is not set")
        self.cfg = CONFIG.search
        self.log = logger or (lambda *a, **k: None)
        self.session = requests.Session()

    # -- upload ---------------------------------------------------------

    def prepare_upload_bytes(self, image_path: Path) -> tuple[bytes, str]:
        """Return image bytes within the provider's size limit.

        Re-encodes only when the file exceeds the limit. This is to satisfy an
        API constraint, not to manufacture a match.
        """
        path = Path(image_path)
        if not path.exists():
            raise SearchUploadError(f"input image not found: {path}")
        if path.suffix.lower() not in self.cfg.allowed_formats:
            raise SearchUploadError(
                f"unsupported format {path.suffix!r}; provider accepts "
                f"{', '.join(self.cfg.allowed_formats)}"
            )

        data = path.read_bytes()
        limit = self.cfg.max_upload_bytes
        if len(data) <= limit:
            return data, "unmodified"

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SearchUploadError(f"cannot decode {path}")

        # Drop quality first, then scale; stop as soon as we are under.
        for quality in (92, 85, 75, 65, 55):
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok and buf.nbytes <= limit:
                return buf.tobytes(), f"re-encoded q={quality} ({len(data)}->{buf.nbytes} bytes)"

        scaled = img
        for _ in range(6):
            scaled = cv2.resize(scaled, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok and buf.nbytes <= limit:
                return buf.tobytes(), (
                    f"downscaled to {scaled.shape[1]}x{scaled.shape[0]} q=80 "
                    f"({len(data)}->{buf.nbytes} bytes)"
                )

        raise SearchUploadError(f"cannot reduce {path} below {limit} bytes")

    def upload(self, image_path: Path) -> str:
        """Upload the image and return its temporary image_id."""
        payload, note = self.prepare_upload_bytes(image_path)
        self.log(f"upload payload {len(payload)} bytes ({note})")

        try:
            r = self.session.post(
                self.cfg.upload_url,
                files={"image": (Path(image_path).name, payload, "image/jpeg")},
                data={"api_key": self.api_key},
                timeout=self.cfg.upload_timeout,
            )
        except requests.Timeout as exc:
            raise SearchTransportError(f"upload timed out after {self.cfg.upload_timeout}s") from exc
        except requests.RequestException as exc:
            raise SearchTransportError(f"upload transport error: {exc}") from exc

        self._raise_for_api_status(r, "upload")

        try:
            body = r.json()
        except ValueError as exc:
            raise SearchUploadError(f"upload returned non-JSON: {r.text[:200]}") from exc

        image_id = body.get("image_id")
        if not image_id:
            raise SearchUploadError(f"upload response had no image_id: {body}")
        return image_id

    # -- search ---------------------------------------------------------

    def search(self, image_path: Path) -> SearchResult:
        started = time.perf_counter()

        image_id = self.upload(image_path)
        self.log(f"image_id {image_id[:32]}...")

        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "api_key": self.api_key,
            "hl": self.cfg.hl,
            "country": self.cfg.country,
        }
        try:
            r = self.session.get(
                self.cfg.search_url, params=params, timeout=self.cfg.search_timeout
            )
        except requests.Timeout as exc:
            raise SearchTransportError(f"search timed out after {self.cfg.search_timeout}s") from exc
        except requests.RequestException as exc:
            raise SearchTransportError(f"search transport error: {exc}") from exc

        self._raise_for_api_status(r, "search")

        try:
            raw = r.json()
        except ValueError as exc:
            raise SearchProviderError(f"search returned non-JSON: {r.text[:200]}") from exc

        if isinstance(raw.get("error"), str):
            raise SearchProviderError(raw["error"])

        meta = raw.get("search_metadata") or {}
        status = meta.get("status")
        if status and status != "Success":
            raise SearchProviderError(f"provider reported status {status!r}")

        candidates = self.normalize(raw)
        return SearchResult(
            provider=self.name,
            live=True,
            raw=raw,
            candidates=candidates,
            image_id=image_id,
            search_id=meta.get("id"),
            raw_result_count=normalizer.count_raw_results(raw),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def normalize(self, raw: dict) -> list[SearchCandidate]:
        return normalizer.normalize(raw)

    # -- errors ---------------------------------------------------------

    @staticmethod
    def _raise_for_api_status(response: requests.Response, stage: str) -> None:
        code = response.status_code
        if code == 200:
            return
        snippet = response.text[:300].replace("\n", " ")
        if code in (401, 403):
            raise SearchAuthError(f"{stage}: HTTP {code} - key rejected: {snippet}")
        if code == 429:
            raise SearchRateLimitError(f"{stage}: HTTP 429 - rate limit/quota: {snippet}")
        raise SearchProviderError(f"{stage}: HTTP {code}: {snippet}")
