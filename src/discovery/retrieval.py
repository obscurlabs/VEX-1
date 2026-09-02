"""Candidate image retrieval.

Each candidate is isolated: any failure becomes a recorded status on that
candidate, never an exception that ends the run. Concurrency is bounded so we
do not hammer other people's servers.

Retrieval stops at a decoded image. Face detection and matching are Phase 2.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import requests

from ..config import CONFIG
from ..models import CandidateResult, CandidateStatus, ImageStatus, SearchCandidate
from ..vision import quality


def _classify_http(code: int) -> CandidateStatus:
    if code == 403:
        return CandidateStatus.HTTP_403
    if code == 404:
        return CandidateStatus.HTTP_404
    return CandidateStatus.HTTP_ERROR


class CandidateRetriever:
    """Downloads and decodes candidate images, one isolated attempt each."""

    def __init__(self, logger: Callable[[str], None] | None = None):
        self.cfg = CONFIG.retrieval
        self.log = logger or (lambda *_a, **_k: None)

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": self.cfg.user_agent,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        })
        return s

    def fetch_one(self, candidate: SearchCandidate, session: requests.Session) -> CandidateResult:
        """Retrieve a single candidate. Never raises."""
        result = CandidateResult(candidate=candidate)
        url = candidate.best_image_url
        started = time.perf_counter()

        if not url:
            result.status = CandidateStatus.NO_IMAGE_URL
            result.detail = "provider gave no image or thumbnail URL"
            return result

        try:
            with session.get(
                url, timeout=self.cfg.timeout, stream=True, allow_redirects=True
            ) as r:
                result.http_status = r.status_code
                result.content_type = r.headers.get("Content-Type", "").split(";")[0] or None

                if r.status_code != 200:
                    result.status = _classify_http(r.status_code)
                    result.detail = f"HTTP {r.status_code}"
                    return result

                chunks, size = [], 0
                for chunk in r.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > self.cfg.max_bytes:
                        result.status = CandidateStatus.TOO_LARGE
                        result.detail = f"exceeded {self.cfg.max_bytes} bytes"
                        result.bytes_downloaded = size
                        return result
                    chunks.append(chunk)
                body = b"".join(chunks)
                result.bytes_downloaded = len(body)

        except requests.Timeout:
            result.status = CandidateStatus.TIMEOUT
            result.detail = (
                f"connect>{self.cfg.connect_timeout}s or read>{self.cfg.read_timeout}s"
            )
            return result
        except requests.TooManyRedirects as exc:
            result.status = CandidateStatus.FETCH_FAILED
            result.detail = f"too many redirects: {exc}"
            return result
        except requests.RequestException as exc:
            result.status = CandidateStatus.FETCH_FAILED
            result.detail = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            result.elapsed_ms = (time.perf_counter() - started) * 1000

        status, img, err = quality.decode_bytes(body)
        if status is not ImageStatus.OK or img is None:
            result.status = CandidateStatus.INVALID_IMAGE
            result.detail = err or status.value
            return result

        result.image = img
        result.image_size = (img.shape[1], img.shape[0])
        result.status = CandidateStatus.RETRIEVED
        return result

    def fetch_all(self, candidates: Iterable[SearchCandidate]) -> list[CandidateResult]:
        """Retrieve every candidate with bounded concurrency, preserving order."""
        items = list(candidates)
        if not items:
            return []

        workers = max(1, min(self.cfg.concurrency, len(items)))
        session = self._session()
        results: list[CandidateResult | None] = [None] * len(items)

        def work(pair):
            i, candidate = pair
            r = self.fetch_one(candidate, session)
            results[i] = r
            self.log(self.format_line(i + 1, r))
            return r

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, enumerate(items)))

        session.close()
        return [r for r in results if r is not None]

    @staticmethod
    def format_line(index: int, r: CandidateResult) -> str:
        """One-line structured log entry for a finished candidate."""
        domain = r.candidate.source_domain or "?"
        head = f"[Candidate {index:02d}] {domain:<28}"
        if r.ok:
            w, h = r.image_size or (0, 0)
            kb = r.bytes_downloaded / 1024
            return f"{head} -> {w}x{h}  {kb:6.1f} KB  {r.elapsed_ms:5.0f} ms  RETRIEVED"
        return f"{head} -> {r.status.value}  ({r.detail})  {r.elapsed_ms:5.0f} ms  SKIP"
