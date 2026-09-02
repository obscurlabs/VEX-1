"""Phase 2: match retrieved candidate images against the target embedding.

Per candidate:

    retrieved image
        -> face detection (all faces)
        -> per-face quality gate
        -> ArcFace embedding for every usable face
        -> cosine similarity of each against the target
        -> best face decides, threshold decides the label
        -> ranked by similarity

Multiple-face policy (deterministic, and deliberately conservative):

    Every usable face is embedded and scored. The candidate's score is the
    HIGHEST per-face similarity, and the winning face index is recorded. When
    the image holds more than one face the status is MULTIPLE_FACE_MATCH
    rather than MATCH, so the record states "one face in this image matches
    the target" and never "this image is the target". Ties resolve to the
    lower face index; faces arrive largest-first, so the larger face wins.

Nothing here selects a candidate by URL, domain, or position. The only
ordering signal is measured identity similarity.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from ..config import CONFIG
from ..matching.similarity import cosine_similarity
from ..models import (
    CandidateMatch,
    CandidateResult,
    CandidateStatus,
    FaceEmbedding,
    FaceMatch,
)
from ..vision.embedder import ArcFaceEmbedder


class CandidateMatcher:
    """Scores retrieved candidates against one target embedding."""

    def __init__(
        self,
        embedder: ArcFaceEmbedder | None = None,
        threshold: float | None = None,
        logger: Callable[[str], None] | None = None,
        input_sha256: str | None = None,
    ):
        self.embedder = embedder or ArcFaceEmbedder()
        # Configurable, never adjusted at runtime to manufacture a match.
        self.threshold = CONFIG.match.threshold if threshold is None else threshold
        self.log = logger or (lambda *_a, **_k: None)
        # Lets a candidate that is byte-identical to the input be flagged.
        self.input_sha256 = input_sha256

    def evaluate(
        self, target: FaceEmbedding, result: CandidateResult
    ) -> CandidateMatch:
        """Score one candidate. Never raises: every failure becomes a status."""
        started = time.perf_counter()
        match = CandidateMatch(
            candidate=result.candidate,
            status=result.status,
            threshold=self.threshold,
            image_size=result.image_size,
            detail=result.detail,
            identical_to_input=bool(
                self.input_sha256
                and result.content_sha256
                and result.content_sha256 == self.input_sha256
            ),
        )

        # Retrieval already failed - carry the reason through unchanged.
        if not result.ok or result.image is None:
            match.elapsed_ms = (time.perf_counter() - started) * 1000
            return match

        try:
            faces = self.embedder.detector.detect(result.image)
        except Exception as exc:  # a corrupt decode can still upset the detector
            match.status = CandidateStatus.INVALID_IMAGE
            match.detail = f"detection failed: {type(exc).__name__}: {exc}"
            match.elapsed_ms = (time.perf_counter() - started) * 1000
            return match

        match.faces_detected = len(faces)
        if not faces:
            match.status = CandidateStatus.NO_FACE
            match.detail = "no face detected"
            match.elapsed_ms = (time.perf_counter() - started) * 1000
            return match

        scored: list[FaceMatch] = []
        rejected: list[str] = []
        for face in faces:
            usable, why = self.embedder.detector.is_usable(face)
            if not usable:
                rejected.append(str(why))
                continue
            try:
                emb = self.embedder.embed(result.image, face)
                similarity = cosine_similarity(target, emb)
            except Exception as exc:
                rejected.append(f"face {face.index}: {type(exc).__name__}: {exc}")
                continue
            scored.append(
                FaceMatch(
                    face_index=face.index,
                    similarity=similarity,
                    bbox=face.bbox,
                    det_score=face.det_score,
                    face_px=(face.width, face.height),
                )
            )

        match.faces_embedded = len(scored)
        match.faces = scored

        if not scored:
            match.status = CandidateStatus.LOW_QUALITY
            match.detail = "; ".join(rejected) or "no usable face"
            match.elapsed_ms = (time.perf_counter() - started) * 1000
            return match

        # Strict > keeps the first (largest) face on a tie: deterministic.
        best = scored[0]
        for f in scored[1:]:
            if f.similarity > best.similarity:
                best = f
        others = sorted((f.similarity for f in scored if f is not best), reverse=True)

        match.best_face_index = best.face_index
        match.best_similarity = best.similarity
        match.runner_up_similarity = others[0] if others else None

        if best.similarity >= self.threshold:
            match.status = (
                CandidateStatus.MATCH
                if match.faces_detected == 1
                else CandidateStatus.MULTIPLE_FACE_MATCH
            )
        else:
            match.status = CandidateStatus.REJECTED
            match.detail = (
                f"best similarity {best.similarity:.4f} below threshold {self.threshold}"
            )
        if rejected:
            note = f"{len(rejected)} face(s) skipped by quality gate"
            match.detail = f"{match.detail}; {note}" if match.detail else note

        match.elapsed_ms = (time.perf_counter() - started) * 1000
        return match

    def evaluate_all(
        self, target: FaceEmbedding, results: Iterable[CandidateResult]
    ) -> list[CandidateMatch]:
        """Score every candidate independently, preserving discovery order.

        Runs serially: ONNX Runtime already uses the available cores, and
        threading the inference here would contend rather than help.
        """
        matches: list[CandidateMatch] = []
        for i, result in enumerate(results, start=1):
            match = self.evaluate(target, result)
            matches.append(match)
            self.log(format_line(i, match))
        return matches


def rank(matches: Iterable[CandidateMatch]) -> list[CandidateMatch]:
    """Rank by measured identity similarity, highest first.

    Candidates with no similarity (retrieval or detection failures) sort last
    while keeping their original discovery order.
    """
    items = list(matches)
    scored = [m for m in items if m.best_similarity is not None]
    unscored = [m for m in items if m.best_similarity is None]
    scored.sort(key=lambda m: (-float(m.best_similarity), m.candidate.position))
    return scored + unscored


def best_match(matches: Iterable[CandidateMatch]) -> CandidateMatch | None:
    """The highest-scoring candidate that actually cleared the threshold."""
    passing = [m for m in matches if m.is_match]
    if not passing:
        return None
    return rank(passing)[0]


def best_independent_match(matches: Iterable[CandidateMatch]) -> CandidateMatch | None:
    """Highest-scoring match that is NOT simply the input file rediscovered.

    Re-finding the source image proves where the file lives; another
    photograph of the same person is what corroborates an identity claim.
    """
    passing = [m for m in matches if m.is_match and not m.identical_to_input]
    if not passing:
        return None
    return rank(passing)[0]


def distribution(matches: Iterable[CandidateMatch]) -> dict[str, float | int]:
    """Summary statistics over the candidates that produced a similarity."""
    scores = [
        float(m.best_similarity) for m in matches if m.best_similarity is not None
    ]
    if not scores:
        return {"n": 0}
    a = np.asarray(scores)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "sd": float(a.std()),
    }


def format_line(index: int, m: CandidateMatch) -> str:
    """One structured terminal line per evaluated candidate."""
    domain = (m.candidate.source_domain or "?")[:30]
    head = f"[Candidate {index:02d}] {domain:<30}"

    if m.best_similarity is None:
        return f"{head} -> {m.status.value:<20} ({m.detail})"

    faces = f"{m.faces_detected} face{'s' if m.faces_detected != 1 else ''}"
    detail = f"sim {m.best_similarity:.4f}  face #{m.best_face_index} of {faces}"
    if m.runner_up_similarity is not None:
        detail += f"  (next {m.runner_up_similarity:.4f})"
    if m.identical_to_input:
        detail += "  [SAME FILE AS INPUT]"
    return f"{head} -> {m.status.value:<20} {detail}"
