"""The canonical evidence manifest.

The manifest is the thing that gets fingerprinted. It is a compact, explicit
description of what the pipeline observed, and it covers every other file in
the bundle by SHA-256 - so altering any artifact, binary or not, changes the
manifest hash.

Deliberately excluded
---------------------
* face embeddings (biometric data never leaves the machine, never gets hashed)
* API keys, private keys, .env contents, any credential
* absolute filesystem paths (machine-dependent)
* wall-clock times that describe the *process* rather than the *evidence*

Timestamps that ARE included (``created_at``, ``requested_at``,
``retrieved_at``) are part of the evidence record: they state when the search
ran and when the candidate was fetched. They are written once, at bundle
creation, and read back verbatim on verification - never regenerated.
"""
from __future__ import annotations

from typing import Any

from ..models import CandidateMatch, SearchResult
from . import hashing

SCHEMA = "hhgoa-task3/evidence-manifest"
SCHEMA_VERSION = "1.0.0"

# Files covered by the manifest's artifact digests, in a fixed order that
# never depends on directory listing order.
ARTIFACT_FILES = (
    "input.jpg",
    "source-image.jpg",
    "search-request.json",
    "search-response.json",
    "candidates.json",
    "retrieval.json",
    "matching.json",
)

# Anything matching these must never appear in the manifest.
SECRET_KEYS = ("api_key", "apikey", "private_key", "secret", "token", "password",
               "authorization", "serpapi_key")


def _face_record(match: CandidateMatch) -> dict[str, Any] | None:
    """The selected face inside a candidate image."""
    if match.best_face_index is None:
        return None
    for f in match.faces:
        if f.face_index == match.best_face_index:
            return {
                "face_index": f.face_index,
                "bbox": list(f.bbox),
                "det_score": hashing.decimal_str(f.det_score, hashing.SCORE_PLACES),
                "width": f.face_px[0],
                "height": f.face_px[1],
            }
    return None


def _match_record(match: CandidateMatch, role: str,
                  image_file: str | None = None) -> dict[str, Any]:
    """One matched candidate, with everything needed to re-examine it."""
    record: dict[str, Any] = {
        "role": role,
        "status": match.status.value,
        "source_url": match.candidate.url,
        "source_domain": match.candidate.source_domain,
        "title": match.candidate.title,
        "search_position": match.candidate.position,
        "provider": match.candidate.provider,
        "image_url": match.candidate.image_url,
        "identical_to_input": match.identical_to_input,
        "faces_detected": match.faces_detected,
        "faces_embedded": match.faces_embedded,
        "selected_face_index": match.best_face_index,
        "selected_face": _face_record(match),
        "similarity": (
            hashing.decimal_str(match.best_similarity)
            if match.best_similarity is not None else None
        ),
        "runner_up_similarity": (
            hashing.decimal_str(match.runner_up_similarity)
            if match.runner_up_similarity is not None else None
        ),
        "threshold": (
            hashing.decimal_str(match.threshold)
            if match.threshold is not None else None
        ),
        "image_size": list(match.image_size) if match.image_size else None,
    }
    if image_file:
        record["image_file"] = image_file
    return record


def build(
    *,
    investigation_id: str,
    created_at: str,
    pipeline_version: str,
    input_sha256: str,
    input_bytes: int,
    input_size: tuple[int, int],
    input_face: dict[str, Any],
    search: SearchResult,
    search_requested_at: str,
    normalized_candidate_count: int,
    evaluated_count: int,
    selected: CandidateMatch | None,
    independent: CandidateMatch | None,
    retrieval: dict[str, Any],
    model: dict[str, str],
    threshold: float,
    artifact_digests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the canonical manifest.

    Every real number arrives here already quantized to a decimal string;
    hashing.canonical_bytes() rejects raw floats as a backstop.
    """
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": pipeline_version,
        "investigation_id": investigation_id,
        "created_at": created_at,

        "input": {
            "file": "input.jpg",
            "sha256": input_sha256,
            "bytes": input_bytes,
            "width": input_size[0],
            "height": input_size[1],
            "face": input_face,
        },

        "search": {
            "provider": search.provider,
            "engine": "google_lens",
            "live": search.live,
            "image_id": search.image_id,
            "search_id": search.search_id,
            "requested_at": search_requested_at,
            "raw_response_file": "search-response.json",
            "raw_result_count": search.raw_result_count,
            "normalized_candidate_count": normalized_candidate_count,
            "candidates_evaluated": evaluated_count,
        },

        "matching": {
            "threshold": hashing.decimal_str(threshold),
            "similarity_metric": "cosine",
            "selected_match": (
                _match_record(selected, "selected_match") if selected else None
            ),
            "best_independent_match": (
                _match_record(independent, "best_independent_match", "source-image.jpg")
                if independent else None
            ),
        },

        "retrieval": retrieval,
        "model": model,
        "artifacts": artifact_digests,
    }

    assert_no_secrets(manifest)
    return manifest


def assert_no_secrets(node: Any, path: str = "") -> None:
    """Fail loudly if anything credential-shaped reached the manifest."""
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            if any(s in lowered for s in SECRET_KEYS):
                raise ValueError(f"secret-shaped key in manifest: {path}/{key}")
            assert_no_secrets(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            assert_no_secrets(value, f"{path}[{i}]")
    elif isinstance(node, str):
        if "api_key=" in node.lower() or "apikey=" in node.lower():
            raise ValueError(f"secret-shaped value in manifest at {path}")
