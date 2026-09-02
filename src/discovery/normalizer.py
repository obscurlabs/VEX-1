"""Normalize Google Lens responses into SearchCandidate objects.

Shape confirmed against a live response captured 2026-09-02:

    top level : search_metadata, search_parameters, ai_overview,
                visual_matches, related_content, organic_results

    visual_matches[] : position, title, link, source, source_icon,
                       thumbnail, thumbnail_width, thumbnail_height,
                       image, image_width, image_height
                       (+ price, in_stock on shopping results)

    organic_results[] : position, title, link, redirect_link, displayed_link,
                        thumbnail, favicon, snippet, source

visual_matches is the useful array: every entry carried both a full-resolution
`image` URL and a `thumbnail`. organic_results are page hits with only a
thumbnail, so they are normalized too but ranked after the visual matches.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ..models import SearchCandidate

PROVIDER = "google_lens"


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _clean(value: Any) -> str | None:
    """Provider fields are occasionally absent, null, or the wrong type."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _from_visual_match(item: dict, index: int) -> SearchCandidate | None:
    link = _clean(item.get("link"))
    if not link:
        return None
    return SearchCandidate(
        url=link,
        title=_clean(item.get("title")) or "(untitled)",
        source_domain=_domain(link),
        image_url=_clean(item.get("image")),
        thumbnail_url=_clean(item.get("thumbnail")),
        position=int(item.get("position") or index),
        provider=PROVIDER,
        raw_metadata={
            "result_type": "visual_match",
            "source": _clean(item.get("source")),
            "image_width": item.get("image_width"),
            "image_height": item.get("image_height"),
            "thumbnail_width": item.get("thumbnail_width"),
            "thumbnail_height": item.get("thumbnail_height"),
        },
    )


def _from_organic_result(item: dict, index: int, offset: int) -> SearchCandidate | None:
    link = _clean(item.get("link"))
    if not link:
        return None
    return SearchCandidate(
        url=link,
        title=_clean(item.get("title")) or "(untitled)",
        source_domain=_domain(link),
        # organic results carry no full-resolution image, only a thumbnail
        image_url=None,
        thumbnail_url=_clean(item.get("thumbnail")),
        position=offset + int(item.get("position") or index),
        provider=PROVIDER,
        raw_metadata={
            "result_type": "organic_result",
            "source": _clean(item.get("source")),
            "snippet": _clean(item.get("snippet")),
            "displayed_link": _clean(item.get("displayed_link")),
        },
    )


def normalize(raw: dict, include_organic: bool = True) -> list[SearchCandidate]:
    """Extract candidates from a Google Lens response.

    Tolerates a missing or malformed array rather than raising: a provider
    that changes shape should degrade to fewer candidates, not crash the run.
    """
    if not isinstance(raw, dict):
        return []

    candidates: list[SearchCandidate] = []
    seen: set[str] = set()

    visual = raw.get("visual_matches")
    if isinstance(visual, list):
        for i, item in enumerate(visual, start=1):
            if not isinstance(item, dict):
                continue
            c = _from_visual_match(item, i)
            if c and c.url not in seen:
                seen.add(c.url)
                candidates.append(c)

    if include_organic:
        organic = raw.get("organic_results")
        if isinstance(organic, list):
            offset = len(candidates)
            for i, item in enumerate(organic, start=1):
                if not isinstance(item, dict):
                    continue
                c = _from_organic_result(item, i, offset)
                if c and c.url not in seen:
                    seen.add(c.url)
                    candidates.append(c)

    return candidates


def count_raw_results(raw: dict) -> int:
    """How many result rows the provider actually returned, before filtering."""
    if not isinstance(raw, dict):
        return 0
    total = 0
    for key in ("visual_matches", "organic_results"):
        v = raw.get(key)
        if isinstance(v, list):
            total += len(v)
    return total
