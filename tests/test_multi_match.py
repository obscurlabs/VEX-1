"""Multi-match output: ranking, thresholding, duplicate grouping, census.

These run on synthetic CandidateMatch objects so the ranking rules can be
exercised exactly. Nothing here re-implements matching - the objects are the
same ones ``CandidateMatcher.evaluate()`` produces.
"""
from __future__ import annotations

import pytest

from src.matching import ranker
from src.models import (
    CandidateMatch,
    CandidateResult,
    CandidateStatus,
    SearchCandidate,
)


def make_match(similarity: float | None, domain: str, position: int = 1,
               status: CandidateStatus | None = None, url: str | None = None,
               content_sha256: str | None = None, faces: int = 1,
               identical: bool = False, threshold: float = 0.30) -> CandidateMatch:
    """One scored candidate, shaped exactly as the matcher emits it."""
    candidate = SearchCandidate(
        url=url or f"https://{domain}/page-{position}",
        title=f"{domain} story {position}",
        source_domain=domain,
        image_url=f"https://{domain}/img-{position}.jpg",
        thumbnail_url=None,
        position=position,
        provider="google_lens",
    )
    if status is None:
        if similarity is None:
            status = CandidateStatus.HTTP_403
        elif similarity >= threshold:
            status = (CandidateStatus.MATCH if faces == 1
                      else CandidateStatus.MULTIPLE_FACE_MATCH)
        else:
            status = CandidateStatus.REJECTED

    retrieval = CandidateResult(candidate=candidate)
    retrieval.status = (CandidateStatus.RETRIEVED if similarity is not None
                        else status)
    retrieval.content_sha256 = content_sha256 or f"{abs(hash((domain, position))):064x}"[:64]
    retrieval.bytes_downloaded = 2048
    retrieval.content_type = "image/jpeg"

    return CandidateMatch(
        candidate=candidate, status=status, best_similarity=similarity,
        threshold=threshold, faces_detected=faces, faces_embedded=faces,
        best_face_index=0 if similarity is not None else None,
        identical_to_input=identical, retrieval=retrieval,
    )


# --- ranking -----------------------------------------------------------

def test_all_qualifying_matches_are_retained():
    matches = [make_match(0.55, "a.example", 1), make_match(0.91, "b.example", 2),
               make_match(0.72, "c.example", 3)]
    assert len(ranker.qualifying(matches)) == 3


def test_qualifying_is_ranked_by_similarity():
    matches = [make_match(0.55, "a.example", 1), make_match(0.91, "b.example", 2),
               make_match(0.72, "c.example", 3)]
    scores = [m.best_similarity for m in ranker.qualifying(matches)]
    assert scores == sorted(scores, reverse=True)
    assert scores == [0.91, 0.72, 0.55]


def test_ranking_ties_break_on_search_position():
    matches = [make_match(0.80, "b.example", 7), make_match(0.80, "a.example", 2)]
    assert [m.candidate.position for m in ranker.qualifying(matches)] == [2, 7]


def test_ranking_is_stable_across_calls():
    matches = [make_match(0.9, f"d{i}.example", i) for i in range(1, 8)]
    first = [m.candidate.url for m in ranker.qualifying(matches)]
    for _ in range(5):
        assert [m.candidate.url for m in ranker.qualifying(matches)] == first


# --- thresholding ------------------------------------------------------

def test_below_threshold_candidates_are_excluded():
    matches = [make_match(0.91, "a.example", 1), make_match(0.12, "b.example", 2),
               make_match(0.29, "c.example", 3)]
    qualifying = ranker.qualifying(matches)
    assert [m.candidate.source_domain for m in qualifying] == ["a.example"]
    assert all(m.best_similarity >= 0.30 for m in qualifying)


def test_threshold_is_inclusive():
    matches = [make_match(0.30, "a.example", 1)]
    assert len(ranker.qualifying(matches)) == 1


def test_unscored_candidates_never_qualify():
    """A 403 or a no-face candidate is not a match at any threshold."""
    matches = [make_match(None, "a.example", 1),
               make_match(None, "b.example", 2, status=CandidateStatus.NO_FACE)]
    assert ranker.qualifying(matches) == []
    assert ranker.independent_matches(matches) == []


# --- duplicate grouping ------------------------------------------------

def test_same_domain_collapses_into_one_source():
    """Three thumbnail sizes of one video are one source, not three."""
    matches = [make_match(0.95, "youtube.com", 1), make_match(0.94, "youtube.com", 2),
               make_match(0.90, "youtube.com", 3), make_match(0.80, "bbc.com", 4)]
    groups = ranker.independent_matches(matches)
    assert len(groups) == 2
    assert groups[0].representative.candidate.source_domain == "youtube.com"
    assert groups[0].size == 3
    assert len(groups[0].duplicates) == 2


def test_the_strongest_member_represents_its_group():
    matches = [make_match(0.70, "x.example", 1), make_match(0.95, "x.example", 2),
               make_match(0.80, "x.example", 3)]
    (group,) = ranker.independent_matches(matches)
    assert group.representative.best_similarity == 0.95
    assert [m.best_similarity for m, _ in group.duplicates] == [0.80, 0.70]


def test_nothing_is_discarded_by_grouping():
    matches = [make_match(0.9, "a.example", i) for i in range(1, 5)]
    groups = ranker.independent_matches(matches)
    retained = sum(g.size for g in groups)
    assert retained == len(ranker.qualifying(matches)) == 4


def test_identical_bytes_are_reported_as_such():
    digest = "c" * 64
    matches = [make_match(0.95, "a.example", 1, content_sha256=digest),
               make_match(0.90, "a.example", 2, content_sha256=digest)]
    (group,) = ranker.independent_matches(matches)
    assert group.duplicates[0][1] == "identical image bytes"


def test_same_page_is_reported_as_such():
    url = "https://a.example/story"
    matches = [make_match(0.95, "a.example", 1, url=url),
               make_match(0.90, "a.example", 2, url=url)]
    (group,) = ranker.independent_matches(matches)
    assert group.duplicates[0][1] == "same page"


def test_same_domain_different_page_is_reported_as_such():
    matches = [make_match(0.95, "a.example", 1), make_match(0.90, "a.example", 2)]
    (group,) = ranker.independent_matches(matches)
    assert group.duplicates[0][1] == "same source domain"


def test_distinct_domains_are_never_merged():
    matches = [make_match(0.9, f"site{i}.example", i) for i in range(1, 6)]
    assert len(ranker.independent_matches(matches)) == 5


def test_subdomains_stay_separate_documented_limitation():
    """Documented in _group_key: exact-string comparison, no public-suffix list."""
    matches = [make_match(0.9, "en.wikipedia.org", 1),
               make_match(0.8, "pap.wikipedia.org", 2)]
    assert len(ranker.independent_matches(matches)) == 2


# --- independence from the input ---------------------------------------

def test_the_input_file_rediscovered_is_not_an_independent_source():
    matches = [make_match(1.0, "wikipedia.org", 1, identical=True),
               make_match(0.88, "bbc.com", 2)]
    groups = ranker.independent_matches(matches)
    assert [g.representative.candidate.source_domain for g in groups] == ["bbc.com"]
    assert ranker.best_match(matches).identical_to_input is True


def test_backwards_compatible_best_independent_match():
    """The historical accessor must return the very same object."""
    matches = [make_match(1.0, "src.example", 1, identical=True),
               make_match(0.93, "a.example", 2), make_match(0.91, "a.example", 3),
               make_match(0.88, "b.example", 4)]
    groups = ranker.independent_matches(matches)
    assert ranker.best_independent_match(matches) is groups[0].representative


def test_best_match_still_returns_the_overall_top():
    matches = [make_match(1.0, "src.example", 1, identical=True),
               make_match(0.88, "b.example", 2)]
    assert ranker.best_match(matches).best_similarity == 1.0


# --- counts -------------------------------------------------------------

def test_zero_matches():
    matches = [make_match(0.10, "a.example", 1), make_match(None, "b.example", 2)]
    assert ranker.qualifying(matches) == []
    assert ranker.independent_matches(matches) == []
    assert ranker.best_independent_match(matches) is None
    census = ranker.census(matches, discovered=40)
    assert census.qualifying == 0 and census.independent == 0
    assert census.face_matched == 1


def test_exactly_one_match():
    matches = [make_match(0.77, "only.example", 1), make_match(0.10, "b.example", 2)]
    groups = ranker.independent_matches(matches)
    assert len(groups) == 1
    assert groups[0].size == 1
    assert groups[0].duplicates == []


def test_multiple_independent_matches():
    matches = [make_match(0.95, "a.example", 1), make_match(0.91, "b.example", 2),
               make_match(0.88, "c.example", 3), make_match(0.72, "d.example", 4)]
    groups = ranker.independent_matches(matches)
    assert len(groups) == 4
    assert all(g.size == 1 for g in groups)
    assert [g.similarity for g in groups] == [0.95, 0.91, 0.88, 0.72]


def test_census_counts_every_stage_separately():
    matches = [
        make_match(0.95, "a.example", 1),                       # qualifies
        make_match(0.91, "a.example", 2),                       # duplicate source
        make_match(0.20, "b.example", 3),                       # below threshold
        make_match(None, "c.example", 4),                       # 403
        make_match(None, "d.example", 5, status=CandidateStatus.NO_FACE),
    ]
    census = ranker.census(matches, discovered=66, evaluated=5)
    assert census.discovered == 66
    assert census.evaluated == 5
    assert census.retrieved == 3
    assert census.face_matched == 3
    assert census.qualifying == 2
    assert census.independent == 1
    assert set(census.to_dict()) == {
        "discovered", "evaluated", "retrieved", "face_matched",
        "qualifying", "independent"}


def test_census_never_reports_more_independent_than_qualifying():
    matches = [make_match(0.9, "a.example", i) for i in range(1, 6)]
    census = ranker.census(matches, discovered=5, evaluated=5)
    assert census.independent <= census.qualifying


def test_top_independent_limits_without_losing_the_rest():
    matches = [make_match(0.9 - i / 100, f"s{i}.example", i) for i in range(1, 10)]
    top = ranker.top_independent(matches, limit=3)
    assert len(top) == 3
    assert len(ranker.independent_matches(matches)) == 9
    assert [g.similarity for g in top] == [0.89, 0.88, 0.87]


# --- serialization -------------------------------------------------------

def test_group_serializes_with_its_duplicates():
    import json

    matches = [make_match(0.95, "a.example", 1), make_match(0.91, "a.example", 2)]
    (group,) = ranker.independent_matches(matches)
    payload = json.loads(json.dumps(group.to_dict()))
    assert payload["size"] == 2
    assert payload["representative"]["best_similarity"] == 0.95
    assert payload["duplicates"][0]["duplicate_reason"] == "same source domain"


def test_multiple_face_matches_still_qualify():
    matches = [make_match(0.85, "a.example", 1, faces=3)]
    (group,) = ranker.independent_matches(matches)
    assert group.representative.status is CandidateStatus.MULTIPLE_FACE_MATCH
    assert group.representative.status.is_match


# --- manifest ------------------------------------------------------------

def test_manifest_record_carries_retrieval_facts():
    """Every retained match must be re-examinable without re-fetching."""
    from src.evidence.manifest import _match_record

    record = _match_record(make_match(0.9, "a.example", 1), "independent_match")
    for field in ("source_url", "source_domain", "title", "search_position",
                  "image_url", "retrieval_status", "content_sha256",
                  "similarity", "selected_face", "status"):
        assert field in record, f"missing {field}"
    assert record["retrieval_status"] == "RETRIEVED"
    assert len(record["content_sha256"]) == 64


def test_manifest_group_record_nests_duplicates():
    from src.evidence.manifest import _group_record

    matches = [make_match(0.95, "a.example", 1), make_match(0.91, "a.example", 2)]
    (group,) = ranker.independent_matches(matches)
    record = _group_record(group, 1)
    assert record["rank"] == 1
    assert record["group_size"] == 2
    assert record["role"] == "independent_match"
    assert record["duplicates"][0]["role"] == "duplicate"
    assert record["duplicates"][0]["duplicate_reason"] == "same source domain"


def test_manifest_similarity_is_a_fixed_precision_string():
    """Floats are forbidden in the canonical manifest."""
    from src.evidence.manifest import _match_record

    record = _match_record(make_match(0.9534, "a.example", 1), "independent_match")
    assert record["similarity"] == "0.953400"
    assert isinstance(record["similarity"], str)


def test_manifest_with_ranked_matches_is_canonicalizable():
    from src.evidence import hashing
    from src.evidence.manifest import _group_record

    matches = [make_match(0.95, "a.example", 1), make_match(0.91, "b.example", 2)]
    groups = ranker.independent_matches(matches)
    payload = {"matching": {"ranked_matches": [
        _group_record(g, i) for i, g in enumerate(groups, start=1)]}}
    digest, canonical = hashing.fingerprint(payload)
    assert len(digest) == 64
    assert hashing.fingerprint(payload)[0] == digest, "must be deterministic"


def test_schema_version_records_the_new_shape():
    from src.evidence import manifest

    assert manifest.SCHEMA_VERSION == "1.2.0"


def test_match_record_states_which_url_produced_the_bytes():
    """1.2.0: provenance of the retrieved image is part of the evidence.

    A platform CDN corroborates that the media belongs to that platform; the
    search provider's cached thumbnail does not. A reader must be able to tell
    them apart without re-fetching anything.
    """
    from src.evidence import manifest

    match = make_match(0.9, "example.com")
    match.retrieval.image_url_used = "https://cdn.example.com/real.jpg"
    record = manifest._match_record(match, "selected_match")

    assert record["retrieved_from"] == "https://cdn.example.com/real.jpg"


def test_match_record_reports_absent_provenance_as_none():
    from src.evidence import manifest

    match = make_match(0.9, "example.com")
    match.retrieval = None
    assert manifest._match_record(match, "selected_match")["retrieved_from"] is None


# --- similarity is never a probability ------------------------------------

def test_no_probability_language_in_the_ranking_layer():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("src/matching/ranker.py", "src/evidence/manifest.py"):
        text = (root / name).read_text(encoding="utf-8").lower()
        for phrase in ("% likely", "probability this", "confidence score",
                       "% match", "% confidence"):
            assert phrase not in text, f"{name} frames similarity as {phrase!r}"
