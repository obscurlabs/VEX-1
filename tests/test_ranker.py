"""Phase 2: candidate matching, multi-face policy, and ranking.

Runs against the confirmed local corpus - real images, real embeddings, real
similarities. Nothing here asserts that a particular URL is the answer.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.matching import ranker
from src.matching.ranker import CandidateMatcher
from src.models import (
    CandidateResult,
    CandidateStatus,
    SearchCandidate,
)


def _candidate(pos: int = 1, domain: str = "example.com") -> SearchCandidate:
    return SearchCandidate(
        url=f"https://{domain}/p{pos}", title=f"c{pos}", source_domain=domain,
        image_url=f"https://{domain}/i{pos}.jpg", thumbnail_url=None,
        position=pos, provider="google_lens",
    )


def _retrieved(img, pos: int = 1, domain: str = "example.com") -> CandidateResult:
    r = CandidateResult(candidate=_candidate(pos, domain))
    r.status = CandidateStatus.RETRIEVED
    r.http_status = 200
    r.image = img
    r.image_size = (img.shape[1], img.shape[0])
    r.bytes_downloaded = 1234
    return r


def _failed(status: CandidateStatus, pos: int = 1, detail: str = "x") -> CandidateResult:
    r = CandidateResult(candidate=_candidate(pos))
    r.status = status
    r.detail = detail
    return r


@pytest.fixture(scope="module")
def matcher(embedder):
    return CandidateMatcher(embedder=embedder)


@pytest.fixture(scope="module")
def target(embedder, corpus):
    """Target embedding taken from the first Obama photograph."""
    result = embedder.process_path(corpus["barack_obama"][0], all_faces=False)
    assert result.ok
    return result.embeddings[0]


# --- decisions --------------------------------------------------------

def test_same_person_is_a_match(matcher, target, corpus):
    img = cv2.imread(str(corpus["barack_obama"][1]))
    m = matcher.evaluate(target, _retrieved(img))
    assert m.is_match
    assert m.best_similarity >= m.threshold
    assert m.faces_embedded >= 1
    assert m.best_face_index is not None


def test_different_person_is_rejected(matcher, target, corpus):
    img = cv2.imread(str(corpus["joe_biden"][0]))
    m = matcher.evaluate(target, _retrieved(img))
    assert m.status is CandidateStatus.REJECTED
    assert not m.is_match
    assert m.best_similarity < m.threshold
    assert "below threshold" in m.detail


def test_no_face_image_is_no_face(matcher, target, no_face_image):
    m = matcher.evaluate(target, _retrieved(no_face_image))
    assert m.status is CandidateStatus.NO_FACE
    assert m.faces_detected == 0
    assert m.best_similarity is None
    assert m.faces == []


def test_similarity_is_actually_computed_not_invented(matcher, target, corpus):
    """The recorded score must equal a direct cosine of the two embeddings."""
    from src.matching.similarity import cosine_similarity

    path = corpus["barack_obama"][1]
    img = cv2.imread(str(path))
    m = matcher.evaluate(target, _retrieved(img))

    faces = matcher.embedder.detector.detect(img)
    best = faces[m.best_face_index]
    direct = cosine_similarity(target, matcher.embedder.embed(img, best))
    assert m.best_similarity == pytest.approx(direct, abs=1e-6)


# --- multiple faces ---------------------------------------------------

def test_multi_face_image_uses_a_distinct_status(matcher, target, group_image, corpus):
    """A group photo containing the target must not be labelled a plain MATCH."""
    obama = cv2.imread(str(corpus["barack_obama"][0]))
    h, w = obama.shape[:2]
    group = cv2.resize(group_image, (w, int(group_image.shape[0] * w / group_image.shape[1])))
    canvas = np.vstack([obama, group[: min(group.shape[0], 400)]])

    m = matcher.evaluate(target, _retrieved(canvas))
    assert m.faces_detected > 1
    assert m.status is CandidateStatus.MULTIPLE_FACE_MATCH
    assert m.status.is_match, "it is still a match, just a qualified one"
    assert m.status is not CandidateStatus.MATCH


def test_every_face_is_scored_and_recorded(matcher, target, group_image):
    m = matcher.evaluate(target, _retrieved(group_image))
    assert m.faces_detected >= 4
    assert len(m.faces) == m.faces_embedded
    assert {f.face_index for f in m.faces} == set(range(m.faces_embedded))
    for f in m.faces:
        assert -1.0 <= f.similarity <= 1.0
        assert f.face_px[0] > 0 and f.face_px[1] > 0
        assert len(f.bbox) == 4


def test_best_face_is_the_highest_scoring_one(matcher, target, group_image):
    m = matcher.evaluate(target, _retrieved(group_image))
    if m.best_similarity is None:
        pytest.skip("no usable face")
    assert m.best_similarity == max(f.similarity for f in m.faces)
    assert m.best_face_index == max(m.faces, key=lambda f: f.similarity).face_index


def test_runner_up_is_recorded_for_multi_face(matcher, target, group_image):
    m = matcher.evaluate(target, _retrieved(group_image))
    if m.faces_embedded < 2:
        pytest.skip("need >=2 embedded faces")
    assert m.runner_up_similarity is not None
    assert m.runner_up_similarity <= m.best_similarity


def test_single_face_has_no_runner_up(matcher, target, corpus):
    img = cv2.imread(str(corpus["ursula_von_der_leyen"][0]))
    m = matcher.evaluate(target, _retrieved(img))
    if m.faces_detected != 1:
        pytest.skip("image is not single-face")
    assert m.runner_up_similarity is None


# --- isolation --------------------------------------------------------

def test_retrieval_failures_are_carried_through(matcher, target):
    for status in (CandidateStatus.HTTP_403, CandidateStatus.HTTP_404,
                   CandidateStatus.TIMEOUT, CandidateStatus.INVALID_IMAGE,
                   CandidateStatus.FETCH_FAILED, CandidateStatus.HTTP_ERROR):
        m = matcher.evaluate(target, _failed(status))
        assert m.status is status, "matching must not overwrite the retrieval reason"
        assert m.best_similarity is None
        assert m.faces_detected == 0


def test_one_bad_candidate_does_not_stop_the_rest(matcher, target, corpus, no_face_image):
    results = [
        _failed(CandidateStatus.HTTP_403, 1),
        _retrieved(cv2.imread(str(corpus["barack_obama"][1])), 2),
        _retrieved(no_face_image, 3),
        _failed(CandidateStatus.TIMEOUT, 4),
        _retrieved(cv2.imread(str(corpus["joe_biden"][0])), 5),
    ]
    matches = matcher.evaluate_all(target, results)
    assert len(matches) == len(results)
    assert [m.candidate.position for m in matches] == [1, 2, 3, 4, 5], "order preserved"
    assert matches[0].status is CandidateStatus.HTTP_403
    assert matches[1].is_match
    assert matches[2].status is CandidateStatus.NO_FACE
    assert matches[3].status is CandidateStatus.TIMEOUT
    assert matches[4].status is CandidateStatus.REJECTED


def test_source_metadata_survives_matching(matcher, target, corpus):
    img = cv2.imread(str(corpus["barack_obama"][1]))
    r = _retrieved(img, pos=7, domain="news.example.org")
    m = matcher.evaluate(target, r)
    assert m.candidate.position == 7
    assert m.candidate.source_domain == "news.example.org"
    assert m.candidate.url == "https://news.example.org/p7"


# --- threshold --------------------------------------------------------

def test_threshold_is_configurable_and_respected(embedder, target, corpus):
    img = cv2.imread(str(corpus["barack_obama"][1]))
    loose = CandidateMatcher(embedder=embedder, threshold=0.10)
    strict = CandidateMatcher(embedder=embedder, threshold=0.99)

    m_loose = loose.evaluate(target, _retrieved(img))
    m_strict = strict.evaluate(target, _retrieved(img))

    assert m_loose.best_similarity == pytest.approx(m_strict.best_similarity, abs=1e-9), \
        "the score does not depend on the threshold"
    assert m_loose.is_match
    assert not m_strict.is_match
    assert m_strict.status is CandidateStatus.REJECTED
    assert m_loose.threshold == 0.10 and m_strict.threshold == 0.99


def test_threshold_is_recorded_on_every_result(matcher, target, corpus):
    img = cv2.imread(str(corpus["joe_biden"][0]))
    m = matcher.evaluate(target, _retrieved(img))
    assert m.threshold == matcher.threshold
    assert m.to_dict()["threshold"] == matcher.threshold


def test_default_threshold_comes_from_config(embedder):
    from src.config import CONFIG
    assert CandidateMatcher(embedder=embedder).threshold == CONFIG.match.threshold


# --- ranking ----------------------------------------------------------

def test_rank_orders_by_similarity_descending(matcher, target, corpus, no_face_image):
    results = [
        _retrieved(cv2.imread(str(corpus["joe_biden"][0])), 1),
        _retrieved(cv2.imread(str(corpus["barack_obama"][1])), 2),
        _retrieved(no_face_image, 3),
    ]
    ranked = ranker.rank(matcher.evaluate_all(target, results))
    scored = [m for m in ranked if m.best_similarity is not None]
    assert [m.best_similarity for m in scored] == sorted(
        (m.best_similarity for m in scored), reverse=True)
    assert ranked[-1].best_similarity is None, "unscored candidates sort last"


def test_best_match_returns_none_when_nothing_passes(embedder, target, corpus):
    strict = CandidateMatcher(embedder=embedder, threshold=0.999)
    results = [_retrieved(cv2.imread(str(corpus["barack_obama"][1])), 1)]
    assert ranker.best_match(strict.evaluate_all(target, results)) is None


def test_best_match_picks_the_highest_passing_candidate(matcher, target, corpus):
    results = [
        _retrieved(cv2.imread(str(corpus["joe_biden"][0])), 1, "a.com"),
        _retrieved(cv2.imread(str(corpus["barack_obama"][1])), 2, "b.com"),
    ]
    best = ranker.best_match(matcher.evaluate_all(target, results))
    assert best is not None
    assert best.candidate.source_domain == "b.com"
    assert best.is_match


def test_distribution_reports_real_statistics(matcher, target, corpus):
    results = [
        _retrieved(cv2.imread(str(corpus["barack_obama"][1])), 1),
        _retrieved(cv2.imread(str(corpus["joe_biden"][0])), 2),
    ]
    matches = matcher.evaluate_all(target, results)
    d = ranker.distribution(matches)
    scores = [m.best_similarity for m in matches if m.best_similarity is not None]
    assert d["n"] == len(scores)
    assert d["min"] == pytest.approx(min(scores))
    assert d["max"] == pytest.approx(max(scores))


def test_distribution_of_nothing_is_empty():
    assert ranker.distribution([]) == {"n": 0}


def test_rank_of_nothing_is_empty():
    assert ranker.rank([]) == []
    assert ranker.best_match([]) is None


# --- serialization ----------------------------------------------------

def test_match_serializes_without_pixel_data(matcher, target, corpus):
    import json

    img = cv2.imread(str(corpus["barack_obama"][1]))
    m = matcher.evaluate(target, _retrieved(img))
    payload = json.loads(json.dumps(m.to_dict()))
    assert payload["status"] in ("MATCH", "MULTIPLE_FACE_MATCH")
    assert payload["best_similarity"] is not None
    assert "image" not in payload
    assert isinstance(payload["face_similarities"], list)
    assert payload["face_similarities"][0]["similarity"] is not None


# --- identical-file flagging ------------------------------------------

def test_identical_input_is_flagged(embedder, target, corpus):
    """Re-finding the input file must be marked, not passed off as
    independent corroboration."""
    import hashlib

    path = corpus["barack_obama"][0]
    blob = path.read_bytes()
    img = cv2.imread(str(path))

    r = _retrieved(img)
    r.content_sha256 = hashlib.sha256(blob).hexdigest()

    m = CandidateMatcher(
        embedder=embedder, input_sha256=hashlib.sha256(blob).hexdigest()
    ).evaluate(target, r)
    assert m.identical_to_input is True
    assert m.to_dict()["identical_to_input"] is True


def test_different_file_is_not_flagged(embedder, target, corpus):
    import hashlib

    img = cv2.imread(str(corpus["barack_obama"][1]))
    r = _retrieved(img)
    r.content_sha256 = hashlib.sha256(b"different bytes").hexdigest()
    m = CandidateMatcher(
        embedder=embedder, input_sha256=hashlib.sha256(b"input bytes").hexdigest()
    ).evaluate(target, r)
    assert m.identical_to_input is False


def test_best_independent_match_skips_the_input_file(embedder, target, corpus):
    import hashlib

    same_bytes = corpus["barack_obama"][0].read_bytes()
    same_sha = hashlib.sha256(same_bytes).hexdigest()

    r_same = _retrieved(cv2.imread(str(corpus["barack_obama"][0])), 1, "source.example")
    r_same.content_sha256 = same_sha
    r_other = _retrieved(cv2.imread(str(corpus["barack_obama"][1])), 2, "other.example")
    r_other.content_sha256 = hashlib.sha256(b"other").hexdigest()

    matcher = CandidateMatcher(embedder=embedder, input_sha256=same_sha)
    matches = matcher.evaluate_all(target, [r_same, r_other])

    top = ranker.best_match(matches)
    independent = ranker.best_independent_match(matches)
    assert top.candidate.source_domain == "source.example"
    assert top.identical_to_input
    assert independent is not None
    assert independent.candidate.source_domain == "other.example"
    assert not independent.identical_to_input
    assert independent.best_similarity < top.best_similarity


def test_best_independent_match_is_none_when_all_are_the_input(embedder, target, corpus):
    import hashlib

    blob = corpus["barack_obama"][0].read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    r = _retrieved(cv2.imread(str(corpus["barack_obama"][0])))
    r.content_sha256 = sha
    matches = CandidateMatcher(embedder=embedder, input_sha256=sha).evaluate_all(target, [r])
    assert ranker.best_match(matches) is not None
    assert ranker.best_independent_match(matches) is None


def test_flag_is_false_without_an_input_hash(matcher, target, corpus):
    """The default matcher has no input hash, so nothing is ever flagged."""
    r = _retrieved(cv2.imread(str(corpus["barack_obama"][1])))
    r.content_sha256 = "abc123"
    assert matcher.evaluate(target, r).identical_to_input is False
