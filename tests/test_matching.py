"""Phase 0: cosine similarity behaviour and the same/different separation.

These assertions are deliberately loose. They check that the model separates
identities at all, not that a particular threshold is correct - the corpus is
far too small to fix a threshold. See scripts/threshold_report.py.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from src.matching.similarity import best_match, cosine_similarity, is_match
from src.vision.embedder import EMBEDDING_DIM


def _embed_file(embedder, path):
    result = embedder.process_path(path, all_faces=False)
    assert result.ok, f"{path}: {result.face_status} {result.error}"
    return result.embeddings[0]


@pytest.fixture(scope="session")
def identity_embeddings(embedder, corpus):
    """slug -> list of embeddings, one per confirmed image."""
    return {slug: [_embed_file(embedder, p) for p in paths] for slug, paths in corpus.items()}


@pytest.fixture(scope="session")
def similarity_pairs(identity_embeddings):
    """(same_person_scores, different_person_scores)."""
    same, diff = [], []
    for embs in identity_embeddings.values():
        if len(embs) >= 2:
            same.extend(cosine_similarity(a, b) for a, b in itertools.combinations(embs, 2))
    for a_slug, b_slug in itertools.combinations(identity_embeddings, 2):
        for ea in identity_embeddings[a_slug]:
            for eb in identity_embeddings[b_slug]:
                diff.append(cosine_similarity(ea, eb))
    return same, diff


# --- the metric itself ------------------------------------------------

def test_self_similarity_is_one(embedder, portrait):
    emb = embedder.process_image(portrait).embeddings[0]
    assert cosine_similarity(emb, emb) == pytest.approx(1.0, abs=1e-5)


def test_similarity_is_symmetric(identity_embeddings):
    embs = [e for v in identity_embeddings.values() for e in v][:4]
    for a, b in itertools.combinations(embs, 2):
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-6)


def test_similarity_stays_in_range(similarity_pairs):
    for s in similarity_pairs[0] + similarity_pairs[1]:
        assert -1.0 <= s <= 1.0


def test_opposite_vector_scores_minus_one():
    v = np.ones(EMBEDDING_DIM, dtype=np.float32) / np.sqrt(EMBEDDING_DIM)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(512, dtype=np.float32), np.ones(128, dtype=np.float32))


def test_zero_vector_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(512, dtype=np.float32), np.ones(512, dtype=np.float32))


# --- identity separation ----------------------------------------------

def test_same_person_scores_above_different_person(similarity_pairs):
    same, diff = similarity_pairs
    assert same and diff
    assert min(same) > max(diff), (
        f"overlap: worst same-person {min(same):.3f} <= best different-person {max(diff):.3f}"
    )


def test_different_people_score_near_zero(similarity_pairs):
    _, diff = similarity_pairs
    assert float(np.mean(diff)) < 0.15


def test_same_person_scores_are_clearly_positive(similarity_pairs):
    same, _ = similarity_pairs
    assert float(np.mean(same)) > 0.35


def test_best_match_picks_the_right_identity(identity_embeddings):
    """A probe must rank its own identity's other photo above every stranger."""
    slugs = sorted(identity_embeddings)
    for slug in slugs:
        if len(identity_embeddings[slug]) < 2:
            continue  # no second photo to recover
        probe = identity_embeddings[slug][0]
        gallery = [identity_embeddings[slug][1]]
        owner = [slug]
        for other in slugs:
            if other != slug:
                gallery.append(identity_embeddings[other][0])
                owner.append(other)
        idx, score = best_match(probe, gallery)
        assert owner[idx] == slug, f"{slug} probe matched {owner[idx]} at {score:.3f}"


def test_compression_lowers_but_preserves_identity(embedder, corpus):
    """Heavy JPEG compression must not destroy the match - it is the main
    real-world degradation for images retrieved from the web."""
    path = corpus[sorted(corpus)[0]][0]
    img = cv2.imread(str(path))
    original = embedder.process_image(img, all_faces=False).embeddings[0]

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
    assert ok
    degraded_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    degraded = embedder.process_image(degraded_img, all_faces=False).embeddings[0]

    assert cosine_similarity(original, degraded) > 0.7


# --- threshold plumbing ------------------------------------------------

def test_is_match_respects_explicit_threshold():
    assert is_match(0.62, threshold=0.60)
    assert not is_match(0.59, threshold=0.60)
    assert is_match(0.60, threshold=0.60), "threshold is inclusive"


def test_is_match_uses_configured_default():
    from src.config import CONFIG

    t = CONFIG.match.threshold
    assert is_match(t + 0.01)
    assert not is_match(t - 0.01)


def test_best_match_on_empty_gallery():
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    idx, score = best_match(v, [])
    assert idx == -1 and score == float("-inf")
