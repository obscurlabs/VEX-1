"""Phase 1: response normalization and candidate retrieval.

The normalizer is exercised against the field shape observed on the live API.
Retrieval is exercised against a real local HTTP server that returns real
status codes - no request mocking, so the isolation logic is genuinely tested.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.discovery import normalizer
from src.discovery.retrieval import CandidateRetriever
from src.models import CandidateStatus, SearchCandidate

FIXTURE = Path(__file__).parent / "fixtures" / "google-lens-response.json"


def _visual_match(pos: int, link: str, image: str | None = "https://img.example/a.jpg") -> dict:
    return {
        "position": pos,
        "title": f"result {pos}",
        "link": link,
        "source": "Example",
        "source_icon": "https://icon",
        "thumbnail": "https://thumb.example/t.jpg",
        "thumbnail_width": 253,
        "thumbnail_height": 199,
        "image": image,
        "image_width": 1500,
        "image_height": 1184,
    }


# --- normalizer -------------------------------------------------------

def test_normalizes_visual_matches():
    raw = {"visual_matches": [_visual_match(1, "https://www.bbc.com/news/story")]}
    (c,) = normalizer.normalize(raw)
    assert c.url == "https://www.bbc.com/news/story"
    assert c.source_domain == "bbc.com", "www. is stripped"
    assert c.image_url == "https://img.example/a.jpg"
    assert c.thumbnail_url == "https://thumb.example/t.jpg"
    assert c.position == 1
    assert c.provider == "google_lens"
    assert c.raw_metadata["result_type"] == "visual_match"


def test_zero_results_is_not_an_error():
    for raw in ({"visual_matches": []}, {}, {"search_metadata": {"status": "Success"}}):
        assert normalizer.normalize(raw) == []
        assert normalizer.count_raw_results(raw) == 0


def test_malformed_response_degrades_without_raising():
    raw = {
        "visual_matches": [
            _visual_match(1, "https://ok.example/a"),
            "not a dict",
            {"position": 3},
            {"link": None, "title": "x"},
            {"link": "https://ok.example/b"},
        ]
    }
    out = normalizer.normalize(raw)
    assert [c.url for c in out] == ["https://ok.example/a", "https://ok.example/b"]
    assert out[1].title == "(untitled)"
    assert out[1].image_url is None


def test_non_dict_response_yields_nothing():
    for junk in (None, [], "text", 42):
        assert normalizer.normalize(junk) == []
        assert normalizer.count_raw_results(junk) == 0


def test_duplicate_urls_are_collapsed():
    raw = {"visual_matches": [_visual_match(1, "https://a.example/x"),
                              _visual_match(2, "https://a.example/x")]}
    assert len(normalizer.normalize(raw)) == 1


def test_organic_results_are_included_after_visual_matches():
    raw = {
        "visual_matches": [_visual_match(1, "https://vm.example/a")],
        "organic_results": [{"position": 1, "title": "page", "link": "https://or.example/b",
                             "thumbnail": "https://t/x.jpg", "snippet": "s"}],
    }
    out = normalizer.normalize(raw)
    assert [c.raw_metadata["result_type"] for c in out] == ["visual_match", "organic_result"]
    assert out[1].image_url is None, "organic results carry no full-resolution image"
    assert out[1].best_image_url == "https://t/x.jpg", "falls back to the thumbnail"


def test_organic_results_can_be_excluded():
    raw = {"visual_matches": [_visual_match(1, "https://vm.example/a")],
           "organic_results": [{"position": 1, "link": "https://or.example/b"}]}
    assert len(normalizer.normalize(raw, include_organic=False)) == 1


def test_count_raw_results_counts_both_arrays():
    raw = {"visual_matches": [{}, {}, {}], "organic_results": [{}, {}]}
    assert normalizer.count_raw_results(raw) == 5


@pytest.mark.skipif(not FIXTURE.exists(), reason="live response fixture not captured")
def test_normalizes_the_real_captured_response():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = normalizer.normalize(raw)
    assert out, "captured response should yield candidates"
    assert all(c.url.startswith("http") for c in out)
    assert all(c.provider == "google_lens" for c in out)
    assert any(c.image_url for c in out), "visual matches carry full-size images"
    assert len({c.url for c in out}) == len(out), "urls are unique"


# --- retrieval server -------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves the outcomes a real candidate host can produce."""

    png = None

    def log_message(self, *_args):
        pass

    def do_GET(self):
        route = self.path
        if route == "/ok.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
        elif route == "/forbidden":
            self.send_error(403, "Forbidden")
        elif route == "/missing":
            self.send_error(404, "Not Found")
        elif route == "/boom":
            self.send_error(500, "Server Error")
        elif route == "/slow":
            time.sleep(3)
            self.send_response(200)
            self.end_headers()
        elif route == "/notanimage":
            body = b"<html>this is not an image</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


@pytest.fixture(scope="module")
def server():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :] = (40, 90, 160)
    _Handler.png = cv2.imencode(".png", img)[1].tobytes()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _fetch(candidate) -> "CandidateResult":
    """Retrieve one candidate using the retriever's own configured session."""
    retriever = CandidateRetriever()
    session = retriever._session()
    try:
        return retriever.fetch_one(candidate, session)
    finally:
        session.close()


def _candidate(url: str, pos: int = 1) -> SearchCandidate:
    return SearchCandidate(
        url=url, title="t", source_domain="127.0.0.1",
        image_url=url, thumbnail_url=None, position=pos, provider="google_lens",
    )


# --- retrieval --------------------------------------------------------

def test_retrieves_a_valid_image(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/ok.png")])[0]
    assert r.status is CandidateStatus.RETRIEVED
    assert r.ok and r.http_status == 200
    assert r.image_size == (160, 120)
    assert r.bytes_downloaded > 0
    assert r.image is not None


def test_403_is_classified_and_isolated(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/forbidden")])[0]
    assert r.status is CandidateStatus.HTTP_403
    assert r.http_status == 403
    assert r.status.is_failure


def test_404_is_classified(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/missing")])[0]
    assert r.status is CandidateStatus.HTTP_404


def test_other_http_errors_are_classified(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/boom")])[0]
    assert r.status is CandidateStatus.HTTP_ERROR
    assert r.http_status == 500


def test_timeout_is_classified(server):
    from src.config import RetrievalConfig

    retriever = CandidateRetriever()
    retriever.cfg = RetrievalConfig(connect_timeout=2.0, read_timeout=0.5)
    r = retriever.fetch_one(_candidate(f"{server}/slow"), retriever._session())
    assert r.status is CandidateStatus.TIMEOUT
    assert "read>0.5s" in r.detail
    # the whole point of splitting the timeouts: a predictable upper bound
    assert r.elapsed_ms < retriever.cfg.worst_case_seconds * 1000


def test_non_image_body_is_invalid_image(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/notanimage")])[0]
    assert r.status is CandidateStatus.INVALID_IMAGE
    assert r.http_status == 200, "the request succeeded; the payload was not an image"


def test_unreachable_host_is_fetch_failed():
    r = CandidateRetriever().fetch_all([_candidate("http://127.0.0.1:9/never")])[0]
    assert r.status is CandidateStatus.FETCH_FAILED


def test_candidate_without_image_url_is_flagged():
    c = SearchCandidate(url="https://x.example/page", title="t", source_domain="x.example",
                        image_url=None, thumbnail_url=None, position=1, provider="google_lens")
    r = CandidateRetriever().fetch_all([c])[0]
    assert r.status is CandidateStatus.NO_IMAGE_URL


def test_one_failure_does_not_stop_the_others(server):
    """The core isolation guarantee."""
    candidates = [
        _candidate(f"{server}/forbidden", 1),
        _candidate(f"{server}/ok.png", 2),
        _candidate(f"{server}/missing", 3),
        _candidate(f"{server}/ok.png", 4),
        _candidate(f"{server}/notanimage", 5),
        _candidate("http://127.0.0.1:9/never", 6),
    ]
    results = CandidateRetriever().fetch_all(candidates)
    assert len(results) == len(candidates), "every candidate produces a result"
    assert [r.status for r in results] == [
        CandidateStatus.HTTP_403,
        CandidateStatus.RETRIEVED,
        CandidateStatus.HTTP_404,
        CandidateStatus.RETRIEVED,
        CandidateStatus.INVALID_IMAGE,
        CandidateStatus.FETCH_FAILED,
    ], "results stay in input order"
    assert sum(r.ok for r in results) == 2


def test_empty_candidate_list_returns_empty():
    assert CandidateRetriever().fetch_all([]) == []


def test_oversized_body_is_rejected(server):
    from src.config import RetrievalConfig

    retriever = CandidateRetriever()
    retriever.cfg = RetrievalConfig(max_bytes=10)
    r = retriever.fetch_one(_candidate(f"{server}/ok.png"), retriever._session())
    assert r.status is CandidateStatus.TOO_LARGE


def test_result_serialization_is_json_safe(server):
    r = CandidateRetriever().fetch_all([_candidate(f"{server}/ok.png")])[0]
    payload = json.dumps(r.to_dict())
    assert "RETRIEVED" in payload
    assert "image" not in json.loads(payload), "decoded pixels are not serialized"


# --- ordered image-URL fallback -----------------------------------------
#
# The provider hands us two URLs per result. Choosing on presence rather than
# on success is what silenced every Meta and TikTok candidate: their primary
# URL serves a consent wall while a usable thumbnail sits beside it.

def _two_url_candidate(primary: str, thumb: str) -> SearchCandidate:
    return SearchCandidate(
        url="https://social.example/post/1", title="t",
        source_domain="social.example", image_url=primary,
        thumbnail_url=thumb, position=1, provider="google_lens")


def test_image_urls_are_ordered_primary_then_thumbnail():
    c = _two_url_candidate("https://cdn/a.jpg", "https://tbn/b.jpg")
    assert c.image_urls == ["https://cdn/a.jpg", "https://tbn/b.jpg"]


def test_image_urls_deduplicates_identical_urls():
    c = _two_url_candidate("https://cdn/a.jpg", "https://cdn/a.jpg")
    assert c.image_urls == ["https://cdn/a.jpg"]


def test_image_urls_is_empty_when_the_provider_gave_nothing():
    c = SearchCandidate(url="https://a.example/p", title="t",
                        source_domain="a.example", image_url=None,
                        thumbnail_url=None, position=1, provider="google_lens")
    assert c.image_urls == []
    assert c.best_image_url is None


def test_html_primary_falls_back_to_the_thumbnail(server):
    """The Meta case: primary returns a consent page, thumbnail is an image."""
    base = server
    c = _two_url_candidate(f"{base}/notanimage", f"{base}/ok.png")
    result = _fetch(c)

    assert result.status is CandidateStatus.RETRIEVED
    assert result.image_url_used == f"{base}/ok.png"


def test_the_failed_attempt_is_preserved_not_discarded(server):
    base = server
    c = _two_url_candidate(f"{base}/notanimage", f"{base}/ok.png")
    result = _fetch(c)

    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt["url"] == f"{base}/notanimage"
    assert attempt["status"] == CandidateStatus.INVALID_IMAGE.value
    assert attempt["content_type"] == "text/html"


def test_a_403_primary_also_falls_back(server):
    """TikTok's case: the primary is blocked outright."""
    base = server
    c = _two_url_candidate(f"{base}/forbidden", f"{base}/ok.png")
    result = _fetch(c)

    assert result.status is CandidateStatus.RETRIEVED
    assert result.attempts[0]["status"] == CandidateStatus.HTTP_403.value


def test_a_working_primary_is_never_second_guessed(server):
    """No extra request when the first URL already worked."""
    base = server
    c = _two_url_candidate(f"{base}/ok.png", f"{base}/ok.png")
    result = _fetch(c)

    assert result.status is CandidateStatus.RETRIEVED
    assert result.image_url_used == f"{base}/ok.png"
    assert result.attempts == []


def test_when_every_url_fails_the_last_status_is_reported(server):
    base = server
    c = _two_url_candidate(f"{base}/forbidden", f"{base}/missing")
    result = _fetch(c)

    assert not result.ok
    assert result.status is CandidateStatus.HTTP_404
    assert [a["status"] for a in result.attempts] == [
        CandidateStatus.HTTP_403.value, CandidateStatus.HTTP_404.value]


def test_no_image_url_still_reported_when_provider_gave_none():
    c = SearchCandidate(url="https://a.example/p", title="t",
                        source_domain="a.example", image_url=None,
                        thumbnail_url=None, position=1, provider="google_lens")
    result = _fetch(c)
    assert result.status is CandidateStatus.NO_IMAGE_URL


def test_retrieval_records_which_url_produced_the_bytes(server):
    base = server
    c = _two_url_candidate(f"{base}/notanimage", f"{base}/ok.png")
    payload = _fetch(c).to_dict()
    assert payload["image_url_used"] == f"{base}/ok.png"
    assert payload["attempts"][0]["status"] == CandidateStatus.INVALID_IMAGE.value
