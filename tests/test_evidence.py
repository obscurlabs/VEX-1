"""Phase 3: canonicalization, deterministic hashing, verification, tampering.

These tests operate on real bundles built by the collector, and on hand-built
manifests for the canonicalization edge cases.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.evidence import hashing
from src.evidence import manifest as manifest_mod
from src.evidence.collector import (
    FINGERPRINT_FILE,
    MANIFEST_FILE,
    ArtifactStore,
    new_investigation_id,
    verify_bundle,
    write_fingerprint,
)

ROOT = Path(__file__).resolve().parent.parent


# --- canonicalization -------------------------------------------------

def test_hash_is_deterministic_across_many_runs():
    m = {"b": 1, "a": "x", "n": [3, 2, 1], "d": {"z": True, "y": None}}
    digests = {hashing.fingerprint(m)[0] for _ in range(50)}
    assert len(digests) == 1


def test_key_order_does_not_affect_the_hash():
    a = {"alpha": 1, "beta": {"x": "1", "y": "2"}, "gamma": [1, 2]}
    b = {"gamma": [1, 2], "beta": {"y": "2", "x": "1"}, "alpha": 1}
    assert hashing.canonical_bytes(a) == hashing.canonical_bytes(b)
    assert hashing.fingerprint(a)[0] == hashing.fingerprint(b)[0]


def test_list_order_does_affect_the_hash():
    """Sequence is meaningful evidence; only mapping order is normalized."""
    assert hashing.fingerprint({"x": [1, 2]})[0] != hashing.fingerprint({"x": [2, 1]})[0]


def test_canonical_form_has_no_incidental_whitespace():
    assert hashing.canonical_bytes({"a": 1, "b": "c"}) == b'{"a":1,"b":"c"}'


def test_unicode_is_encoded_as_utf8_not_escaped():
    out = hashing.canonical_bytes({"name": "Ursula von der Leyen é"})
    assert b"\\u00e9" not in out
    assert "é".encode("utf-8") in out
    assert out.decode("utf-8")


def test_unicode_normalization_makes_equivalent_strings_hash_alike():
    composed = "café"            # é as one code point
    decomposed = "café"          # e + combining acute
    assert composed != decomposed
    assert hashing.fingerprint({"t": composed})[0] == hashing.fingerprint({"t": decomposed})[0]


def test_unicode_keys_are_normalized_too():
    assert (hashing.canonical_bytes({"café": 1})
            == hashing.canonical_bytes({"café": 1}))


def test_non_ascii_sorts_stably():
    m = {"zebra": 1, "éclair": 2, "apple": 3}
    assert hashing.canonical_bytes(m) == hashing.canonical_bytes(dict(reversed(list(m.items()))))


def test_floats_are_rejected_outright():
    with pytest.raises(TypeError, match="floats are not permitted"):
        hashing.canonical_bytes({"similarity": 0.994378})


def test_nested_floats_are_rejected():
    with pytest.raises(TypeError):
        hashing.canonical_bytes({"a": {"b": [1, 2, 3.5]}})


def test_decimal_must_be_rendered_first():
    from decimal import Decimal
    with pytest.raises(TypeError):
        hashing.canonical_bytes({"x": Decimal("1.5")})


def test_unsupported_types_are_rejected():
    with pytest.raises(TypeError):
        hashing.canonical_bytes({"when": object()})


def test_non_string_keys_are_rejected():
    with pytest.raises(TypeError, match="keys must be strings"):
        hashing.canonical_bytes({1: "a"})


# --- numeric representation -------------------------------------------

def test_decimal_str_is_fixed_precision():
    assert hashing.decimal_str(0.7) == "0.700000"
    assert hashing.decimal_str(0.70) == "0.700000"
    assert hashing.decimal_str(1) == "1.000000"
    assert hashing.decimal_str(0.9943781) == "0.994378"


def test_decimal_str_collapses_negative_zero():
    assert hashing.decimal_str(-0.0000001) == "0.000000"
    assert not hashing.decimal_str(-0.0000001).startswith("-")


def test_decimal_str_handles_negative_similarity():
    """Cosine similarity is legitimately negative for unrelated faces."""
    assert hashing.decimal_str(-0.1381) == "-0.138100"


def test_float_noise_collapses_to_one_representation():
    assert hashing.decimal_str(0.1 + 0.2) == hashing.decimal_str(0.3)
    assert hashing.decimal_str(0.30000000000000004) == "0.300000"


def test_equivalent_numbers_produce_the_same_hash():
    a = {"s": hashing.decimal_str(0.1 + 0.2)}
    b = {"s": hashing.decimal_str(0.3)}
    assert hashing.fingerprint(a)[0] == hashing.fingerprint(b)[0]


def test_a_changed_digit_changes_the_hash():
    a = {"s": hashing.decimal_str(0.994378)}
    b = {"s": hashing.decimal_str(0.994379)}
    assert a != b
    assert hashing.fingerprint(a)[0] != hashing.fingerprint(b)[0]


def test_integers_stay_json_numbers():
    assert hashing.canonical_bytes({"n": 512}) == b'{"n":512}'


def test_booleans_are_not_treated_as_integers():
    assert hashing.canonical_bytes({"x": True}) == b'{"x":true}'
    assert hashing.canonical_bytes({"x": True}) != hashing.canonical_bytes({"x": 1})


# --- secrets ----------------------------------------------------------

def test_secret_shaped_keys_are_rejected():
    for key in ("api_key", "PRIVATE_KEY", "serpapi_key", "authorization", "password"):
        with pytest.raises(ValueError, match="secret-shaped key"):
            manifest_mod.assert_no_secrets({key: "x"})


def test_secret_shaped_nested_keys_are_rejected():
    with pytest.raises(ValueError, match="secret-shaped key"):
        manifest_mod.assert_no_secrets({"search": {"params": {"api_key": "abc"}}})


def test_urls_carrying_a_key_are_rejected():
    with pytest.raises(ValueError, match="secret-shaped value"):
        manifest_mod.assert_no_secrets({"url": "https://serpapi.com/search?api_key=abc123"})


def test_clean_manifest_passes_the_secret_scan():
    manifest_mod.assert_no_secrets({"search": {"image_id": "abc", "engine": "google_lens"}})


# --- bundles ----------------------------------------------------------

@pytest.fixture
def bundle(tmp_path) -> Path:
    """A minimal but structurally real evidence bundle."""
    store = ArtifactStore(new_investigation_id(), root=tmp_path)
    (store.root / "input.jpg").write_bytes(b"fake-input-image-bytes")
    (store.root / "source-image.jpg").write_bytes(b"fake-candidate-image-bytes")
    store.write_json("candidates.json", [{"position": 1, "url": "https://a.example/x"}])
    store.write_json("retrieval.json", [{"position": 1, "status": "RETRIEVED"}])
    store.write_json("matching.json", {"threshold": 0.3, "candidates": []})
    store.write_bytes("search-response.json", b'{"visual_matches":[]}')

    digests = {}
    for name in manifest_mod.ARTIFACT_FILES:
        p = store.root / name
        if p.exists():
            digests[name] = {"sha256": hashing.sha256_file(p), "bytes": p.stat().st_size}

    m = {
        "schema": manifest_mod.SCHEMA,
        "schema_version": manifest_mod.SCHEMA_VERSION,
        "investigation_id": store.investigation_id,
        "created_at": "2026-09-02T00:00:00+00:00",
        "matching": {
            "threshold": hashing.decimal_str(0.3),
            "best_independent_match": {
                "source_url": "https://news.example/story",
                "source_domain": "news.example",
                "similarity": hashing.decimal_str(0.9944),
                "status": "MATCH",
            },
        },
        "artifacts": digests,
    }
    write_fingerprint(store, m)
    return store.root


def test_a_fresh_bundle_verifies(bundle):
    r = verify_bundle(bundle)
    assert r.verified
    assert r.problems == []
    assert r.expected == r.computed
    assert len(r.checked) == 6


def test_manifest_on_disk_is_the_canonical_bytes(bundle):
    raw = (bundle / MANIFEST_FILE).read_bytes()
    reparsed = json.loads(raw.decode("utf-8"))
    assert hashing.canonical_bytes(reparsed) == raw, \
        "the stored file must BE the bytes that were hashed"


def test_recorded_hash_matches_the_stored_manifest(bundle):
    fp = json.loads((bundle / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    raw = (bundle / MANIFEST_FILE).read_bytes()
    assert fp["evidence_sha256"] == hashing.sha256_bytes(raw)
    assert fp["algorithm"] == "SHA-256"
    assert fp["manifest_bytes"] == len(raw)


def test_verification_is_repeatable(bundle):
    assert {verify_bundle(bundle).computed for _ in range(10)} == {
        verify_bundle(bundle).expected}


# --- tampering --------------------------------------------------------

def test_modified_metadata_fails_verification(bundle):
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["matching"]["best_independent_match"]["similarity"] = "0.994500"
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))
    r = verify_bundle(bundle)
    assert not r.verified
    assert "manifest hash mismatch" in r.problems


def test_modified_url_fails_verification(bundle):
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["matching"]["best_independent_match"]["source_url"] = "https://evil.example/other"
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))
    assert not verify_bundle(bundle).verified


def test_modified_artifact_json_fails_verification(bundle):
    path = bundle / "matching.json"
    path.write_text(path.read_text(encoding="utf-8").replace("0.3", "0.9"), encoding="utf-8")
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("matching.json" in p for p in r.problems)


def test_modified_candidate_image_fails_verification(bundle):
    path = bundle / "source-image.jpg"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("source-image.jpg" in p for p in r.problems)


def test_modified_input_image_fails_verification(bundle):
    (bundle / "input.jpg").write_bytes(b"different-input-bytes!")
    assert not verify_bundle(bundle).verified


def test_modified_raw_response_fails_verification(bundle):
    (bundle / "search-response.json").write_bytes(b'{"visual_matches":[{"a":1}]}')
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("search-response.json" in p for p in r.problems)


def test_missing_artifact_fails_verification(bundle):
    (bundle / "input.jpg").unlink()
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("missing artifact" in p for p in r.problems)


def test_missing_manifest_fails_verification(bundle):
    (bundle / MANIFEST_FILE).unlink()
    r = verify_bundle(bundle)
    assert not r.verified
    assert any(MANIFEST_FILE in p for p in r.problems)


def test_missing_fingerprint_fails_verification(bundle):
    (bundle / FINGERPRINT_FILE).unlink()
    assert not verify_bundle(bundle).verified


def test_missing_bundle_directory_fails(tmp_path):
    r = verify_bundle(tmp_path / "no-such-bundle")
    assert not r.verified
    assert any("not found" in p for p in r.problems)


def test_corrupt_manifest_json_fails(bundle):
    (bundle / MANIFEST_FILE).write_bytes(b"{not json")
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("not readable JSON" in p for p in r.problems)


def test_swapped_artifact_of_equal_size_is_still_detected(bundle):
    """Same byte count, different content - the digest still catches it."""
    path = bundle / "input.jpg"
    original = path.read_bytes()
    path.write_bytes(b"X" * len(original))
    r = verify_bundle(bundle)
    assert not r.verified
    assert any("modified" in p for p in r.problems)


# --- CLI --------------------------------------------------------------

def test_verify_cli_returns_zero_for_a_good_bundle(bundle):
    proc = subprocess.run(
        [sys.executable, "verify.py", str(bundle), "--quiet"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFIED" in proc.stdout


def test_verify_cli_returns_nonzero_for_a_tampered_bundle(bundle):
    (bundle / "input.jpg").write_bytes(b"tampered")
    proc = subprocess.run(
        [sys.executable, "verify.py", str(bundle), "--quiet"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "FAILED" in proc.stdout


def test_tamper_script_reports_success(bundle):
    proc = subprocess.run(
        [sys.executable, "scripts/tamper_test.py", str(bundle)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TAMPER DETECTED IN EVERY CASE" in proc.stdout


# --- copies -----------------------------------------------------------

def test_a_copied_bundle_verifies_identically(bundle, tmp_path):
    """No filesystem-dependent state: a copy hashes the same."""
    copy = tmp_path / "copied-bundle"
    shutil.copytree(bundle, copy)
    a, b = verify_bundle(bundle), verify_bundle(copy)
    assert a.verified and b.verified
    assert a.computed == b.computed


# --- raw response preservation ----------------------------------------

def test_raw_response_bytes_are_stored_verbatim(tmp_path):
    """The provider response must survive byte-for-byte.

    The payload below uses key order and whitespace that json.dumps would
    never reproduce, so an equal comparison proves no re-serialization.
    """
    store = ArtifactStore(new_investigation_id(), root=tmp_path)
    wire = b'{"zebra":1,\r\n   "alpha" :  [2,3],\t"caf\xc3\xa9":"\xc3\xa9"}'

    path = store.write_bytes("search-response.json", wire)
    assert path.read_bytes() == wire

    # sanity: any re-serialization really would have changed these bytes
    reserialized = json.dumps(json.loads(wire.decode("utf-8"))).encode("utf-8")
    assert reserialized != wire


def test_raw_bytes_survive_the_search_result_object():
    from src.models import SearchResult

    wire = b'{"visual_matches":[],\n\n  "x":  1}'
    r = SearchResult(provider="google_lens", live=True, raw={"x": 1}, raw_bytes=wire)
    assert r.raw_bytes == wire


def test_raw_response_digest_covers_the_stored_bytes(bundle):
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    entry = m["artifacts"]["search-response.json"]
    stored = (bundle / "search-response.json").read_bytes()
    assert entry["sha256"] == hashing.sha256_bytes(stored)
    assert entry["bytes"] == len(stored)
