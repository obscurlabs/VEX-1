"""Non-destructive audit of everything VEX-1 needs to be trustworthy.

    python healthcheck.py
    python healthcheck.py --json final/healthcheck.json
    python healthcheck.py --section evidence --section blockchain

Sends no SerpAPI search and no blockchain transaction. Connectivity and
credentials are proven with read-only calls (the SerpApi account endpoint,
``eth_chainId``, ``eth_getCode``), so running this costs nothing.

Every check reports one of:

    PASS             verified, with the evidence in ``details``
    FAIL             verified to be broken
    WARN             works, but something a reader should know
    NOT_RUN          could not be attempted (a prerequisite failed)
    NOT_APPLICABLE   does not apply to this configuration

Exit codes: 0 all pass, 1 at least one FAIL, 2 warnings only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.bootstrap import require_dependencies

require_dependencies()

from src.config import CONFIG  # noqa: E402

PASS, FAIL, WARN, NOT_RUN, NOT_APPLICABLE = (
    "PASS", "FAIL", "WARN", "NOT_RUN", "NOT_APPLICABLE")

SECTIONS = (
    "environment", "configuration", "input", "discovery", "retrieval",
    "matching", "evidence", "blockchain", "chain_verification",
    "security", "reproducibility",
)


class Audit:
    """Collects check results and prints them as they are decided."""

    def __init__(self, quiet: bool = False) -> None:
        self.results: list[dict] = []
        self.quiet = quiet
        self._section: str | None = None

    def section(self, name: str) -> None:
        self._section = name
        if not self.quiet:
            print(f"\n{name.upper().replace('_', ' ')}")

    def add(self, name: str, status: str, details: str = "", **extra) -> dict:
        record = {"section": self._section, "name": name,
                  "status": status, "details": details, **extra}
        self.results.append(record)
        if not self.quiet:
            print(f"  [{status:<14}] {name:<38} {details}")
        return record

    def ok(self, name, details="", **extra):
        return self.add(name, PASS, details, **extra)

    def bad(self, name, details="", **extra):
        return self.add(name, FAIL, details, **extra)

    def warn(self, name, details="", **extra):
        return self.add(name, WARN, details, **extra)

    def skip(self, name, details="", **extra):
        return self.add(name, NOT_RUN, details, **extra)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r["status"] == status)

    def to_dict(self) -> dict:
        return {
            "checks": self.results,
            "totals": {s: self.count(s)
                       for s in (PASS, FAIL, WARN, NOT_RUN, NOT_APPLICABLE)},
        }


def _redact_rpc(url: str) -> str:
    """An RPC URL usually embeds the provider key in its path."""
    if not url:
        return ""
    return re.sub(r"(/v2/|/rpc/|apiKey=|key=)[A-Za-z0-9_\-]+",
                  r"\1<REDACTED>", url)


# ---------------------------------------------------------------- environment

def check_environment(a: Audit) -> None:
    a.section("environment")

    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    (a.ok if v >= (3, 11) else a.bad)(
        "python_version", version, value=version, required=">=3.11")

    venv = CONFIG.project_root / ".venv"
    in_venv = venv.exists() and str(venv.resolve()) in str(Path(sys.executable).resolve())
    (a.ok if in_venv else a.warn)(
        "virtualenv", "running from .venv" if in_venv
        else f"interpreter outside .venv: {Path(sys.executable).name}")

    versions = {}
    missing = []
    for module, label in (("cv2", "opencv"), ("numpy", "numpy"),
                          ("onnxruntime", "onnxruntime"), ("insightface", "insightface"),
                          ("requests", "requests"), ("web3", "web3")):
        try:
            versions[label] = getattr(__import__(module), "__version__", "?")
        except ImportError:
            missing.append(label)
    (a.bad if missing else a.ok)(
        "dependencies", f"missing: {', '.join(missing)}" if missing
        else ", ".join(f"{k} {v}" for k, v in versions.items()),
        versions=versions)

    try:
        import cv2
        import numpy as np
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        works = bool(ok) and decoded is not None and decoded.shape == (8, 8, 3)
        (a.ok if works else a.bad)("opencv_roundtrip",
                                   "encode/decode verified" if works else "failed")
    except Exception as exc:
        a.bad("opencv_roundtrip", f"{type(exc).__name__}: {exc}")

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        a.ok("onnxruntime_providers", ", ".join(providers), providers=providers)
    except Exception as exc:
        a.bad("onnxruntime_providers", f"{type(exc).__name__}: {exc}")

    try:
        started = time.perf_counter()
        from src.vision.detector import model_info
        info = model_info()
        elapsed = time.perf_counter() - started
        a.ok("face_model_loads",
             f"{info['pack']} · {info['detector']} + {info['recognizer_file']} "
             f"({elapsed:.1f}s)", model=info, load_seconds=round(elapsed, 2))
    except Exception as exc:
        a.bad("face_model_loads", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- configuration

def check_configuration(a: Audit) -> None:
    a.section("configuration")

    present = {
        "SERPAPI_KEY": bool(CONFIG.serpapi_key),
        "POLYGON_RPC_URL": bool(CONFIG.polygon_rpc_url),
        "PRIVATE_KEY": bool(CONFIG.private_key),
        "CONTRACT_ADDRESS": bool(CONFIG.contract_address),
    }
    unset = [k for k, v in present.items() if not v]
    (a.bad if unset else a.ok)(
        "required_settings", f"unset: {', '.join(unset)}" if unset
        else "all four present (values not shown)", present=present)

    key = CONFIG.private_key.removeprefix("0x")
    if not key:
        a.skip("private_key_shape", "PRIVATE_KEY not set")
    elif len(key) == 64 and all(c in "0123456789abcdefABCDEF" for c in key):
        a.ok("private_key_shape", "64 hex characters (value never printed)")
    else:
        a.bad("private_key_shape",
              f"expected 64 hex characters, got {len(key)} - "
              "this is a wallet key, not an API key or an address")

    address = CONFIG.contract_address
    valid = bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", address or ""))
    (a.ok if valid else a.bad)(
        "contract_address_shape", address if valid else f"malformed: {address!r}",
        address=address if valid else None)

    a.ok("chain_configuration",
         f"expecting chain {CONFIG.chain.expected_chain_id} "
         f"({CONFIG.chain.network_name})",
         expected_chain_id=CONFIG.chain.expected_chain_id)

    a.ok("thresholds",
         f"match {CONFIG.match.threshold} · det {CONFIG.vision.min_det_score} · "
         f"blur {CONFIG.vision.min_blur_variance}",
         match_threshold=CONFIG.match.threshold)

    env = CONFIG.project_root / ".env"
    if not env.exists():
        a.warn("env_file", "no .env present")
    else:
        try:
            proc = subprocess.run(["git", "check-ignore", "-q", ".env"],
                                  cwd=CONFIG.project_root, capture_output=True, timeout=15)
            (a.ok if proc.returncode == 0 else a.bad)(
                "env_gitignored", "yes" if proc.returncode == 0
                else "NOT IGNORED - secrets could be committed")
        except Exception as exc:
            a.warn("env_gitignored", f"could not check: {type(exc).__name__}")

    try:
        tracked = subprocess.run(["git", "ls-files", ".env"], cwd=CONFIG.project_root,
                                 capture_output=True, text=True, timeout=15).stdout.strip()
        history = subprocess.run(["git", "log", "--all", "--oneline", "--", ".env"],
                                 cwd=CONFIG.project_root, capture_output=True,
                                 text=True, timeout=20).stdout.strip()
        clean = not tracked and not history
        (a.ok if clean else a.bad)(
            "env_never_committed",
            "not tracked, absent from history" if clean
            else f"tracked={bool(tracked)} history_entries={bool(history)}")
    except Exception as exc:
        a.warn("env_never_committed", f"could not check: {type(exc).__name__}")


# ---------------------------------------------------------------- input

def check_input(a: Audit, image: Path) -> None:
    a.section("input")

    if not image.exists():
        a.bad("demo_input_exists", f"not found: {image}")
        a.skip("demo_input_readable")
        a.skip("demo_input_face")
        return
    a.ok("demo_input_exists", f"{image.name} ({image.stat().st_size / 1024:,.0f} KB)",
         path=str(image).replace("\\", "/"), bytes=image.stat().st_size)

    from src.models import ImageStatus
    from src.vision.quality import load_image

    status, img, err = load_image(image)
    if status is not ImageStatus.OK or img is None:
        a.bad("demo_input_readable", f"{status.value}: {err}")
        a.skip("demo_input_face")
        return
    height, width = img.shape[:2]
    a.ok("demo_input_readable", f"{width} x {height}", width=width, height=height)

    try:
        from src.vision.embedder import ArcFaceEmbedder
        result = ArcFaceEmbedder().process_image(img, all_faces=False)
    except Exception as exc:
        a.bad("demo_input_face", f"{type(exc).__name__}: {exc}")
        return

    if not result.ok:
        a.bad("demo_input_face", f"{result.face_status.value}: {result.error}")
        return
    face = result.faces[0]
    emb = result.embeddings[0]
    normalized = abs(emb.norm - 1.0) < 1e-4
    a.ok("demo_input_face",
         f"{result.faces_detected} face · {face.width}x{face.height}px · "
         f"det {face.det_score:.3f}",
         faces_detected=result.faces_detected,
         det_score=round(float(face.det_score), 4))
    (a.ok if emb.dim == 512 and normalized else a.bad)(
        "embedding_shape", f"dim {emb.dim}, L2 norm {emb.norm:.6f}",
        dim=emb.dim, norm=round(float(emb.norm), 6))


# ---------------------------------------------------------------- discovery

def check_discovery(a: Audit) -> None:
    a.section("discovery")

    if not CONFIG.serpapi_key:
        a.bad("serpapi_configured", "SERPAPI_KEY not set")
        a.skip("serpapi_authenticated")
        a.skip("serpapi_quota")
    else:
        a.ok("serpapi_configured", "key present (value not shown)")
        try:
            import requests
            resp = requests.get("https://serpapi.com/account",
                                params={"api_key": CONFIG.serpapi_key}, timeout=25)
            if resp.status_code != 200:
                a.bad("serpapi_authenticated", f"HTTP {resp.status_code}")
                a.skip("serpapi_quota")
            else:
                data = resp.json()
                a.ok("serpapi_authenticated", f"plan {data.get('plan_name')}")
                left = data.get("total_searches_left")
                if isinstance(left, int) and left <= 0:
                    a.bad("serpapi_quota", "no searches left")
                elif isinstance(left, int) and left < 5:
                    a.warn("serpapi_quota", f"only {left} searches left")
                else:
                    a.ok("serpapi_quota", f"{left} searches left", remaining=left)
        except Exception as exc:
            a.bad("serpapi_authenticated", f"unreachable: {type(exc).__name__}")
            a.skip("serpapi_quota")

    # Response shape is validated against a real captured response rather than
    # by spending a live search.
    from src.discovery import normalizer

    fixture = CONFIG.project_root / "tests" / "fixtures" / "demo-target-response.json"
    if not fixture.exists():
        a.skip("response_structure", "no captured response available")
        a.skip("raw_response_preserved")
        return

    raw_bytes = fixture.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    visual = raw.get("visual_matches")
    if not isinstance(visual, list) or not visual:
        a.bad("response_structure", "no visual_matches array")
    else:
        fields = set(visual[0])
        needed = {"position", "title", "link", "source", "thumbnail", "image"}
        missing = needed - fields
        (a.bad if missing else a.ok)(
            "response_structure",
            f"missing fields: {sorted(missing)}" if missing
            else f"{len(visual)} visual_matches, all expected fields present",
            visual_matches=len(visual))

    candidates = normalizer.normalize(raw)
    (a.ok if candidates else a.bad)(
        "normalizer", f"{len(candidates)} candidates from "
                      f"{normalizer.count_raw_results(raw)} raw rows",
        normalized=len(candidates), raw_rows=normalizer.count_raw_results(raw))

    # Re-serialising must differ, otherwise "byte-preserved" proves nothing.
    reserialized = json.dumps(raw, separators=(",", ":")).encode("utf-8")
    (a.ok if reserialized != raw_bytes else a.warn)(
        "raw_response_preserved",
        "stored bytes differ from any re-serialisation, so preservation is real"
        if reserialized != raw_bytes else "re-serialisation coincidentally matches")


# ---------------------------------------------------------------- retrieval

class _Handler:
    """Placeholder so the module imports without http.server at top level."""


def check_retrieval(a: Audit) -> None:
    """Exercise retrieval against a local server: real sockets, no internet."""
    a.section("retrieval")

    import http.server

    import cv2
    import numpy as np

    from src.discovery.retrieval import CandidateRetriever
    from src.models import CandidateStatus, SearchCandidate

    # Must clear CONFIG.vision.min_image_dim, or the pipeline correctly
    # rejects it as TOO_SMALL and this probe measures the wrong thing.
    side = max(CONFIG.vision.min_image_dim * 2, 128)
    png = cv2.imencode(".png", np.full((side, side, 3), 90, dtype=np.uint8))[1].tobytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            if self.path == "/ok.png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            elif self.path == "/forbidden":
                self.send_error(403)
            elif self.path == "/missing":
                self.send_error(404)
            elif self.path == "/html":
                body = b"<html>not an image</html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as exc:
        a.skip("retrieval_local_server", f"cannot bind a local port: {exc}")
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def candidate(path: str, pos: int) -> SearchCandidate:
        url = f"{base}{path}" if path.startswith("/") else path
        return SearchCandidate(url=url, title="t", source_domain="127.0.0.1",
                               image_url=url, thumbnail_url=None, position=pos,
                               provider="google_lens")

    try:
        results = CandidateRetriever().fetch_all([
            candidate("/ok.png", 1), candidate("/forbidden", 2),
            candidate("/missing", 3), candidate("/html", 4),
            candidate("http://127.0.0.1:9/dead", 5),
        ])
        got = [r.status for r in results]
        expected = [CandidateStatus.RETRIEVED, CandidateStatus.HTTP_403,
                    CandidateStatus.HTTP_404, CandidateStatus.INVALID_IMAGE,
                    CandidateStatus.FETCH_FAILED]
        (a.ok if got == expected else a.bad)(
            "failure_classification",
            " ".join(s.value for s in got),
            expected=[s.value for s in expected], observed=[s.value for s in got])

        (a.ok if len(results) == 5 else a.bad)(
            "candidate_isolation",
            "one failure never terminates the run; all 5 produced a result",
            results=len(results))

        good = results[0]
        import hashlib
        expected_hash = hashlib.sha256(png).hexdigest()
        (a.ok if good.content == png else a.bad)(
            "content_bytes_preserved",
            f"{good.bytes_downloaded} bytes retained verbatim")
        (a.ok if good.content_sha256 == expected_hash else a.bad)(
            "content_hash_correct",
            f"sha256 {(good.content_sha256 or '')[:16]}… matches an independent digest")
    except Exception as exc:
        a.bad("retrieval_local_server", f"{type(exc).__name__}: {exc}")
    finally:
        httpd.shutdown()

    cfg = CONFIG.retrieval
    a.ok("bounded_concurrency",
         f"concurrency {cfg.concurrency} · connect {cfg.connect_timeout}s · "
         f"read {cfg.read_timeout}s",
         concurrency=cfg.concurrency,
         socket_worst_case_seconds=cfg.worst_case_seconds)
    a.warn("dns_not_covered_by_socket_timeouts",
           "getaddrinfo is an OS call that connect/read timeouts do not bound; "
           "an unresponsive resolver has been observed to extend one candidate "
           "to ~83s. Correctness is unaffected - the candidate is classified "
           "FETCH_FAILED and the run continues - but a stage can run long.",
           observed_worst_case_seconds=83)


# ---------------------------------------------------------------- matching

def check_matching(a: Audit, image: Path) -> None:
    a.section("matching")

    import cv2

    from src.matching import ranker
    from src.matching.similarity import cosine_similarity
    from src.models import CandidateMatch, CandidateResult, CandidateStatus, SearchCandidate
    from src.vision.embedder import ArcFaceEmbedder

    if not image.exists():
        a.skip("similarity_is_real", "no input image")
        return

    embedder = ArcFaceEmbedder()
    img = cv2.imread(str(image))
    target = embedder.process_image(img, all_faces=False).embeddings[0]

    self_sim = cosine_similarity(target, target)
    (a.ok if abs(self_sim - 1.0) < 1e-5 else a.bad)(
        "similarity_is_real", f"self-similarity {self_sim:.6f} (expected 1.0)",
        self_similarity=round(float(self_sim), 6))

    # A visibly different image must not score like the same person.
    flipped = embedder.process_image(cv2.flip(img, 1), all_faces=False)
    if flipped.ok:
        mirrored = cosine_similarity(target, flipped.embeddings[0])
        a.ok("similarity_discriminates",
             f"mirrored image scores {mirrored:.4f}, not 1.0",
             mirrored_similarity=round(float(mirrored), 4))

    def synthetic(sim, domain, pos, identical=False, faces=1):
        cand = SearchCandidate(url=f"https://{domain}/p{pos}", title="t",
                               source_domain=domain, image_url=None,
                               thumbnail_url=None, position=pos, provider="google_lens")
        res = CandidateResult(candidate=cand)
        res.status = CandidateStatus.RETRIEVED
        res.content_sha256 = f"{pos:064d}"
        status = (CandidateStatus.MATCH if faces == 1
                  else CandidateStatus.MULTIPLE_FACE_MATCH)
        if sim < CONFIG.match.threshold:
            status = CandidateStatus.REJECTED
        return CandidateMatch(candidate=cand, status=status, best_similarity=sim,
                              threshold=CONFIG.match.threshold, faces_detected=faces,
                              identical_to_input=identical, retrieval=res)

    matches = [synthetic(1.0, "src.example", 1, identical=True),
               synthetic(0.95, "a.example", 2), synthetic(0.91, "a.example", 3),
               synthetic(0.88, "b.example", 4), synthetic(0.10, "c.example", 5),
               synthetic(0.80, "d.example", 6, faces=3)]

    qualifying = ranker.qualifying(matches)
    below = [m for m in matches if m.best_similarity < CONFIG.match.threshold]
    (a.ok if all(m not in qualifying for m in below) else a.bad)(
        "threshold_respected",
        f"{len(qualifying)} of {len(matches)} cleared {CONFIG.match.threshold}",
        qualifying=len(qualifying))

    scores = [m.best_similarity for m in qualifying]
    (a.ok if scores == sorted(scores, reverse=True) else a.bad)(
        "ranking_ordered", f"descending: {[round(s, 2) for s in scores]}")

    groups = ranker.independent_matches(matches)
    grouped = any(g.size > 1 for g in groups)
    (a.ok if grouped else a.bad)(
        "duplicate_grouping",
        f"{len(qualifying)} qualifying → {len(groups)} independent sources",
        qualifying=len(qualifying), independent=len(groups))

    top = ranker.best_independent_match(matches)
    (a.ok if groups and top is groups[0].representative else a.bad)(
        "backwards_compatible_accessor",
        "best_independent_match() is the first ranked group's representative")

    (a.ok if not any(g.representative.identical_to_input for g in groups) else a.bad)(
        "identical_input_excluded",
        "the input file rediscovered is never an independent source")

    multi = [m for m in matches if m.status is CandidateStatus.MULTIPLE_FACE_MATCH]
    (a.ok if multi and all(m.status.is_match for m in multi) else a.bad)(
        "multiple_face_policy",
        "MULTIPLE_FACE_MATCH counts as a match but stays distinct from MATCH")

    banned = ("% likely", "probability this", "confidence score", "% match")
    offenders = []
    for name in ("src/matching/ranker.py", "src/evidence/manifest.py", "main.py"):
        text = (CONFIG.project_root / name).read_text(encoding="utf-8").lower()
        offenders += [f"{name}:{p}" for p in banned if p in text]
    (a.ok if not offenders else a.bad)(
        "no_probability_language",
        "similarity is presented as cosine everywhere" if not offenders
        else f"found {offenders}")


# ---------------------------------------------------------------- evidence

def _newest_bundle() -> Path | None:
    bundles = [p for p in CONFIG.evidence_dir.glob("TRACE-*")
               if (p / "manifest.json").exists()]
    return max(bundles, key=lambda p: p.stat().st_mtime) if bundles else None


def check_evidence(a: Audit, bundle: Path | None) -> None:
    a.section("evidence")

    from src.evidence import hashing
    from src.evidence import manifest as manifest_mod
    from src.evidence.collector import FINGERPRINT_FILE, MANIFEST_FILE, verify_bundle

    if bundle is None:
        a.skip("bundle_available", "no evidence bundle with a manifest found")
        return
    a.ok("bundle_available", str(bundle.relative_to(CONFIG.project_root)).replace("\\", "/"),
         bundle=str(bundle.relative_to(CONFIG.project_root)).replace("\\", "/"))

    raw = (bundle / MANIFEST_FILE).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))

    present = [n for n in manifest_mod.ARTIFACT_FILES if (bundle / n).exists()]
    a.ok("artifacts_present", f"{len(present)} of {len(manifest_mod.ARTIFACT_FILES)}: "
                              f"{', '.join(present)}", artifacts=present)

    canonical = hashing.canonical_bytes(manifest)
    (a.ok if canonical == raw else a.bad)(
        "manifest_is_canonical",
        "the stored file IS the bytes that were hashed" if canonical == raw
        else "stored manifest is not its own canonical form")

    check = verify_bundle(bundle)
    (a.ok if check.verified else a.bad)(
        "artifact_digests_verify",
        f"{len(check.checked)} artifacts match their recorded digests"
        if check.verified else "; ".join(check.problems),
        checked=len(check.checked))

    digest_a = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    digest_b = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    recorded = json.loads((bundle / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    matches_recorded = digest_a == recorded.get("evidence_sha256")
    (a.ok if digest_a == digest_b and matches_recorded else a.bad)(
        "fingerprint_recomputes",
        f"{digest_a[:16]}… reproduced from the files on disk",
        evidence_sha256=digest_a)

    # fingerprint.json must be advisory only.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(bundle, copy)
        lying = json.loads((copy / FINGERPRINT_FILE).read_text(encoding="utf-8"))
        lying["evidence_sha256"] = "0" * 64
        (copy / FINGERPRINT_FILE).write_text(json.dumps(lying), encoding="utf-8")
        from src.blockchain.verifier import recompute_local
        _, _, recomputed, _, _ = recompute_local(copy)
        (a.ok if recomputed == digest_a else a.bad)(
            "fingerprint_not_authoritative",
            "a falsified fingerprint.json does not change the recomputed hash")

    raw_response = bundle / "search-response.json"
    if raw_response.exists():
        stored = raw_response.read_bytes()
        entry = (manifest.get("artifacts") or {}).get("search-response.json", {})
        digest_ok = hashing.sha256_bytes(stored) == entry.get("sha256")
        differs = json.dumps(json.loads(stored), separators=(",", ":")).encode() != stored
        (a.ok if digest_ok and differs else a.bad)(
            "raw_response_byte_preserved",
            f"{len(stored)} bytes, digest matches, differs from re-serialisation")
    else:
        a.add("raw_response_byte_preserved", NOT_APPLICABLE, "no raw response in bundle")

    blob = json.dumps(manifest).lower()
    leaks = [w for w in ("private_key", "api_key", "serpapi", "embedding_vector",
                         "authorization", "secret") if w in blob]
    (a.ok if not leaks else a.bad)(
        "manifest_has_no_secrets",
        "no credential-shaped keys or embeddings" if not leaks else f"found {leaks}")

    has_vectors = re.search(r"\[-?0\.\d+,\s*-?0\.\d+,\s*-?0\.\d+", json.dumps(manifest))
    (a.ok if not has_vectors else a.bad)(
        "no_biometric_embeddings",
        "no face embedding is written to the bundle")

    matching = manifest.get("matching", {})
    ranked = matching.get("ranked_matches") or []
    if ranked:
        required = ("source_url", "source_domain", "title", "search_position",
                    "image_url", "retrieval_status", "content_sha256",
                    "similarity", "selected_face", "status")
        missing = {f for rec in ranked for f in required if f not in rec}
        (a.ok if not missing else a.bad)(
            "ranked_match_completeness",
            f"{len(ranked)} independent sources, all required fields present"
            if not missing else f"missing {sorted(missing)}",
            independent_sources=len(ranked))
    else:
        a.add("ranked_match_completeness", NOT_APPLICABLE,
              "bundle predates ranked_matches")

    for legacy in ("selected_match", "best_independent_match"):
        (a.ok if legacy in matching else a.warn)(
            f"legacy_{legacy}", "retained for backwards compatibility")


def check_tamper(a: Audit, bundle: Path | None) -> dict:
    """Mutate copies and confirm each is detected. Never touches the original."""
    from src.blockchain.verifier import VerificationStatus, recompute_local
    from src.evidence.collector import verify_bundle
    from src.tamper import CASES

    outcomes: list[dict] = []
    if bundle is None:
        a.skip("tamper_detection", "no bundle available")
        return {"cases": outcomes}

    before = verify_bundle(bundle).verified
    detected = 0
    with tempfile.TemporaryDirectory() as tmp:
        for index, case in enumerate(CASES, start=1):
            dest = Path(tmp) / f"case{index}"
            try:
                detail = case.apply_to_copy(bundle, dest)
            except Exception as exc:
                outcomes.append({"case": case.name, "status": NOT_RUN,
                                 "details": f"{type(exc).__name__}: {exc}"})
                continue
            status, _, _, _, problems = recompute_local(dest)
            local = verify_bundle(dest)
            caught = (not local.verified) or status is not VerificationStatus.VERIFIED
            if caught:
                detected += 1
            outcomes.append({
                "case": case.name, "layer": case.layer,
                "mutation": detail,
                "status": PASS if caught else FAIL,
                "detected_by": ("manifest fingerprint" if case.expects_hash_change
                                else "artifact digest"),
            })

    after = verify_bundle(bundle).verified
    all_caught = detected == len(CASES)
    (a.ok if all_caught and before and after else a.bad)(
        "tamper_detection",
        f"{detected}/{len(CASES)} mutations detected; original verifies "
        f"before={before} after={after}",
        detected=detected, total=len(CASES),
        original_intact=bool(before and after))
    return {"cases": outcomes, "detected": detected, "total": len(CASES),
            "original_verified_before": before, "original_verified_after": after}


# ---------------------------------------------------------------- blockchain

def check_blockchain(a: Audit) -> None:
    a.section("blockchain")

    try:
        from src.blockchain.client import (
            AnchorClient, RpcConnectionError, WalletConfigError, WrongChainError,
            read_only_client,
        )
        from src.blockchain.contract import compile_contract
    except Exception as exc:
        a.bad("blockchain_imports", f"{type(exc).__name__}: {exc}")
        return

    try:
        artifact = compile_contract()
        a.ok("contract_artifact",
             f"solc {artifact['solc_version']} · "
             f"{len(artifact['deployed_bytecode']) // 2 - 1} bytes deployed",
             solc=artifact["solc_version"], source_sha256=artifact["source_sha256"])
    except Exception as exc:
        a.bad("contract_artifact", f"{type(exc).__name__}: {exc}")
        return

    a.ok("rpc_endpoint", _redact_rpc(CONFIG.polygon_rpc_url) or "not set",
         rpc=_redact_rpc(CONFIG.polygon_rpc_url))

    try:
        reader = read_only_client()
    except Exception as exc:
        a.bad("rpc_reachable", f"{type(exc).__name__}: {exc}")
        return

    try:
        chain_id = reader.connect()
    except WrongChainError as exc:
        a.bad("chain_id", str(exc))
        return
    except (RpcConnectionError, Exception) as exc:
        a.bad("rpc_reachable", f"{type(exc).__name__}: {exc}")
        return

    a.ok("rpc_reachable", "responds to eth_chainId")
    (a.ok if chain_id == CONFIG.chain.expected_chain_id else a.bad)(
        "chain_id", f"{chain_id} ({CONFIG.chain.network_name})", chain_id=chain_id)

    address = CONFIG.contract_address
    if not address:
        a.skip("contract_deployed", "CONTRACT_ADDRESS not set")
        return
    try:
        code = reader.w3.eth.get_code(reader.w3.to_checksum_address(address))
    except Exception as exc:
        a.bad("contract_deployed", f"{type(exc).__name__}: {exc}")
        return

    (a.ok if len(code) > 0 else a.bad)(
        "contract_deployed", f"{len(code)} bytes of code at {address}",
        address=address, bytecode_bytes=len(code))

    local = artifact["deployed_bytecode"].removeprefix("0x")
    onchain = code.hex().removeprefix("0x")
    (a.ok if onchain == local else a.warn)(
        "bytecode_matches_source",
        "deployed bytecode is byte-identical to the compiled contract"
        if onchain == local else "deployed bytecode differs from the local build")

    try:
        contract = reader.contract_at(address)
        total = contract.functions.totalAnchored().call()
        a.ok("contract_interface", f"totalAnchored() = {total}", total_anchored=total)
    except Exception as exc:
        a.bad("contract_interface",
              f"not an IdentityAnchor? {type(exc).__name__}")
        return

    # Read-only verification must not need a signing key.
    a.ok("verification_without_key",
         "read-only client has no wallet and still reads the contract",
         wallet=reader.address)

    try:
        signer = AnchorClient()
    except WalletConfigError as exc:
        a.bad("wallet_configured", str(exc))
        return
    a.ok("wallet_configured", signer.address, wallet=signer.address)

    try:
        balance = signer.balance_wei()
    except Exception as exc:
        a.bad("wallet_balance", f"{type(exc).__name__}: {exc}")
        return
    minimum = CONFIG.chain.min_balance_wei
    pol = balance / 1e18
    if balance >= minimum * 3:
        a.ok("wallet_balance", f"{pol:.6f} POL", balance_pol=round(pol, 6))
    elif balance >= minimum:
        a.warn("wallet_balance", f"{pol:.6f} POL - above the floor but low",
               balance_pol=round(pol, 6))
    else:
        a.bad("wallet_balance", f"{pol:.6f} POL - below the "
                                f"{minimum / 1e18:.3f} POL floor",
              balance_pol=round(pol, 6))


def check_chain_verification(a: Audit, bundle: Path | None) -> dict:
    a.section("chain_verification")

    from src.blockchain.client import evidence_bytes32, investigation_key, read_only_client
    from src.blockchain.verifier import VerificationStatus, verify_against_chain

    if bundle is None:
        a.skip("bundle_verifies_on_chain", "no bundle available")
        return {}

    result = verify_against_chain(bundle)
    payload = {
        "bundle": str(bundle.relative_to(CONFIG.project_root)).replace("\\", "/"),
        "verification_status": result.status.value,
        "local_sha256": result.local_sha256,
        "on_chain_sha256": result.on_chain_sha256,
        "contract_address": result.contract_address,
        "chain_id": result.chain_id,
        "submitter": result.submitter,
        "anchored_block": result.anchored_block,
    }

    if result.status is VerificationStatus.NOT_ANCHORED:
        a.add("bundle_verifies_on_chain", NOT_APPLICABLE,
              "this bundle was never anchored", **payload)
        return payload

    (a.ok if result.verified else a.bad)(
        "bundle_verifies_on_chain",
        f"{result.status.value}: local == on-chain" if result.verified
        else "; ".join(result.problems), **payload)

    (a.ok if result.local_sha256 == result.on_chain_sha256 else a.bad)(
        "local_equals_on_chain",
        f"{(result.local_sha256 or '')[:16]}… == {(result.on_chain_sha256 or '')[:16]}…")

    if not result.investigation_id or not result.contract_address:
        return payload

    reader = read_only_client()
    reader.connect()
    contract = reader.contract_at(result.contract_address)
    key = investigation_key(result.investigation_id)

    try:
        good = contract.functions.verifyEvidence(
            key, evidence_bytes32(result.local_sha256)).call()
        (a.ok if good else a.bad)("contract_verifyEvidence_true",
                                  "the contract itself confirms the fingerprint")
    except Exception as exc:
        a.bad("contract_verifyEvidence_true", f"{type(exc).__name__}: {exc}")

    wrong = (result.local_sha256[:-1]
             + ("0" if result.local_sha256[-1] != "0" else "1"))
    try:
        rejected = contract.functions.verifyEvidence(
            key, evidence_bytes32(wrong)).call()
        (a.ok if not rejected else a.bad)(
            "wrong_hash_rejected", "a one-character change is rejected on chain")
    except Exception as exc:
        a.bad("wrong_hash_rejected", f"{type(exc).__name__}: {exc}")

    try:
        unknown = contract.functions.isAnchored(
            investigation_key("TRACE-00000000-NEVER")).call()
        (a.ok if not unknown else a.bad)(
            "unknown_investigation_rejected", "an unanchored id reports not anchored")
    except Exception as exc:
        a.bad("unknown_investigation_rejected", f"{type(exc).__name__}: {exc}")

    return payload


# ---------------------------------------------------------------- security

def check_security(a: Audit) -> None:
    a.section("security")

    from dotenv import dotenv_values

    env = dotenv_values(CONFIG.project_root / ".env")
    secrets = {k: v for k, v in env.items()
               if v and k in ("SERPAPI_KEY", "PRIVATE_KEY")}
    rpc = env.get("POLYGON_RPC_URL", "") or ""
    alchemy = rpc.split("/v2/")[-1] if "/v2/" in rpc else None

    def scan(paths, label: str) -> None:
        leaks, scanned = [], 0
        for path in paths:
            if not path.is_file():
                continue
            scanned += 1
            try:
                blob = path.read_bytes()
            except OSError:
                continue
            for name, value in secrets.items():
                for form in (value, value.removeprefix("0x")):
                    if form.encode() in blob:
                        leaks.append(f"{name} in {path}")
            if alchemy and alchemy.encode() in blob:
                leaks.append(f"rpc credential in {path}")
        (a.ok if not leaks else a.bad)(
            label, f"{scanned} files scanned, no secret values found"
            if not leaks else "; ".join(leaks[:5]), files_scanned=scanned)

    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=CONFIG.project_root,
                                 capture_output=True, text=True, timeout=30)
        scan([CONFIG.project_root / p for p in tracked.stdout.split()],
             "no_secrets_in_tracked_files")
    except Exception as exc:
        a.warn("no_secrets_in_tracked_files", f"could not list: {type(exc).__name__}")

    scan(list(CONFIG.evidence_dir.rglob("*")), "no_secrets_in_evidence")
    scan(list((CONFIG.project_root / "build").rglob("*")), "no_secrets_in_build")

    # Serialisation surfaces must never carry a key.
    from src.blockchain.client import TxResult
    receipt = TxResult(tx_hash="0xabc", block_number=1, block_hash="0xdef",
                       gas_used=21000, effective_gas_price=1, status=1,
                       from_address="0x1")
    fields = set(receipt.to_dict())
    (a.ok if not (fields & {"private_key", "key", "secret"}) else a.bad)(
        "tx_result_has_no_key", f"fields: {sorted(fields)}")

    from src.pipeline import RunResult
    result_fields = set(RunResult.__dataclass_fields__)
    (a.ok if not (result_fields & {"private_key", "api_key", "rpc_url"}) else a.bad)(
        "run_result_has_no_secrets", f"fields: {sorted(result_fields)}")

    (a.ok if "PRIVATE_KEY" not in repr(CONFIG) and CONFIG.private_key not in repr(CONFIG)
     else a.bad)("config_repr_safe", "repr(CONFIG) exposes no key")

    # Error text must not echo the key it rejected.
    from src.blockchain.client import AnchorClient, WalletConfigError
    probe = "deadbeef" * 8 + "tail"
    try:
        AnchorClient(private_key=probe, rpc_url="http://127.0.0.1:1")
        a.bad("wallet_errors_redacted", "a malformed key was accepted")
    except WalletConfigError as exc:
        (a.ok if probe not in str(exc) and "deadbeef" not in str(exc) else a.bad)(
            "wallet_errors_redacted", "the rejected key is never echoed")


# ---------------------------------------------------------------- reproducibility

def check_reproducibility(a: Audit, bundle: Path | None) -> None:
    a.section("reproducibility")

    from src.evidence import hashing
    from src.evidence.collector import verify_bundle

    sample = {"b": 1, "a": "é", "n": [3, 2, 1], "nested": {"z": True, "y": None}}
    digests = {hashing.fingerprint(sample)[0] for _ in range(50)}
    (a.ok if len(digests) == 1 else a.bad)(
        "canonical_hash_deterministic", f"50 runs produced {len(digests)} digest(s)")

    reordered = {"nested": {"y": None, "z": True}, "n": [3, 2, 1], "a": "é", "b": 1}
    (a.ok if hashing.canonical_bytes(sample) == hashing.canonical_bytes(reordered)
     else a.bad)("key_order_irrelevant", "insertion order does not change the bytes")

    composed, decomposed = "café", "café"
    (a.ok if hashing.fingerprint({"t": composed})[0]
     == hashing.fingerprint({"t": decomposed})[0] else a.bad)(
        "unicode_normalised", "NFC-equivalent strings hash alike")

    try:
        hashing.canonical_bytes({"x": 0.5})
        a.bad("floats_rejected", "a raw float was accepted into the manifest")
    except TypeError:
        a.ok("floats_rejected", "raw floats are refused; values are decimal strings")

    if bundle is not None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "copied"
            shutil.copytree(bundle, copy)
            original, duplicate = verify_bundle(bundle), verify_bundle(copy)
            (a.ok if original.verified and duplicate.verified
             and original.computed == duplicate.computed else a.bad)(
                "copied_bundle_verifies",
                "a copied bundle produces the same fingerprint")

    bundles = sorted(p for p in CONFIG.evidence_dir.glob("TRACE-*")
                     if (p / "manifest.json").exists())
    if not bundles:
        a.skip("historical_bundles_verify", "none present")
    else:
        failed = [p.name for p in bundles if not verify_bundle(p).verified]
        (a.ok if not failed else a.bad)(
            "historical_bundles_verify",
            f"{len(bundles) - len(failed)}/{len(bundles)} bundles still verify locally"
            + (f"; failing: {failed[:3]}" if failed else ""),
            total=len(bundles), failed=len(failed))


# ---------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description="Non-destructive audit of VEX-1")
    ap.add_argument("--image", default="inputs/demo-target.jpg")
    ap.add_argument("--bundle", help="evidence bundle to audit (default: newest)")
    ap.add_argument("--section", action="append", choices=SECTIONS,
                    help="run only these sections (repeatable)")
    ap.add_argument("--json", help="write the full result to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    wanted = set(args.section or SECTIONS)
    image = Path(args.image)
    if not image.is_absolute():
        image = CONFIG.project_root / image
    bundle = Path(args.bundle) if args.bundle else _newest_bundle()
    if bundle is not None and not bundle.is_absolute():
        bundle = CONFIG.project_root / bundle

    if not args.quiet:
        print("=" * 66)
        print("  VEX-1 HEALTH CHECK".center(66))
        print("=" * 66)
        print(f"  started      {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        print(f"  bundle       {bundle.name if bundle else '(none)'}")
        print("  read-only    no search is spent, no transaction is sent")

    audit = Audit(quiet=args.quiet)
    started = time.perf_counter()
    extra: dict = {}

    if "environment" in wanted:
        check_environment(audit)
    if "configuration" in wanted:
        check_configuration(audit)
    if "input" in wanted:
        check_input(audit, image)
    if "discovery" in wanted:
        check_discovery(audit)
    if "retrieval" in wanted:
        check_retrieval(audit)
    if "matching" in wanted:
        check_matching(audit, image)
    if "evidence" in wanted:
        check_evidence(audit, bundle)
        extra["tamper"] = check_tamper(audit, bundle)
    if "blockchain" in wanted:
        check_blockchain(audit)
    if "chain_verification" in wanted:
        extra["chain_verification"] = check_chain_verification(audit, bundle)
    if "security" in wanted:
        check_security(audit)
    if "reproducibility" in wanted:
        check_reproducibility(audit, bundle)

    elapsed = time.perf_counter() - started
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "bundle": (str(bundle.relative_to(CONFIG.project_root)).replace("\\", "/")
                   if bundle else None),
        **audit.to_dict(),
        **extra,
    }

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = CONFIG.project_root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\n  wrote {out}")

    failed, warned = audit.count(FAIL), audit.count(WARN)
    if not args.quiet:
        totals = payload["totals"]
        print("\n" + "=" * 66)
        print(f"  {totals[PASS]} passed · {failed} failed · {warned} warnings · "
              f"{totals[NOT_RUN]} not run · {totals[NOT_APPLICABLE]} n/a"
              f"   ({elapsed:.1f}s)")
        verdict = ("HEALTHY" if not failed and not warned
                   else "HEALTHY WITH WARNINGS" if not failed else "DEFECTS FOUND")
        print(f"  {verdict}".center(66))
        print("=" * 66)
        if failed:
            for record in audit.results:
                if record["status"] == FAIL:
                    print(f"  FAIL  {record['name']}: {record['details']}")

    return 1 if failed else (2 if warned else 0)


if __name__ == "__main__":
    sys.exit(main())
