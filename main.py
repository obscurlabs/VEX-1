"""HH Goa 2026 Task 3 - face evidence pipeline.

    python main.py --image inputs/target.jpg --mode live
    python main.py --image inputs/target.jpg --mode diagnostic

Implemented through Phase 1 (live discovery + candidate retrieval).
Matching, evidence bundling and the blockchain anchor are not wired yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.config import CONFIG
from src.discovery.base import (
    SearchAuthError,
    SearchError,
    SearchRateLimitError,
)
from src.discovery.google_lens import GoogleLensProvider
from src.discovery.retrieval import CandidateRetriever
from src.discovery import normalizer
from src.matching import ranker
from src.evidence import hashing, manifest as manifest_mod
from src.evidence.collector import (
    ArtifactStore,
    new_investigation_id,
    utc_now_iso,
    verify_bundle,
    write_fingerprint,
)
from src.matching.ranker import CandidateMatcher
from src.models import CandidateStatus, ImageStatus, SearchResult
from src.pipeline import banner, die, fail, info, ok, stage, warn
from src.vision.detector import model_info
from src.vision.embedder import ArcFaceEmbedder
from src.vision.quality import load_image


def run_discovery_live(image_path: Path, store: ArtifactStore) -> SearchResult:
    """Genuinely live search. No cached fallback exists on this path."""
    provider = GoogleLensProvider(logger=info)
    info(f"provider   {provider.name} (SerpApi)")
    info(f"upload     {provider.cfg.upload_url}")
    info(f"search     {provider.cfg.search_url}?engine=google_lens")

    result = provider.search(image_path)

    # Byte-for-byte, not a re-serialisation of the parsed dict.
    store.write_bytes("search-response.json", result.raw_bytes)
    store.write_json("search-request.json", {
        "provider": provider.name,
        "engine": "google_lens",
        "image_id": result.image_id,
        "search_id": result.search_id,
        "hl": provider.cfg.hl,
        "country": provider.cfg.country,
        "requested_at": utc_now_iso(),
        "live": True,
    })
    return result


def run_discovery_diagnostic(path: Path) -> SearchResult:
    """Replay a previously saved response. Clearly labelled, never automatic."""
    if not path.exists():
        die(f"diagnostic mode needs a saved response; {path} does not exist")
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    candidates = normalizer.normalize(raw)
    return SearchResult(
        provider="google_lens",
        live=False,
        raw=raw,
        raw_bytes=raw_bytes,
        candidates=candidates,
        image_id=(raw.get("search_parameters") or {}).get("image_id"),
        search_id=(raw.get("search_metadata") or {}).get("id"),
        raw_result_count=normalizer.count_raw_results(raw),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Face evidence pipeline")
    ap.add_argument("--image", required=True, help="input face image")
    ap.add_argument("--mode", choices=("live", "diagnostic"), default="live")
    ap.add_argument("--from-response", help="saved search-response.json (diagnostic mode)")
    ap.add_argument("--max-candidates", type=int, default=CONFIG.retrieval.max_candidates)
    ap.add_argument("--no-retrieval", action="store_true", help="stop after discovery")
    args = ap.parse_args()

    image_path = Path(args.image)
    investigation_id = new_investigation_id()
    started_at = utc_now_iso()

    banner("TRACE • FACE EVIDENCE")
    print(f"  investigation  {investigation_id}")
    print(f"  mode           {args.mode.upper()}"
          + ("" if args.mode == "live" else "   ** REPLAYING A SAVED RESPONSE **"))
    print(f"  started        {started_at}")

    store = ArtifactStore(investigation_id)

    # -- 01 input ----------------------------------------------------
    stage("01", "INPUT")
    status, img, err = load_image(image_path)
    if status is not ImageStatus.OK or img is None:
        fail(f"{status.value}: {err}")
        return 1
    ok(f"image loaded  {img.shape[1]}x{img.shape[0]}  ({image_path})")

    embedder = ArcFaceEmbedder()
    vision = embedder.process_image(img, all_faces=False)
    if not vision.ok:
        fail(f"{vision.face_status.value}: {vision.error}")
        return 1
    face = vision.faces[0]
    ok(f"face detected  {face.width}x{face.height}px  conf={face.det_score:.3f}")
    store.copy_input(image_path)

    # -- 02 identity -------------------------------------------------
    stage("02", "IDENTITY")
    emb = vision.embeddings[0]
    mi = model_info()
    ok(f"ArcFace embedding  dim={emb.dim}  norm={emb.norm:.6f}")
    info(f"detector {mi['detector']} ({mi['detector_file']}), "
         f"recognizer {mi['recognizer_file']}")

    # -- 03 discovery ------------------------------------------------
    stage("03", "LIVE DISCOVERY" if args.mode == "live" else "DIAGNOSTIC DISCOVERY (CACHED)")
    try:
        if args.mode == "live":
            result = run_discovery_live(image_path, store)
        else:
            src = Path(args.from_response) if args.from_response else CONFIG.cache_dir / "search-response.json"
            warn(f"NOT LIVE - replaying {src}")
            result = run_discovery_diagnostic(src)
    except SearchAuthError as exc:
        fail(f"SEARCH FAILED - authentication: {exc}")
        return 2
    except SearchRateLimitError as exc:
        fail(f"SEARCH FAILED - rate limited: {exc}")
        return 3
    except SearchError as exc:
        fail(f"SEARCH FAILED - {type(exc).__name__}: {exc}")
        return 4

    search_requested_at = utc_now_iso()
    if args.mode == "live":
        ok(f"image uploaded   image_id={result.image_id[:28]}...")
        ok(f"Google Lens queried  search_id={result.search_id}  ({result.elapsed_ms:.0f} ms)")
    ok(f"{result.raw_result_count} raw result rows")
    ok(f"{len(result.candidates)} candidates normalized")

    if not result.candidates:
        warn("SEARCH SUCCEEDED WITH ZERO CANDIDATES")
        info("nothing to retrieve or match; exiting cleanly")
        store.write_json("candidates.json", [])
        print(f"\nartifacts: {store.relative(store.root)}")
        return 0

    domains = sorted({c.source_domain for c in result.candidates if c.source_domain})
    info(f"{len(domains)} distinct domains: {', '.join(domains[:8])}"
         + (" ..." if len(domains) > 8 else ""))
    store.write_json("candidates.json", [c.to_dict() for c in result.candidates])

    if args.no_retrieval:
        print(f"\nartifacts: {store.relative(store.root)}")
        return 0

    # -- 04 retrieval ------------------------------------------------
    selected = result.candidates[: args.max_candidates]
    stage("04", f"CANDIDATE RETRIEVAL  ({len(selected)} of {len(result.candidates)}, "
                f"concurrency {CONFIG.retrieval.concurrency})")
    retrieved_at = utc_now_iso()
    retriever = CandidateRetriever(logger=lambda line: print("  " + line))
    fetched = retriever.fetch_all(selected)

    counts: dict[str, int] = {}
    for r in fetched:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    retrieved = [r for r in fetched if r.ok]

    print()
    ok(f"{len(retrieved)}/{len(fetched)} candidate images retrieved")
    for state, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        info(f"{state:<16} {n}")

    store.write_json("retrieval.json", [r.to_dict() for r in fetched])

    # -- 05 matching -------------------------------------------------
    stage("05", f"FACE MATCHING  (threshold {CONFIG.match.threshold})")
    if not retrieved:
        warn("no candidate images were retrievable; nothing to match")
        store.write_json("matching.json",
                         {"threshold": CONFIG.match.threshold, "candidates": []})
        print(f"\nartifacts: {store.relative(store.root)}")
        return 0

    input_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    matcher = CandidateMatcher(embedder=embedder, input_sha256=input_sha256,
                               logger=lambda line: print("  " + line))
    matches = matcher.evaluate_all(emb, fetched)
    ranked = ranker.rank(matches)
    dist = ranker.distribution(matches)

    tally: dict[str, int] = {}
    for m in matches:
        tally[m.status.value] = tally.get(m.status.value, 0) + 1

    print()
    ok(f"{sum(1 for m in matches if m.best_similarity is not None)} candidates scored")
    for state, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        info(f"{state:<22} {n}")
    if dist.get("n"):
        info(f"similarity  n={dist['n']}  min={dist['min']:.4f}  "
             f"median={dist['median']:.4f}  max={dist['max']:.4f}  sd={dist['sd']:.4f}")

    # -- 06 ranking --------------------------------------------------
    stage("06", "RANKING")
    for i, m in enumerate(ranked[:10], start=1):
        if m.best_similarity is None:
            continue
        flag = "MATCH" if m.is_match else "     "
        same = "  [SAME FILE AS INPUT]" if m.identical_to_input else ""
        print(f"  #{i:02d}  {m.best_similarity:.4f}  {flag}  "
              f"{m.candidate.source_domain:<30} face #{m.best_face_index}"
              f"/{m.faces_detected}{same}")

    selected = ranker.best_match(matches)
    independent = ranker.best_independent_match(matches)
    store.write_json("matching.json", {
        "threshold": CONFIG.match.threshold,
        "input_sha256": input_sha256,
        "target_image": str(image_path).replace("\\", "/"),
        "detector": mi["detector"],
        "face_model": mi["recognizer_file"],
        "distribution": dist,
        "selected": selected.to_dict() if selected else None,
        "best_independent": independent.to_dict() if independent else None,
        "candidates": [m.to_dict() for m in ranked],
    })

    print()
    if selected is None:
        warn(f"NO CANDIDATE REACHED THE THRESHOLD ({CONFIG.match.threshold})")
        info("candidates were scored; none matched")
        info("the threshold is NOT lowered to force a result")
    else:
        ok(f"selected candidate  similarity {selected.best_similarity:.4f}"
           f"  ({selected.status.value})")
        info(f"url     {selected.candidate.url}")
        info(f"domain  {selected.candidate.source_domain}")
        info(f"face    #{selected.best_face_index} of {selected.faces_detected} "
             f"detected in the candidate image")
        if selected.status is CandidateStatus.MULTIPLE_FACE_MATCH:
            info("NOTE: this image contains several faces. One of them matches "
                 "the target; the image as a whole does not.")
        if selected.identical_to_input:
            warn("this candidate is BYTE-IDENTICAL to the input image")
            info("that locates the source file; it is not independent corroboration")
            if independent is not None:
                info("")
                ok(f"best independent match  similarity "
                   f"{independent.best_similarity:.4f}  ({independent.status.value})")
                info(f"url     {independent.candidate.url}")
                info(f"domain  {independent.candidate.source_domain}")
                info(f"face    #{independent.best_face_index} of "
                     f"{independent.faces_detected} detected")

    # -- 07 evidence -------------------------------------------------
    stage("07", "EVIDENCE BUNDLE")

    anchor = independent or selected
    if anchor is None:
        warn("no candidate matched; no evidence bundle to build")
        print(f"\nartifacts: {store.relative(store.root)}")
        return 0
    if independent is None:
        warn("the only match is the input file itself; bundling it, but it is "
             "not independent corroboration")

    if anchor.retrieval is not None and anchor.retrieval.content:
        store.write_bytes("source-image.jpg", anchor.retrieval.content)
        ok(f"candidate image saved verbatim ({len(anchor.retrieval.content)} bytes)")

    digests = {}
    for name in manifest_mod.ARTIFACT_FILES:
        path = store.root / name
        if path.exists():
            digests[name] = {"sha256": hashing.sha256_file(path),
                             "bytes": path.stat().st_size}

    input_face = {
        "bbox": list(face.bbox),
        "det_score": hashing.decimal_str(face.det_score, hashing.SCORE_PLACES),
        "width": face.width,
        "height": face.height,
        "faces_detected": vision.faces_detected,
    }
    retrieval_meta = {
        "http_status": anchor.retrieval.http_status if anchor.retrieval else None,
        "content_type": anchor.retrieval.content_type if anchor.retrieval else None,
        "content_sha256": anchor.retrieval.content_sha256 if anchor.retrieval else None,
        "bytes_downloaded": anchor.retrieval.bytes_downloaded if anchor.retrieval else 0,
        "retrieved_at": retrieved_at,
        "candidates_retrieved": len(retrieved),
        "candidates_attempted": len(fetched),
    }
    model_meta = {
        "model_pack": mi["pack"],
        "detector": mi["detector"],
        "detector_file": mi["detector_file"],
        "recognizer": "ArcFace",
        "recognizer_file": mi["recognizer_file"],
        "embedding_dim": emb.dim,
        "providers": mi["providers"],
    }

    evidence_manifest = manifest_mod.build(
        investigation_id=investigation_id,
        created_at=started_at,
        pipeline_version=CONFIG.pipeline_version,
        input_sha256=input_sha256,
        input_bytes=image_path.stat().st_size,
        input_size=(img.shape[1], img.shape[0]),
        input_face=input_face,
        search=result,
        search_requested_at=search_requested_at,
        normalized_candidate_count=len(result.candidates),
        evaluated_count=len(fetched),
        selected=selected,
        independent=independent,
        retrieval=retrieval_meta,
        model=model_meta,
        threshold=CONFIG.match.threshold,
        artifact_digests=digests,
    )

    digest, manifest_path, _ = write_fingerprint(store, evidence_manifest)
    ok(f"canonical manifest  {manifest_path.stat().st_size} bytes")
    ok(f"SHA-256  {digest}")
    info(f"covers {len(digests)} artifacts + every manifest field")
    info(f"anchored candidate: {anchor.candidate.source_domain} "
         f"(similarity {anchor.best_similarity:.6f})")

    # Determinism is asserted on the real manifest, on every run.
    repeats = {hashing.fingerprint(evidence_manifest)[0] for _ in range(5)}
    if len(repeats) != 1 or repeats.pop() != digest:
        fail("CANONICALIZATION IS NOT DETERMINISTIC")
        return 5
    ok("canonicalization deterministic (5x identical)")

    # -- 08 verification ---------------------------------------------
    stage("08", "VERIFICATION")
    check = verify_bundle(store.root)
    if not check.verified:
        fail("EVIDENCE INTEGRITY FAILED")
        for problem in check.problems:
            info(problem)
        return 6
    ok(f"{len(check.checked)} artifact digests match what is on disk")
    ok("recomputed manifest hash matches the recorded fingerprint")
    print()
    print("  ╔" + "═" * 44 + "╗")
    print("  ║" + "✓ EVIDENCE INTEGRITY VERIFIED".center(44) + "║")
    print("  ╚" + "═" * 44 + "╝")

    print(f"\nartifacts: {store.relative(store.root)}")
    for f in sorted(store.root.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

    print("\n  Phase 4 (Polygon Amoy anchoring) is not wired yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
