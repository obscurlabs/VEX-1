"""HH Goa 2026 Task 3 - face evidence pipeline.

One command runs the whole thing:

    python main.py --image inputs/demo-target.jpg --mode live

    FACE SCAN -> WEB DISCOVERY -> CANDIDATE RETRIEVAL -> FACE MATCHING
      -> EVIDENCE BUNDLE -> SHA-256 -> POLYGON AMOY ANCHOR -> VERIFICATION

Flags:
    --mode live|diagnostic  live never touches a cached response
    --verbose               per-candidate detail instead of aggregate counts
    --debug                 let unexpected exceptions surface with a traceback
    --no-chain              stop after the evidence fingerprint
    --max-candidates N      how many discovered candidates to retrieve
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Fail with a readable message, not an import traceback, when the entry
# point is run with an interpreter that lacks the dependencies.
from src.bootstrap import require_dependencies

require_dependencies()

from src.blockchain.client import (
    AnchorClient,
    ChainError,
    InsufficientFundsError,
    ReceiptTimeout,
    RpcConnectionError,
    WalletConfigError,
    WrongChainError,
)
from src.blockchain.verifier import VerificationStatus, verify_against_chain
from src.config import CONFIG
from src.discovery import normalizer
from src.discovery.base import SearchAuthError, SearchError, SearchRateLimitError
from src.discovery.google_lens import GoogleLensProvider
from src.discovery.retrieval import CandidateRetriever
from src.evidence import hashing, manifest as manifest_mod
from src.evidence.collector import (
    ArtifactStore,
    new_investigation_id,
    utc_now_iso,
    verify_bundle,
    write_fingerprint,
)
from src.matching import ranker
from src.matching.ranker import CandidateMatcher
from src.models import CandidateStatus, ImageStatus, SearchResult
from src.pipeline import (
    EXIT_CHAIN,
    EXIT_EVIDENCE,
    EXIT_INPUT,
    EXIT_NO_MATCH,
    EXIT_OK,
    EXIT_SEARCH,
    EXIT_SEARCH_AUTH,
    EXIT_SEARCH_LIMIT,
    EXIT_VERIFY,
    counts,
    fail,
    header,
    info,
    ok,
    stage,
    summary,
    verdict,
    warn,
)
from src.vision.detector import model_info
from src.vision.embedder import ArcFaceEmbedder
from src.vision.quality import load_image

TITLE = "HH GOA 2026 · FACE EVIDENCE PIPELINE"


# ---------------------------------------------------------------- discovery

def run_discovery_live(image_path: Path, store: ArtifactStore, verbose: bool) -> SearchResult:
    """Genuinely live search. There is no cached path in this function."""
    provider = GoogleLensProvider(logger=info if verbose else (lambda *_a: None))
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
    """Replay a saved response. Only ever reached in diagnostic mode."""
    if not path.exists():
        raise FileNotFoundError(f"diagnostic mode needs a saved response; {path} does not exist")
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    return SearchResult(
        provider="google_lens",
        live=False,
        raw=raw,
        raw_bytes=raw_bytes,
        candidates=normalizer.normalize(raw),
        image_id=(raw.get("search_parameters") or {}).get("image_id"),
        search_id=(raw.get("search_metadata") or {}).get("id"),
        raw_result_count=normalizer.count_raw_results(raw),
    )


# ---------------------------------------------------------------- the run

def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    image_path = Path(args.image)
    investigation_id = new_investigation_id()
    started_at = utc_now_iso()
    live = args.mode == "live"

    header(TITLE)
    print(f"  MODE: {args.mode.upper()}"
          + ("" if live else "   ** REPLAYING A SAVED RESPONSE - NOT LIVE **"))
    print(f"  investigation: {investigation_id}")
    print(f"  started:       {started_at}")

    store = ArtifactStore(investigation_id)

    # ---------------------------------------------------------- [01]
    stage("01", "FACE SCAN")
    status, img, err = load_image(image_path)
    if status is not ImageStatus.OK or img is None:
        fail(f"{status.value}: {err}")
        return EXIT_INPUT
    ok(f"image loaded ({img.shape[1]}x{img.shape[0]}, {image_path.name})")

    embedder = ArcFaceEmbedder()          # model loads once, reused throughout
    vision = embedder.process_image(img, all_faces=False)
    if not vision.ok:
        fail(f"{vision.face_status.value}: {vision.error}")
        return EXIT_INPUT

    face = vision.faces[0]
    emb = vision.embeddings[0]
    mi = model_info()
    ok(f"{vision.faces_detected} face detected "
       f"({face.width}x{face.height}px, confidence {face.det_score:.3f})")
    ok(f"ArcFace embedding generated (dim {emb.dim}, norm {emb.norm:.4f})")
    if args.verbose:
        info(f"detector {mi['detector']} ({mi['detector_file']}), "
             f"recognizer {mi['recognizer_file']}")
    store.copy_input(image_path)

    # ---------------------------------------------------------- [02]
    stage("02", "WEB DISCOVERY")
    try:
        if live:
            info("Google Lens via SerpApi (live request)")
            result = run_discovery_live(image_path, store, args.verbose)
        else:
            src = (Path(args.from_response) if args.from_response
                   else CONFIG.cache_dir / "search-response.json")
            warn(f"NOT LIVE - replaying {src}")
            result = run_discovery_diagnostic(src)
    except SearchAuthError as exc:
        fail(f"[DISCOVERY] SerpAPI authentication failed: {exc}")
        return EXIT_SEARCH_AUTH
    except SearchRateLimitError as exc:
        fail(f"[DISCOVERY] SerpAPI rate limit / quota reached: {exc}")
        return EXIT_SEARCH_LIMIT
    except SearchError as exc:
        fail(f"[DISCOVERY] SerpAPI request failed: {exc}")
        return EXIT_SEARCH
    except FileNotFoundError as exc:
        fail(f"[DISCOVERY] {exc}")
        return EXIT_SEARCH

    if live:
        ok(f"Google Lens search completed ({result.elapsed_ms / 1000:.1f}s)")
        if args.verbose:
            info(f"image_id  {result.image_id}")
            info(f"search_id {result.search_id}")

    domains = sorted({c.source_domain for c in result.candidates if c.source_domain})
    ok(f"{len(result.candidates)} candidates discovered "
       f"across {len(domains)} domains")
    store.write_json("candidates.json", [c.to_dict() for c in result.candidates])

    if not result.candidates:
        warn("search succeeded with zero candidates")
        info("nothing to retrieve; exiting cleanly")
        verdict("NO CANDIDATES DISCOVERED", good=False)
        return EXIT_NO_MATCH

    if args.no_retrieval:
        info(f"artifacts: {store.relative(store.root)}")
        return EXIT_OK

    # ---------------------------------------------------------- [03]
    selected = result.candidates[: args.max_candidates]
    stage("03", "CANDIDATE RETRIEVAL")
    info(f"{len(selected)} of {len(result.candidates)} candidates, "
         f"concurrency {CONFIG.retrieval.concurrency}")
    retrieved_at = utc_now_iso()
    retriever = CandidateRetriever(
        logger=(lambda line: info(line)) if args.verbose else None
    )
    fetched = retriever.fetch_all(selected)

    tally: dict[str, int] = {}
    for r in fetched:
        tally[r.status.value] = tally.get(r.status.value, 0) + 1
    retrieved = [r for r in fetched if r.ok]
    skipped = len(fetched) - len(retrieved)

    ok(f"{len(retrieved)} usable images")
    if skipped:
        warn(f"{skipped} candidates skipped")
        counts(tally, skip=CandidateStatus.RETRIEVED.value)
    store.write_json("retrieval.json", [r.to_dict() for r in fetched])

    # ---------------------------------------------------------- [04]
    stage("04", "FACE MATCHING")
    info(f"threshold {CONFIG.match.threshold} (cosine similarity, configurable)")
    if not retrieved:
        fail("no candidate images were retrievable; nothing to match")
        store.write_json("matching.json",
                         {"threshold": CONFIG.match.threshold, "candidates": []})
        verdict("NO USABLE CANDIDATES", good=False)
        return EXIT_NO_MATCH

    input_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    matcher = CandidateMatcher(
        embedder=embedder, input_sha256=input_sha256,
        logger=(lambda line: info(line)) if args.verbose else None,
    )
    matches = matcher.evaluate_all(emb, fetched)
    ranked = ranker.rank(matches)
    dist = ranker.distribution(matches)

    match_tally: dict[str, int] = {}
    for m in matches:
        match_tally[m.status.value] = match_tally.get(m.status.value, 0) + 1

    scored = [m for m in matches if m.best_similarity is not None]
    ok(f"{len(scored)} candidates embedded and scored")
    if dist.get("n"):
        info(f"similarity  min {dist['min']:.4f}  median {dist['median']:.4f}  "
             f"max {dist['max']:.4f}")

    selected_match = ranker.best_match(matches)
    independent = ranker.best_independent_match(matches)
    store.write_json("matching.json", {
        "threshold": CONFIG.match.threshold,
        "input_sha256": input_sha256,
        "target_image": str(image_path).replace("\\", "/"),
        "detector": mi["detector"],
        "face_model": mi["recognizer_file"],
        "distribution": dist,
        "selected": selected_match.to_dict() if selected_match else None,
        "best_independent": independent.to_dict() if independent else None,
        "candidates": [m.to_dict() for m in ranked],
    })

    if selected_match is None:
        fail(f"no candidate reached the threshold ({CONFIG.match.threshold})")
        info("the threshold is NOT lowered to force a result")
        verdict("NO MATCH FOUND", good=False)
        return EXIT_NO_MATCH

    if independent is not None:
        ok("independent match found")
        ok(f"similarity: {independent.best_similarity:.4f}")
        ok(f"source: {independent.candidate.source_domain}")
        if selected_match.identical_to_input:
            info(f"(top hit {selected_match.candidate.source_domain} is the input "
                 "file itself - not independent corroboration)")
    else:
        warn("the only match is the input file rediscovered")
        ok(f"similarity: {selected_match.best_similarity:.4f}")
        ok(f"source: {selected_match.candidate.source_domain}")
        info("this locates the source file; it is not independent corroboration")

    if args.verbose:
        info("")
        for i, m in enumerate(ranked[:10], start=1):
            if m.best_similarity is None:
                continue
            same = "  [SAME FILE AS INPUT]" if m.identical_to_input else ""
            info(f"#{i:02d}  {m.best_similarity:.4f}  {m.status.value:<20} "
                 f"{m.candidate.source_domain}{same}")

    # ---------------------------------------------------------- [05]
    stage("05", "EVIDENCE")
    anchor_match = independent or selected_match
    if anchor_match.retrieval is not None and anchor_match.retrieval.content:
        store.write_bytes("source-image.jpg", anchor_match.retrieval.content)

    digests = {}
    for name in manifest_mod.ARTIFACT_FILES:
        path = store.root / name
        if path.exists():
            digests[name] = {"sha256": hashing.sha256_file(path),
                             "bytes": path.stat().st_size}

    evidence_manifest = manifest_mod.build(
        investigation_id=investigation_id,
        created_at=started_at,
        pipeline_version=CONFIG.pipeline_version,
        input_sha256=input_sha256,
        input_bytes=image_path.stat().st_size,
        input_size=(img.shape[1], img.shape[0]),
        input_face={
            "bbox": list(face.bbox),
            "det_score": hashing.decimal_str(face.det_score, hashing.SCORE_PLACES),
            "width": face.width,
            "height": face.height,
            "faces_detected": vision.faces_detected,
        },
        search=result,
        search_requested_at=started_at,
        normalized_candidate_count=len(result.candidates),
        evaluated_count=len(fetched),
        selected=selected_match,
        independent=independent,
        retrieval={
            "http_status": anchor_match.retrieval.http_status if anchor_match.retrieval else None,
            "content_type": anchor_match.retrieval.content_type if anchor_match.retrieval else None,
            "content_sha256": anchor_match.retrieval.content_sha256 if anchor_match.retrieval else None,
            "bytes_downloaded": anchor_match.retrieval.bytes_downloaded if anchor_match.retrieval else 0,
            "retrieved_at": retrieved_at,
            "candidates_retrieved": len(retrieved),
            "candidates_attempted": len(fetched),
        },
        model={
            "model_pack": mi["pack"],
            "detector": mi["detector"],
            "detector_file": mi["detector_file"],
            "recognizer": "ArcFace",
            "recognizer_file": mi["recognizer_file"],
            "embedding_dim": emb.dim,
            "providers": mi["providers"],
        },
        threshold=CONFIG.match.threshold,
        artifact_digests=digests,
    )

    digest, manifest_path, _ = write_fingerprint(store, evidence_manifest)
    ok(f"evidence bundle created ({len(digests)} artifacts)")
    ok(f"SHA-256: {digest[:8]}...{digest[-4:]}")
    if args.verbose:
        info(digest)
        info(f"canonical manifest {manifest_path.stat().st_size} bytes")

    # Determinism is asserted on the real manifest, every run.
    repeats = {hashing.fingerprint(evidence_manifest)[0] for _ in range(5)}
    if len(repeats) != 1 or repeats.pop() != digest:
        fail("canonicalization is not deterministic")
        return EXIT_EVIDENCE

    local_check = verify_bundle(store.root)
    if not local_check.verified:
        fail("the freshly written bundle does not verify")
        for problem in local_check.problems:
            info(problem)
        return EXIT_EVIDENCE
    ok("local integrity verified")

    if args.no_chain:
        info("--no-chain: stopping before the blockchain anchor")
        _final_summary(investigation_id, independent or selected_match, digest,
                       None, None, store, started)
        verdict("EVIDENCE FINGERPRINT READY")
        return EXIT_OK

    # ---------------------------------------------------------- [06]
    stage("06", "BLOCKCHAIN")
    try:
        client = AnchorClient(logger=info if args.verbose else (lambda *_a: None))
        chain_id = client.connect()
        ok(f"{CONFIG.chain.network_name} (chain id {chain_id})")
        if args.verbose:
            info(f"wallet {client.address}")
    except WalletConfigError as exc:
        fail(f"[CHAIN] wallet configuration error: {exc}")
        return EXIT_CHAIN
    except WrongChainError as exc:
        fail(f"[CHAIN] wrong chain: {exc}")
        return EXIT_CHAIN
    except RpcConnectionError as exc:
        fail(f"[CHAIN] RPC unavailable: {exc}")
        return EXIT_CHAIN

    contract = CONFIG.contract_address
    if not contract:
        fail("[CHAIN] CONTRACT_ADDRESS is not set; run scripts/deploy.py first")
        return EXIT_CHAIN
    info(f"contract {contract}")

    anchor_receipt = None
    try:
        # The contract rejects duplicates by design. Check first rather than
        # spending gas on a transaction that would revert.
        if client.is_anchored(contract, investigation_id):
            warn("this investigation is already anchored; skipping the transaction")
        else:
            client.require_funds()
            ok("wallet balance sufficient")
            anchor_receipt = client.anchor(contract, investigation_id, digest)
            ok("evidence anchored")
            ok(f"transaction confirmed in block {anchor_receipt.block_number}")
            if args.verbose:
                info(f"tx  {anchor_receipt.tx_hash}")
                info(f"gas {anchor_receipt.gas_used} "
                     f"({anchor_receipt.fee_wei / 1e18:.9f} POL)")
    except InsufficientFundsError as exc:
        fail(f"[CHAIN] insufficient funds: {exc}")
        return EXIT_CHAIN
    except ReceiptTimeout as exc:
        ok(f"transaction submitted  TX: {exc.tx_hash}")
        warn(f"confirmation delayed: {exc}")
        info("not reporting this as confirmed")
        return EXIT_CHAIN
    except ChainError as exc:
        fail(f"[CHAIN] anchoring failed: {exc}")
        return EXIT_CHAIN

    # ---------------------------------------------------------- [07]
    stage("07", "VERIFICATION")
    check = verify_against_chain(store.root, contract_address=contract)
    ok(f"local fingerprint:    {check.local_sha256}")
    if check.on_chain_sha256:
        ok(f"on-chain fingerprint: {check.on_chain_sha256}")

    if check.status is not VerificationStatus.VERIFIED:
        fail(f"[VERIFY] {check.status.value}")
        for problem in check.problems:
            info(problem)
        verdict("VERIFICATION FAILED", good=False)
        return EXIT_VERIFY

    ok("hash match")
    _final_summary(investigation_id, independent or selected_match, digest,
                   anchor_receipt, check, store, started)
    verdict("BLOCKCHAIN VERIFIED")
    return EXIT_OK


def _final_summary(investigation_id, match, digest, receipt, check, store, started) -> None:
    rows = [
        ("investigation", investigation_id),
        ("matched source", match.candidate.url[:58]),
        ("domain", match.candidate.source_domain),
        ("similarity", f"{match.best_similarity:.6f}"),
        ("match status", match.status.value),
        ("evidence SHA-256", digest),
    ]
    if check is not None:
        rows += [
            ("network", f"{CONFIG.chain.network_name} (chain id {check.chain_id})"),
            ("contract", check.contract_address),
        ]
        if receipt is not None:
            rows += [
                ("anchoring tx", receipt.tx_hash),
                ("anchoring block", str(receipt.block_number)),
                ("gas used", f"{receipt.gas_used} ({receipt.fee_wei / 1e18:.9f} POL)"),
            ]
        else:
            rows.append(("anchoring tx", "already anchored (no new transaction)"))
        rows += [
            ("on-chain hash", check.on_chain_sha256),
            ("local == on-chain", "YES" if check.verified else "NO"),
        ]
    rows += [
        ("evidence bundle", store.relative(store.root)),
        ("elapsed", f"{time.perf_counter() - started:.1f}s"),
    ]
    summary(rows, title="RESULT")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Face evidence pipeline: discovery, matching, evidence, anchoring")
    ap.add_argument("--image", required=True, help="input face image")
    ap.add_argument("--mode", choices=("live", "diagnostic"), default="live",
                    help="live performs real requests; diagnostic replays a saved response")
    ap.add_argument("--from-response", help="saved search-response.json (diagnostic mode)")
    ap.add_argument("--max-candidates", type=int, default=CONFIG.retrieval.max_candidates)
    ap.add_argument("--verbose", action="store_true", help="per-candidate detail")
    ap.add_argument("--debug", action="store_true",
                    help="let unexpected exceptions surface with a traceback")
    ap.add_argument("--no-chain", action="store_true",
                    help="stop after the evidence fingerprint")
    ap.add_argument("--no-retrieval", action="store_true", help="stop after discovery")
    args = ap.parse_args()

    try:
        return run(args)
    except KeyboardInterrupt:
        print()
        fail("interrupted by user")
        info("no partial result is reported as successful")
        return 130
    except Exception as exc:                      # noqa: BLE001 - CLI boundary
        if args.debug:
            raise
        fail(f"unexpected error: {type(exc).__name__}: {exc}")
        info("re-run with --debug for the full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
