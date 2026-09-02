"""HH Goa 2026 Task 3 - face evidence pipeline.

    python main.py --image inputs/target.jpg --mode live
    python main.py --image inputs/target.jpg --mode diagnostic

Implemented through Phase 1 (live discovery + candidate retrieval).
Matching, evidence bundling and the blockchain anchor are not wired yet.
"""
from __future__ import annotations

import argparse
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
from src.evidence.collector import ArtifactStore, new_investigation_id, utc_now_iso
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

    store.write_json("search-response.json", result.raw)
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
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = normalizer.normalize(raw)
    return SearchResult(
        provider="google_lens",
        live=False,
        raw=raw,
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

    banner("TRACE • FACE EVIDENCE")
    print(f"  investigation  {investigation_id}")
    print(f"  mode           {args.mode.upper()}"
          + ("" if args.mode == "live" else "   ** REPLAYING A SAVED RESPONSE **"))
    print(f"  started        {utc_now_iso()}")

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

    print(f"\nartifacts: {store.relative(store.root)}")
    for f in sorted(store.root.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

    print("\n  Phase 2 (face matching against retrieved candidates) is not wired yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
