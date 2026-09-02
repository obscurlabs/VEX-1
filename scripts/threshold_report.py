"""Report the observed cosine-similarity distribution on the test corpus.

This informs FACE_MATCH_THRESHOLD. It does not set it: the corpus is small,
so treat the numbers as a sanity check and a starting point, not a calibration.

    python scripts/threshold_report.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.matching.similarity import cosine_similarity  # noqa: E402
from src.vision.detector import model_info  # noqa: E402
from src.vision.embedder import ArcFaceEmbedder  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


def load_corpus() -> dict[str, list[Path]]:
    manifest = DATA / "confirmed.json"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest} - run scripts/fetch_test_images.py first")
    out = {}
    for slug in json.loads(manifest.read_text()):
        paths = sorted((DATA / slug).glob("*.jpg"))
        if paths:
            out[slug] = paths
    return out


def describe(name: str, xs: list[float]) -> None:
    a = np.asarray(xs)
    print(
        f"  {name:<18} n={a.size:<4} "
        f"min={a.min():.4f}  p05={np.percentile(a,5):.4f}  "
        f"mean={a.mean():.4f}  p95={np.percentile(a,95):.4f}  max={a.max():.4f}  sd={a.std():.4f}"
    )


def main() -> int:
    embedder = ArcFaceEmbedder()
    corpus = load_corpus()

    print("=" * 78)
    print("  SIMILARITY DISTRIBUTION - buffalo_l / w600k_r50 (ArcFace), cosine")
    print("=" * 78)
    print(f"  model: {model_info()}")

    embeddings: dict[str, list] = {}
    for slug, paths in corpus.items():
        vecs = []
        for p in paths:
            r = embedder.process_path(p, all_faces=False)
            if not r.ok:
                print(f"  !! {p}: {r.face_status} {r.error}")
                continue
            vecs.append(r.embeddings[0])
        embeddings[slug] = vecs
        note = "" if len(vecs) >= 2 else "   (different-person pairs only)"
        print(f"  {slug:<22} {len(vecs)} embeddings{note}")

    same, diff = [], []
    labelled_same: list[tuple[str, float]] = []
    for slug, vecs in embeddings.items():
        if len(vecs) >= 2:
            for (i, a), (j, b) in itertools.combinations(enumerate(vecs, 1), 2):
                score = cosine_similarity(a, b)
                same.append(score)
                labelled_same.append((f"{slug} {i:02d}-{j:02d}", score))
    for a_slug, b_slug in itertools.combinations(embeddings, 2):
        for ea in embeddings[a_slug]:
            for eb in embeddings[b_slug]:
                diff.append(cosine_similarity(ea, eb))

    print("\n  SAME-PERSON PAIRS")
    for name, score in sorted(labelled_same, key=lambda x: x[1]):
        print(f"    {name:<28} {score:.4f}")

    print("\n  OBSERVED")
    describe("same person", same)
    describe("different person", diff)

    lo, hi = max(diff), min(same)
    print(f"\n  separation gap: {lo:.4f} .. {hi:.4f}  (width {hi - lo:+.4f})")
    if hi > lo:
        print(f"  any threshold in that band separates this corpus perfectly")
        print(f"  midpoint: {(lo + hi) / 2:.4f}")
    else:
        print("  DISTRIBUTIONS OVERLAP - no threshold separates this corpus cleanly")

    print(f"\n  configured FACE_MATCH_THRESHOLD = {CONFIG.match.threshold}")
    tp = sum(s >= CONFIG.match.threshold for s in same)
    fp = sum(s >= CONFIG.match.threshold for s in diff)
    print(f"    same-person pairs accepted:      {tp}/{len(same)}")
    print(f"    different-person pairs accepted: {fp}/{len(diff)}")

    print("\n  Corpus is small; these numbers do not calibrate a production threshold.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
