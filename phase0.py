"""Phase 0 demonstration: input image -> face detection -> ArcFace -> vector.

    python phase0.py                          # runs on the bundled group photo
    python phase0.py --image inputs/target.jpg
    python phase0.py --image a.jpg --compare b.jpg
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.config import CONFIG
from src.matching.similarity import cosine_similarity, is_match
from src.models import ImageStatus
from src.vision.detector import model_info
from src.vision.embedder import ArcFaceEmbedder
from src.vision.quality import load_image

BAR = "=" * 60


def _load(path: str):
    status, img, err = load_image(path)
    if status is not ImageStatus.OK or img is None:
        print(f"✗ {status.value}: {err}")
        sys.exit(1)
    return img


def report(embedder: ArcFaceEmbedder, img, label: str):
    print(f"\n[{label}]")
    t = time.perf_counter()
    result = embedder.process_image(img)
    dt = (time.perf_counter() - t) * 1000

    q = result.quality
    print(f"  resolution      {q.width}x{q.height}")
    print(f"  blur variance   {q.blur_variance:.1f}" + ("  (BLURRY)" if q.is_blurry else ""))
    print(f"  brightness      {q.brightness:.1f}   contrast {q.contrast:.1f}")

    if not result.ok:
        print(f"  ✗ {result.face_status.value}: {result.error}")
        return result

    print(f"  faces detected  {result.faces_detected}   ({dt:.0f} ms)")
    for f in result.faces:
        print(
            f"    #{f.index}  bbox={f.bbox}  {f.width}x{f.height}px  "
            f"conf={f.det_score:.3f}  blur={f.quality.blur_variance:.0f}"
        )
    for e in result.embeddings:
        print(f"    #{e.face_index}  embedding dim={e.dim}  L2 norm={e.norm:.8f}  dtype={e.vector.dtype}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 local vision flow")
    ap.add_argument("--image", help="input image (defaults to the bundled group photo)")
    ap.add_argument("--compare", help="second image to compare against --image")
    args = ap.parse_args()

    print(BAR)
    print("  PHASE 0 - LOCAL FACE DETECTION + ARCFACE EMBEDDING")
    print(BAR)

    t = time.perf_counter()
    embedder = ArcFaceEmbedder()
    print(f"\nmodel loaded in {time.perf_counter() - t:.1f}s")
    for k, v in model_info().items():
        print(f"  {k:<16} {v}")
    print(f"  threshold        {CONFIG.match.threshold}  (configurable, provisional)")

    if args.image:
        img_a = _load(args.image)
        label_a = args.image
    else:
        from insightface.data import get_image

        img_a, label_a = get_image("t1"), "bundled group photo (t1.jpg)"

    res_a = report(embedder, img_a, label_a)

    if args.compare:
        res_b = report(embedder, _load(args.compare), args.compare)
        if res_a.ok and res_b.ok:
            s = cosine_similarity(res_a.embeddings[0], res_b.embeddings[0])
            print(f"\n  Similarity: {s:.4f}")
            print(f"  Decision:   {'MATCH' if is_match(s) else 'NO MATCH'}"
                  f"   (threshold {CONFIG.match.threshold})")
    elif res_a.ok and len(res_a.embeddings) > 1:
        n = len(res_a.embeddings)
        print(f"\n  pairwise cosine similarity ({n} faces in this image):")
        print("        " + "".join(f"{i:>8}" for i in range(n)))
        for i in range(n):
            row = "".join(
                f"{cosine_similarity(res_a.embeddings[i], res_a.embeddings[j]):>8.3f}"
                for j in range(n)
            )
            print(f"    #{i}  {row}")

    print(f"\n{BAR}")
    print("  ✓ PHASE 0 FLOW COMPLETE")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
