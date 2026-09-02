"""Warm-path timing for the Phase 0 vision stage (CPU, ONNX Runtime)."""
from __future__ import annotations

import glob
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.embedder import ArcFaceEmbedder  # noqa: E402


def main() -> int:
    embedder = ArcFaceEmbedder()
    paths = sorted(glob.glob("tests/data/*/*.jpg"))
    if not paths:
        print("no corpus images")
        return 1

    images = [(p, cv2.imread(p)) for p in paths]
    embedder.process_image(images[0][1])  # warm up the ONNX session

    print(f"{'image':<34}{'px':>12}{'faces':>7}{'ms':>9}")
    times = []
    for p, img in images:
        t = time.perf_counter()
        r = embedder.process_image(img)
        ms = (time.perf_counter() - t) * 1000
        times.append(ms)
        label = str(Path(p).parent.name + "/" + Path(p).name)
        print(f"{label:<34}{f'{img.shape[1]}x{img.shape[0]}':>12}{r.faces_detected:>7}{ms:>9.0f}")

    print(f"\nn={len(times)}  median={statistics.median(times):.0f} ms  "
          f"mean={statistics.mean(times):.0f} ms  "
          f"min={min(times):.0f} ms  max={max(times):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
