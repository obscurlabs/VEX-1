"""Add a small, explicitly-labeled face set to the candidate corpus.

Wikimedia categories require human verification because membership does not
imply depiction. These images instead come from a face-recognition project's
example folder, where the filename is the project's own identity label - a
much stronger signal. They are still confirmed by eye before promotion.

    python scripts/fetch_labeled_examples.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ImageStatus  # noqa: E402
from src.vision.embedder import ArcFaceEmbedder  # noqa: E402
from src.vision.quality import decode_bytes  # noqa: E402

RAW = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples"
STAGE = Path(__file__).resolve().parent.parent / "tests" / "data" / "_candidates"

# slug -> filenames. Only genuinely distinct photographs; the -240p/-480p
# variants are the same shot downscaled and would make same-person pairs
# meaninglessly easy.
SETS = {
    "barack_obama_gh": ["obama.jpg", "obama2.jpg", "obama_small.jpg"],
    "joe_biden": ["biden.jpg"],
    "alex_lacamoire": ["alex-lacamoire.png"],
    "lin_manuel_miranda": ["lin-manuel-miranda.png"],
}


def main() -> int:
    embedder = ArcFaceEmbedder()
    for slug, names in SETS.items():
        dest = STAGE / slug
        dest.mkdir(parents=True, exist_ok=True)
        sources: dict[str, str] = {}
        kept = 0
        print(f"\n[{slug}]")
        for name in names:
            url = f"{RAW}/{name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "hhgoa-task3"})
                blob = urllib.request.urlopen(req, timeout=30).read()
            except Exception as exc:
                print(f"   fetch failed {name}: {exc}")
                continue

            status, img, err = decode_bytes(blob)
            if status is not ImageStatus.OK or img is None:
                print(f"   skip {name}: {err}")
                continue
            faces = embedder.detector.detect(img, assess_quality=False)
            if len(faces) != 1:
                print(f"   skip {name}: {len(faces)} faces")
                continue

            kept += 1
            out = f"{kept:02d}.jpg"
            (dest / out).write_bytes(blob)
            sources[out] = f"{RAW}/{name}"
            print(f"   staged {slug}/{out}  <- {name}  ({img.shape[1]}x{img.shape[0]})")
        (dest / "sources.json").write_text(json.dumps(sources, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
