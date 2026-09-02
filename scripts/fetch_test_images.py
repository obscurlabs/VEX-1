"""Build the Phase 0 test corpus from Wikimedia Commons.

Category membership does not guarantee a photo depicts the named person, so
this runs in two stages:

    python scripts/fetch_test_images.py fetch    # download candidates
    <a human confirms identity by looking at them>
    python scripts/fetch_test_images.py promote  # accept the confirmed ones

Candidates land in tests/data/_candidates/<slug>/, confirmed images in
tests/data/<slug>/. Only single-face images large enough to embed are kept.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ImageStatus  # noqa: E402
from src.vision.embedder import ArcFaceEmbedder  # noqa: E402
from src.vision.quality import decode_bytes  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = "HHGoa2026-Task3-Phase0/1.0 (face-pipeline test corpus; local dev)"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data"
STAGE = OUT / "_candidates"

IDENTITIES = [
    "Angela Merkel",
    "Barack Obama",
    "Serena Williams",
    "Narendra Modi",
    "Ursula von der Leyen",
]
WANT = 8          # candidates to stage per identity
MIN_FACE_PX = 110  # a face smaller than this is not worth embedding
THROTTLE = 0.4    # seconds between API calls; Commons returns 429 if pushed

# Commons categories are full of derivative works - drawings, statues,
# caricatures, signatures - which are useless as identity references.
# Cheap title filter first; a human still confirms what survives.
BLOCKLIST = (
    "drawing", "sketch", "painting", "portrait of", "caricature", "cartoon",
    "statue", "bust", "wax", "mural", "graffiti", "signature", "logo",
    "coin", "stamp", "poster", "banner", "plakat", "zeichnung", "karikatur",
    "gemalde", "gemälde", "sculpture", "puppet", "doll", "mask", "protest",
    "demonstration", "artwork", "illustration", "art ", "grave", "memorial",
)


def _get(params: dict, attempt: int = 0) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        time.sleep(THROTTLE)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and attempt < 4:
            wait = 5 * (attempt + 1)
            print(f"      429 - backing off {wait}s")
            time.sleep(wait)
            return _get(params, attempt + 1)
        raise


def category_members(name: str, kind: str, limit: int = 50) -> list[str]:
    data = _get({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{name}",
        "cmtype": kind,
        "cmlimit": limit,
    })
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def collect_files(identity: str, budget: int) -> list[str]:
    """Prefer per-year subcategories: those hold press photographs rather
    than the artwork and memorabilia that clutter the parent category."""
    subcats = category_members(identity, "subcat", limit=50)
    years = [c for c in subcats if re.search(r"(19|20)\d{2}", c)]
    # Spread across years so the corpus is not all from one shoot.
    years.sort(reverse=True)
    picked = years[:10] if years else subcats[:8]

    files: list[str] = []
    for sub in picked:
        files.extend(category_members(sub.split(":", 1)[1], "file", limit=12))
        if len(files) >= budget * 6:
            break
    if not files:
        files = category_members(identity, "file")
    return files


def looks_like_a_photo(title: str) -> bool:
    low = title.lower()
    return not any(word in low for word in BLOCKLIST)


def file_url(title: str, width: int = 900) -> str | None:
    data = _get({
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": width,
    })
    for page in data.get("query", {}).get("pages", {}).values():
        info = page.get("imageinfo")
        if info:
            return info[0].get("thumburl") or info[0].get("url")
    return None


def download(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception as exc:
        print(f"      fetch failed: {type(exc).__name__}: {exc}")
        return None


def stage_candidates() -> int:
    embedder = ArcFaceEmbedder()
    STAGE.mkdir(parents=True, exist_ok=True)

    for identity in IDENTITIES:
        slug = identity.lower().replace(" ", "_")
        dest = STAGE / slug
        dest.mkdir(exist_ok=True)
        sources: dict[str, str] = {}
        kept = 0
        print(f"\n[{identity}]")

        for title in collect_files(identity, WANT):
            if kept >= WANT:
                break
            if not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            if not looks_like_a_photo(title):
                print(f"   skip {title[:60]}: title suggests a derivative work")
                continue
            url = file_url(title)
            if not url:
                continue
            blob = download(url)
            if not blob:
                continue

            status, img, err = decode_bytes(blob)
            if status is not ImageStatus.OK or img is None:
                print(f"   skip {title[:60]}: {err}")
                continue

            faces = embedder.detector.detect(img, assess_quality=False)
            big = [f for f in faces if min(f.width, f.height) >= MIN_FACE_PX]
            if len(big) != 1:
                print(f"   skip {title[:60]}: {len(big)} usable faces")
                continue

            kept += 1
            name = f"{kept:02d}.jpg"
            (dest / name).write_bytes(blob)
            sources[name] = title
            print(f"   staged {slug}/{name}  <- {title[:60]}")

        (dest / "sources.json").write_text(json.dumps(sources, indent=2))

    print(f"\nStaged under {STAGE}")
    print("Inspect them, then list the confirmed ones in tests/data/confirmed.json")
    return 0


def promote() -> int:
    """Copy the human-confirmed candidates into the corpus proper."""
    manifest = OUT / "confirmed.json"
    if not manifest.exists():
        print(f"missing {manifest}")
        return 1

    confirmed: dict[str, dict] = json.loads(manifest.read_text())
    total = 0
    for slug, spec in confirmed.items():
        # "from" names the staged directory explicitly. Inferring it caused a
        # real mix-up: two sources staged under similar slugs, and unverified
        # images were promoted in place of the ones a human had checked.
        source, names = spec["from"], spec["images"]
        src_dir = STAGE / source
        if not src_dir.is_dir():
            print(f"   MISSING staged directory {src_dir}")
            continue

        staged = [src_dir / n for n in names]
        missing = [p for p in staged if not p.exists()]
        if missing:
            for p in missing:
                print(f"   MISSING {p}")
            continue  # promote a set completely or not at all

        dest = OUT / slug
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.jpg"):
            old.unlink()
        for i, src in enumerate(staged, start=1):
            shutil.copyfile(src, dest / f"{i:02d}.jpg")
            total += 1
        print(f"[{slug}] {len(names)} images from _candidates/{source}")

    print(f"\n{total} confirmed images in {OUT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    raise SystemExit(stage_candidates() if cmd == "fetch" else promote())
