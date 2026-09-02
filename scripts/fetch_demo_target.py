"""Fetch a final-demo target image with a strong public web presence.

Uses the Wikipedia lead (infobox) image for a named person. That image is the
one the article itself uses to depict the subject, so it is a far stronger
identity signal than category membership - and it is among the most widely
reproduced photographs of that person, which is what live discovery needs.

The downloaded image is validated locally (decodable, exactly one usable face)
and must still be confirmed by eye before it is used.

    python scripts/fetch_demo_target.py "Justin Trudeau" inputs/demo-target.jpg
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ImageStatus  # noqa: E402
from src.vision.embedder import ArcFaceEmbedder  # noqa: E402
from src.vision.quality import decode_bytes  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
UA = "HHGoa2026-Task3/1.0 (demo target fetcher; local dev)"
MIN_FACE_PX = 110


def lead_image_url(person: str) -> tuple[str, str] | None:
    """Return (image_url, page_title) for the article's lead image."""
    params = {
        "action": "query",
        "titles": person,
        "prop": "pageimages",
        "piprop": "original",
        "format": "json",
        "redirects": "1",
    }
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": UA}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)

    for page in data.get("query", {}).get("pages", {}).values():
        original = page.get("original") or {}
        if original.get("source"):
            return original["source"], page.get("title", person)
    return None


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    person = sys.argv[1] if len(sys.argv) > 1 else "Justin Trudeau"
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "inputs/demo-target.jpg")

    found = lead_image_url(person)
    if not found:
        print(f"no lead image for {person!r}")
        return 1
    url, title = found
    print(f"person      {title}")
    print(f"source      {url}")

    blob = download(url)
    status, img, err = decode_bytes(blob)
    if status is not ImageStatus.OK or img is None:
        print(f"decode failed: {err}")
        return 1
    print(f"decoded     {img.shape[1]}x{img.shape[0]}  ({len(blob)} bytes)")

    embedder = ArcFaceEmbedder()
    faces = embedder.detector.detect(img)
    usable = [f for f in faces if min(f.width, f.height) >= MIN_FACE_PX]
    print(f"faces       {len(faces)} detected, {len(usable)} usable (>={MIN_FACE_PX}px)")
    for f in faces:
        print(f"   #{f.index} {f.width}x{f.height}px conf={f.det_score:.3f}")

    if len(usable) != 1:
        print(f"REJECTED: need exactly one clearly visible face, got {len(usable)}")
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    print(f"\nwrote {dest}")
    print("CONFIRM BY EYE before using this as the demo target.")

    sidecar = dest.with_suffix(dest.suffix + ".source.json")
    sidecar.write_text(json.dumps({
        "person": title,
        "source_url": url,
        "origin": "Wikipedia lead image",
        "width": img.shape[1],
        "height": img.shape[0],
        "bytes": len(blob),
        "faces_detected": len(faces),
        "usable_faces": len(usable),
    }, indent=2), encoding="utf-8")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
