"""Demonstrate that tampering with an evidence bundle is detectable.

Copies a bundle to a temporary directory, alters exactly one value in one
non-binary file, and shows that the recomputed hash no longer matches. The
original bundle is never modified.

    python scripts/tamper_test.py evidence/TRACE-20260902-A41E7A
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import require_dependencies  # noqa: E402

require_dependencies()

from src.evidence import hashing  # noqa: E402
from src.evidence.collector import MANIFEST_FILE, verify_bundle  # noqa: E402

BAR = "=" * 70


def _report(label: str, bundle: Path) -> bool:
    r = verify_bundle(bundle)
    verdict = "VERIFIED" if r.verified else "FAILED"
    print(f"  {label:<34} {verdict}")
    if not r.verified:
        for p in r.problems:
            print(f"      - {p}")
    return r.verified


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/tamper_test.py <evidence/TRACE-...>")
        return 2
    original = Path(sys.argv[1])
    if not (original / MANIFEST_FILE).exists():
        print(f"no {MANIFEST_FILE} in {original}")
        return 2

    print(BAR)
    print("  TAMPER TEST")
    print(BAR)

    manifest = json.loads((original / MANIFEST_FILE).read_bytes().decode("utf-8"))
    baseline = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    print(f"  original bundle   {original}")
    print(f"  recorded hash     {baseline}")
    print()

    print("  BEFORE TAMPERING")
    if not _report("original bundle", original):
        print("\n  original bundle does not verify; aborting")
        return 1
    print()

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        # --- 1. alter a value inside the manifest itself ----------------
        case1 = Path(tmp) / "case1-manifest-value"
        shutil.copytree(original, case1)
        tampered = json.loads((case1 / MANIFEST_FILE).read_bytes().decode("utf-8"))
        before = tampered["matching"]["best_independent_match"]["similarity"]
        # smallest possible change: last decimal digit +1
        after = before[:-1] + str((int(before[-1]) + 1) % 10)
        tampered["matching"]["best_independent_match"]["similarity"] = after
        (case1 / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(tampered))
        new_hash = hashing.sha256_bytes(hashing.canonical_bytes(tampered))

        print("  CASE 1 - one digit changed in the manifest")
        print(f"    similarity      {before} -> {after}")
        print(f"    original hash   {baseline}")
        print(f"    tampered hash   {new_hash}")
        print(f"    hashes differ   {new_hash != baseline}")
        if new_hash == baseline:
            print("    *** HASH DID NOT CHANGE ***")
            failures += 1
        if _report("tampered bundle", case1):
            print("    *** TAMPERED BUNDLE STILL VERIFIED ***")
            failures += 1
        print()

        # --- 2. alter a byte in a covered non-binary artifact -----------
        case2 = Path(tmp) / "case2-artifact-byte"
        shutil.copytree(original, case2)
        target = None
        for name in ("matching.json", "candidates.json", "retrieval.json"):
            if (case2 / name).exists():
                target = case2 / name
                break
        if target:
            raw = target.read_bytes()
            mutated = bytearray(raw)
            # Prefer flipping a digit past the midpoint; fall back to any
            # digit, then to any byte at all, so this works on small files too.
            idx = next((i for i, b in enumerate(raw)
                        if chr(b).isdigit() and i > len(raw) // 2), None)
            if idx is None:
                idx = next((i for i, b in enumerate(raw) if chr(b).isdigit()), None)
            if idx is None:
                idx = len(raw) // 2
                mutated[idx] ^= 0x01
            else:
                mutated[idx] = ord(str((int(chr(raw[idx])) + 1) % 10))
            target.write_bytes(bytes(mutated))
            print(f"  CASE 2 - one byte changed in {target.name} (offset {idx})")
            print(f"    {chr(raw[idx])!r} -> {chr(mutated[idx])!r}")
            if _report("tampered bundle", case2):
                print("    *** TAMPERED BUNDLE STILL VERIFIED ***")
                failures += 1
            print()

        # --- 3. alter the candidate image -------------------------------
        case3 = Path(tmp) / "case3-candidate-image"
        shutil.copytree(original, case3)
        img = case3 / "source-image.jpg"
        if img.exists():
            data = bytearray(img.read_bytes())
            data[-1] ^= 0xFF
            img.write_bytes(bytes(data))
            print("  CASE 3 - final byte of source-image.jpg flipped")
            if _report("tampered bundle", case3):
                print("    *** TAMPERED BUNDLE STILL VERIFIED ***")
                failures += 1
            print()

        # --- 4. remove an artifact --------------------------------------
        case4 = Path(tmp) / "case4-missing-artifact"
        shutil.copytree(original, case4)
        (case4 / "input.jpg").unlink()
        print("  CASE 4 - input.jpg deleted")
        if _report("tampered bundle", case4):
            print("    *** TAMPERED BUNDLE STILL VERIFIED ***")
            failures += 1
        print()

    print("  AFTER TAMPERING")
    if not _report("original bundle (untouched)", original):
        print("    *** ORIGINAL BUNDLE WAS MODIFIED ***")
        failures += 1

    print()
    print(BAR)
    if failures:
        print(f"  ✗ TAMPER TEST FAILED ({failures} problem(s))")
        print(BAR)
        return 1
    print("  ✓ TAMPER DETECTED IN EVERY CASE; ORIGINAL STILL VERIFIES")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
