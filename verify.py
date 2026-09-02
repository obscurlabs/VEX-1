"""Verify an evidence bundle against its recorded fingerprint.

    python verify.py evidence/TRACE-20260902-A41E7A
    python verify.py evidence/TRACE-20260902-A41E7A --show-manifest

Exit codes: 0 VERIFIED, 1 FAILED, 2 usage error.

Two independent checks must both pass:

  1. every artifact's SHA-256 on disk matches the digest the manifest recorded
  2. re-canonicalizing manifest.json reproduces the recorded evidence_sha256

Phase 4 will compare the same recomputed hash against the on-chain record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evidence import hashing
from src.evidence.collector import MANIFEST_FILE, verify_bundle

BAR = "=" * 66


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify an evidence bundle")
    ap.add_argument("bundle", help="path to evidence/TRACE-...")
    ap.add_argument("--show-manifest", action="store_true",
                    help="print the canonical manifest")
    ap.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = ap.parse_args()

    root = Path(args.bundle)
    result = verify_bundle(root)

    if not args.quiet:
        print(BAR)
        print("  EVIDENCE VERIFICATION")
        print(BAR)
        print(f"  bundle      {root}")

        manifest_path = root / MANIFEST_FILE
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
                print(f"  id          {manifest.get('investigation_id')}")
                print(f"  created     {manifest.get('created_at')}")
                m = manifest.get("matching") or {}
                ind = m.get("best_independent_match") or {}
                if ind:
                    print(f"  anchored    {ind.get('source_domain')}  "
                          f"similarity {ind.get('similarity')}  "
                          f"({ind.get('status')})")
                print(f"  threshold   {m.get('threshold')}")
            except Exception:
                pass

        print(f"  algorithm   {hashing.ALGORITHM}")
        print(f"  canonical   {hashing.CANONICALIZATION}")
        print()
        for name in result.checked:
            print(f"  ✓ artifact  {name}")
        for problem in result.problems:
            print(f"  ✗ {problem}")
        print()
        print(f"  expected    {result.expected}")
        print(f"  computed    {result.computed}")
        print()

        if args.show_manifest and manifest_path.exists():
            print(BAR)
            print(manifest_path.read_bytes().decode("utf-8"))
            print(BAR)
            print()

    if result.verified:
        print("  ╔" + "═" * 44 + "╗")
        print("  ║" + "✓ EVIDENCE INTEGRITY VERIFIED".center(44) + "║")
        print("  ╚" + "═" * 44 + "╝")
        return 0

    print("  ╔" + "═" * 44 + "╗")
    print("  ║" + "✗ EVIDENCE INTEGRITY FAILED".center(44) + "║")
    print("  ╚" + "═" * 44 + "╝")
    for problem in result.problems:
        print(f"    {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
