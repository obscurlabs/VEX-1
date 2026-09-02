"""Demonstrate that tampering is detectable against the live on-chain anchor.

Copies the anchored bundle to a temporary directory, alters it, recomputes the
fingerprint from the altered files, and compares that against the unchanged
fingerprint stored on Polygon Amoy.

The real bundle is never modified - every mutation happens on a copy inside a
TemporaryDirectory, and the original is re-verified against the chain at the
end to prove it.

    python scripts/tamper_chain_demo.py evidence/TRACE-20260902-F53AF4
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blockchain.verifier import verify_against_chain  # noqa: E402
from src.evidence import hashing  # noqa: E402
from src.evidence.collector import MANIFEST_FILE  # noqa: E402

BAR = "=" * 74


def show(label: str, bundle: Path, contract: str | None = None) -> "object":
    r = verify_against_chain(bundle, contract_address=contract)
    verdict = "VERIFIED" if r.verified else r.status.value
    print(f"    {label:<36} {verdict}")
    if r.local_sha256:
        print(f"      local    {r.local_sha256}")
    if r.on_chain_sha256:
        print(f"      on-chain {r.on_chain_sha256}")
    if r.local_sha256 and r.on_chain_sha256:
        print(f"      equal    {r.local_sha256 == r.on_chain_sha256}")
    for p in r.problems:
        print(f"      - {p}")
    return r


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/tamper_chain_demo.py <evidence/TRACE-...>")
        return 2
    original = Path(sys.argv[1])
    contract = sys.argv[2] if len(sys.argv) > 2 else None

    if not (original / MANIFEST_FILE).exists():
        print(f"no {MANIFEST_FILE} in {original}")
        return 2

    print(BAR)
    print("  TAMPER DEMONSTRATION vs LIVE ON-CHAIN ANCHOR")
    print(BAR)
    print(f"  bundle   {original}")
    print(f"  note     the original is never modified; all edits happen on copies")
    print()

    print("  BASELINE")
    baseline = show("original bundle", original, contract)
    if not baseline.verified:
        print("\n  the original bundle does not verify; aborting the demonstration")
        return 1
    anchored_hash = baseline.on_chain_sha256
    print()

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        # Two independent detection layers, and a case is expected to trip a
        # specific one:
        #   "manifest" - the manifest itself changed, so its SHA-256 changed
        #                and no longer matches the anchored value
        #   "artifact" - a covered file changed while the manifest did not.
        #                The manifest hash is UNCHANGED by design; detection
        #                comes from the per-artifact digests the manifest
        #                carries. Expecting the hash to move here would be
        #                wrong.
        cases: list[tuple[str, str, callable]] = []

        def case(name, layer):
            def deco(fn):
                cases.append((name, layer, fn))
                return fn
            return deco

        @case("one digit changed in the manifest", "manifest")
        def _(dest: Path) -> str:
            manifest = json.loads((dest / MANIFEST_FILE).read_bytes().decode("utf-8"))
            node = manifest["matching"]["best_independent_match"]
            before = node["similarity"]
            node["similarity"] = before[:-1] + str((int(before[-1]) + 1) % 10)
            (dest / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(manifest))
            return f"similarity {before} -> {node['similarity']}"

        @case("source URL changed in the manifest", "manifest")
        def _(dest: Path) -> str:
            manifest = json.loads((dest / MANIFEST_FILE).read_bytes().decode("utf-8"))
            node = manifest["matching"]["best_independent_match"]
            before = node["source_url"]
            node["source_url"] = "https://attacker.example/substituted"
            (dest / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(manifest))
            return f"{before[:44]}... -> attacker.example"

        @case("candidate image byte flipped", "artifact")
        def _(dest: Path) -> str:
            img = dest / "source-image.jpg"
            data = bytearray(img.read_bytes())
            data[-1] ^= 0xFF
            img.write_bytes(bytes(data))
            return "last byte of source-image.jpg XOR 0xFF"

        @case("raw search response altered", "artifact")
        def _(dest: Path) -> str:
            path = dest / "search-response.json"
            data = path.read_bytes().replace(b'"position": 1', b'"position": 9', 1)
            path.write_bytes(data)
            return "one position field rewritten"

        @case("artifact deleted", "artifact")
        def _(dest: Path) -> str:
            (dest / "input.jpg").unlink()
            return "input.jpg removed"

        @case("manifest digest updated to cover a modified artifact", "manifest")
        def _(dest: Path) -> str:
            # The sophisticated attempt: change an artifact AND fix the
            # manifest so local self-consistency is restored. The chain still
            # catches it, because the manifest hash changed.
            img = dest / "source-image.jpg"
            data = bytearray(img.read_bytes())
            data[-1] ^= 0xFF
            img.write_bytes(bytes(data))
            manifest = json.loads((dest / MANIFEST_FILE).read_bytes().decode("utf-8"))
            manifest["artifacts"]["source-image.jpg"] = {
                "sha256": hashing.sha256_file(img),
                "bytes": img.stat().st_size,
            }
            (dest / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(manifest))
            return "image altered AND manifest digest re-signed"

        for i, (name, layer, mutate) in enumerate(cases, start=1):
            dest = Path(tmp) / f"case{i}"
            shutil.copytree(original, dest)
            detail = mutate(dest)
            print(f"  CASE {i} - {name}  [{layer} layer]")
            print(f"    change: {detail}")
            r = show("tampered copy", dest, contract)

            if r.verified:
                print("    *** TAMPERED BUNDLE STILL VERIFIED ***")
                failures += 1
            elif layer == "manifest":
                if r.local_sha256 == anchored_hash:
                    print("    *** MANIFEST CHANGED BUT ITS HASH DID NOT ***")
                    failures += 1
                else:
                    print("    detected: recomputed hash no longer matches the chain")
            else:
                if r.local_sha256 != anchored_hash:
                    print("    note: manifest hash also moved")
                print("    detected: artifact digest recorded in the manifest")
            print()

    print("  AFTER TAMPERING")
    final = show("original bundle (untouched)", original, contract)
    if not final.verified:
        print("    *** THE ORIGINAL BUNDLE WAS MODIFIED ***")
        failures += 1
    print()

    print(BAR)
    if failures:
        print(f"  ✗ DEMONSTRATION FAILED ({failures} problem(s))")
        print(BAR)
        return 1
    print("  ✓ TAMPER DETECTED IN EVERY CASE")
    print("  ✓ ORIGINAL BUNDLE STILL VERIFIES AGAINST THE ON-CHAIN ANCHOR")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
