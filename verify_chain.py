"""Standalone end-to-end verification: evidence bundle vs Polygon Amoy.

    python verify_chain.py evidence/TRACE-20260902-F53AF4
    python verify_chain.py <bundle> --contract 0x...
    python verify_chain.py <bundle> --json

Recomputes the evidence fingerprint from the files on disk, reads the anchored
fingerprint from IdentityAnchor, and compares them.

fingerprint.json is NOT trusted as the source of truth - it is shown only as a
cross-check. Anyone who edited the manifest would have edited that file too.

Read-only: no private key is required or used.

Exit codes: 0 VERIFIED, 1 FAILED, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fail with a readable message, not an import traceback, when the entry
# point is run with an interpreter that lacks the dependencies.
from src.bootstrap import require_dependencies

require_dependencies()

from src.blockchain.verifier import (
    VerificationStatus,
    investigation_record_key,
    verify_against_chain,
)
from src.config import CONFIG

BAR = "=" * 70


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify evidence against Polygon Amoy")
    ap.add_argument("bundle", help="path to evidence/TRACE-...")
    ap.add_argument("--contract", default=None,
                    help="IdentityAnchor address (defaults to CONTRACT_ADDRESS)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    quiet = args.json
    log = (lambda _m: None) if quiet else print

    if not quiet:
        print(BAR)
        print("  END-TO-END EVIDENCE VERIFICATION")
        print(BAR)
        print(f"  bundle    {args.bundle}")
        print(f"  network   {CONFIG.chain.network_name} "
              f"(expecting chain id {CONFIG.chain.expected_chain_id})")
        print()

    result = verify_against_chain(args.bundle, contract_address=args.contract, logger=log)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.verified else 1

    if result.artifacts_checked:
        print(f"[VERIFY] {len(result.artifacts_checked)} artifacts match their recorded digests")
    if result.investigation_id:
        print(f"[VERIFY] Record key: {investigation_record_key(result.investigation_id)}")
    if result.recorded_sha256:
        agrees = result.recorded_sha256 == result.local_sha256
        print(f"[VERIFY] fingerprint.json says: {result.recorded_sha256}"
              f"  ({'agrees' if agrees else 'DISAGREES'} - not authoritative)")
    if result.chain_id is not None:
        print(f"[CHAIN] Chain id: {result.chain_id}")
    if result.submitter:
        stamp = datetime.fromtimestamp(result.chain_timestamp, timezone.utc).isoformat()
        print(f"[CHAIN] Submitter: {result.submitter}")
        print(f"[CHAIN] Anchored in block {result.anchored_block} at {stamp}")
    if result.contract_verify_evidence is not None:
        print(f"[CHAIN] contract verifyEvidence(): {result.contract_verify_evidence}")

    print()
    if result.verified:
        print("[VERIFY] ✓ HASH MATCH")
        print()
        print("  ╔" + "═" * 46 + "╗")
        print("  ║" + "✓ BLOCKCHAIN VERIFICATION".center(46) + "║")
        print("  ╚" + "═" * 46 + "╝")
        return 0

    print(f"[VERIFY] ✗ {result.status.value}")
    for problem in result.problems:
        print(f"           {problem}")
    if result.local_sha256 and result.on_chain_sha256:
        print()
        print(f"           local:    {result.local_sha256}")
        print(f"           on-chain: {result.on_chain_sha256}")
    print()
    headline = "✗ TAMPER DETECTED" if result.status.is_tamper else "✗ VERIFICATION FAILED"
    print("  ╔" + "═" * 46 + "╗")
    print("  ║" + headline.center(46) + "║")
    print("  ╚" + "═" * 46 + "╝")
    return 1


if __name__ == "__main__":
    sys.exit(main())
