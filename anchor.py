"""Anchor an evidence bundle's fingerprint on Polygon Amoy, then verify it.

    python anchor.py evidence/TRACE-20260902-F53AF4
    python anchor.py evidence/TRACE-20260902-F53AF4 --verify-only

Flow:

    LOCAL EVIDENCE HASH -> POLYGON AMOY TRANSACTION -> ON-CHAIN HASH
                        -> LOCAL HASH COMPARISON -> VERIFIED

The hash is always recomputed from the bundle on disk, never read from a
cached field, so anchoring and verification both reflect what the files
actually contain. The private key is never printed.
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

from src.blockchain.client import (
    AnchorClient,
    ChainError,
    InsufficientFundsError,
    ReceiptTimeout,
    RpcConnectionError,
    WalletConfigError,
    WrongChainError,
    evidence_bytes32,
    investigation_key,
)
from src.blockchain.contract import BUILD_DIR
from src.config import CONFIG
from src.evidence import hashing
from src.evidence.collector import MANIFEST_FILE, verify_bundle

BAR = "=" * 70


def recompute(bundle: Path) -> tuple[str, str]:
    """Recompute the fingerprint from the bundle. Returns (id, sha256)."""
    manifest_path = bundle / MANIFEST_FILE
    if not manifest_path.exists():
        raise SystemExit(f"no {MANIFEST_FILE} in {bundle}")
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    digest = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    return manifest["investigation_id"], digest


def main() -> int:
    ap = argparse.ArgumentParser(description="Anchor evidence on Polygon Amoy")
    ap.add_argument("bundle", help="path to evidence/TRACE-...")
    ap.add_argument("--contract", default=CONFIG.contract_address,
                    help="IdentityAnchor address (defaults to CONTRACT_ADDRESS)")
    ap.add_argument("--verify-only", action="store_true",
                    help="read and compare without sending a transaction")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    print(BAR)
    print("  EVIDENCE ANCHORING -> Polygon Amoy")
    print(BAR)

    # -- local integrity first --------------------------------------
    local = verify_bundle(bundle)
    if not local.verified:
        print("  ✗ the local bundle does not verify; refusing to anchor it")
        for problem in local.problems:
            print(f"    {problem}")
        return 1
    investigation_id, local_hash = recompute(bundle)
    print(f"  bundle          {bundle}")
    print(f"  investigation   {investigation_id}")
    print(f"  local hash      {local_hash}")
    print(f"  artifacts       {len(local.checked)} verified")
    print()

    if not args.contract:
        print("  ✗ no contract address; run scripts/deploy.py or set CONTRACT_ADDRESS")
        return 2

    # -- chain ------------------------------------------------------
    try:
        client = AnchorClient(logger=print)
    except WalletConfigError as exc:
        print(f"[CHAIN] ✗ wallet configuration error: {exc}")
        return 2
    try:
        chain_id = client.connect()
        print(f"[CHAIN] wallet verified {client.address}")
        balance = client.balance_wei()
        print(f"[CHAIN] balance verified {balance / 1e18:.6f} POL")
    except WrongChainError as exc:
        print(f"[CHAIN] ✗ wrong chain: {exc}")
        return 3
    except RpcConnectionError as exc:
        print(f"[CHAIN] ✗ RPC failure: {exc}")
        return 5

    print(f"[CHAIN] contract        {args.contract}")
    key = investigation_key(investigation_id)
    already = client.is_anchored(args.contract, investigation_id)

    anchor_receipt = None
    if args.verify_only:
        print("[CHAIN] verify-only: no transaction will be sent")
        if not already:
            print("[CHAIN] ✗ this investigation is not anchored on chain")
            return 8
    elif already:
        print("[CHAIN] already anchored; contract rejects duplicates, skipping the write")
    else:
        try:
            client.require_funds()
            anchor_receipt = client.anchor(args.contract, investigation_id, local_hash)
        except InsufficientFundsError as exc:
            print(f"[CHAIN] ✗ insufficient funds: {exc}")
            return 4
        except ReceiptTimeout as exc:
            print(f"[CHAIN] ✓ transaction submitted  TX: {exc.tx_hash}")
            print(f"[CHAIN] ⚠ confirmation delayed: {exc}")
            print("[CHAIN]   NOT reporting this as confirmed")
            return 6
        except ChainError as exc:
            print(f"[CHAIN] ✗ anchoring failed: {exc}")
            return 7

    # -- read back --------------------------------------------------
    print("[CHAIN] reading on-chain fingerprint")
    try:
        record = client.read_evidence(args.contract, investigation_id)
    except Exception as exc:
        print(f"[CHAIN] ✗ read failed: {type(exc).__name__}: {exc}")
        return 9

    on_chain_hash = record["evidence_sha256"]
    contract_says = client.verify_on_chain(args.contract, investigation_id, local_hash)

    print()
    print(f"[CHAIN] local fingerprint:    {local_hash}")
    print(f"[CHAIN] on-chain fingerprint: {on_chain_hash}")
    print(f"[CHAIN] contract verifyEvidence(): {contract_says}")
    print(f"[CHAIN] submitter             {record['submitter']}")
    print(f"[CHAIN] anchored in block     {record['block_number']}")
    print(f"[CHAIN] chain timestamp       {record['timestamp']} "
          f"({datetime.fromtimestamp(record['timestamp'], timezone.utc).isoformat()})")

    matched = (on_chain_hash == local_hash) and contract_says
    receipt_record = {
        "investigation_id": investigation_id,
        "investigation_key": "0x" + key.hex(),
        "network": CONFIG.chain.network_name,
        "chain_id": chain_id,
        "contract_address": args.contract,
        "local_sha256": local_hash,
        "on_chain_sha256": on_chain_hash,
        "contract_verify_evidence": contract_says,
        "verified": matched,
        "submitter": record["submitter"],
        "anchored_block": record["block_number"],
        "chain_timestamp": record["timestamp"],
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if anchor_receipt is not None:
        receipt_record["anchor_tx"] = anchor_receipt.to_dict()
        receipt_record["explorer"] = f"{CONFIG.chain.explorer}/tx/{anchor_receipt.tx_hash}"

    # Written into the bundle, but NOT covered by the Phase 3 fingerprint -
    # the anchor references the evidence, never the other way round.
    out = bundle / "anchor.json"
    out.write_text(json.dumps(receipt_record, indent=2), encoding="utf-8")

    print()
    if matched:
        print("  ╔" + "═" * 46 + "╗")
        print("  ║" + "✓ BLOCKCHAIN VERIFICATION".center(46) + "║")
        print("  ╚" + "═" * 46 + "╝")
    else:
        print("  ╔" + "═" * 46 + "╗")
        print("  ║" + "✗ HASH MISMATCH".center(46) + "║")
        print("  ╚" + "═" * 46 + "╝")

    if anchor_receipt is not None:
        print()
        print(f"  anchoring tx    {anchor_receipt.tx_hash}")
        print(f"  block           {anchor_receipt.block_number}")
        print(f"  gas used        {anchor_receipt.gas_used}")
        print(f"  fee             {anchor_receipt.fee_wei / 1e18:.9f} POL")
        print(f"  explorer        {CONFIG.chain.explorer}/tx/{anchor_receipt.tx_hash}")
    print(f"\n  wrote {out}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
