"""Deploy IdentityAnchor to Polygon Amoy.

    python scripts/deploy.py

Writes build/deployment.json with the real address, transaction hash and block
number taken from the mined receipt. Nothing is fabricated: if the receipt does
not arrive, the transaction hash is reported as submitted-but-unconfirmed and
the script exits non-zero.

The private key is read from the environment and never printed. Only the
derived public address is shown.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blockchain.client import (  # noqa: E402
    AnchorClient,
    ChainError,
    InsufficientFundsError,
    ReceiptTimeout,
    RpcConnectionError,
    WalletConfigError,
    WrongChainError,
)
from src.blockchain.contract import ARTIFACT, BUILD_DIR, compile_contract  # noqa: E402
from src.config import CONFIG  # noqa: E402

DEPLOYMENT = BUILD_DIR / "deployment.json"


def main() -> int:
    print("=" * 70)
    print("  DEPLOY IdentityAnchor -> Polygon Amoy")
    print("=" * 70)

    artifact = compile_contract()
    print(f"[BUILD] solc {artifact['solc_version']}  "
          f"optimizer runs={artifact['optimizer']['runs']}  evm={artifact['evm_version']}")
    print(f"[BUILD] source   contracts/IdentityAnchor.sol")
    print(f"[BUILD] sha256   {artifact['source_sha256']}")
    print(f"[BUILD] deployed bytecode {len(artifact['deployed_bytecode']) // 2 - 1} bytes")
    print(f"[BUILD] artifact {ARTIFACT.relative_to(CONFIG.project_root)}")

    try:
        client = AnchorClient(logger=print)
    except WalletConfigError as exc:
        print(f"[CHAIN] ✗ wallet configuration error: {exc}")
        return 2

    try:
        chain_id = client.connect()
        print(f"[CHAIN] wallet verified {client.address}")
        balance = client.require_funds()
        print(f"[CHAIN] balance verified {balance / 1e18:.6f} POL")
    except WrongChainError as exc:
        print(f"[CHAIN] ✗ wrong chain: {exc}")
        return 3
    except InsufficientFundsError as exc:
        print(f"[CHAIN] ✗ insufficient funds: {exc}")
        return 4
    except RpcConnectionError as exc:
        print(f"[CHAIN] ✗ RPC failure: {exc}")
        return 5

    try:
        address, receipt = client.deploy()
    except ReceiptTimeout as exc:
        print(f"[CHAIN] ✓ transaction submitted  TX: {exc.tx_hash}")
        print(f"[CHAIN] ⚠ confirmation delayed: {exc}")
        print("[CHAIN]   not treating this as a successful deployment")
        return 6
    except InsufficientFundsError as exc:
        print(f"[CHAIN] ✗ insufficient funds: {exc}")
        return 4
    except ChainError as exc:
        print(f"[CHAIN] ✗ deployment failed: {exc}")
        return 7

    record = {
        "contract": "IdentityAnchor",
        "network": CONFIG.chain.network_name,
        "chain_id": chain_id,
        "address": address,
        "deployer": client.address,
        "deployment_tx": receipt.tx_hash,
        "block_number": receipt.block_number,
        "block_hash": receipt.block_hash,
        "gas_used": receipt.gas_used,
        "effective_gas_price_wei": receipt.effective_gas_price,
        "fee_wei": receipt.fee_wei,
        "fee_pol": f"{receipt.fee_wei / 1e18:.9f}",
        "solc_version": artifact["solc_version"],
        "source_sha256": artifact["source_sha256"],
        "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "explorer": f"{CONFIG.chain.explorer}/address/{address}",
    }
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print()
    print(f"[CHAIN] address        {address}")
    print(f"[CHAIN] deployment tx  {receipt.tx_hash}")
    print(f"[CHAIN] block          {receipt.block_number}")
    print(f"[CHAIN] chain id       {chain_id}")
    print(f"[CHAIN] gas used       {receipt.gas_used}")
    print(f"[CHAIN] fee            {receipt.fee_wei / 1e18:.9f} POL")
    print(f"[CHAIN] explorer       {record['explorer']}")
    print()
    print(f"  wrote {DEPLOYMENT.relative_to(CONFIG.project_root)}")
    print(f"  set CONTRACT_ADDRESS={address} in .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
