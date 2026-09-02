"""Compile IdentityAnchor.sol and expose its ABI and bytecode.

Compilation output is cached in build/ so deployment and tests use the same
artifact, and so the compiler version that produced it is recorded alongside.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import CONFIG

SOLC_VERSION = "0.8.26"
CONTRACT_NAME = "IdentityAnchor"
SOURCE = CONFIG.project_root / "contracts" / f"{CONTRACT_NAME}.sol"
BUILD_DIR = CONFIG.project_root / "build"
ARTIFACT = BUILD_DIR / f"{CONTRACT_NAME}.json"

# Matches the settings recorded in the build artifact.
OPTIMIZER_RUNS = 200


def _ensure_solc() -> None:
    import solcx

    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        solcx.install_solc(SOLC_VERSION)


def compile_contract(force: bool = False) -> dict[str, Any]:
    """Compile the contract, returning {abi, bytecode, solc_version, ...}."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"contract source not found: {SOURCE}")

    source_text = SOURCE.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    if ARTIFACT.exists() and not force:
        cached = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        if cached.get("source_sha256") == source_sha256:
            return cached

    import solcx

    _ensure_solc()
    compiled = solcx.compile_standard(
        {
            "language": "Solidity",
            "sources": {f"{CONTRACT_NAME}.sol": {"content": source_text}},
            "settings": {
                "optimizer": {"enabled": True, "runs": OPTIMIZER_RUNS},
                "evmVersion": "paris",  # widely supported on Polygon Amoy
                "outputSelection": {
                    "*": {"*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]}
                },
            },
        },
        solc_version=SOLC_VERSION,
    )

    contract = compiled["contracts"][f"{CONTRACT_NAME}.sol"][CONTRACT_NAME]
    artifact = {
        "contract": CONTRACT_NAME,
        "solc_version": SOLC_VERSION,
        "optimizer": {"enabled": True, "runs": OPTIMIZER_RUNS},
        "evm_version": "paris",
        "source_file": f"contracts/{CONTRACT_NAME}.sol",
        "source_sha256": source_sha256,
        "abi": contract["abi"],
        "bytecode": "0x" + contract["evm"]["bytecode"]["object"],
        "deployed_bytecode": "0x" + contract["evm"]["deployedBytecode"]["object"],
    }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def load_abi() -> list[dict[str, Any]]:
    return compile_contract()["abi"]
