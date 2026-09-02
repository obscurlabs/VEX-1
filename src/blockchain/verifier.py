"""End-to-end verification: evidence bundle on disk vs the on-chain anchor.

The data path, and the reason each step exists:

    artifact files            -> SHA-256 each, compared to manifest.artifacts
                                 (catches a modified or missing artifact)
    manifest.json             -> canonical bytes -> SHA-256 = LOCAL FINGERPRINT
                                 (recomputed, never read from fingerprint.json)
    investigation_id          -> keccak256 -> the on-chain record key
    IdentityAnchor.getEvidence -> ON-CHAIN FINGERPRINT
    local == on-chain          -> VERIFIED

fingerprint.json is read only as a cross-check and is explicitly NOT the
source of truth: a tamperer who edits the manifest would edit that file too.
The authority is the recomputation plus the chain.

Verification is read-only and needs no private key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..config import CONFIG
from ..evidence import hashing
from ..evidence.collector import FINGERPRINT_FILE, MANIFEST_FILE
from .client import (
    AnchorClient,
    RpcConnectionError,
    WrongChainError,
    investigation_key,
    read_only_client,
)


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    # local problems
    BUNDLE_NOT_FOUND = "BUNDLE_NOT_FOUND"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_MALFORMED = "MANIFEST_MALFORMED"
    INVESTIGATION_ID_MISSING = "INVESTIGATION_ID_MISSING"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_MODIFIED = "ARTIFACT_MODIFIED"
    # chain problems
    CONTRACT_NOT_CONFIGURED = "CONTRACT_NOT_CONFIGURED"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    WRONG_CHAIN = "WRONG_CHAIN"
    RPC_FAILURE = "RPC_FAILURE"
    NOT_ANCHORED = "NOT_ANCHORED"
    # the verdict that matters
    HASH_MISMATCH = "HASH_MISMATCH"

    @property
    def ok(self) -> bool:
        return self is VerificationStatus.VERIFIED

    @property
    def is_tamper(self) -> bool:
        """States that mean the evidence no longer matches what was anchored."""
        return self in (
            VerificationStatus.HASH_MISMATCH,
            VerificationStatus.ARTIFACT_MODIFIED,
            VerificationStatus.ARTIFACT_MISSING,
        )


@dataclass
class ChainVerification:
    status: VerificationStatus
    bundle: Path
    investigation_id: str | None = None
    local_sha256: str | None = None
    on_chain_sha256: str | None = None
    recorded_sha256: str | None = None      # what fingerprint.json claims
    contract_address: str | None = None
    chain_id: int | None = None
    submitter: str | None = None
    anchored_block: int | None = None
    chain_timestamp: int | None = None
    contract_verify_evidence: bool | None = None
    artifacts_checked: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.status.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "verified": self.verified,
            "bundle": str(self.bundle).replace("\\", "/"),
            "investigation_id": self.investigation_id,
            "local_sha256": self.local_sha256,
            "on_chain_sha256": self.on_chain_sha256,
            "recorded_sha256": self.recorded_sha256,
            "contract_address": self.contract_address,
            "chain_id": self.chain_id,
            "submitter": self.submitter,
            "anchored_block": self.anchored_block,
            "chain_timestamp": self.chain_timestamp,
            "contract_verify_evidence": self.contract_verify_evidence,
            "artifacts_checked": self.artifacts_checked,
            "problems": self.problems,
        }


def recompute_local(bundle: Path) -> tuple[VerificationStatus, dict | None, str | None, list[str], list[str]]:
    """Recompute the fingerprint from what is on disk.

    Returns (status, manifest, local_sha256, artifacts_checked, problems).
    """
    problems: list[str] = []
    checked: list[str] = []

    if not bundle.is_dir():
        return VerificationStatus.BUNDLE_NOT_FOUND, None, None, checked, [
            f"bundle directory not found: {bundle}"
        ]

    manifest_path = bundle / MANIFEST_FILE
    if not manifest_path.exists():
        return VerificationStatus.MANIFEST_MISSING, None, None, checked, [
            f"missing {MANIFEST_FILE}"
        ]

    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except Exception as exc:
        return VerificationStatus.MANIFEST_MALFORMED, None, None, checked, [
            f"{MANIFEST_FILE} is not readable JSON: {exc}"
        ]
    if not isinstance(manifest, dict):
        return VerificationStatus.MANIFEST_MALFORMED, None, None, checked, [
            f"{MANIFEST_FILE} is not a JSON object"
        ]

    investigation_id = manifest.get("investigation_id")
    if not investigation_id or not isinstance(investigation_id, str):
        return VerificationStatus.INVESTIGATION_ID_MISSING, manifest, None, checked, [
            "manifest has no investigation_id"
        ]

    # Artifact digests: catches a modified or deleted evidence file.
    artifact_status = None
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        path = bundle / name
        if not path.exists():
            problems.append(f"missing artifact: {name}")
            artifact_status = artifact_status or VerificationStatus.ARTIFACT_MISSING
            continue
        actual = hashing.sha256_file(path)
        if actual != entry.get("sha256"):
            problems.append(f"artifact modified: {name}")
            artifact_status = VerificationStatus.ARTIFACT_MODIFIED
        elif path.stat().st_size != entry.get("bytes"):
            problems.append(f"artifact size changed: {name}")
            artifact_status = VerificationStatus.ARTIFACT_MODIFIED
        else:
            checked.append(name)

    # The fingerprint, recomputed. Never taken from fingerprint.json.
    try:
        local_sha256 = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    except Exception as exc:
        problems.append(f"manifest is not canonicalizable: {exc}")
        return VerificationStatus.MANIFEST_MALFORMED, manifest, None, checked, problems

    if artifact_status is not None:
        return artifact_status, manifest, local_sha256, checked, problems

    return VerificationStatus.VERIFIED, manifest, local_sha256, checked, problems


def _recorded_fingerprint(bundle: Path) -> str | None:
    """Whatever fingerprint.json claims. Advisory only."""
    path = bundle / FINGERPRINT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("evidence_sha256")
    except Exception:
        return None


def verify_against_chain(
    bundle: str | Path,
    contract_address: str | None = None,
    client: AnchorClient | None = None,
    logger: Callable[[str], None] | None = None,
) -> ChainVerification:
    """Full local-then-chain verification. Never raises for expected failures."""
    log = logger or (lambda *_a, **_k: None)
    bundle = Path(bundle)

    status, manifest, local_sha, checked, problems = recompute_local(bundle)
    result = ChainVerification(
        status=status,
        bundle=bundle,
        local_sha256=local_sha,
        artifacts_checked=checked,
        problems=list(problems),
        recorded_sha256=_recorded_fingerprint(bundle),
    )
    if manifest is not None:
        result.investigation_id = manifest.get("investigation_id")

    if status is not VerificationStatus.VERIFIED:
        # Local evidence is already inconsistent. Still try to show the
        # on-chain value for contrast when we have enough to look it up.
        if result.investigation_id and local_sha:
            _attach_chain_facts(result, contract_address, client, log, compare=True)
        return result

    log("[VERIFY] Evidence loaded")
    log(f"[VERIFY] Local SHA-256: {local_sha}")

    _attach_chain_facts(result, contract_address, client, log, compare=True)
    return result


def _attach_chain_facts(
    result: ChainVerification,
    contract_address: str | None,
    client: AnchorClient | None,
    log: Callable[[str], None],
    compare: bool,
) -> None:
    """Read the anchor and compare. Mutates result; swallows expected errors."""
    address = contract_address or CONFIG.contract_address
    if not address:
        result.status = VerificationStatus.CONTRACT_NOT_CONFIGURED
        result.problems.append(
            "no contract address: pass --contract or set CONTRACT_ADDRESS"
        )
        return
    result.contract_address = address

    try:
        chain = client or read_only_client()
    except Exception as exc:
        result.status = VerificationStatus.RPC_FAILURE
        result.problems.append(f"client construction failed: {type(exc).__name__}: {exc}")
        return

    try:
        result.chain_id = chain.connect()
    except WrongChainError as exc:
        result.status = VerificationStatus.WRONG_CHAIN
        result.problems.append(str(exc))
        return
    except RpcConnectionError as exc:
        result.status = VerificationStatus.RPC_FAILURE
        result.problems.append(str(exc))
        return
    except Exception as exc:
        result.status = VerificationStatus.RPC_FAILURE
        result.problems.append(f"{type(exc).__name__}: {exc}")
        return

    # A wrong address is usually an EOA or an unrelated contract.
    try:
        code = chain.w3.eth.get_code(chain.w3.to_checksum_address(address))
    except Exception as exc:
        result.status = VerificationStatus.CONTRACT_INVALID
        result.problems.append(f"cannot read code at {address}: {exc}")
        return
    if len(code) == 0:
        result.status = VerificationStatus.CONTRACT_INVALID
        result.problems.append(f"no contract code at {address}")
        return

    log(f"[CHAIN] Investigation: {result.investigation_id}")
    log(f"[CHAIN] Contract: {address}")

    try:
        anchored = chain.is_anchored(address, result.investigation_id)
    except Exception as exc:
        result.status = VerificationStatus.CONTRACT_INVALID
        result.problems.append(
            f"isAnchored() failed at {address}; is this an IdentityAnchor? "
            f"({type(exc).__name__})"
        )
        return

    if not anchored:
        result.status = VerificationStatus.NOT_ANCHORED
        result.problems.append(
            f"investigation {result.investigation_id} is not anchored at {address}"
        )
        return

    try:
        record = chain.read_evidence(address, result.investigation_id)
    except Exception as exc:
        result.status = VerificationStatus.RPC_FAILURE
        result.problems.append(f"getEvidence() failed: {type(exc).__name__}: {exc}")
        return

    result.on_chain_sha256 = record["evidence_sha256"]
    result.submitter = record["submitter"]
    result.anchored_block = record["block_number"]
    result.chain_timestamp = record["timestamp"]
    log(f"[CHAIN] On-chain SHA-256: {result.on_chain_sha256}")

    if not compare or not result.local_sha256:
        return

    try:
        result.contract_verify_evidence = chain.verify_on_chain(
            address, result.investigation_id, result.local_sha256
        )
    except Exception:
        result.contract_verify_evidence = None

    matched = result.on_chain_sha256 == result.local_sha256
    if not matched:
        # Only downgrade to HASH_MISMATCH when the local side was otherwise
        # consistent; an ARTIFACT_MODIFIED verdict is more specific.
        if result.status is VerificationStatus.VERIFIED:
            result.status = VerificationStatus.HASH_MISMATCH
        result.problems.append(
            "local fingerprint does not match the anchored fingerprint"
        )
        return

    if result.contract_verify_evidence is False:
        result.status = VerificationStatus.HASH_MISMATCH
        result.problems.append("contract verifyEvidence() returned false")


def investigation_record_key(investigation_id: str) -> str:
    """The on-chain key for an investigation id, as hex."""
    return "0x" + investigation_key(investigation_id).hex()
