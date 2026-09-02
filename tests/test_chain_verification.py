"""Phase 5: end-to-end verification states.

Every failure state is exercised. Chain-backed states run against a real
in-process EVM with IdentityAnchor actually deployed and the fingerprint
actually anchored, so the comparison is a genuine contract read.

Nothing here touches Polygon Amoy.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from web3 import EthereumTesterProvider, Web3

from src.blockchain.client import AnchorClient, evidence_bytes32, investigation_key
from src.blockchain.contract import compile_contract
from src.blockchain.verifier import (
    VerificationStatus,
    investigation_record_key,
    recompute_local,
    verify_against_chain,
)
from src.evidence import hashing
from src.evidence import manifest as manifest_mod
from src.evidence.collector import (
    FINGERPRINT_FILE,
    MANIFEST_FILE,
    ArtifactStore,
    new_investigation_id,
    write_fingerprint,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_KEY = "0x" + "22" * 32


# --- a real bundle + a real chain -------------------------------------

@pytest.fixture
def bundle(tmp_path) -> Path:
    store = ArtifactStore(new_investigation_id(), root=tmp_path / "evidence")
    (store.root / "input.jpg").write_bytes(b"input-image-bytes")
    (store.root / "source-image.jpg").write_bytes(b"candidate-image-bytes")
    store.write_json("candidates.json", [{"position": 1, "url": "https://a.example/x"}])
    store.write_json("retrieval.json", [{"position": 1, "status": "RETRIEVED"}])
    store.write_json("matching.json", {"threshold": "0.300000"})
    store.write_bytes("search-response.json", b'{"visual_matches":[]}')

    digests = {}
    for name in manifest_mod.ARTIFACT_FILES:
        p = store.root / name
        if p.exists():
            digests[name] = {"sha256": hashing.sha256_file(p), "bytes": p.stat().st_size}

    write_fingerprint(store, {
        "schema": manifest_mod.SCHEMA,
        "investigation_id": store.investigation_id,
        "created_at": "2026-09-02T00:00:00+00:00",
        "matching": {
            "threshold": hashing.decimal_str(0.3),
            "best_independent_match": {
                "source_url": "https://news.example/story",
                "source_domain": "news.example",
                "similarity": hashing.decimal_str(0.9944),
                "status": "MATCH",
            },
        },
        "artifacts": digests,
    })
    return store.root


@pytest.fixture
def chain(monkeypatch):
    """A local EVM with IdentityAnchor deployed, presented as chain 80002."""
    w3 = Web3(EthereumTesterProvider())
    artifact = compile_contract()
    deployer = w3.eth.accounts[0]
    factory = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    receipt = w3.eth.wait_for_transaction_receipt(factory.constructor().transact({"from": deployer}))
    address = receipt["contractAddress"]

    client = AnchorClient(private_key=TEST_KEY, web3=w3)
    # The chain-id assertion is tested separately; here we let the tester chain
    # stand in for Amoy so contract reads are genuine.
    monkeypatch.setattr(type(client.cfg), "expected_chain_id",
                        property(lambda _s: w3.eth.chain_id), raising=False)
    client.contract_address = address
    client.deployer = deployer
    client.w3_accounts_contract = w3.eth.contract(address=address, abi=artifact["abi"])
    return client


def _anchor(chain, bundle: Path) -> str:
    """Anchor the bundle's real fingerprint from a funded tester account."""
    _, manifest, local_sha, _, _ = recompute_local(bundle)
    contract = chain.w3_accounts_contract
    tx = contract.functions.anchorEvidence(
        investigation_key(manifest["investigation_id"]), evidence_bytes32(local_sha)
    ).transact({"from": chain.deployer})
    chain.w3.eth.wait_for_transaction_receipt(tx)
    return local_sha


# --- the happy path ---------------------------------------------------

def test_verified_end_to_end(bundle, chain):
    local_sha = _anchor(chain, bundle)
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.VERIFIED
    assert r.verified
    assert r.local_sha256 == local_sha
    assert r.on_chain_sha256 == local_sha
    assert r.contract_verify_evidence is True
    assert r.submitter == chain.deployer
    assert r.anchored_block > 0
    assert r.chain_timestamp > 0
    assert len(r.artifacts_checked) == 6
    assert r.problems == []


def test_hash_is_recomputed_not_read_from_fingerprint_json(bundle, chain):
    """fingerprint.json is never the source of truth."""
    _anchor(chain, bundle)
    # Corrupt the cached value; verification must be unaffected.
    fp = json.loads((bundle / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    real = fp["evidence_sha256"]
    fp["evidence_sha256"] = "0" * 64
    (bundle / FINGERPRINT_FILE).write_text(json.dumps(fp), encoding="utf-8")

    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.verified, "a lying fingerprint.json must not affect the verdict"
    assert r.local_sha256 == real
    assert r.recorded_sha256 == "0" * 64, "the false claim is still surfaced"


def test_record_key_is_keccak_of_the_investigation_id(bundle):
    _, manifest, _, _, _ = recompute_local(bundle)
    inv = manifest["investigation_id"]
    assert investigation_record_key(inv) == "0x" + Web3.keccak(text=inv).hex()


# --- local failure states ---------------------------------------------

def test_missing_bundle(tmp_path):
    r = verify_against_chain(tmp_path / "nope")
    assert r.status is VerificationStatus.BUNDLE_NOT_FOUND
    assert not r.verified


def test_missing_manifest(bundle):
    (bundle / MANIFEST_FILE).unlink()
    r = verify_against_chain(bundle)
    assert r.status is VerificationStatus.MANIFEST_MISSING


def test_malformed_manifest(bundle):
    (bundle / MANIFEST_FILE).write_bytes(b"{ not json at all")
    r = verify_against_chain(bundle)
    assert r.status is VerificationStatus.MANIFEST_MALFORMED


def test_manifest_that_is_not_an_object(bundle):
    (bundle / MANIFEST_FILE).write_bytes(b'["a","list"]')
    r = verify_against_chain(bundle)
    assert r.status is VerificationStatus.MANIFEST_MALFORMED


def test_missing_investigation_id(bundle):
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    del m["investigation_id"]
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))
    r = verify_against_chain(bundle)
    assert r.status is VerificationStatus.INVESTIGATION_ID_MISSING


def test_empty_investigation_id(bundle):
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["investigation_id"] = ""
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))
    assert verify_against_chain(bundle).status is VerificationStatus.INVESTIGATION_ID_MISSING


def test_missing_artifact(bundle, chain):
    _anchor(chain, bundle)
    (bundle / "input.jpg").unlink()
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.ARTIFACT_MISSING
    assert any("missing artifact" in p for p in r.problems)


def test_corrupted_artifact(bundle, chain):
    _anchor(chain, bundle)
    (bundle / "source-image.jpg").write_bytes(b"replaced-content-entirely")
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.ARTIFACT_MODIFIED
    assert any("source-image.jpg" in p for p in r.problems)


def test_artifact_change_keeps_the_manifest_hash(bundle, chain):
    """Detection of an artifact edit comes from the digest layer, not the
    manifest hash - the manifest itself is untouched, so its hash must not
    move. This is the design, and the test pins it."""
    local_sha = _anchor(chain, bundle)
    (bundle / "input.jpg").write_bytes(b"tampered-input-bytes")
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.ARTIFACT_MODIFIED
    assert r.local_sha256 == local_sha
    assert r.on_chain_sha256 == local_sha


# --- chain failure states ---------------------------------------------

def test_contract_not_configured(bundle, monkeypatch):
    from src.config import CONFIG
    monkeypatch.setattr(type(CONFIG), "contract_address", property(lambda _s: ""), raising=False)
    r = verify_against_chain(bundle)
    assert r.status is VerificationStatus.CONTRACT_NOT_CONFIGURED


def test_wrong_contract_address_with_no_code(bundle, chain):
    _anchor(chain, bundle)
    eoa = chain.w3.eth.accounts[3]  # an account, not a contract
    r = verify_against_chain(bundle, contract_address=eoa, client=chain)
    assert r.status is VerificationStatus.CONTRACT_INVALID
    assert any("no contract code" in p for p in r.problems)


def test_unknown_investigation_is_not_anchored(bundle, chain):
    """The contract is right, but this bundle was never anchored."""
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.NOT_ANCHORED
    assert r.on_chain_sha256 is None
    assert any("not anchored" in p for p in r.problems)


def test_on_chain_hash_mismatch(bundle, chain):
    """Anchor one fingerprint, then change the manifest so they diverge."""
    _anchor(chain, bundle)
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["matching"]["best_independent_match"]["similarity"] = "0.111111"
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))

    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.HASH_MISMATCH
    assert r.local_sha256 != r.on_chain_sha256
    assert r.contract_verify_evidence is False
    assert r.status.is_tamper


def test_modified_url_causes_mismatch(bundle, chain):
    _anchor(chain, bundle)
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["matching"]["best_independent_match"]["source_url"] = "https://attacker.example/x"
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.HASH_MISMATCH


def test_resigned_manifest_still_fails_against_the_chain(bundle, chain):
    """Change an artifact AND fix the manifest digest: locally consistent,
    but the anchored fingerprint no longer matches."""
    _anchor(chain, bundle)
    img = bundle / "source-image.jpg"
    img.write_bytes(b"substituted-image-content")
    m = json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))
    m["artifacts"]["source-image.jpg"] = {
        "sha256": hashing.sha256_file(img), "bytes": img.stat().st_size}
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(m))

    local_status, _, _, _, problems = recompute_local(bundle)
    assert local_status is VerificationStatus.VERIFIED, "locally self-consistent"

    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    assert r.status is VerificationStatus.HASH_MISMATCH, "the chain still catches it"


def test_wrong_chain_id(bundle, monkeypatch):
    """A real client against a chain that is not 80002 must refuse."""
    w3 = Web3(EthereumTesterProvider())
    client = AnchorClient(private_key=TEST_KEY, web3=w3)
    r = verify_against_chain(bundle, contract_address="0x" + "11" * 20, client=client)
    assert r.status is VerificationStatus.WRONG_CHAIN
    assert any("expected 80002" in p for p in r.problems)


def test_rpc_failure(bundle):
    client = AnchorClient(private_key=TEST_KEY, rpc_url="http://127.0.0.1:9/")
    r = verify_against_chain(bundle, contract_address="0x" + "11" * 20, client=client)
    assert r.status is VerificationStatus.RPC_FAILURE
    assert not r.verified


# --- read-only ---------------------------------------------------------

def test_verification_needs_no_private_key(bundle, chain, monkeypatch):
    """A third party with only the bundle must be able to verify."""
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    from src.blockchain.client import read_only_client

    reader = read_only_client(web3=chain.w3)
    monkeypatch.setattr(type(reader.cfg), "expected_chain_id",
                        property(lambda _s: chain.w3.eth.chain_id), raising=False)
    assert reader.address is None

    _anchor(chain, bundle)
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=reader)
    assert r.verified


def test_read_only_client_refuses_to_sign(chain):
    from src.blockchain.client import WalletConfigError, read_only_client

    reader = read_only_client(web3=chain.w3)
    with pytest.raises(WalletConfigError, match="read-only"):
        reader.balance_wei()


# --- serialization and CLI --------------------------------------------

def test_result_serializes_to_json(bundle, chain):
    _anchor(chain, bundle)
    r = verify_against_chain(bundle, contract_address=chain.contract_address, client=chain)
    payload = json.loads(json.dumps(r.to_dict()))
    assert payload["status"] == "VERIFIED"
    assert payload["verified"] is True
    assert payload["local_sha256"] == payload["on_chain_sha256"]
    assert "private_key" not in payload


def test_status_flags():
    assert VerificationStatus.VERIFIED.ok
    assert not VerificationStatus.HASH_MISMATCH.ok
    assert VerificationStatus.HASH_MISMATCH.is_tamper
    assert VerificationStatus.ARTIFACT_MODIFIED.is_tamper
    assert VerificationStatus.ARTIFACT_MISSING.is_tamper
    assert not VerificationStatus.RPC_FAILURE.is_tamper
    assert not VerificationStatus.NOT_ANCHORED.is_tamper


def test_cli_reports_failure_for_a_missing_bundle(tmp_path):
    proc = subprocess.run(
        [sys.executable, "verify_chain.py", str(tmp_path / "nope"), "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["status"] == "BUNDLE_NOT_FOUND"


def test_cli_json_output_is_machine_readable(bundle):
    """No chain access needed: this bundle has no anchor, so it fails early
    with a structured result."""
    proc = subprocess.run(
        [sys.executable, "verify_chain.py", str(bundle), "--json",
         "--contract", "0x" + "00" * 20],
        cwd=ROOT, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["verified"] is False
    assert payload["local_sha256"]
    assert payload["investigation_id"]


# --- the original bundle is never harmed ------------------------------

def test_tamper_demo_leaves_the_original_intact(bundle, chain):
    _anchor(chain, bundle)
    before = {p.name: hashing.sha256_file(p) for p in sorted(bundle.iterdir()) if p.is_file()}

    copy = bundle.parent / "copy-for-tampering"
    shutil.copytree(bundle, copy)
    (copy / "input.jpg").write_bytes(b"tampered")

    after = {p.name: hashing.sha256_file(p) for p in sorted(bundle.iterdir()) if p.is_file()}
    assert before == after
    assert verify_against_chain(
        bundle, contract_address=chain.contract_address, client=chain).verified
    assert not verify_against_chain(
        copy, contract_address=chain.contract_address, client=chain).verified
