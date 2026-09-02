"""Phase 4: contract behaviour and client safety.

Contract tests run against a real in-process EVM (eth-tester + py-evm), so
deployment, anchoring, retrieval, reverts and events are genuinely executed -
no mocking of contract logic. Client tests cover the failure paths that must
never be exercised against a live network: wrong chain, bad key, no funds,
dead RPC, receipt timeout.

Nothing here touches Polygon Amoy.
"""
from __future__ import annotations

import ast

import pytest
from eth_tester.exceptions import TransactionFailed
from web3 import EthereumTesterProvider, Web3
from web3.exceptions import ContractCustomError, Web3RPCError

# eth-tester surfaces a custom-error revert as TransactionFailed carrying the
# raw ABI-encoded error; web3 raises ContractCustomError over a real node.
REVERTS = (ContractCustomError, Web3RPCError, TransactionFailed)


def selector(signature: str) -> bytes:
    """4-byte selector of a Solidity custom error, e.g. "NotAnchored(bytes32)"."""
    return Web3.keccak(text=signature)[:4]


def revert_data(exc: BaseException) -> bytes:
    """Pull the ABI-encoded revert payload out of whichever wrapper we got.

    web3 gives HexBytes; eth-tester formats the raw bytes into its message
    string, so the literal has to be recovered from that repr.
    """
    arg = exc.args[0] if exc.args else b""
    while isinstance(arg, BaseException):
        arg = arg.args[0] if arg.args else b""
    if isinstance(arg, (bytes, bytearray)):
        return bytes(arg)
    if isinstance(arg, str):
        text = arg.split("execution reverted:", 1)[-1].strip()
        if text.startswith("0x"):
            return bytes.fromhex(text[2:])
        if text.startswith(("b'", 'b"')):
            try:
                return bytes(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return b""
    return b""


def assert_reverts_with(signature: str, fn):
    """Assert a call reverts with one specific custom error."""
    with pytest.raises(REVERTS) as exc:
        fn()
    data = revert_data(exc.value)
    expected = selector(signature)
    assert data[:4] == expected, (
        f"expected {signature} (selector {expected.hex()}), got {data[:4].hex()}"
    )

from src.blockchain import client as chain
from src.blockchain.contract import compile_contract
from src.blockchain.client import (
    AnchorClient,
    InsufficientFundsError,
    RpcConnectionError,
    WalletConfigError,
    WrongChainError,
    evidence_bytes32,
    investigation_key,
)

# A throwaway key used only by the in-process EVM in these tests.
TEST_KEY = "0x" + "11" * 32
HASH_A = "92c214ad728df9af1887fb185075b73ab2171ad814bfcef03afae4ea8909c5f6"
HASH_B = "0000111122223333444455556666777788889999aaaabbbbccccddddeeeeffff"


@pytest.fixture(scope="module")
def artifact():
    return compile_contract()


@pytest.fixture
def w3():
    return Web3(EthereumTesterProvider())


@pytest.fixture
def deployed(w3, artifact):
    """Deploy IdentityAnchor to the in-process EVM."""
    acct = w3.eth.accounts[0]
    factory = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    tx = factory.constructor().transact({"from": acct})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt["status"] == 1
    return w3.eth.contract(address=receipt["contractAddress"], abi=artifact["abi"]), acct, receipt


# --- compilation ------------------------------------------------------

def test_contract_compiles(artifact):
    assert artifact["solc_version"] == "0.8.26"
    assert artifact["bytecode"].startswith("0x") and len(artifact["bytecode"]) > 100
    names = {f["name"] for f in artifact["abi"] if f["type"] == "function"}
    assert {"anchorEvidence", "getEvidence", "verifyEvidence", "isAnchored"} <= names


def test_contract_stays_small(artifact):
    """A large contract means evidence data leaked on-chain."""
    deployed_size = len(artifact["deployed_bytecode"]) // 2 - 1
    assert deployed_size < 4096, f"deployed bytecode is {deployed_size} bytes"


def test_abi_declares_the_event(artifact):
    events = [e for e in artifact["abi"] if e["type"] == "event"]
    assert len(events) == 1
    assert events[0]["name"] == "EvidenceAnchored"


# --- deployment -------------------------------------------------------

def test_deployment_succeeds(deployed):
    contract, _, receipt = deployed
    assert receipt["contractAddress"]
    assert receipt["status"] == 1
    assert contract.functions.totalAnchored().call() == 0


def test_deployed_code_is_on_chain(w3, deployed):
    contract, _, _ = deployed
    assert len(w3.eth.get_code(contract.address)) > 0


# --- anchoring --------------------------------------------------------

def test_anchor_stores_and_retrieves_the_hash(w3, deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-20260902-F53AF4")
    digest = evidence_bytes32(HASH_A)

    tx = contract.functions.anchorEvidence(key, digest).transact({"from": acct})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt["status"] == 1

    stored_hash, submitter, timestamp, block_number = contract.functions.getEvidence(key).call()
    assert stored_hash.hex() == HASH_A
    assert submitter == acct
    assert timestamp > 0
    assert block_number > 0
    assert contract.functions.totalAnchored().call() == 1


def test_correct_hash_verifies(deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    assert contract.functions.verifyEvidence(key, evidence_bytes32(HASH_A)).call() is True


def test_incorrect_hash_is_rejected(deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    assert contract.functions.verifyEvidence(key, evidence_bytes32(HASH_B)).call() is False


def test_single_bit_difference_is_rejected(deployed):
    """A one-character change in the fingerprint must not verify."""
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    tampered = HASH_A[:-1] + ("0" if HASH_A[-1] != "0" else "1")
    assert contract.functions.verifyEvidence(key, evidence_bytes32(tampered)).call() is False


def test_zero_hash_never_verifies(deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    assert contract.functions.verifyEvidence(key, b"\x00" * 32).call() is False


def test_unknown_investigation_is_not_anchored(deployed):
    contract, _, _ = deployed
    key = investigation_key("TRACE-NEVER-SEEN")
    assert contract.functions.isAnchored(key).call() is False
    assert contract.functions.verifyEvidence(key, evidence_bytes32(HASH_A)).call() is False


def test_reading_an_unknown_investigation_reverts(deployed):
    contract, _, _ = deployed
    assert_reverts_with(
        "NotAnchored(bytes32)",
        lambda: contract.functions.getEvidence(investigation_key("TRACE-NEVER-SEEN")).call(),
    )


def test_hash_resolves_back_to_its_investigation(deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    assert contract.functions.investigationForHash(evidence_bytes32(HASH_A)).call() == key
    assert contract.functions.investigationForHash(evidence_bytes32(HASH_B)).call() == b"\x00" * 32


# --- duplicates -------------------------------------------------------

def test_duplicate_investigation_is_rejected(deployed):
    """Documented policy: one investigation anchors exactly once."""
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    assert_reverts_with(
        "InvestigationAlreadyAnchored(bytes32)",
        lambda: contract.functions.anchorEvidence(
            key, evidence_bytes32(HASH_B)).call({"from": acct}),
    )


def test_duplicate_evidence_hash_is_rejected(deployed):
    """A fingerprint can never map to two conflicting records."""
    contract, acct, _ = deployed
    contract.functions.anchorEvidence(
        investigation_key("TRACE-A"), evidence_bytes32(HASH_A)
    ).transact({"from": acct})
    assert_reverts_with(
        "EvidenceHashAlreadyAnchored(bytes32,bytes32)",
        lambda: contract.functions.anchorEvidence(
            investigation_key("TRACE-B"), evidence_bytes32(HASH_A)).call({"from": acct}),
    )


def test_zero_values_are_rejected(deployed):
    contract, acct, _ = deployed
    assert_reverts_with(
        "ZeroValue()",
        lambda: contract.functions.anchorEvidence(
            b"\x00" * 32, evidence_bytes32(HASH_A)).call({"from": acct}),
    )
    assert_reverts_with(
        "ZeroValue()",
        lambda: contract.functions.anchorEvidence(
            investigation_key("TRACE-A"), b"\x00" * 32).call({"from": acct}),
    )


def test_distinct_investigations_can_both_anchor(deployed):
    contract, acct, _ = deployed
    contract.functions.anchorEvidence(
        investigation_key("TRACE-A"), evidence_bytes32(HASH_A)).transact({"from": acct})
    contract.functions.anchorEvidence(
        investigation_key("TRACE-B"), evidence_bytes32(HASH_B)).transact({"from": acct})
    assert contract.functions.totalAnchored().call() == 2


# --- events -----------------------------------------------------------

def test_event_is_emitted_with_the_right_values(w3, deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-20260902-F53AF4")
    digest = evidence_bytes32(HASH_A)

    tx = contract.functions.anchorEvidence(key, digest).transact({"from": acct})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    events = contract.events.EvidenceAnchored().process_receipt(receipt)

    assert len(events) == 1
    args = events[0]["args"]
    assert args["investigationId"] == key
    assert args["evidenceHash"] == digest
    assert args["submitter"] == acct
    assert args["timestamp"] > 0
    assert args["blockNumber"] > 0


def test_no_event_when_the_call_reverts(w3, deployed):
    contract, acct, _ = deployed
    key = investigation_key("TRACE-A")
    contract.functions.anchorEvidence(key, evidence_bytes32(HASH_A)).transact({"from": acct})
    before = contract.functions.totalAnchored().call()
    assert_reverts_with(
        "InvestigationAlreadyAnchored(bytes32)",
        lambda: contract.functions.anchorEvidence(
            key, evidence_bytes32(HASH_B)).call({"from": acct}),
    )
    assert contract.functions.totalAnchored().call() == before


# --- hash helpers -----------------------------------------------------

def test_evidence_bytes32_round_trips():
    assert evidence_bytes32(HASH_A).hex() == HASH_A
    assert evidence_bytes32("0x" + HASH_A).hex() == HASH_A
    assert evidence_bytes32(HASH_A.upper()).hex() == HASH_A


def test_evidence_bytes32_rejects_bad_input():
    for bad in ("", "abc", HASH_A[:-1], HASH_A + "0", "z" * 64):
        with pytest.raises(ValueError):
            evidence_bytes32(bad)


def test_investigation_key_is_deterministic():
    assert investigation_key("TRACE-X") == investigation_key("TRACE-X")
    assert investigation_key("TRACE-X") != investigation_key("TRACE-Y")
    assert len(investigation_key("TRACE-X")) == 32


# --- client safety ----------------------------------------------------

def test_wrong_chain_is_refused(w3):
    """eth-tester reports a chain id that is not 80002."""
    c = AnchorClient(private_key=TEST_KEY, web3=w3)
    with pytest.raises(WrongChainError) as exc:
        c.connect()
    assert str(CONFIG_EXPECTED) in str(exc.value)


from src.config import CONFIG  # noqa: E402

CONFIG_EXPECTED = CONFIG.chain.expected_chain_id


def test_missing_private_key_is_reported():
    with pytest.raises(WalletConfigError, match="not set"):
        AnchorClient(private_key="", rpc_url="http://localhost:1")


def test_malformed_private_key_is_reported():
    with pytest.raises(WalletConfigError, match="64 hex"):
        AnchorClient(private_key="not-a-key", rpc_url="http://localhost:1")


def test_api_key_shaped_value_is_rejected_as_a_private_key():
    """The exact mistake of pasting an Alchemy key into PRIVATE_KEY."""
    with pytest.raises(WalletConfigError):
        AnchorClient(private_key="alcht_" + "a" * 30, rpc_url="http://localhost:1")


def test_address_shaped_value_is_rejected_as_a_private_key():
    with pytest.raises(WalletConfigError, match="64 hex"):
        AnchorClient(private_key="0x" + "a" * 40, rpc_url="http://localhost:1")


def test_non_hex_key_of_right_length_is_rejected():
    with pytest.raises(WalletConfigError, match="hexadecimal"):
        AnchorClient(private_key="z" * 64, rpc_url="http://localhost:1")


def test_address_is_derived_not_configured(w3):
    c = AnchorClient(private_key=TEST_KEY, web3=w3)
    assert Web3.is_checksum_address(c.address)
    assert c.address == "0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A"


def test_dead_rpc_is_reported():
    c = AnchorClient(private_key=TEST_KEY, rpc_url="http://127.0.0.1:9/")
    with pytest.raises(RpcConnectionError):
        c.connect()


def test_insufficient_balance_is_reported(w3):
    c = AnchorClient(private_key=TEST_KEY, web3=w3)
    # This key controls no funds on the tester chain.
    assert c.balance_wei() == 0
    with pytest.raises(InsufficientFundsError, match="below the required"):
        c.require_funds()


def test_sufficient_balance_passes(w3):
    c = AnchorClient(private_key=TEST_KEY, web3=w3)
    funder = w3.eth.accounts[0]
    w3.eth.send_transaction({"from": funder, "to": c.address, "value": w3.to_wei(1, "ether")})
    assert c.require_funds() >= c.cfg.min_balance_wei


def test_receipt_timeout_carries_the_tx_hash(w3, monkeypatch):
    """A timeout must surface the hash so the caller can report it as
    submitted-but-unconfirmed rather than failed."""
    from web3.exceptions import TimeExhausted

    c = AnchorClient(private_key=TEST_KEY, web3=w3)

    def boom(*_a, **_k):
        raise TimeExhausted("no receipt")

    monkeypatch.setattr(c.w3.eth, "wait_for_transaction_receipt", boom)
    with pytest.raises(chain.ReceiptTimeout) as exc:
        c._await_receipt("0x" + "ab" * 32, timeout=0.1)
    assert exc.value.tx_hash == "0x" + "ab" * 32
    assert "may still confirm" in str(exc.value)


# --- secret hygiene ---------------------------------------------------

def test_client_never_exposes_the_private_key(w3):
    c = AnchorClient(private_key=TEST_KEY, web3=w3)
    bare = TEST_KEY.removeprefix("0x")
    for blob in (repr(c), str(c.__dict__), repr(c.account), str(c.artifact)):
        assert bare not in blob
        assert TEST_KEY not in blob


def test_wallet_errors_never_echo_the_key():
    secret = "deadbeef" * 8 + "extra"
    try:
        AnchorClient(private_key=secret, rpc_url="http://localhost:1")
    except WalletConfigError as exc:
        assert secret not in str(exc)
        assert "deadbeef" not in str(exc)
    else:
        pytest.fail("expected WalletConfigError")


def test_tx_result_dict_carries_no_key(w3):
    from src.blockchain.client import TxResult

    r = TxResult(tx_hash="0xabc", block_number=1, block_hash="0xdef", gas_used=21000,
                 effective_gas_price=1, status=1, from_address="0x1")
    assert TEST_KEY.removeprefix("0x") not in str(r.to_dict())
    assert set(r.to_dict()) & {"private_key", "key", "secret"} == set()


def test_config_repr_does_not_leak_the_key():
    from src.config import CONFIG as cfg
    assert "PRIVATE_KEY" not in repr(cfg)
    if cfg.private_key:
        assert cfg.private_key not in repr(cfg)
