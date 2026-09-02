"""Polygon Amoy anchoring client.

Anchors the SHA-256 fingerprint of a canonical evidence manifest and reads it
back. Does no face recognition, no search, no evidence processing.

Safety rules enforced here:
  * the private key is read from the environment, never logged, never returned,
    never written to an artifact - only the derived public address is exposed
  * the connected chain id is asserted before any state-changing call
  * balance is checked before broadcasting
  * receipt waiting is bounded; a delayed confirmation is reported as delayed,
    never as success
  * a broadcast transaction is never blindly re-sent
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from eth_account import Account
from web3 import Web3
from web3.exceptions import TimeExhausted, Web3RPCError
from web3.middleware import ExtraDataToPOAMiddleware

from ..config import CONFIG
from .contract import compile_contract


class ChainError(RuntimeError):
    """Base class for anchoring failures."""


class RpcConnectionError(ChainError):
    """The RPC endpoint is unreachable or not responding."""


class WrongChainError(ChainError):
    """Connected to a different chain than expected. Never proceed."""


class WalletConfigError(ChainError):
    """The configured private key is missing or malformed."""


class InsufficientFundsError(ChainError):
    """Not enough native token to pay for the transaction."""


class ReceiptTimeout(ChainError):
    """Broadcast succeeded but no receipt arrived in time.

    Carries the transaction hash: the transaction may still confirm, and the
    caller must report it as submitted-but-unconfirmed rather than failed.
    """

    def __init__(self, message: str, tx_hash: str):
        super().__init__(message)
        self.tx_hash = tx_hash


class TransactionReverted(ChainError):
    """The transaction was mined with status 0."""


@dataclass
class TxResult:
    """Facts about a mined transaction. Every field comes from a real receipt."""

    tx_hash: str
    block_number: int
    block_hash: str
    gas_used: int
    effective_gas_price: int
    status: int
    from_address: str
    to_address: str | None = None
    contract_address: str | None = None
    logs: list[Any] = field(default_factory=list)

    @property
    def fee_wei(self) -> int:
        return self.gas_used * self.effective_gas_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "gas_used": self.gas_used,
            "effective_gas_price_wei": self.effective_gas_price,
            "fee_wei": self.fee_wei,
            "fee_pol": f"{self.fee_wei / 1e18:.9f}",
            "status": self.status,
            "from": self.from_address,
            "to": self.to_address,
            "contract_address": self.contract_address,
        }


def investigation_key(investigation_id: str) -> bytes:
    """keccak256 of the investigation id string - the on-chain record key."""
    return Web3.keccak(text=investigation_id)


def evidence_bytes32(sha256_hex: str) -> bytes:
    """A SHA-256 hex digest as bytes32."""
    cleaned = sha256_hex.lower().removeprefix("0x")
    if len(cleaned) != 64:
        raise ValueError(f"expected a 64-character SHA-256 hex digest, got {len(cleaned)}")
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(f"not a hex digest: {exc}") from exc


class AnchorClient:
    """Talks to Polygon Amoy. One instance per run."""

    def __init__(
        self,
        rpc_url: str | None = None,
        private_key: str | None = None,
        logger: Callable[[str], None] | None = None,
        web3: Web3 | None = None,
        read_only: bool = False,
    ):
        # Verification is a read-only operation: anyone holding a bundle must
        # be able to check it against the chain without the submitter's key.
        self.read_only = read_only
        self.cfg = CONFIG.chain
        self.log = logger or (lambda *_a, **_k: None)
        self.rpc_url = rpc_url if rpc_url is not None else CONFIG.polygon_rpc_url

        if web3 is not None:
            self.w3 = web3
        else:
            if not self.rpc_url:
                raise RpcConnectionError("POLYGON_RPC_URL is not set")
            self.w3 = Web3(Web3.HTTPProvider(
                self.rpc_url, request_kwargs={"timeout": self.cfg.rpc_timeout}
            ))
            # Polygon is proof-of-authority: its block extraData is longer
            # than the 32 bytes web3 expects, so every get_block() would fail
            # without this.
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if read_only:
            self.account = None
        else:
            key = private_key if private_key is not None else CONFIG.private_key
            self.account = self._load_account(key)
        self.artifact = compile_contract()

    # -- wallet ---------------------------------------------------------

    @staticmethod
    def _load_account(key: str):
        """Derive the account. The key itself never leaves this function."""
        if not key:
            raise WalletConfigError(
                "PRIVATE_KEY is not set; see .env.example for the required value"
            )
        cleaned = key.strip().removeprefix("0x")
        if len(cleaned) != 64:
            raise WalletConfigError(
                f"PRIVATE_KEY must be 64 hex characters (got {len(cleaned)}). "
                "It is a wallet private key, not an API key or an address."
            )
        try:
            int(cleaned, 16)
        except ValueError:
            raise WalletConfigError("PRIVATE_KEY is not hexadecimal") from None
        try:
            return Account.from_key(cleaned)
        except Exception as exc:
            # Deliberately does not include the key or the raw exception text.
            raise WalletConfigError(f"PRIVATE_KEY is not a valid key ({type(exc).__name__})") from None

    @property
    def address(self) -> str | None:
        """The derived public address. Safe to print. None when read-only."""
        return self.account.address if self.account else None

    def _require_signer(self) -> None:
        if self.account is None:
            raise WalletConfigError("this client is read-only; no signing key is loaded")

    # -- connection -----------------------------------------------------

    def connect(self) -> int:
        """Connect and assert the chain id. Returns the actual chain id."""
        self.log("[CHAIN] connecting")
        try:
            connected = self.w3.is_connected()
        except Exception as exc:
            raise RpcConnectionError(f"RPC unreachable: {type(exc).__name__}: {exc}") from exc
        if not connected:
            raise RpcConnectionError("RPC endpoint did not respond to a liveness check")

        try:
            chain_id = self.w3.eth.chain_id
        except Exception as exc:
            raise RpcConnectionError(f"eth_chainId failed: {type(exc).__name__}: {exc}") from exc

        if chain_id != self.cfg.expected_chain_id:
            raise WrongChainError(
                f"connected to chain {chain_id}, expected {self.cfg.expected_chain_id} "
                f"({self.cfg.network_name}); refusing to continue"
            )
        self.log(f"[CHAIN] chain id {chain_id} ({self.cfg.network_name})")
        return chain_id

    def balance_wei(self) -> int:
        self._require_signer()
        try:
            return self.w3.eth.get_balance(self.address)
        except Exception as exc:
            raise RpcConnectionError(f"eth_getBalance failed: {type(exc).__name__}: {exc}") from exc

    def require_funds(self, minimum: int | None = None) -> int:
        """Fail clearly rather than broadcasting a transaction that cannot pay."""
        minimum = self.cfg.min_balance_wei if minimum is None else minimum
        balance = self.balance_wei()
        if balance < minimum:
            raise InsufficientFundsError(
                f"wallet {self.address} holds {balance / 1e18:.6f} POL, "
                f"below the required {minimum / 1e18:.6f} POL; fund it from an Amoy faucet"
            )
        return balance

    # -- fees -----------------------------------------------------------

    def _fee_fields(self) -> dict[str, int]:
        """EIP-1559 fees obtained from the network, never hardcoded."""
        latest = self.w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas")
        try:
            priority = self.w3.eth.max_priority_fee
        except Exception:
            priority = self.w3.to_wei(25, "gwei")

        if base_fee is None:
            # Pre-1559 network: fall back to a legacy gas price.
            return {"gasPrice": self.w3.eth.gas_price}

        # Headroom for a couple of base-fee doublings.
        max_fee = base_fee * 2 + priority
        return {"maxFeePerGas": int(max_fee), "maxPriorityFeePerGas": int(priority)}

    def _build_common(self) -> dict[str, Any]:
        self._require_signer()
        return {
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
            **self._fee_fields(),
        }

    # -- sending --------------------------------------------------------

    def _send(self, tx: dict[str, Any]) -> str:
        """Sign and broadcast. Returns the tx hash immediately after broadcast."""
        self._require_signer()
        signed = self.account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        try:
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        except Web3RPCError as exc:
            message = str(exc).lower()
            if "insufficient funds" in message:
                raise InsufficientFundsError(f"insufficient funds: {exc}") from exc
            raise ChainError(f"broadcast rejected: {exc}") from exc
        except Exception as exc:
            raise ChainError(f"broadcast failed: {type(exc).__name__}: {exc}") from exc
        return tx_hash.hex()

    def _await_receipt(self, tx_hash: str, timeout: float | None = None) -> TxResult:
        """Bounded wait. Raises ReceiptTimeout carrying the hash, never retries."""
        timeout = self.cfg.receipt_timeout if timeout is None else timeout
        normalized = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                normalized, timeout=timeout, poll_latency=self.cfg.poll_interval
            )
        except TimeExhausted as exc:
            raise ReceiptTimeout(
                f"no receipt within {timeout:.0f}s; the transaction may still confirm",
                normalized,
            ) from exc
        except Exception as exc:
            raise ChainError(f"receipt lookup failed: {type(exc).__name__}: {exc}") from exc

        result = TxResult(
            tx_hash=normalized,
            block_number=int(receipt["blockNumber"]),
            block_hash=receipt["blockHash"].hex() if hasattr(receipt["blockHash"], "hex")
            else str(receipt["blockHash"]),
            gas_used=int(receipt["gasUsed"]),
            effective_gas_price=int(receipt.get("effectiveGasPrice", 0)),
            status=int(receipt["status"]),
            from_address=receipt["from"],
            to_address=receipt.get("to"),
            contract_address=receipt.get("contractAddress"),
            logs=list(receipt.get("logs", [])),
        )
        if result.status != 1:
            raise TransactionReverted(
                f"transaction {normalized} was mined but reverted (status 0)"
            )
        return result

    # -- deployment -----------------------------------------------------

    def deploy(self) -> tuple[str, TxResult]:
        """Deploy IdentityAnchor. Returns (address, receipt facts)."""
        self.log("[CHAIN] contract deploying")
        contract = self.w3.eth.contract(
            abi=self.artifact["abi"], bytecode=self.artifact["bytecode"]
        )
        tx = contract.constructor().build_transaction(self._build_common())
        tx["gas"] = self._estimate(tx)

        tx_hash = self._send(tx)
        self.log(f"[CHAIN] deployment tx submitted {tx_hash}")
        result = self._await_receipt(tx_hash)

        if not result.contract_address:
            raise ChainError("receipt contained no contract address")
        self.log(f"[CHAIN] deployment confirmed in block {result.block_number}")
        return result.contract_address, result

    def _estimate(self, tx: dict[str, Any]) -> int:
        """Estimate gas from the network and add a buffer."""
        try:
            estimate = self.w3.eth.estimate_gas(tx)
        except Exception as exc:
            raise ChainError(f"gas estimation failed: {type(exc).__name__}: {exc}") from exc
        return int(estimate * (100 + self.cfg.gas_buffer_percent) / 100)

    # -- contract -------------------------------------------------------

    def contract_at(self, address: str):
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(address), abi=self.artifact["abi"]
        )

    def anchor(self, address: str, investigation_id: str, evidence_sha256: str) -> TxResult:
        """Anchor a fingerprint. Returns the mined receipt facts."""
        contract = self.contract_at(address)
        key = investigation_key(investigation_id)
        digest = evidence_bytes32(evidence_sha256)

        self.log("[CHAIN] anchoring evidence")
        tx = contract.functions.anchorEvidence(key, digest).build_transaction(
            self._build_common()
        )
        tx["gas"] = self._estimate(tx)

        tx_hash = self._send(tx)
        self.log(f"[CHAIN] transaction submitted {tx_hash}")
        result = self._await_receipt(tx_hash)
        self.log(f"[CHAIN] confirmation received in block {result.block_number}")
        return result

    def read_evidence(self, address: str, investigation_id: str) -> dict[str, Any]:
        """Read the anchored record back from the chain."""
        contract = self.contract_at(address)
        key = investigation_key(investigation_id)
        evidence_hash, submitter, timestamp, block_number = (
            contract.functions.getEvidence(key).call()
        )
        return {
            "evidence_sha256": evidence_hash.hex(),
            "submitter": submitter,
            "timestamp": int(timestamp),
            "block_number": int(block_number),
        }

    def verify_on_chain(self, address: str, investigation_id: str, evidence_sha256: str) -> bool:
        """Ask the contract itself whether the supplied hash matches."""
        contract = self.contract_at(address)
        return bool(
            contract.functions.verifyEvidence(
                investigation_key(investigation_id), evidence_bytes32(evidence_sha256)
            ).call()
        )

    def is_anchored(self, address: str, investigation_id: str) -> bool:
        contract = self.contract_at(address)
        return bool(
            contract.functions.isAnchored(investigation_key(investigation_id)).call()
        )


def read_only_client(rpc_url: str | None = None, logger=None, web3=None) -> AnchorClient:
    """A client that can read the chain but holds no signing key."""
    return AnchorClient(rpc_url=rpc_url, logger=logger, web3=web3, read_only=True)
