"""Deterministic canonicalization and SHA-256 fingerprinting.

The same evidence must always produce the same bytes and therefore the same
hash, on any machine, in any process, in any Python 3.11+ build. Changing any
covered field must change the hash.

Canonical form
--------------
JSON with:

  * ``sort_keys=True``       - key order never depends on insertion order
  * ``separators=(",", ":")`` - no incidental whitespace
  * ``ensure_ascii=False``   - real characters, then an explicit UTF-8 encode
  * ``allow_nan=False``      - NaN/Infinity are not valid JSON
  * NFC-normalized strings   - "é" composed and decomposed hash the same
  * **no floats at all**     - see below

Why floats are rejected
-----------------------
``repr(0.1 + 0.2)`` and the shortest-round-trip rules are stable in CPython,
but binary floating point still makes "the same number" ambiguous across
languages and serializers: 0.9944 may be emitted as 0.9944, 0.99440000000001
or 9.944e-1. A fingerprint that other tools must be able to reproduce cannot
depend on that. Every real-valued quantity is therefore quantized to a
fixed-precision decimal **string** by :func:`decimal_str` before it enters the
manifest, and :func:`canonical_bytes` raises if a raw float slips through.
Integers are exact and are kept as JSON numbers.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

# Fixed precision for every real-valued field in the manifest.
SIMILARITY_PLACES = 6
SCORE_PLACES = 4

CANONICALIZATION = (
    "json;sort_keys=true;separators=(',',':');ensure_ascii=false;"
    "allow_nan=false;unicode=NFC;encoding=utf-8;floats=forbidden"
)
ALGORITHM = "SHA-256"


def decimal_str(value: float | int | Decimal | str, places: int = SIMILARITY_PLACES) -> str:
    """Quantize a real value to a fixed-precision decimal string.

    Banker's rounding, an explicit sign for nothing, and a stable number of
    decimal places, so 0.7 and 0.70 and 0.7000001 all become "0.700000".
    """
    d = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    # Decimal renders -0 as "-0.000000"; collapse it so sign never flips a hash.
    if d == 0:
        d = abs(d)
    return f"{d:f}"


def _normalize(node: Any) -> Any:
    """Recursively prepare a value for canonical serialization."""
    if isinstance(node, bool):
        return node
    if isinstance(node, int):
        return node
    if isinstance(node, float):
        raise TypeError(
            "floats are not permitted in the canonical manifest; "
            "use hashing.decimal_str() to emit a fixed-precision string"
        )
    if isinstance(node, Decimal):
        raise TypeError("Decimal must be rendered with hashing.decimal_str() first")
    if node is None:
        return None
    if isinstance(node, str):
        return unicodedata.normalize("NFC", node)
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if not isinstance(key, str):
                raise TypeError(f"manifest keys must be strings, got {type(key).__name__}")
            out[unicodedata.normalize("NFC", key)] = _normalize(value)
        return out
    if isinstance(node, (list, tuple)):
        return [_normalize(v) for v in node]
    raise TypeError(f"unsupported type in canonical manifest: {type(node).__name__}")


def canonical_bytes(manifest: dict) -> bytes:
    """Serialize a manifest to its one canonical byte representation."""
    return json.dumps(
        _normalize(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fingerprint(manifest: dict) -> tuple[str, bytes]:
    """Return (sha256_hex, canonical_bytes) for a manifest."""
    data = canonical_bytes(manifest)
    return sha256_bytes(data), data
