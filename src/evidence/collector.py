"""Investigation identity and on-disk artifacts.

Phase 1 scope: allocate the investigation ID and persist the raw provider
responses so later phases can package them. The full evidence bundle and its
canonical manifest are Phase 3.
"""
from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import CONFIG


def new_investigation_id(now: datetime | None = None) -> str:
    """e.g. TRACE-20260902-A7F31C"""
    now = now or datetime.now(timezone.utc)
    return f"TRACE-{now:%Y%m%d}-{secrets.token_hex(3).upper()}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ArtifactStore:
    """The directory holding one investigation's artifacts."""

    def __init__(self, investigation_id: str, root: Path | None = None):
        self.investigation_id = investigation_id
        self.root = (root or CONFIG.evidence_dir) / investigation_id
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, name: str, payload: bytes) -> Path:
        """Persist bytes verbatim. Used for provider responses, which must be
        preserved exactly as received."""
        path = self.root / name
        path.write_bytes(payload)
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def copy_input(self, image_path: Path) -> Path:
        dest = self.root / f"input{Path(image_path).suffix.lower()}"
        shutil.copyfile(image_path, dest)
        return dest

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(CONFIG.project_root)).replace("\\", "/")
        except ValueError:
            return str(path)


FINGERPRINT_FILE = "fingerprint.json"
MANIFEST_FILE = "manifest.json"


def write_fingerprint(store: "ArtifactStore", manifest: dict) -> tuple[str, Path, Path]:
    """Write manifest.json plus its fingerprint. Returns (sha256, paths)."""
    from . import hashing

    digest, canonical = hashing.fingerprint(manifest)

    # manifest.json is stored in canonical form, so the file on disk IS the
    # bytes that were hashed - no ambiguity about what was fingerprinted.
    manifest_path = store.root / MANIFEST_FILE
    manifest_path.write_bytes(canonical)

    fingerprint_path = store.write_json(FINGERPRINT_FILE, {
        "investigation_id": manifest["investigation_id"],
        "algorithm": hashing.ALGORITHM,
        "canonicalization": hashing.CANONICALIZATION,
        "manifest_file": MANIFEST_FILE,
        "manifest_bytes": len(canonical),
        "evidence_sha256": digest,
    })
    return digest, manifest_path, fingerprint_path


class VerificationResult:
    """Outcome of checking a bundle against its recorded fingerprint."""

    def __init__(self, bundle: Path):
        self.bundle = bundle
        self.problems: list[str] = []
        self.checked: list[str] = []
        self.expected: str | None = None
        self.computed: str | None = None

    @property
    def verified(self) -> bool:
        return not self.problems and self.expected is not None \
            and self.expected == self.computed

    def fail(self, msg: str) -> None:
        self.problems.append(msg)


def verify_bundle(bundle: str | Path) -> VerificationResult:
    """Recompute a bundle's fingerprint from what is actually on disk.

    Two independent checks, both required:

      1. every artifact's on-disk SHA-256 matches the digest recorded in the
         manifest - catches tampering with any file, binary included
      2. the canonical re-serialization of manifest.json matches the recorded
         evidence_sha256 - catches tampering with the manifest itself
    """
    from . import hashing

    root = Path(bundle)
    result = VerificationResult(root)

    if not root.is_dir():
        result.fail(f"bundle directory not found: {root}")
        return result

    manifest_path = root / MANIFEST_FILE
    fingerprint_path = root / FINGERPRINT_FILE
    for path in (manifest_path, fingerprint_path):
        if not path.exists():
            result.fail(f"missing {path.name}")
    if result.problems:
        return result

    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except Exception as exc:
        result.fail(f"{MANIFEST_FILE} is not readable JSON: {exc}")
        return result
    try:
        recorded = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.fail(f"{FINGERPRINT_FILE} is not readable JSON: {exc}")
        return result

    result.expected = recorded.get("evidence_sha256")
    if not result.expected:
        result.fail(f"{FINGERPRINT_FILE} has no evidence_sha256")

    # 1. artifacts on disk must match the digests the manifest recorded
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        path = root / name
        if not path.exists():
            result.fail(f"missing artifact: {name}")
            continue
        actual = hashing.sha256_file(path)
        size = path.stat().st_size
        if actual != entry.get("sha256"):
            result.fail(f"artifact modified: {name}")
        elif size != entry.get("bytes"):
            result.fail(f"artifact size changed: {name}")
        else:
            result.checked.append(name)

    # 2. the manifest itself must re-canonicalize to the recorded fingerprint
    try:
        result.computed = hashing.sha256_bytes(hashing.canonical_bytes(manifest))
    except Exception as exc:
        result.fail(f"manifest is not canonicalizable: {exc}")
        return result

    if result.expected and result.computed != result.expected:
        result.fail("manifest hash mismatch")

    return result
