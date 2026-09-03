"""Tamper case definitions, shared by the demo script and the GUI panel.

These are demonstration mutations, not pipeline logic. Verification itself is
not reimplemented here - callers run the real
:func:`src.blockchain.verifier.verify_against_chain` against the mutated copy.

Two independent detection layers, and each case is expected to trip a
specific one:

``manifest``
    The manifest itself changed, so its SHA-256 changed and no longer matches
    the anchored value.

``artifact``
    A covered file changed while the manifest did not. The manifest hash is
    UNCHANGED **by design**; detection comes from the per-artifact digests the
    manifest carries. Expecting the hash to move here would be wrong.

Every mutation is applied to a COPY. Nothing here writes to the bundle it is
given - :func:`apply_to_copy` does the copying.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .evidence import hashing
from .evidence.collector import MANIFEST_FILE

MANIFEST_LAYER = "manifest"
ARTIFACT_LAYER = "artifact"


@dataclass(frozen=True)
class TamperCase:
    """One demonstration mutation."""

    name: str
    layer: str          # MANIFEST_LAYER or ARTIFACT_LAYER
    mutate: Callable[[Path], str]

    @property
    def expects_hash_change(self) -> bool:
        """Whether the recomputed manifest fingerprint should move.

        False for artifact-layer cases: the manifest is untouched, so its hash
        must stay the same and the artifact digests are what catch the change.
        """
        return self.layer == MANIFEST_LAYER

    def apply_to_copy(self, bundle: Path, destination: Path) -> str:
        """Copy the bundle, mutate the copy, and describe what changed."""
        shutil.copytree(bundle, destination)
        return self.mutate(destination)


def _read_manifest(bundle: Path) -> dict:
    return json.loads((bundle / MANIFEST_FILE).read_bytes().decode("utf-8"))


def _write_manifest(bundle: Path, manifest: dict) -> None:
    (bundle / MANIFEST_FILE).write_bytes(hashing.canonical_bytes(manifest))


# -- the mutations ----------------------------------------------------

def _change_one_digit(dest: Path) -> str:
    manifest = _read_manifest(dest)
    node = manifest["matching"]["best_independent_match"]
    before = node["similarity"]
    node["similarity"] = before[:-1] + str((int(before[-1]) + 1) % 10)
    _write_manifest(dest, manifest)
    return f"similarity {before} -> {node['similarity']}"


def _change_source_url(dest: Path) -> str:
    manifest = _read_manifest(dest)
    node = manifest["matching"]["best_independent_match"]
    before = node["source_url"]
    node["source_url"] = "https://attacker.example/substituted"
    _write_manifest(dest, manifest)
    return f"{before[:44]}... -> attacker.example"


def _flip_candidate_image_byte(dest: Path) -> str:
    img = dest / "source-image.jpg"
    data = bytearray(img.read_bytes())
    data[-1] ^= 0xFF
    img.write_bytes(bytes(data))
    return "last byte of source-image.jpg XOR 0xFF"


def _alter_raw_response(dest: Path) -> str:
    path = dest / "search-response.json"
    path.write_bytes(path.read_bytes().replace(b'"position": 1', b'"position": 9', 1))
    return "one position field rewritten"


def _delete_artifact(dest: Path) -> str:
    (dest / "input.jpg").unlink()
    return "input.jpg removed"


def _resign_manifest_over_modified_artifact(dest: Path) -> str:
    # The sophisticated attempt: change an artifact AND fix the manifest so
    # local self-consistency is restored. The chain still catches it, because
    # the manifest hash changed.
    img = dest / "source-image.jpg"
    data = bytearray(img.read_bytes())
    data[-1] ^= 0xFF
    img.write_bytes(bytes(data))

    manifest = _read_manifest(dest)
    manifest["artifacts"]["source-image.jpg"] = {
        "sha256": hashing.sha256_file(img),
        "bytes": img.stat().st_size,
    }
    _write_manifest(dest, manifest)
    return "image altered AND manifest digest re-signed"


CASES: tuple[TamperCase, ...] = (
    TamperCase("one digit changed in the manifest", MANIFEST_LAYER, _change_one_digit),
    TamperCase("source URL changed in the manifest", MANIFEST_LAYER, _change_source_url),
    TamperCase("candidate image byte flipped", ARTIFACT_LAYER, _flip_candidate_image_byte),
    TamperCase("raw search response altered", ARTIFACT_LAYER, _alter_raw_response),
    TamperCase("artifact deleted", ARTIFACT_LAYER, _delete_artifact),
    TamperCase("manifest digest updated to cover a modified artifact",
               MANIFEST_LAYER, _resign_manifest_over_modified_artifact),
)
