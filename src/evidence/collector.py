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
