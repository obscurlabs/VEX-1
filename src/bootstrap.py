"""Interpreter guard for the CLI entry points.

The dependencies live in the project virtualenv. Running an entry point with
a different interpreter fails deep inside an import with a ModuleNotFoundError
traceback, which is exactly the kind of noise the CLI is supposed to never
show. This module is imported first, uses only the standard library, and turns
that into a one-line explanation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# module name -> what the user would recognise it as
REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "insightface": "insightface",
    "requests": "requests",
    "web3": "web3",
    "eth_account": "web3 (eth-account)",
}


def _venv_python() -> Path | None:
    for candidate in (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
        PROJECT_ROOT / ".venv" / "bin" / "python",           # POSIX
    ):
        if candidate.exists():
            return candidate
    return None


def _running_in_project_venv() -> bool:
    venv = _venv_python()
    if venv is None:
        return False
    try:
        return Path(sys.executable).resolve() == venv.resolve()
    except OSError:
        return False


def missing_dependencies() -> list[str]:
    """Names of required packages that cannot be imported."""
    import importlib.util

    missing = []
    for module, label in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(label)
    return missing


def require_dependencies() -> None:
    """Exit with a readable message instead of an import traceback."""
    missing = missing_dependencies()
    if not missing:
        return

    venv = _venv_python()
    print("=" * 60, file=sys.stderr)
    print("  WRONG PYTHON INTERPRETER", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  running: {sys.executable}", file=sys.stderr)
    print(f"  version: {sys.version.split()[0]}", file=sys.stderr)
    print(f"  missing: {', '.join(sorted(set(missing)))}", file=sys.stderr)
    print(file=sys.stderr)

    if venv is not None and not _running_in_project_venv():
        activate = (
            ".venv\\Scripts\\activate" if os.name == "nt"
            else "source .venv/bin/activate"
        )
        print("  The dependencies are installed in this project's virtualenv.", file=sys.stderr)
        print("  Activate it once:", file=sys.stderr)
        print(f"      {activate}", file=sys.stderr)
        print("  or call it directly:", file=sys.stderr)
        try:
            shown = venv.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = venv
        print(f"      {shown} {Path(sys.argv[0]).name} ...", file=sys.stderr)
    else:
        print("  Install the dependencies:", file=sys.stderr)
        print("      uv venv --python 3.12.13 .venv", file=sys.stderr)
        print("      VIRTUAL_ENV=.venv uv pip install -r requirements.txt", file=sys.stderr)

    print("=" * 60, file=sys.stderr)
    raise SystemExit(1)
