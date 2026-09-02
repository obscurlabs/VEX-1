"""Terminal presentation helpers shared by the CLI stages."""
from __future__ import annotations

import sys

WIDTH = 62


def banner(title: str) -> None:
    print("╔" + "═" * WIDTH + "╗")
    print("║" + title.center(WIDTH) + "║")
    print("╚" + "═" * WIDTH + "╝")


def stage(number: str, title: str) -> None:
    print(f"\n[{number}] {title}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


def die(msg: str, code: int = 1) -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    raise SystemExit(code)
