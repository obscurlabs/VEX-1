"""Terminal presentation for the CLI.

Deliberately plain: no animations, no spinners, no fake progress. Every line
printed corresponds to work that actually completed.
"""
from __future__ import annotations

import sys

WIDTH = 60
PAD = "     "


# Exit codes. Stable, so scripts and the demo can rely on them.
EXIT_OK = 0
EXIT_INPUT = 1          # unreadable image, no face
EXIT_SEARCH_AUTH = 2    # provider rejected the key
EXIT_SEARCH_LIMIT = 3   # provider quota / rate limit
EXIT_SEARCH = 4         # any other discovery failure
EXIT_NO_MATCH = 5       # ran fine, nothing matched
EXIT_EVIDENCE = 6       # bundle or fingerprint problem
EXIT_CHAIN = 7          # RPC, wallet, or transaction failure
EXIT_VERIFY = 8         # hash mismatch


def header(title: str, subtitle: str | None = None) -> None:
    print("=" * WIDTH)
    print(title.center(WIDTH))
    if subtitle:
        print(subtitle.center(WIDTH))
    print("=" * WIDTH)


def stage(number: str, title: str) -> None:
    print(f"\n[{number}] {title}")


def ok(msg: str) -> None:
    print(f"{PAD}✓ {msg}")


def warn(msg: str) -> None:
    print(f"{PAD}! {msg}")


def fail(msg: str) -> None:
    print(f"{PAD}✗ {msg}")


def info(msg: str) -> None:
    print(f"{PAD}  {msg}")


def detail(msg: str) -> None:
    """Verbose-only line, indented under a stage."""
    print(f"{PAD}  {msg}")


def counts(mapping: dict[str, int], skip: str | None = None) -> None:
    """Aggregate tally, most common first. Quiet when there is nothing to say."""
    for key, n in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0])):
        if key == skip:
            continue
        print(f"{PAD}  {key:<22} {n}")


def verdict(text: str, good: bool = True) -> None:
    mark = "✓" if good else "✗"
    line = f"{mark} {text}"
    print()
    print("=" * WIDTH)
    print(line.center(WIDTH))
    print("=" * WIDTH)


def summary(rows: list[tuple[str, str]], title: str = "SUMMARY") -> None:
    """Compact key/value block for the end of a run."""
    print()
    print("-" * WIDTH)
    print(f"  {title}")
    print("-" * WIDTH)
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        print(f"  {key:<{width}}  {value}")
    print("-" * WIDTH)


def die(msg: str, code: int = 1) -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    raise SystemExit(code)
