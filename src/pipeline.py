"""Terminal presentation for the CLI, and the seam a GUI reports through.

Deliberately plain: no animations, no spinners, no fake progress. Every line
printed corresponds to work that actually completed.

The reporter seam
-----------------
``main.run()`` reports by calling the module-level functions below. Those
functions delegate to a *thread-local* active reporter, which defaults to
:class:`ConsoleReporter` - byte-for-byte the behaviour the CLI has always had.

A front end (the PySide6 GUI) swaps in its own reporter for the duration of a
run and receives the same events as Qt signals instead of stdout::

    with pipeline.use_reporter(QtReporter(signals)):
        exit_code = main.run(args)

Thread-local, so a GUI worker thread cannot affect the console output of
anything else in the process, and concurrent runs cannot interleave.

Nothing here computes anything. A reporter only ever receives state the
pipeline has already established, so a front end cannot invent progress,
scores, transactions or verdicts that the pipeline did not produce.

Only the standard library is imported, so this module is safe to import early.
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

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


@dataclass
class RunResult:
    """Structured outcome of a completed run.

    Every field is an object the pipeline already produced. This exists so a
    front end does not have to re-parse formatted text; it carries no value
    that is not also visible in the console output.
    """

    investigation_id: str
    evidence_sha256: str
    bundle_path: str
    elapsed_seconds: float
    # src.models.CandidateMatch - the anchored candidate
    match: Any = None
    # src.blockchain.client.TxResult, or None when nothing was broadcast
    receipt: Any = None
    # src.blockchain.verifier.ChainVerification, or None with --no-chain
    verification: Any = None

    @property
    def verified(self) -> bool:
        return bool(self.verification is not None and self.verification.verified)


class Reporter:
    """Receives pipeline events. Subclass to render them somewhere else.

    The default implementations do nothing, so a front end only overrides the
    events it cares about.
    """

    def header(self, title: str, subtitle: str | None = None) -> None: ...
    def line(self, text: str) -> None: ...
    def stage(self, number: str, title: str) -> None: ...
    def ok(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def fail(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def detail(self, msg: str) -> None: ...
    def counts(self, mapping: dict[str, int], skip: str | None = None) -> None: ...
    def verdict(self, text: str, good: bool = True) -> None: ...
    def summary(self, rows: list[tuple[str, str]], title: str = "SUMMARY") -> None: ...
    def result(self, result: RunResult) -> None: ...


class ConsoleReporter(Reporter):
    """The CLI's rendering. Bodies are unchanged from the original functions."""

    def header(self, title: str, subtitle: str | None = None) -> None:
        print("=" * WIDTH)
        print(title.center(WIDTH))
        if subtitle:
            print(subtitle.center(WIDTH))
        print("=" * WIDTH)

    def line(self, text: str = "") -> None:
        print(text)

    def stage(self, number: str, title: str) -> None:
        print(f"\n[{number}] {title}")

    def ok(self, msg: str) -> None:
        print(f"{PAD}✓ {msg}")

    def warn(self, msg: str) -> None:
        print(f"{PAD}! {msg}")

    def fail(self, msg: str) -> None:
        print(f"{PAD}✗ {msg}")

    def info(self, msg: str) -> None:
        print(f"{PAD}  {msg}")

    def detail(self, msg: str) -> None:
        print(f"{PAD}  {msg}")

    def counts(self, mapping: dict[str, int], skip: str | None = None) -> None:
        for key, n in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0])):
            if key == skip:
                continue
            print(f"{PAD}  {key:<22} {n}")

    def verdict(self, text: str, good: bool = True) -> None:
        mark = "✓" if good else "✗"
        line = f"{mark} {text}"
        print()
        print("=" * WIDTH)
        print(line.center(WIDTH))
        print("=" * WIDTH)

    def summary(self, rows: list[tuple[str, str]], title: str = "SUMMARY") -> None:
        print()
        print("-" * WIDTH)
        print(f"  {title}")
        print("-" * WIDTH)
        width = max((len(k) for k, _ in rows), default=0)
        for key, value in rows:
            print(f"  {key:<{width}}  {value}")
        print("-" * WIDTH)

    def result(self, result: RunResult) -> None:
        """The console already printed everything in the summary block."""


CONSOLE = ConsoleReporter()
_local = threading.local()


def active_reporter() -> Reporter:
    """The reporter for the calling thread. Console unless one was set."""
    return getattr(_local, "reporter", CONSOLE)


def set_reporter(reporter: Reporter | None) -> Reporter:
    """Install a reporter for the calling thread. Returns the previous one."""
    previous = active_reporter()
    if reporter is None:
        _local.__dict__.pop("reporter", None)
    else:
        _local.reporter = reporter
    return previous


@contextmanager
def use_reporter(reporter: Reporter) -> Iterator[Reporter]:
    """Scope a reporter to a block, restoring the previous one afterwards."""
    previous = set_reporter(reporter)
    try:
        yield reporter
    finally:
        set_reporter(None if previous is CONSOLE else previous)


# -- module-level API -------------------------------------------------
# main.run() calls these. They forward to whichever reporter is active on
# this thread, so the CLI is unaffected and a GUI needs no changes in main.py.

def header(title: str, subtitle: str | None = None) -> None:
    active_reporter().header(title, subtitle)


def line(text: str = "") -> None:
    active_reporter().line(text)


def stage(number: str, title: str) -> None:
    active_reporter().stage(number, title)


def ok(msg: str) -> None:
    active_reporter().ok(msg)


def warn(msg: str) -> None:
    active_reporter().warn(msg)


def fail(msg: str) -> None:
    active_reporter().fail(msg)


def info(msg: str) -> None:
    active_reporter().info(msg)


def detail(msg: str) -> None:
    active_reporter().detail(msg)


def counts(mapping: dict[str, int], skip: str | None = None) -> None:
    active_reporter().counts(mapping, skip)


def verdict(text: str, good: bool = True) -> None:
    active_reporter().verdict(text, good)


def summary(rows: list[tuple[str, str]], title: str = "SUMMARY") -> None:
    active_reporter().summary(rows, title)


def result(payload: RunResult) -> None:
    active_reporter().result(payload)


def die(msg: str, code: int = 1) -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    raise SystemExit(code)
