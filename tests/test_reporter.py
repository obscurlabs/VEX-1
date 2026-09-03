"""Phase A: the reporter seam.

Two things must hold:
  1. the console rendering is byte-for-byte what it always was
  2. an injected reporter receives the same events, on its own thread only
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src import pipeline
from src.pipeline import ConsoleReporter, Reporter, RunResult


class Recorder(Reporter):
    """Captures events instead of printing them."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def header(self, title, subtitle=None): self.events.append(("header", title, subtitle))
    def line(self, text=""): self.events.append(("line", text))
    def stage(self, number, title): self.events.append(("stage", number, title))
    def ok(self, msg): self.events.append(("ok", msg))
    def warn(self, msg): self.events.append(("warn", msg))
    def fail(self, msg): self.events.append(("fail", msg))
    def info(self, msg): self.events.append(("info", msg))
    def detail(self, msg): self.events.append(("detail", msg))
    def counts(self, mapping, skip=None): self.events.append(("counts", dict(mapping), skip))
    def verdict(self, text, good=True): self.events.append(("verdict", text, good))
    def summary(self, rows, title="SUMMARY"): self.events.append(("summary", list(rows), title))
    def result(self, result): self.events.append(("result", result))

    def kinds(self) -> list[str]:
        return [e[0] for e in self.events]


@pytest.fixture(autouse=True)
def _clean_reporter():
    """No test may leak a reporter into another."""
    pipeline.set_reporter(None)
    yield
    pipeline.set_reporter(None)


# --- console rendering is unchanged -----------------------------------

def test_console_is_the_default(capsys):
    assert isinstance(pipeline.active_reporter(), ConsoleReporter)
    pipeline.ok("image loaded")
    assert capsys.readouterr().out == f"{pipeline.PAD}✓ image loaded\n"


@pytest.mark.parametrize("call, expected", [
    (lambda: pipeline.ok("x"), "     ✓ x\n"),
    (lambda: pipeline.warn("x"), "     ! x\n"),
    (lambda: pipeline.fail("x"), "     ✗ x\n"),
    (lambda: pipeline.info("x"), "       x\n"),
    (lambda: pipeline.detail("x"), "       x\n"),
    (lambda: pipeline.stage("01", "FACE SCAN"), "\n[01] FACE SCAN\n"),
    (lambda: pipeline.line("raw"), "raw\n"),
    (lambda: pipeline.line(), "\n"),
])
def test_each_console_line_is_exact(capsys, call, expected):
    call()
    assert capsys.readouterr().out == expected


def test_header_rendering(capsys):
    pipeline.header("TITLE")
    out = capsys.readouterr().out
    assert out == "=" * 60 + "\n" + "TITLE".center(60) + "\n" + "=" * 60 + "\n"


def test_header_with_subtitle(capsys):
    pipeline.header("T", "S")
    assert "S".center(60) in capsys.readouterr().out


def test_counts_orders_by_frequency_then_name(capsys):
    pipeline.counts({"B": 1, "A": 5, "C": 1})
    lines = capsys.readouterr().out.splitlines()
    assert [l.split()[0] for l in lines] == ["A", "B", "C"]


def test_counts_honours_skip(capsys):
    pipeline.counts({"RETRIEVED": 9, "HTTP_403": 1}, skip="RETRIEVED")
    out = capsys.readouterr().out
    assert "RETRIEVED" not in out and "HTTP_403" in out


def test_counts_of_nothing_prints_nothing(capsys):
    pipeline.counts({})
    assert capsys.readouterr().out == ""


def test_summary_pads_to_the_widest_key(capsys):
    pipeline.summary([("a", "1"), ("longer", "2")], title="RESULT")
    out = capsys.readouterr().out
    assert "  RESULT" in out
    assert "  a       1" in out
    assert "  longer  2" in out


def test_summary_of_nothing_does_not_crash(capsys):
    pipeline.summary([])
    assert "SUMMARY" in capsys.readouterr().out


def test_verdict_rendering(capsys):
    pipeline.verdict("BLOCKCHAIN VERIFIED")
    out = capsys.readouterr().out
    assert "✓ BLOCKCHAIN VERIFIED".center(60) in out
    pipeline.verdict("FAILED", good=False)
    assert "✗ FAILED".center(60) in capsys.readouterr().out


def test_console_result_prints_nothing(capsys):
    """The console already showed the summary block; result() is for front ends."""
    pipeline.result(RunResult("TRACE-X", "a" * 64, "evidence/x", 1.0))
    assert capsys.readouterr().out == ""


# --- injection ---------------------------------------------------------

def test_injected_reporter_receives_events_and_console_stays_silent(capsys):
    rec = Recorder()
    with pipeline.use_reporter(rec):
        pipeline.stage("01", "FACE SCAN")
        pipeline.ok("1 face detected")
        pipeline.warn("6 skipped")
        pipeline.fail("nope")
        pipeline.counts({"A": 1})
        pipeline.verdict("DONE")
        pipeline.summary([("k", "v")])
    assert capsys.readouterr().out == "", "nothing should reach stdout"
    assert rec.kinds() == ["stage", "ok", "warn", "fail", "counts", "verdict", "summary"]
    assert rec.events[1] == ("ok", "1 face detected")


def test_reporter_is_restored_after_the_block(capsys):
    with pipeline.use_reporter(Recorder()):
        pass
    assert isinstance(pipeline.active_reporter(), ConsoleReporter)
    pipeline.ok("back to console")
    assert "back to console" in capsys.readouterr().out


def test_reporter_is_restored_even_after_an_exception():
    rec = Recorder()
    with pytest.raises(RuntimeError):
        with pipeline.use_reporter(rec):
            raise RuntimeError("boom")
    assert isinstance(pipeline.active_reporter(), ConsoleReporter)


def test_nested_reporters_restore_in_order():
    outer, inner = Recorder(), Recorder()
    with pipeline.use_reporter(outer):
        pipeline.ok("a")
        with pipeline.use_reporter(inner):
            pipeline.ok("b")
        pipeline.ok("c")
    assert [e[1] for e in outer.events] == ["a", "c"]
    assert [e[1] for e in inner.events] == ["b"]


def test_set_reporter_returns_the_previous_one():
    rec = Recorder()
    previous = pipeline.set_reporter(rec)
    assert isinstance(previous, ConsoleReporter)
    assert pipeline.active_reporter() is rec
    pipeline.set_reporter(None)
    assert isinstance(pipeline.active_reporter(), ConsoleReporter)


def test_base_reporter_swallows_every_event():
    """A front end may override only what it needs."""
    base = Reporter()
    with pipeline.use_reporter(base):
        pipeline.header("t")
        pipeline.stage("01", "x")
        pipeline.ok("x")
        pipeline.counts({"A": 1})
        pipeline.summary([("k", "v")])
        pipeline.result(RunResult("T", "a" * 64, "b", 0.0))


# --- thread isolation --------------------------------------------------

def test_a_reporter_on_one_thread_does_not_affect_another(capsys):
    rec = Recorder()
    seen: dict[str, object] = {}

    def worker():
        with pipeline.use_reporter(rec):
            pipeline.ok("from worker")
        seen["worker_default"] = pipeline.active_reporter()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # The main thread was never switched.
    assert isinstance(pipeline.active_reporter(), ConsoleReporter)
    pipeline.ok("from main")
    assert "from main" in capsys.readouterr().out

    assert [e[1] for e in rec.events] == ["from worker"]
    assert isinstance(seen["worker_default"], ConsoleReporter)


def test_two_worker_threads_do_not_interleave():
    a, b = Recorder(), Recorder()
    barrier = threading.Barrier(2)

    def run(rec, tag):
        with pipeline.use_reporter(rec):
            barrier.wait()          # force overlap
            for i in range(20):
                pipeline.ok(f"{tag}{i}")

    threads = [threading.Thread(target=run, args=(a, "a")),
               threading.Thread(target=run, args=(b, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [e[1] for e in a.events] == [f"a{i}" for i in range(20)]
    assert [e[1] for e in b.events] == [f"b{i}" for i in range(20)]


# --- RunResult ---------------------------------------------------------

def test_run_result_verified_reflects_the_verification_object():
    class V:
        verified = True

    assert RunResult("T", "a" * 64, "b", 1.0, verification=V()).verified is True
    V.verified = False
    assert RunResult("T", "a" * 64, "b", 1.0, verification=V()).verified is False
    assert RunResult("T", "a" * 64, "b", 1.0, verification=None).verified is False


def test_run_result_carries_no_secret_fields():
    fields = RunResult.__dataclass_fields__
    for banned in ("private_key", "api_key", "serpapi_key", "rpc_url", "key"):
        assert banned not in fields


# --- main.py still uses the seam ---------------------------------------

def test_main_has_no_raw_prints_left():
    """Every line main.run() emits must go through the reporter, or a GUI
    would silently miss it."""
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    offenders = [l.strip() for l in src.splitlines()
                 if l.strip().startswith("print(")]
    assert offenders == [], f"raw print() bypasses the reporter: {offenders}"


def test_pipeline_module_is_stdlib_only():
    """It is imported early; it must not drag in heavy dependencies."""
    src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
    for banned in ("import cv2", "import numpy", "import web3", "from .config",
                   "from .models", "from .blockchain"):
        assert banned not in src, f"pipeline.py must not import {banned}"
