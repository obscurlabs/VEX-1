"""Phase 7.B: GUI shell integration.

Deterministic by construction: the one test that executes the real pipeline
runs in diagnostic mode with --no-chain against a captured fixture, so no
SerpAPI credit is spent and no transaction is sent. Everything else drives the
widgets with synthetic reporter events.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.reporter import PipelineCancelled, QtReporter  # noqa: E402
from gui.widgets.drop_zone import DropZone, inspect_image  # noqa: E402
from gui.widgets.log_panel import LogPanel  # noqa: E402
from gui.widgets.result_panel import UNAVAILABLE, ResultPanel  # noqa: E402
from gui.widgets.stage_list import StageList, StageState  # noqa: E402
from gui.worker import EXIT_CANCELLED, PipelineWorker, RunRequest  # noqa: E402
from src import pipeline  # noqa: E402
from src.pipeline import RunResult  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "inputs" / "demo-target.jpg"
DEMO_FIXTURE = ROOT / "tests" / "fixtures" / "demo-target-response.json"
HAVE_DEMO = DEMO.exists() and DEMO_FIXTURE.exists()


@pytest.fixture
def window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def bad_file(tmp_path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image", encoding="utf-8")
    return path


@pytest.fixture
def corrupt_image(tmp_path) -> Path:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return path


@pytest.fixture
def no_face_image(tmp_path) -> Path:
    path = tmp_path / "no-face.png"
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :] = (30, 90, 160)
    cv2.imwrite(str(path), img)
    return path


def _drop(widget, paths: list[Path]) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    widget.dropEvent(event)


# --- construction ------------------------------------------------------

def test_window_constructs(window):
    assert window.windowTitle()
    assert list(window.stage_list.states()) == ["01", "02", "03", "04", "05", "06", "07"]
    assert not window.is_running


def test_all_stages_start_pending(window):
    assert set(window.stage_list.states().values()) == {StageState.PENDING}


def test_result_panel_starts_empty(window):
    assert "NO INVESTIGATION" in window.result_panel.headline()
    assert window.result_panel.summary.similarity_text() == "—"
    assert window.result_panel.value_of("transaction") == "—"


# --- image selection ---------------------------------------------------

@pytest.mark.skipif(not HAVE_DEMO, reason="demo image missing")
def test_valid_image_is_accepted(window):
    assert window.drop_zone.accept_path(DEMO) is True
    assert window.drop_zone.path == DEMO
    assert window.start_button.isEnabled()


def test_non_image_file_is_rejected(window, bad_file):
    assert window.drop_zone.accept_path(bad_file) is False
    assert window.drop_zone.path is None
    assert not window.start_button.isEnabled()
    assert "unsupported file type" in window.status_bar.text()


def test_corrupt_image_is_rejected(window, corrupt_image):
    assert window.drop_zone.accept_path(corrupt_image) is False
    assert not window.start_button.isEnabled()
    assert "Rejected" in window.status_bar.text()


def test_missing_file_is_rejected(window, tmp_path):
    assert window.drop_zone.accept_path(tmp_path / "nope.jpg") is False
    assert not window.start_button.isEnabled()


def test_validation_uses_the_pipeline_loader(no_face_image):
    """A face-less image is still a valid *image*; the pipeline rejects it
    later at stage 01, not at file selection."""
    ok, message, size = inspect_image(no_face_image)
    assert ok is True and size == (400, 300)


@pytest.mark.skipif(not HAVE_DEMO, reason="demo image missing")
def test_drop_accepts_a_single_image(qtbot, window):
    _drop(window.drop_zone, [DEMO])
    assert window.drop_zone.path == DEMO
    assert window.start_button.isEnabled()


def test_drop_rejects_a_non_image(qtbot, window, bad_file):
    _drop(window.drop_zone, [bad_file])
    assert window.drop_zone.path is None


@pytest.mark.skipif(not HAVE_DEMO, reason="demo image missing")
def test_drop_rejects_multiple_files(qtbot, window):
    _drop(window.drop_zone, [DEMO, DEMO])
    assert window.drop_zone.path is None
    assert "single image" in window.status_bar.text()


def test_drop_zone_emits_rejection_reason(qtbot, bad_file):
    zone = DropZone()
    qtbot.addWidget(zone)
    with qtbot.waitSignal(zone.imageRejected, timeout=2000) as blocker:
        zone.accept_path(bad_file)
    assert "unsupported" in blocker.args[0]


# --- start button gating -----------------------------------------------

def test_start_disabled_without_an_image(window):
    assert not window.start_button.isEnabled()
    assert window.start_investigation() is False


@pytest.mark.skipif(not HAVE_DEMO, reason="demo image missing")
def test_start_is_refused_while_already_running(window, monkeypatch):
    window.drop_zone.accept_path(DEMO)
    window._running = True
    assert window.start_investigation() is False


# --- stage transitions come from real events ---------------------------

def test_stages_advance_only_on_events(qtbot):
    stages = StageList()
    qtbot.addWidget(stages)
    assert stages.state_of("01") is StageState.PENDING

    stages.advance("01")
    assert stages.state_of("01") is StageState.RUNNING
    assert stages.state_of("02") is StageState.PENDING

    stages.advance("02")
    assert stages.state_of("01") is StageState.DONE
    assert stages.state_of("02") is StageState.RUNNING


def test_failure_marks_the_running_stage(qtbot):
    stages = StageList()
    qtbot.addWidget(stages)
    stages.advance("01")
    stages.advance("02")
    stages.mark_failed()
    assert stages.state_of("02") is StageState.FAILED
    assert stages.state_of("01") is StageState.DONE


def test_unreached_stages_are_skipped_on_success(qtbot):
    """--no-chain finishes at 05; 06 and 07 must read as skipped, not done."""
    stages = StageList()
    qtbot.addWidget(stages)
    for number in ("01", "02", "03", "04", "05"):
        stages.advance(number)
    stages.finish(0)
    assert stages.state_of("05") is StageState.DONE
    assert stages.state_of("06") is StageState.SKIPPED
    assert stages.state_of("07") is StageState.SKIPPED


def test_unreached_stages_are_not_reached_on_failure(qtbot):
    stages = StageList()
    qtbot.addWidget(stages)
    stages.advance("01")
    stages.advance("02")
    stages.finish(4)
    assert stages.state_of("02") is StageState.FAILED
    assert stages.state_of("03") is StageState.NOT_REACHED


def test_no_stage_completes_without_an_event(qtbot):
    """The core anti-fake-progress guarantee."""
    stages = StageList()
    qtbot.addWidget(stages)
    qtbot.wait(120)
    assert set(stages.states().values()) == {StageState.PENDING}


# --- reporter -> GUI ----------------------------------------------------

def test_reporter_events_reach_the_widgets(qtbot, window):
    reporter = QtReporter()
    reporter.stageStarted.connect(window._on_stage)
    reporter.okReceived.connect(lambda m: window.log_panel.append_event("ok", m))
    reporter.failReceived.connect(window._on_fail)

    reporter.stage("01", "FACE SCAN")
    reporter.ok("1 face detected")
    assert window.stage_list.state_of("01") is StageState.RUNNING
    assert ("ok", "1 face detected", "01") in window.log_panel.records

    reporter.fail("NO_FACE")
    assert window.stage_list.state_of("01") is StageState.FAILED


def test_log_preserves_order_severity_and_stage(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.set_stage("02", "WEB DISCOVERY")
    panel.append_event("ok", "search completed")
    panel.append_event("warn", "6 skipped")
    panel.set_stage("03", "CANDIDATE RETRIEVAL")
    panel.append_event("info", "concurrency 5")

    assert panel.records == [
        ("stage", "02  WEB DISCOVERY", None),
        ("ok", "search completed", "02"),
        ("warn", "6 skipped", "02"),
        ("stage", "03  CANDIDATE RETRIEVAL", None),
        ("info", "concurrency 5", "03"),
    ]


def test_counts_are_expanded_into_the_log(qtbot):
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.append_counts({"RETRIEVED": 19, "INVALID_IMAGE": 6}, skip="RETRIEVED")
    assert len(panel.records) == 1
    assert "invalid image" in panel.records[0][1]
    assert "6" in panel.records[0][1]


def test_reporter_emits_exactly_what_it_was_given(qtbot):
    reporter = QtReporter()
    with qtbot.waitSignal(reporter.okReceived, timeout=2000) as blocker:
        reporter.ok("similarity: 0.9899")
    assert blocker.args == ["similarity: 0.9899"], "no reformatting, no invention"


# --- result panel -------------------------------------------------------

def _fake_chain_verification(verified: bool):
    class Check:
        chain_id = 80002
        contract_address = "0x9463c096472c67Fe85E931361904CDB6A6546b2E"
        on_chain_sha256 = "a" * 64

        class _S:
            value = "VERIFIED" if verified else "HASH_MISMATCH"
        status = _S()
    Check.verified = verified
    return Check()


def test_result_panel_shows_absent_values_honestly(qtbot):
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult(
        investigation_id="TRACE-20260903-ABCDEF",
        evidence_sha256="b" * 64,
        bundle_path="evidence/TRACE-20260903-ABCDEF",
        elapsed_seconds=12.3,
    ))
    assert panel.value_of("investigation") == "TRACE-20260903-ABCDEF"
    assert panel.value_of("transaction") == UNAVAILABLE
    assert panel.value_of("block") == UNAVAILABLE
    assert panel.value_of("verification") == UNAVAILABLE
    assert "EVIDENCE FINGERPRINT READY" in panel.headline()


def test_result_panel_never_shows_a_verified_badge_without_verification(qtbot):
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "c" * 64, "evidence/T", 1.0,
                                verification=_fake_chain_verification(False)))
    assert panel.headline() != "BLOCKCHAIN VERIFIED"
    assert panel.value_of("verification") == "HASH_MISMATCH"


def test_result_panel_shows_verified_only_when_the_pipeline_says_so(qtbot):
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "d" * 64, "evidence/T", 1.0,
                                verification=_fake_chain_verification(True)))
    assert panel.headline() == "BLOCKCHAIN VERIFIED"


def test_result_panel_renders_a_real_match_object(qtbot):
    from src.models import CandidateMatch, CandidateStatus, SearchCandidate

    candidate = SearchCandidate(
        url="https://news.example/story", title="t", source_domain="news.example",
        image_url=None, thumbnail_url=None, position=1, provider="google_lens")
    match = CandidateMatch(candidate=candidate, status=CandidateStatus.MATCH,
                           best_similarity=0.989948, threshold=0.30)

    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "e" * 64, "evidence/T", 2.0, match=match))
    assert panel.summary.similarity_text() == "0.9899"
    assert panel.summary.status_text() == "MATCH"
    assert panel.source.domain_text() == "news.example"
    assert panel.value_of("threshold") == "0.30"


def test_result_panel_has_no_hardcoded_values():
    """No similarity, domain, hash or tx may be baked into the source."""
    src = (ROOT / "gui" / "widgets" / "result_panel.py").read_text(encoding="utf-8")
    import re
    assert not re.search(r"0\.9\d{4}", src), "a similarity value is hardcoded"
    assert not re.search(r"0x[0-9a-fA-F]{40,}", src), "an address/tx is hardcoded"
    assert "wdayradionow" not in src and "polygonscan" not in src


# --- worker threading ---------------------------------------------------

@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_worker_runs_the_real_pipeline_off_the_ui_thread(qtbot):
    """The one end-to-end test: diagnostic + no-chain, so it is deterministic
    and spends nothing."""
    from PySide6.QtCore import QThread

    request = RunRequest(image=DEMO, mode="diagnostic", no_chain=True,
                         from_response=str(DEMO_FIXTURE), max_candidates=4)
    reporter = QtReporter()
    worker = PipelineWorker(request, reporter)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    stages: list[str] = []
    results: list[RunResult] = []
    reporter.stageStarted.connect(lambda n, t: stages.append(n))
    reporter.resultReceived.connect(results.append)

    ui_thread = threading.current_thread().name
    with qtbot.waitSignal(worker.finished, timeout=240_000) as blocker:
        thread.start()
    thread.quit()
    thread.wait(5000)

    assert blocker.args[0] == 0, blocker.args
    assert worker.thread_name is not None
    assert worker.thread_name != ui_thread, "pipeline ran on the UI thread"
    assert stages[:5] == ["01", "02", "03", "04", "05"]
    assert results, "no RunResult was emitted"

    result = results[0]
    assert result.investigation_id.startswith("TRACE-")
    assert len(result.evidence_sha256) == 64
    assert result.receipt is None, "--no-chain must not produce a receipt"
    assert result.verification is None
    assert result.verified is False


def test_worker_reports_a_failure_without_a_traceback(qtbot, tmp_path):
    request = RunRequest(image=tmp_path / "missing.jpg", mode="diagnostic", no_chain=True)
    worker = PipelineWorker(request, QtReporter())
    with qtbot.waitSignal(worker.finished, timeout=120_000) as blocker:
        worker.run()
    code, meaning = blocker.args
    assert code != 0
    assert "Traceback" not in meaning


def test_worker_never_enables_debug_tracebacks():
    ns = RunRequest(image=Path("x.jpg")).to_namespace(25)
    assert ns.debug is False, "the GUI must not surface raw tracebacks"
    assert ns.no_retrieval is False


def test_run_request_mirrors_the_cli_arguments():
    ns = RunRequest(image=Path("a.jpg"), mode="diagnostic", no_chain=True,
                    verbose=True, max_candidates=7).to_namespace(25)
    assert {"image", "mode", "from_response", "max_candidates",
            "verbose", "debug", "no_chain", "no_retrieval"} == set(vars(ns))
    assert ns.max_candidates == 7 and ns.mode == "diagnostic"


# --- cancellation -------------------------------------------------------

def test_cancel_raises_at_the_next_checkpoint():
    reporter = QtReporter()
    assert reporter.request_cancel() is True
    with pytest.raises(PipelineCancelled):
        reporter.ok("next event")


def test_cancel_is_refused_once_the_chain_stage_started(qtbot):
    """Never claim a run was stopped when a transaction may be in flight."""
    reporter = QtReporter()
    deferred: list[str] = []
    reporter.cancelDeferred.connect(deferred.append)

    reporter.stage("06", "BLOCKCHAIN")
    assert reporter.request_cancel() is False

    reporter._cancel.set()          # even if forced, no exception is raised
    reporter.ok("anchoring")
    assert deferred, "the refusal must be reported, not silently ignored"


def test_fail_events_never_trigger_cancellation():
    reporter = QtReporter()
    reporter.request_cancel()
    reporter.fail("something broke")     # must not raise; exit code wins


def test_cancelled_run_reports_its_own_exit_code(qtbot, tmp_path):
    request = RunRequest(image=tmp_path / "x.jpg", mode="diagnostic", no_chain=True)
    reporter = QtReporter()
    reporter.request_cancel()
    worker = PipelineWorker(request, reporter)
    with qtbot.waitSignal(worker.finished, timeout=120_000) as blocker:
        worker.run()
    assert blocker.args[0] == EXIT_CANCELLED


# --- architecture guards ------------------------------------------------

def test_gui_does_not_parse_stdout():
    for path in (ROOT / "gui").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "subprocess" not in src, f"{path.name} shells out"
        assert "stdout" not in src, f"{path.name} touches stdout"


def _imported_modules(path: Path) -> set[str]:
    """Modules actually imported, ignoring mentions in docs and comments."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def _calls_main_run(path: Path) -> bool:
    """A real ``main.run(...)`` call node, not a docstring mention."""
    import ast

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "main"):
            return True
    return False


def test_gui_does_not_orchestrate_the_pipeline():
    """No widget may drive a pipeline stage itself.

    ``src.vision.quality`` is deliberately allowed: the drop zone validates
    with the pipeline's own loader so the GUI cannot disagree with it about
    what a usable image is. That is reuse of a validator, not orchestration.
    """
    banned = (
        "src.vision.embedder", "src.vision.detector",
        "src.discovery.google_lens", "src.discovery.retrieval",
        "src.discovery.normalizer", "src.matching.ranker",
        "src.matching.similarity", "src.evidence.manifest",
        "src.evidence.collector", "src.evidence.hashing",
        "src.blockchain.client", "src.blockchain.verifier",
    )
    for path in (ROOT / "gui").rglob("*.py"):
        imported = _imported_modules(path)
        for module in banned:
            assert module not in imported, f"{path.name} imports {module}"


def test_only_the_worker_invokes_main_run():
    hits = sorted(p.name for p in (ROOT / "gui").rglob("*.py") if _calls_main_run(p))
    assert hits == ["worker.py"], f"main.run() called from {hits}"


def test_worker_uses_the_reporter_seam():
    src = (ROOT / "gui" / "worker.py").read_text(encoding="utf-8")
    assert "pipeline.use_reporter" in src


def test_gui_contains_no_sleep_or_fake_timer():
    for path in (ROOT / "gui").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "time.sleep" not in src, f"{path.name} sleeps"
        assert "QTimer" not in src, f"{path.name} uses a timer to imply progress"


# --- secrets -------------------------------------------------------------

def test_no_secret_reaches_gui_signals(qtbot, window):
    """Everything shown comes from reporter strings the CLI already vets."""
    from dotenv import dotenv_values

    env = dotenv_values(ROOT / ".env")
    reporter = QtReporter()
    reporter.okReceived.connect(lambda m: window.log_panel.append_event("ok", m))
    reporter.ok("Polygon Amoy (chain id 80002)")
    reporter.ok("wallet balance sufficient")

    blob = "\n".join(r[1] for r in window.log_panel.records)
    for key in ("SERPAPI_KEY", "PRIVATE_KEY"):
        value = env.get(key)
        if value:
            assert value not in blob
            assert value.removeprefix("0x") not in blob


def test_gui_source_contains_no_credentials():
    for path in (ROOT / "gui").rglob("*.py"):
        src = path.read_text(encoding="utf-8").lower()
        for banned in ("private_key", "serpapi_key", "api_key", "alchemy"):
            assert banned not in src, f"{path.name} references {banned}"


def test_gui_never_reads_env_directly():
    """Configuration stays with src.config; the GUI must not reimplement it."""
    for path in (ROOT / "gui").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "dotenv" not in src
        assert "os.environ" not in src


# --- the CLI is untouched -------------------------------------------------

def test_console_reporter_is_still_the_default_after_gui_import():
    """Importing the GUI must not install a reporter globally."""
    assert isinstance(pipeline.active_reporter(), pipeline.ConsoleReporter)


# --- 7.C presentation guarantees ---------------------------------------

def test_similarity_is_never_shown_as_a_percentage(qtbot):
    """Spec rule: a cosine similarity is not a probability. 0.9534, not 95.34%."""
    from src.models import CandidateMatch, CandidateStatus, SearchCandidate

    candidate = SearchCandidate(url="https://a.example/x", title="t",
                                source_domain="a.example", image_url=None,
                                thumbnail_url=None, position=1, provider="google_lens")
    match = CandidateMatch(candidate=candidate, status=CandidateStatus.MATCH,
                           best_similarity=0.9534, threshold=0.30)
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "f" * 64, "evidence/T", 1.0, match=match))

    shown = panel.summary.similarity_text()
    assert "%" not in shown
    assert shown == "0.9534"


def test_no_gui_source_presents_similarity_as_a_probability():
    """Only affirmative framings are banned; the source may say what it is NOT."""
    banned = ("% likely", "% probability", "probability this", "probability that",
              "match confidence", "confidence score", "% match", "% similar")
    for path in (ROOT / "gui").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in text, f"{path.name} frames similarity as {phrase!r}"


def test_similarity_labels_say_cosine_not_score():
    """The caption must name the measure, so nobody reads it as a percentage."""
    src = (ROOT / "gui" / "widgets" / "result_panel.py").read_text(encoding="utf-8")
    assert "COSINE SIMILARITY" in src
    assert "not a probability" in src


def test_stage_durations_are_measured_not_predicted(qtbot):
    """A stage has no duration until it has actually started and ended."""
    from gui.widgets.stage_list import StageList

    stages = StageList()
    qtbot.addWidget(stages)
    assert stages.duration_of("01") is None

    stages.advance("01")
    assert stages.duration_of("01") is None, "still running: no duration yet"

    qtbot.wait(30)
    stages.advance("02")
    measured = stages.duration_of("01")
    assert measured is not None and measured > 0
    assert stages.duration_of("02") is None


def test_unreached_stages_never_get_a_duration(qtbot):
    from gui.widgets.stage_list import StageList

    stages = StageList()
    qtbot.addWidget(stages)
    stages.advance("01")
    stages.finish(0)
    for number in ("02", "03", "04", "05", "06", "07"):
        assert stages.duration_of(number) is None


def test_stage_counter_reflects_completed_stages_only(qtbot):
    from gui.widgets.stage_list import StageBoard

    board = StageBoard()
    qtbot.addWidget(board)
    assert board.counter_text() == "0 / 7"

    board.list.advance("01")
    board.refresh_counter()
    assert board.counter_text() == "0 / 7", "running is not completed"

    board.list.advance("02")
    board.refresh_counter()
    assert board.counter_text() == "1 / 7"


def test_source_card_uses_the_real_retrieved_image(qtbot):
    """The thumbnail is the bytes the pipeline downloaded, not a stock image."""
    from src.models import (
        CandidateMatch, CandidateResult, CandidateStatus, SearchCandidate,
    )

    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :] = (10, 200, 90)
    encoded = cv2.imencode(".png", img)[1].tobytes()

    candidate = SearchCandidate(url="https://a.example/x", title="t",
                                source_domain="a.example", image_url=None,
                                thumbnail_url=None, position=3, provider="google_lens")
    retrieval = CandidateResult(candidate=candidate)
    retrieval.status = CandidateStatus.RETRIEVED
    retrieval.content = encoded
    retrieval.bytes_downloaded = len(encoded)
    match = CandidateMatch(candidate=candidate, status=CandidateStatus.MATCH,
                           best_similarity=0.88, threshold=0.30, retrieval=retrieval)

    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "a" * 64, "evidence/T", 1.0, match=match))
    assert panel.source.has_thumbnail(), "the matched image should be displayed"


def test_source_card_has_no_thumbnail_without_retrieved_bytes(qtbot):
    from src.models import CandidateMatch, CandidateStatus, SearchCandidate

    candidate = SearchCandidate(url="https://a.example/x", title="t",
                                source_domain="a.example", image_url=None,
                                thumbnail_url=None, position=1, provider="google_lens")
    match = CandidateMatch(candidate=candidate, status=CandidateStatus.MATCH,
                           best_similarity=0.88, threshold=0.30)
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "a" * 64, "evidence/T", 1.0, match=match))
    assert not panel.source.has_thumbnail()


def test_identical_to_input_is_flagged_in_the_source_card(qtbot):
    from src.models import CandidateMatch, CandidateStatus, SearchCandidate

    candidate = SearchCandidate(url="https://a.example/x", title="t",
                                source_domain="a.example", image_url=None,
                                thumbnail_url=None, position=1, provider="google_lens")
    match = CandidateMatch(candidate=candidate, status=CandidateStatus.MATCH,
                           best_similarity=1.0, threshold=0.30,
                           identical_to_input=True)
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "a" * 64, "evidence/T", 1.0, match=match))
    assert "not independent corroboration" in panel.source._flag.text()


def test_technical_details_are_collapsed_by_default(qtbot):
    """Progressive disclosure: verdict first, hashes on request."""
    panel = ResultPanel()
    qtbot.addWidget(panel)
    assert panel.details.expanded is False
    panel.details.toggle()
    assert panel.details.expanded is True


def test_face_card_reports_only_what_the_pipeline_said(qtbot, window):
    from gui.reporter import QtReporter

    assert "awaiting" in window.face_card.headline()
    reporter = QtReporter()
    reporter.stageStarted.connect(window._on_stage)
    reporter.okReceived.connect(window._on_ok)

    reporter.stage("01", "FACE SCAN")
    reporter.ok("1 face detected (313x431px, confidence 0.917)")
    assert "1 face detected" in window.face_card.headline()
    assert "0.917" in window.face_card.headline()


def test_multiple_face_match_is_described_honestly(qtbot):
    from src.models import CandidateMatch, CandidateStatus, SearchCandidate

    candidate = SearchCandidate(url="https://a.example/x", title="t",
                                source_domain="a.example", image_url=None,
                                thumbnail_url=None, position=1, provider="google_lens")
    match = CandidateMatch(candidate=candidate,
                           status=CandidateStatus.MULTIPLE_FACE_MATCH,
                           best_similarity=0.79, threshold=0.30, faces_detected=3)
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "a" * 64, "evidence/T", 1.0, match=match))
    note = panel.summary._status_note.text()
    assert "one of 3 faces" in note
    assert "the image as a whole does not" in note


def test_investigation_id_comes_from_the_result_not_the_gui(qtbot, window):
    """The header shows the pipeline's id, never one the GUI made up."""
    assert window.investigation_label.text() == "\u2014"
    window._on_result(RunResult("TRACE-20260903-ABCDEF", "a" * 64, "evidence/x", 1.0))
    assert window.investigation_label.text() == "TRACE-20260903-ABCDEF"


def test_no_image_editing_controls_exist():
    """The searched bytes must be the supplied bytes: the input SHA-256 is
    what the evidence anchors, so no crop/rotate/filter/enhance is offered."""
    banned = ("crop", "rotate", "brightness", "saturation", "enhance", "filter")
    for path in (ROOT / "gui").rglob("*.py"):
        if path.name == "drop_zone.py":
            continue                      # its docstring explains the absence
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} offers image editing: {word}"


def test_hash_is_grouped_for_reading_but_copied_exactly(qtbot):
    """Grouping a digest into blocks is a display convention. The value the
    copy button yields, and the one tests read, must be the exact digest."""
    from gui.widgets.result_panel import DetailRow

    digest = "507b717a837dee6c27cdd96b1d82d2be92a07bc36855a3e654b885d8db37104d"
    row = DetailRow("evidence SHA-256")
    qtbot.addWidget(row)
    row.set_value(digest, copyable=True)

    assert row.text() == digest, "the exact value must be recoverable"
    assert row._value.text() != digest, "the display is grouped for reading"
    assert row._value.text().replace(" ", "") == digest, "grouping adds only spaces"


def test_short_values_are_not_regrouped(qtbot):
    from gui.widgets.result_panel import DetailRow

    row = DetailRow("block")
    qtbot.addWidget(row)
    row.set_value("46599932")
    assert row._value.text() == "46599932"


def test_bundle_paths_are_not_treated_as_digests(qtbot):
    from gui.widgets.result_panel import DetailRow

    row = DetailRow("evidence bundle")
    qtbot.addWidget(row)
    row.set_value("evidence/TRACE-20260903-DAF282")
    assert row._value.text() == "evidence/TRACE-20260903-DAF282"


def test_long_values_wrap_instead_of_widening_the_column(qtbot):
    """A 64-character hash has no spaces to wrap at; without heightForWidth it
    would be clipped rather than wrapped."""
    from gui.widgets.result_panel import DetailRow

    row = DetailRow("evidence SHA-256")
    qtbot.addWidget(row)
    row.set_value("a" * 64)
    policy = row._value.sizePolicy()
    assert policy.hasHeightForWidth()
    assert row._value.wordWrap()
    assert row._value.maximumWidth() < 300


# --- independent sources list -------------------------------------------

def _group(url, domain, similarity, position, duplicates=0, faces=1,
           size=(600, 400), status=None):
    from src.models import (CandidateMatch, CandidateResult, CandidateStatus,
                            MatchGroup, SearchCandidate)

    candidate = SearchCandidate(
        url=url, title="t", source_domain=domain, image_url=None,
        thumbnail_url=None, position=position, provider="google_lens")
    retrieval = CandidateResult(candidate=candidate,
                                status=CandidateStatus.RETRIEVED)
    retrieval.bytes_downloaded = 4096
    match = CandidateMatch(
        candidate=candidate,
        status=status or CandidateStatus.MATCH,
        best_similarity=similarity, threshold=0.30, faces_detected=faces,
        best_face_index=0, image_size=size, retrieval=retrieval)
    return MatchGroup(
        representative=match, key=f"domain:{domain}",
        duplicates=[(match, "same source domain")] * duplicates)


def _panel_with(qtbot, groups):
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "e" * 64, "evidence/T", 2.0,
                                match=groups[0].representative if groups else None,
                                ranked=groups))
    return panel


LONG_URL = ("https://www.hellomagazine.com/celebrities/736617/"
            "meet-a-surprisingly-normal-family-of-three-children/")


def test_every_listed_source_url_is_complete(qtbot):
    """A truncated URL reads as a whole one - none may be elided."""
    groups = [_group(LONG_URL, "hellomagazine.com", 0.83, 1)]
    panel = _panel_with(qtbot, groups)
    assert panel.sources.urls_shown() == [LONG_URL]
    assert "…" not in panel.sources.urls_shown()[0]


def test_each_source_is_openable_and_emits_its_own_url(qtbot):
    groups = [_group(f"https://s{i}.example/page", f"s{i}.example", 0.9 - i / 100, i)
              for i in range(1, 4)]
    panel = _panel_with(qtbot, groups)

    opened = []
    panel.openSourceRequested.connect(opened.append)
    assert len(panel.sources.openable_urls()) == 3

    panel.sources._rows[2].click_open()
    assert opened == ["https://s3.example/page"]


def test_a_source_without_a_url_cannot_be_opened(qtbot):
    panel = _panel_with(qtbot, [_group("", "unknown", 0.9, 1)])
    assert panel.sources.openable_urls() == []


def test_view_all_expands_to_every_source_and_back(qtbot):
    groups = [_group(f"https://s{i}.example/p", f"s{i}.example", 0.9, i)
              for i in range(1, 13)]
    panel = _panel_with(qtbot, groups)
    card = panel.sources

    assert card.source_count() == 12
    assert card.rows_shown() == card.MAX_SHOWN
    assert not card.is_expanded()

    card.toggle()
    assert card.is_expanded()
    assert card.rows_shown() == 12
    assert len(card.openable_urls()) == 12

    card.toggle()
    assert card.rows_shown() == card.MAX_SHOWN


def test_toggle_appears_only_when_sources_are_hidden(qtbot):
    """isVisibleTo, not isVisible: offscreen every widget is invisible, so
    isVisible() would pass this test even if the toggle were always shown."""
    few = _panel_with(qtbot, [_group("https://a.example/p", "a.example", 0.9, 1)])
    assert few.sources.rows_shown() == 1
    assert not few.sources._toggle.isVisibleTo(few.sources)

    many = _panel_with(qtbot, [
        _group(f"https://s{i}.example/p", f"s{i}.example", 0.9, i)
        for i in range(1, 13)])
    assert many.sources._toggle.isVisibleTo(many.sources)
    assert "View all 12" in many.sources._toggle.text()
    many.sources.toggle()
    assert "Show top 5" in many.sources._toggle.text()


def test_rank_figures_come_from_the_match_object(qtbot):
    groups = [_group("https://a.example/p", "a.example", 0.912345, 7,
                     duplicates=2, faces=3, size=(1080, 1080))]
    panel = _panel_with(qtbot, groups)
    row = panel.sources._rows[0]

    assert row.rank_text() == "#1"
    assert row.score_text() == "0.912345"
    figures = row.figures()
    assert "face 1 of 3" in figures
    assert "threshold 0.30" in figures
    assert "1080 × 1080 px" in figures
    assert "search position 7" in figures
    assert "+2 duplicates grouped" in figures


def test_absent_runner_up_is_shown_as_absent_not_zero(qtbot):
    panel = _panel_with(qtbot, [_group("https://a.example/p", "a.example", 0.9, 1)])
    figures = panel.sources._rows[0].figures()
    assert "runner-up —" in figures
    assert "runner-up 0.0" not in figures


def test_sources_list_is_empty_without_a_ranked_set(qtbot):
    panel = ResultPanel()
    qtbot.addWidget(panel)
    panel.show_result(RunResult("T", "e" * 64, "evidence/T", 2.0))
    assert panel.sources.source_count() == 0
    assert panel.sources.rows_shown() == 0


def test_sources_reset_between_runs(qtbot):
    panel = _panel_with(qtbot, [_group("https://a.example/p", "a.example", 0.9, 1)])
    assert panel.sources.rows_shown() == 1
    panel.reset()
    assert panel.sources.rows_shown() == 0
    assert panel.sources.source_count() == 0


def test_source_figures_never_present_similarity_as_a_percentage(qtbot):
    groups = [_group("https://a.example/p", "a.example", 0.912345, 1)]
    panel = _panel_with(qtbot, groups)
    row = panel.sources._rows[0]
    assert "%" not in row.score_text()
    assert "%" not in row.figures()
