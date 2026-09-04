"""The window. Wires widgets to reporter signals; holds no pipeline logic.

Every visible change originates from a signal the pipeline emitted, or from
the exit code ``main.run()`` returned. Nothing is inferred, timed or faked:
there is no widget in this file that advances on its own.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.config import CONFIG
from src.pipeline import RunResult

from .reporter import QtReporter
from .theme import STYLESHEET
from .widgets.drop_zone import DropZone, FaceCard
from .widgets.log_panel import LogPanel
from .widgets.result_panel import ResultPanel
from .widgets.stage_list import StageBoard
from .worker import EXIT_CANCELLED, PipelineWorker, RunRequest

#: Events that describe the input face, surfaced next to the image itself.
FACE_EVENT_HINTS = ("face detected", "ArcFace embedding")


def _restyle(*widgets: QWidget) -> None:
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _column_header(index: str, title: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName("ColumnHead")
    number = QLabel(index)
    number.setObjectName("ColumnIndex")
    label = QLabel(title)
    label.setObjectName("ColumnTitle")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(8)
    layout.addWidget(number)
    layout.addWidget(label)
    layout.addStretch(1)
    return frame


def _column(index: str, title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Column")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(_column_header(index, title))
    body = QVBoxLayout()
    body.setContentsMargins(14, 12, 14, 14)
    body.setSpacing(11)
    outer.addLayout(body, 1)
    return frame, body


class MainWindow(QWidget):
    """Single-window shell around the existing pipeline."""

    runFinished = Signal(int)          # for tests: exit code of the last run

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VEX-1 · Face Evidence Pipeline")
        self.setStyleSheet(STYLESHEET)
        self.resize(1500, 940)
        self.setMinimumSize(1180, 720)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._reporter: QtReporter | None = None
        self._image: Path | None = None
        self._running = False
        self._last_result: RunResult | None = None

        self._build()
        self._update_controls()

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        columns = QHBoxLayout()
        columns.setContentsMargins(14, 14, 14, 12)
        columns.setSpacing(12)
        columns.addWidget(self._build_input_column())
        columns.addWidget(self._build_pipeline_column(), 1)
        columns.addWidget(self._build_result_column())
        root.addLayout(columns, 1)
        root.addWidget(self._build_footer())

    def _build_header(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("HeaderBar")

        wordmark = QLabel("VEX-1")
        wordmark.setObjectName("Wordmark")

        title = QLabel("FACE EVIDENCE PIPELINE")
        title.setObjectName("AppTitle")
        flow = QLabel("Face scan → Web discovery → Matching → Evidence → "
                      "Polygon Amoy → Verification")
        flow.setObjectName("AppFlow")
        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(1)
        titles.addWidget(title)
        titles.addWidget(flow)

        self.run_state = QLabel("● IDLE")
        self.run_state.setObjectName("RunState")
        self.run_state.setProperty("state", "idle")
        self.chain_label = QLabel(
            f"{CONFIG.chain.network_name} · chain {CONFIG.chain.expected_chain_id}")
        self.chain_label.setObjectName("ChainLabel")
        state_box = QVBoxLayout()
        state_box.setContentsMargins(0, 0, 0, 0)
        state_box.setSpacing(2)
        state_box.addWidget(self.run_state, 0, Qt.AlignmentFlag.AlignRight)
        state_box.addWidget(self.chain_label, 0, Qt.AlignmentFlag.AlignRight)

        id_card = QFrame()
        id_card.setObjectName("IdCard")
        caption = QLabel("INVESTIGATION ID")
        caption.setObjectName("IdCaption")
        self.investigation_label = QLabel("—")
        self.investigation_label.setObjectName("IdValue")
        self.investigation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        id_layout = QVBoxLayout(id_card)
        id_layout.setContentsMargins(13, 8, 13, 8)
        id_layout.setSpacing(1)
        id_layout.addWidget(caption)
        id_layout.addWidget(self.investigation_label)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(18)
        layout.addWidget(wordmark)
        layout.addLayout(titles)
        layout.addStretch(1)
        layout.addLayout(state_box)
        layout.addWidget(id_card)
        return bar

    def _build_input_column(self) -> QFrame:
        column, body = _column("1", "INPUT")
        column.setFixedWidth(320)

        self.drop_zone = DropZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        self.drop_zone.imageRejected.connect(self._on_image_rejected)
        self.drop_zone.imageCleared.connect(self._on_image_cleared)

        self.clear_image_button = QPushButton("Clear image")
        self.clear_image_button.setObjectName("Ghost")
        self.clear_image_button.setEnabled(False)
        self.clear_image_button.setToolTip(
            "Forget the selected image. The file itself is untouched and no "
            "existing evidence is affected.")
        self.clear_image_button.clicked.connect(self.drop_zone.clear)

        clear_row = QHBoxLayout()
        clear_row.setContentsMargins(0, 0, 0, 0)
        clear_row.addStretch(1)
        clear_row.addWidget(self.clear_image_button)

        self.face_card = FaceCard()

        self.start_button = QPushButton("▶  START INVESTIGATION")
        self.start_button.setObjectName("Primary")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.clicked.connect(self.start_investigation)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self.request_stop)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.start_button, 3)
        buttons.addWidget(self.stop_button, 1)

        options_caption = QLabel("RUN OPTIONS")
        options_caption.setObjectName("PanelTitle")

        self.diagnostic_check = QCheckBox("Diagnostic mode")
        self.diagnostic_check.setToolTip(
            "Replays cache/search-response.json instead of querying Google Lens. "
            "No SerpAPI credit is spent. The cached response belongs to whichever "
            "image produced it, so matching a different image will honestly "
            "report no match. Development only; never used in live mode.")
        self.no_chain_check = QCheckBox("Skip blockchain anchoring")
        self.no_chain_check.setToolTip(
            "Stop after the evidence fingerprint. No transaction is sent and no "
            "gas is spent; stages 06 and 07 are reported as skipped.")
        self.verbose_check = QCheckBox("Verbose events")
        self.verbose_check.setToolTip(
            "Per-candidate detail instead of aggregate counts.")

        body.addWidget(self.drop_zone)
        body.addLayout(clear_row)
        body.addWidget(self.face_card)
        body.addLayout(buttons)
        body.addSpacing(2)
        body.addWidget(options_caption)
        body.addWidget(self.diagnostic_check)
        body.addWidget(self.no_chain_check)
        body.addWidget(self.verbose_check)
        body.addStretch(1)
        return column

    def _build_pipeline_column(self) -> QFrame:
        column, body = _column("2", "PIPELINE")

        self.stage_board = StageBoard()
        self.stage_list = self.stage_board.list

        events_head = QHBoxLayout()
        events_head.setContentsMargins(0, 0, 0, 0)
        events_caption = QLabel("LIVE EVENTS")
        events_caption.setObjectName("PanelTitle")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("Ghost")
        self.clear_button.clicked.connect(self._on_clear_events)
        events_head.addWidget(events_caption)
        events_head.addStretch(1)
        events_head.addWidget(self.clear_button)

        self.log_panel = LogPanel()
        self.log_panel.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)
        self.log_panel.setMinimumHeight(190)

        body.addWidget(self.stage_board)
        body.addSpacing(2)
        body.addLayout(events_head)
        body.addWidget(self.log_panel, 1)
        return column

    def _build_result_column(self) -> QFrame:
        column, body = _column("3", "RESULTS")
        column.setFixedWidth(478)

        self.result_panel = ResultPanel()
        self.result_panel.openSourceRequested.connect(self._open_url)

        scroll = QScrollArea()
        scroll.setWidget(self.result_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.open_bundle_button = QPushButton("Open evidence folder")
        self.open_bundle_button.setObjectName("Ghost")
        self.open_bundle_button.setEnabled(False)
        self.open_bundle_button.clicked.connect(self._open_bundle)

        body.addWidget(scroll, 1)
        body.addWidget(self.open_bundle_button)
        return column

    def _build_footer(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("FooterBar")

        def pair(key: str, value: str) -> tuple[QLabel, QLabel]:
            k = QLabel(key)
            k.setObjectName("FooterKey")
            v = QLabel(value)
            v.setObjectName("FooterVal")
            return k, v

        status_key, self.status_bar = pair("Status", "Select an image to begin.")
        mode_key, self.mode_label = pair("Mode", "Live")
        net_key, net_value = pair(
            "Network", f"{CONFIG.chain.network_name} ({CONFIG.chain.expected_chain_id})")
        version = QLabel(f"vex {CONFIG.pipeline_version}")
        version.setObjectName("FooterKey")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(8)
        for widget in (status_key, self.status_bar):
            layout.addWidget(widget)
        layout.addSpacing(24)
        for widget in (mode_key, self.mode_label):
            layout.addWidget(widget)
        layout.addSpacing(24)
        for widget in (net_key, net_value):
            layout.addWidget(widget)
        layout.addStretch(1)
        layout.addWidget(version)
        return bar

    # -- input ----------------------------------------------------------

    def _on_image_selected(self, path: Path) -> None:
        self._image = path
        self._set_status(f"Ready: {path.name}")
        self.face_card.reset()
        self._update_controls()

    def _on_image_rejected(self, message: str) -> None:
        self._image = None
        self._set_status(f"Rejected: {message}", "failed")
        self._update_controls()

    def _on_image_cleared(self) -> None:
        self._image = None
        self._set_status("No image selected")
        # The face card describes the cleared image's scan, so it goes too.
        # A completed run's results stay: clearing the input does not retract
        # evidence that was already produced and anchored.
        self.face_card.reset()
        self._update_controls()

    def _on_clear_events(self) -> None:
        self.log_panel.clear_log()

    def _update_controls(self) -> None:
        self.start_button.setEnabled(self._image is not None and not self._running)
        self.stop_button.setEnabled(self._running)
        self.clear_image_button.setEnabled(
            self.drop_zone.path is not None and not self._running)
        self.drop_zone.set_enabled(not self._running)
        for box in (self.diagnostic_check, self.no_chain_check, self.verbose_check):
            box.setEnabled(not self._running)
        self.mode_label.setText(
            "Diagnostic" if self.diagnostic_check.isChecked() else "Live")

    def _set_status(self, text: str, state: str = "") -> None:
        self.status_bar.setText(text)
        self.status_bar.setProperty("state", state)
        _restyle(self.status_bar)

    def _set_run_state(self, text: str, state: str) -> None:
        self.run_state.setText(text)
        self.run_state.setProperty("state", state)
        _restyle(self.run_state)

    # -- run ------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def build_request(self) -> RunRequest:
        return RunRequest(
            image=self._image,
            mode="diagnostic" if self.diagnostic_check.isChecked() else "live",
            no_chain=self.no_chain_check.isChecked(),
            verbose=self.verbose_check.isChecked(),
        )

    def start_investigation(self) -> bool:
        """Start one run. Returns False if it could not be started."""
        if self._running or self._image is None:
            return False

        self._running = True
        self._last_result = None
        self._update_controls()
        self.stage_list.reset()
        self.stage_board.refresh_counter()
        self.log_panel.clear_log()
        self.result_panel.reset()
        self.face_card.reset()
        self.investigation_label.setText("IN PROGRESS")
        self.open_bundle_button.setEnabled(False)
        self._set_run_state("● RUNNING", "running")
        self._set_status("Running…", "running")

        self._reporter = QtReporter()
        self._reporter.stageStarted.connect(self._on_stage)
        self._reporter.okReceived.connect(self._on_ok)
        self._reporter.warnReceived.connect(self._on_warn)
        self._reporter.failReceived.connect(self._on_fail)
        self._reporter.infoReceived.connect(lambda m: self.log_panel.append_event("info", m))
        self._reporter.detailReceived.connect(lambda m: self.log_panel.append_event("detail", m))
        self._reporter.countsReceived.connect(self.log_panel.append_counts)
        self._reporter.verdictReceived.connect(
            lambda text, good: self.log_panel.append_event("verdict", text))
        self._reporter.resultReceived.connect(self._on_result)
        self._reporter.cancelDeferred.connect(self._on_cancel_deferred)

        self._worker = PipelineWorker(self.build_request(), self._reporter)
        self._thread = QThread(self)
        self._thread.setObjectName("pipeline-worker")
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.start()
        return True

    def request_stop(self) -> None:
        """Cooperative stop. Honest when it cannot be honoured."""
        if not self._running or self._reporter is None:
            return
        if self._reporter.request_cancel():
            self._set_status("Stopping at the next checkpoint…", "running")
            self.log_panel.append_event("warn", "stop requested")
        else:
            self._on_cancel_deferred("blockchain stage in progress - cannot stop safely")

    # -- reporter slots (main thread) ------------------------------------

    def _on_stage(self, number: str, title: str) -> None:
        self.stage_list.advance(number)
        self.stage_board.refresh_counter()
        self.log_panel.set_stage(number, title)
        self._set_status(f"[{number}] {title}", "running")

    def _on_ok(self, message: str) -> None:
        self.log_panel.append_event("ok", message)
        self.stage_list.note(message)
        self.stage_board.refresh_counter()
        # Surface what the face scan found next to the image it scanned.
        if (self.stage_list.current == "01"
                and any(hint in message for hint in FACE_EVENT_HINTS)):
            if "face detected" in message:
                self.face_card.report(message)
            else:
                self.face_card.report(self.face_card.headline(), message)

    def _on_warn(self, message: str) -> None:
        self.log_panel.append_event("warn", message)
        self.stage_list.note(message)

    def _on_fail(self, message: str) -> None:
        self.stage_list.mark_failed()
        self.stage_list.note(message)
        self.log_panel.append_event("fail", message)

    def _on_result(self, result: RunResult) -> None:
        self._last_result = result
        self.investigation_label.setText(result.investigation_id)
        self.result_panel.show_result(result)
        self.open_bundle_button.setEnabled(bool(result.bundle_path))

    def _on_cancel_deferred(self, reason: str) -> None:
        self.log_panel.append_event("warn", f"stop not applied: {reason}")
        self._set_status(f"Cannot stop: {reason}", "running")

    # -- worker slots ----------------------------------------------------

    def _on_worker_failed(self, message: str, detail: str) -> None:
        self.stage_list.mark_failed()
        self.log_panel.append_event("fail", message)
        self.result_panel.show_failure(message, detail)

    def _on_cancelled(self) -> None:
        self.log_panel.append_event("warn", "run stopped by user")
        self.result_panel.show_failure("STOPPED BY USER",
                                       "The run was stopped before it completed.")

    def _on_finished(self, code: int, meaning: str) -> None:
        self.stage_list.finish(code)
        self.stage_board.refresh_counter()
        self._running = False
        self._update_controls()

        if code == 0:
            self._set_run_state("● COMPLETE", "done")
            self._set_status("Completed.", "done")
        elif code == EXIT_CANCELLED:
            self._set_run_state("● STOPPED", "idle")
            self._set_status("Stopped by user.", "")
        else:
            self._set_run_state("● FAILED", "failed")
            self._set_status(f"Failed (exit {code}): {meaning}", "failed")
            if self.result_panel.headline() == "INVESTIGATION RUNNING":
                self.result_panel.show_failure(f"FAILED · EXIT {code}", meaning)

        if self._last_result is None:
            self.investigation_label.setText("—")

        self._teardown_thread()
        self.runFinished.emit(code)

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._reporter = None

    # -- actions ----------------------------------------------------------

    def _open_url(self, url: str) -> None:
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _open_bundle(self) -> None:
        if self._last_result is None:
            return
        path = CONFIG.project_root / self._last_result.bundle_path
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # -- lifecycle -------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._running and self._reporter is not None:
            self._reporter.request_cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)
