"""The window. Wires widgets to reporter signals; holds no pipeline logic.

Every visible change originates from a signal the pipeline emitted, or from
the exit code ``main.run()`` returned. Nothing is inferred, timed or faked.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.pipeline import RunResult

from .reporter import QtReporter
from .theme import STYLESHEET
from .widgets.drop_zone import DropZone
from .widgets.log_panel import LogPanel
from .widgets.result_panel import ResultPanel
from .widgets.stage_list import StageList
from .worker import EXIT_CANCELLED, PipelineWorker, RunRequest


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


class MainWindow(QWidget):
    """Single-window shell around the existing pipeline."""

    runFinished = Signal(int)          # for tests: exit code of the last run

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VEX-1 · Face Evidence Pipeline")
        self.setStyleSheet(STYLESHEET)
        self.resize(1180, 760)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._reporter: QtReporter | None = None
        self._image: Path | None = None
        self._running = False

        self._build()
        self._update_controls()

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        title = QLabel("VEX-1 · FACE EVIDENCE PIPELINE")
        title.setObjectName("Title")
        subtitle = QLabel(
            "Face scan → web discovery → matching → evidence → Polygon Amoy → verification")
        subtitle.setObjectName("Subtitle")

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(title)
        header.addWidget(subtitle)

        # -- left: input and controls
        self.drop_zone = DropZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        self.drop_zone.imageRejected.connect(self._on_image_rejected)

        self.start_button = QPushButton("Start Investigation")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_investigation)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self.request_stop)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.stop_button, 1)

        self.diagnostic_check = QCheckBox("Diagnostic mode (replay a saved response)")
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
        self.verbose_check.setToolTip("Per-candidate detail instead of aggregate counts.")

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(_section("INPUT"))
        left.addWidget(self.drop_zone)
        left.addLayout(buttons)
        left.addWidget(_section("RUN OPTIONS"))
        left.addWidget(self.diagnostic_check)
        left.addWidget(self.no_chain_check)
        left.addWidget(self.verbose_check)
        left.addStretch(1)

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_card.setLayout(left)
        left_card.setFixedWidth(360)
        left.setContentsMargins(16, 14, 16, 14)

        # -- middle: stages and log
        self.stage_list = StageList()
        self.log_panel = LogPanel()
        self.log_panel.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)

        middle = QVBoxLayout()
        middle.setContentsMargins(16, 14, 16, 14)
        middle.setSpacing(10)
        middle.addWidget(_section("PIPELINE"))
        middle.addWidget(self.stage_list)
        middle.addWidget(_section("EVENTS"))
        middle.addWidget(self.log_panel, 1)

        middle_card = QFrame()
        middle_card.setObjectName("Card")
        middle_card.setLayout(middle)

        # -- right: result
        self.result_panel = ResultPanel()
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(_section("RESULT"))
        right.addWidget(self.result_panel, 1)

        right_holder = QWidget()
        right_holder.setLayout(right)
        right_holder.setFixedWidth(400)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        columns.addWidget(left_card)
        columns.addWidget(middle_card, 1)
        columns.addWidget(right_holder)

        self.status_bar = QLabel("Select an image to begin.")
        self.status_bar.setObjectName("StatusBar")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(12)
        root.addLayout(header)
        root.addLayout(columns, 1)
        root.addWidget(self.status_bar)

    # -- input ----------------------------------------------------------

    def _on_image_selected(self, path: Path) -> None:
        self._image = path
        self.status_bar.setText(f"Ready: {path.name}")
        self._update_controls()

    def _on_image_rejected(self, message: str) -> None:
        self._image = None
        self.status_bar.setText(f"Rejected: {message}")
        self._update_controls()

    def _update_controls(self) -> None:
        self.start_button.setEnabled(self._image is not None and not self._running)
        self.stop_button.setEnabled(self._running)
        self.drop_zone.set_enabled(not self._running)
        for box in (self.diagnostic_check, self.no_chain_check, self.verbose_check):
            box.setEnabled(not self._running)

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
        self._update_controls()
        self.stage_list.reset()
        self.log_panel.clear_log()
        self.result_panel.reset()
        self.status_bar.setText("Running…")

        self._reporter = QtReporter()
        self._reporter.stageStarted.connect(self._on_stage)
        self._reporter.okReceived.connect(lambda m: self.log_panel.append_event("ok", m))
        self._reporter.warnReceived.connect(lambda m: self.log_panel.append_event("warn", m))
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
            self.status_bar.setText("Stopping at the next checkpoint…")
            self.log_panel.append_event("warn", "stop requested")
        else:
            self._on_cancel_deferred("blockchain stage in progress - cannot stop safely")

    # -- reporter slots (main thread) ------------------------------------

    def _on_stage(self, number: str, title: str) -> None:
        self.stage_list.advance(number)
        self.log_panel.set_stage(number, title)
        self.status_bar.setText(f"[{number}] {title}")

    def _on_fail(self, message: str) -> None:
        self.stage_list.mark_failed()
        self.log_panel.append_event("fail", message)

    def _on_result(self, result: RunResult) -> None:
        self.result_panel.show_result(result)

    def _on_cancel_deferred(self, reason: str) -> None:
        self.log_panel.append_event("warn", f"stop not applied: {reason}")
        self.status_bar.setText(f"Cannot stop: {reason}")

    # -- worker slots ----------------------------------------------------

    def _on_worker_failed(self, message: str, detail: str) -> None:
        self.stage_list.mark_failed()
        self.log_panel.append_event("fail", message)
        self.result_panel.show_failure(message, detail)

    def _on_cancelled(self) -> None:
        self.log_panel.append_event("warn", "run stopped by user")
        self.result_panel.show_failure("Stopped by user")

    def _on_finished(self, code: int, meaning: str) -> None:
        self.stage_list.finish(code)
        self._running = False
        self._update_controls()

        if code == 0:
            self.status_bar.setText("Completed.")
        elif code == EXIT_CANCELLED:
            self.status_bar.setText("Stopped by user.")
        else:
            self.status_bar.setText(f"Failed (exit {code}): {meaning}")
            if self.result_panel.headline() in ("Investigation running…",):
                self.result_panel.show_failure(f"Failed: {meaning}")

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

    # -- lifecycle -------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._running and self._reporter is not None:
            self._reporter.request_cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)
