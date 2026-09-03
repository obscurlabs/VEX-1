"""Qt bridge for the Phase 7.A reporter seam.

``QtReporter`` implements :class:`src.pipeline.Reporter` and re-emits every
event as a Qt signal. It computes nothing: each signal carries exactly the
state ``main.run()`` handed it, so the GUI cannot show progress, a score, a
transaction or a verdict the pipeline did not actually produce.

Signals are emitted from the worker thread. Qt delivers them to slots in the
receiving object's thread via queued connections, which is the supported way
to cross threads.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from src.pipeline import Reporter, RunResult


class PipelineCancelled(Exception):
    """Raised inside a reporter callback to stop a run at the next event.

    Deliberately not a subclass of any exception ``main.run()`` catches, so it
    propagates out to the worker instead of being swallowed as a pipeline
    failure.
    """


class QtReporter(QObject, Reporter):
    """Turns pipeline events into Qt signals."""

    headerReceived = Signal(str, object)        # title, subtitle|None
    lineReceived = Signal(str)
    stageStarted = Signal(str, str)             # number, title
    okReceived = Signal(str)
    warnReceived = Signal(str)
    failReceived = Signal(str)
    infoReceived = Signal(str)
    detailReceived = Signal(str)
    countsReceived = Signal(dict, object)       # mapping, skip|None
    verdictReceived = Signal(str, bool)
    summaryReceived = Signal(list, str)
    resultReceived = Signal(object)             # src.pipeline.RunResult

    # Emitted when a stop was requested while the chain stage was in flight
    # and therefore could not be honoured.
    cancelDeferred = Signal(str)

    #: Stages during which cancellation is refused: a transaction may already
    #: be broadcast, and abandoning the run would misrepresent what happened.
    UNCANCELLABLE_STAGES = ("06", "07")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._current_stage: str | None = None
        self._chain_reached = False

    # -- cancellation ---------------------------------------------------

    def request_cancel(self) -> bool:
        """Ask the run to stop. Returns False if it is too late to honour."""
        with self._lock:
            if self._chain_reached:
                return False
        self._cancel.set()
        return True

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def _checkpoint(self) -> None:
        """Abort between events if a stop was requested and is still safe.

        This is cooperative: it cannot interrupt an HTTP request or an ONNX
        inference already in flight, only the gap between reported events.
        """
        if not self._cancel.is_set():
            return
        with self._lock:
            if self._chain_reached:
                self.cancelDeferred.emit(
                    "blockchain stage in progress - cannot stop safely"
                )
                return
        raise PipelineCancelled("stopped at the next checkpoint")

    # -- Reporter -------------------------------------------------------

    def header(self, title: str, subtitle: str | None = None) -> None:
        self.headerReceived.emit(title, subtitle)

    def line(self, text: str = "") -> None:
        self.lineReceived.emit(text)

    def stage(self, number: str, title: str) -> None:
        with self._lock:
            self._current_stage = number
            if number in self.UNCANCELLABLE_STAGES:
                self._chain_reached = True
        self.stageStarted.emit(number, title)
        self._checkpoint()

    def ok(self, msg: str) -> None:
        self.okReceived.emit(msg)
        self._checkpoint()

    def warn(self, msg: str) -> None:
        self.warnReceived.emit(msg)
        self._checkpoint()

    def fail(self, msg: str) -> None:
        # Never abort on a failure event; the pipeline is already unwinding
        # and its own exit code must be the one that surfaces.
        self.failReceived.emit(msg)

    def info(self, msg: str) -> None:
        self.infoReceived.emit(msg)
        self._checkpoint()

    def detail(self, msg: str) -> None:
        self.detailReceived.emit(msg)

    def counts(self, mapping: dict[str, int], skip: str | None = None) -> None:
        self.countsReceived.emit(dict(mapping), skip)

    def verdict(self, text: str, good: bool = True) -> None:
        self.verdictReceived.emit(text, good)

    def summary(self, rows: list[tuple[str, str]], title: str = "SUMMARY") -> None:
        self.summaryReceived.emit([tuple(r) for r in rows], title)

    def result(self, result: RunResult) -> None:
        self.resultReceived.emit(result)
