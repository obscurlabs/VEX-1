"""Runs the real pipeline on a worker thread.

The worker owns no pipeline logic. It builds the same ``argparse.Namespace``
the CLI builds, installs a :class:`~gui.reporter.QtReporter` for the duration
of the run, and calls ``main.run()``. Everything the GUI displays arrives
through that reporter or through the exit code ``main.run()`` returns.
"""
from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from src import pipeline
from src.pipeline import (
    EXIT_CHAIN,
    EXIT_EVIDENCE,
    EXIT_INPUT,
    EXIT_NO_MATCH,
    EXIT_OK,
    EXIT_SEARCH,
    EXIT_SEARCH_AUTH,
    EXIT_SEARCH_LIMIT,
    EXIT_VERIFY,
)

from .reporter import PipelineCancelled, QtReporter

#: Exit code used when the user stopped the run. Distinct from every pipeline
#: code so a stop is never mistaken for a pipeline failure.
EXIT_CANCELLED = 130

EXIT_MEANING = {
    EXIT_OK: "completed",
    EXIT_INPUT: "the image could not be read, or no usable face was found",
    EXIT_SEARCH_AUTH: "SerpAPI rejected the API key",
    EXIT_SEARCH_LIMIT: "SerpAPI quota or rate limit reached",
    EXIT_SEARCH: "the reverse image search failed",
    EXIT_NO_MATCH: "the pipeline ran, but no candidate matched",
    EXIT_EVIDENCE: "the evidence bundle or its fingerprint failed a check",
    EXIT_CHAIN: "a blockchain step failed (RPC, wallet, or transaction)",
    EXIT_VERIFY: "the on-chain fingerprint did not match the local one",
    EXIT_CANCELLED: "stopped by the user",
}


@dataclass(frozen=True)
class RunRequest:
    """What the user asked for. Mirrors the CLI's arguments exactly."""

    image: Path
    mode: str = "live"                 # "live" | "diagnostic"
    no_chain: bool = False
    verbose: bool = False
    from_response: str | None = None
    max_candidates: int | None = None

    def to_namespace(self, default_max_candidates: int) -> argparse.Namespace:
        return argparse.Namespace(
            image=str(self.image),
            mode=self.mode,
            from_response=self.from_response,
            max_candidates=self.max_candidates or default_max_candidates,
            verbose=self.verbose,
            debug=False,          # the GUI never surfaces raw tracebacks
            no_chain=self.no_chain,
            no_retrieval=False,
        )


class PipelineWorker(QObject):
    """Executes one pipeline run. Move to a QThread and call :meth:`run`."""

    started = Signal()
    finished = Signal(int, str)        # exit code, human-readable meaning
    failed = Signal(str, str)          # short message, detail
    cancelled = Signal()

    def __init__(self, request: RunRequest, reporter: QtReporter) -> None:
        super().__init__()
        self._request = request
        self._reporter = reporter
        self._thread_name: str | None = None

    @property
    def thread_name(self) -> str | None:
        """Name of the thread the pipeline actually executed on."""
        return self._thread_name

    def run(self) -> None:
        """Slot: invoked once the owning QThread has started."""
        self._thread_name = threading.current_thread().name
        self.started.emit()

        # Imported here so constructing a worker never drags the model stack
        # into the UI thread.
        import main
        from src.config import CONFIG

        args = self._request.to_namespace(CONFIG.retrieval.max_candidates)

        try:
            # The reporter is thread-local, so this affects only this thread.
            with pipeline.use_reporter(self._reporter):
                code = main.run(args)
        except PipelineCancelled:
            self.cancelled.emit()
            self.finished.emit(EXIT_CANCELLED, EXIT_MEANING[EXIT_CANCELLED])
            return
        except Exception as exc:                      # noqa: BLE001
            # Mirrors the CLI boundary in main.main(): summarise, never dump a
            # traceback into the interface.
            self.failed.emit(
                f"unexpected error: {type(exc).__name__}",
                str(exc),
            )
            self.finished.emit(1, f"unexpected error: {type(exc).__name__}")
            return

        self.finished.emit(code, EXIT_MEANING.get(code, f"exit code {code}"))
