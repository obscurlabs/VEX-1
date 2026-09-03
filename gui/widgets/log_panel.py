"""Live event log.

Shows reporter events in arrival order with their severity and the stage they
belong to. The text comes verbatim from the pipeline - the panel adds only the
stage tag it already knows from the preceding ``stage()`` event.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget

SEVERITY_MARKER = {
    "ok": "✓",
    "warn": "!",
    "fail": "✗",
    "info": " ",
    "detail": " ",
    "stage": "▸",
    "verdict": "=",
}


class LogPanel(QPlainTextEdit):
    """Append-only view of what the pipeline reported."""

    MAX_BLOCKS = 2000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogPanel")
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._stage: str | None = None
        self._records: list[tuple[str, str, str | None]] = []

    # -- inspection (used by tests) -------------------------------------

    @property
    def records(self) -> list[tuple[str, str, str | None]]:
        """(severity, message, stage) in arrival order."""
        return list(self._records)

    def clear_log(self) -> None:
        self._stage = None
        self._records.clear()
        self.clear()

    # -- events ---------------------------------------------------------

    def set_stage(self, number: str, title: str) -> None:
        self._stage = number
        self.append_event("stage", f"{number}  {title}")

    def append_event(self, severity: str, message: str) -> None:
        stage = None if severity == "stage" else self._stage
        self._records.append((severity, message, stage))

        marker = SEVERITY_MARKER.get(severity, " ")
        prefix = f"[{stage}] " if stage else "     "
        self.appendPlainText(f"{prefix}{marker} {message}")
        self.moveCursor(QTextCursor.MoveOperation.End)

    def append_counts(self, mapping: dict[str, int], skip: str | None = None) -> None:
        for key, value in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0])):
            if key == skip:
                continue
            self.append_event("detail", f"{key:<22} {value}")
