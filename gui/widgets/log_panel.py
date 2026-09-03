"""Live event feed.

Shows reporter events in arrival order with their severity and the stage they
belong to. Message text comes from the pipeline verbatim; the panel adds only
the arrival timestamp and the stage tag it already knows from the preceding
``stage()`` event.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

SEVERITY_DOT = {
    "ok": "●",
    "warn": "▲",
    "fail": "■",
    "info": "·",
    "detail": "·",
    "stage": "◆",
    "verdict": "★",
}


class EventRow(QWidget):
    def __init__(self, stamp: str, severity: str, text: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)

        dot = QLabel(SEVERITY_DOT.get(severity, "·"))
        dot.setObjectName("EventDot")
        dot.setProperty("severity", severity)
        dot.setFixedWidth(12)
        dot.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        time_label = QLabel(stamp)
        time_label.setObjectName("EventTime")
        time_label.setFixedWidth(58)
        time_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        message = QLabel(text)
        message.setObjectName("EventText")
        message.setProperty("severity", severity)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 1, 8, 1)
        layout.setSpacing(7)
        layout.addWidget(dot)
        layout.addWidget(time_label)
        layout.addWidget(message, 1)


class LogPanel(QFrame):
    """Append-only view of what the pipeline reported."""

    MAX_ROWS = 400

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EventArea")
        self._stage: str | None = None
        self._records: list[tuple[str, str, str | None]] = []

        self._canvas = QWidget()
        self._canvas.setObjectName("EventCanvas")
        self._rows = QVBoxLayout(self._canvas)
        self._rows.setContentsMargins(2, 6, 2, 6)
        self._rows.setSpacing(1)
        self._rows.addStretch(1)

        self._empty = QLabel("Events appear here once an investigation starts.")
        self._empty.setObjectName("EventEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rows.insertWidget(0, self._empty)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("EventScroll")
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

    # -- inspection (used by tests) -------------------------------------

    @property
    def records(self) -> list[tuple[str, str, str | None]]:
        """(severity, message, stage) in arrival order."""
        return list(self._records)

    def clear_log(self) -> None:
        self._stage = None
        self._records.clear()
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._empty = QLabel("Events appear here once an investigation starts.")
        self._empty.setObjectName("EventEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rows.insertWidget(0, self._empty)

    # -- events ---------------------------------------------------------

    def set_stage(self, number: str, title: str) -> None:
        self._stage = number
        self.append_event("stage", f"{number}  {title}")

    def append_event(self, severity: str, message: str) -> None:
        stage = None if severity == "stage" else self._stage
        self._records.append((severity, message, stage))

        if self._empty is not None:
            self._empty.deleteLater()
            self._empty = None

        stamp = datetime.now().strftime("%H:%M:%S")
        row = EventRow(stamp, severity, message)
        self._rows.insertWidget(self._rows.count() - 1, row)

        while self._rows.count() - 1 > self.MAX_ROWS:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def append_counts(self, mapping: dict[str, int], skip: str | None = None) -> None:
        for key, value in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0])):
            if key == skip:
                continue
            self.append_event("detail", f"{key.replace('_', ' ').lower()}: {value}")
