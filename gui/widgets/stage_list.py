"""The seven pipeline stages and their real state.

A row changes only when the pipeline reports it. There is no timer, no
percentage and no animation: ``advance()`` is driven by ``stage()`` events and
``finish()`` by the exit code.

When a run ends, stages that were never reached are reported honestly:
SKIPPED when the run succeeded without them (``--no-chain``), NOT REACHED when
the run stopped early because something failed.
"""
from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

#: The stages main.run() reports, in order.
STAGES: tuple[tuple[str, str], ...] = (
    ("01", "FACE SCAN"),
    ("02", "WEB DISCOVERY"),
    ("03", "CANDIDATE RETRIEVAL"),
    ("04", "FACE MATCHING"),
    ("05", "EVIDENCE"),
    ("06", "BLOCKCHAIN"),
    ("07", "VERIFICATION"),
)


class StageState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_REACHED = "NOT REACHED"

    @property
    def marker(self) -> str:
        return {
            StageState.PENDING: "·",
            StageState.RUNNING: "▸",
            StageState.DONE: "✓",
            StageState.FAILED: "✗",
            StageState.SKIPPED: "–",
            StageState.NOT_REACHED: "–",
        }[self]


class StageRow(QWidget):
    def __init__(self, number: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.number = number
        self.title = title
        self._state = StageState.PENDING

        self._marker = QLabel(StageState.PENDING.marker)
        self._marker.setObjectName("StageMarker")
        self._marker.setFixedWidth(18)
        self._marker.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(f"{number}  {title}")
        self._label.setObjectName("StageLabel")

        self._status = QLabel("")
        self._status.setObjectName("StageStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)
        layout.addWidget(self._marker)
        layout.addWidget(self._label, 1)
        layout.addWidget(self._status)
        self._apply()

    @property
    def state(self) -> StageState:
        return self._state

    def set_state(self, state: StageState) -> None:
        self._state = state
        self._apply()

    def _apply(self) -> None:
        self._marker.setText(self._state.marker)
        self._status.setText("" if self._state is StageState.PENDING else self._state.value)
        for widget in (self, self._marker, self._label, self._status):
            widget.setProperty("state", self._state.value)
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class StageList(QWidget):
    """Seven rows, driven entirely by real reporter events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, StageRow] = {}
        self._order: list[str] = [n for n, _ in STAGES]
        self._current: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for number, title in STAGES:
            row = StageRow(number, title)
            self._rows[number] = row
            layout.addWidget(row)
        layout.addStretch(1)

    # -- queries --------------------------------------------------------

    def state_of(self, number: str) -> StageState:
        return self._rows[number].state

    def states(self) -> dict[str, StageState]:
        return {n: r.state for n, r in self._rows.items()}

    @property
    def current(self) -> str | None:
        return self._current

    # -- driven by the pipeline -----------------------------------------

    def reset(self) -> None:
        self._current = None
        for row in self._rows.values():
            row.set_state(StageState.PENDING)

    def advance(self, number: str) -> None:
        """A stage reported that it started."""
        if number not in self._rows:
            return
        # Whatever was running completed, because the pipeline moved on.
        if self._current and self._rows[self._current].state is StageState.RUNNING:
            self._rows[self._current].set_state(StageState.DONE)
        self._current = number
        self._rows[number].set_state(StageState.RUNNING)

    def mark_failed(self) -> None:
        """The pipeline reported a failure inside the running stage."""
        if self._current:
            self._rows[self._current].set_state(StageState.FAILED)

    def finish(self, exit_code: int) -> None:
        """Settle every row once the run is over."""
        succeeded = exit_code == 0
        if self._current:
            row = self._rows[self._current]
            if row.state is StageState.RUNNING:
                row.set_state(StageState.DONE if succeeded else StageState.FAILED)

        started = self._order.index(self._current) if self._current else -1
        for index, number in enumerate(self._order):
            if index <= started:
                continue
            # Never started: deliberately not run, or never got there.
            self._rows[number].set_state(
                StageState.SKIPPED if succeeded else StageState.NOT_REACHED
            )
