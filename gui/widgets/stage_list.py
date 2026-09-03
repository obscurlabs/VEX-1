"""The seven pipeline stages, drawn as a timeline.

A row changes only when the pipeline reports it. There is no timer driving
appearance, no percentage and no animation.

Durations are *measured*, not predicted: the clock starts when a stage reports
that it began and stops when the next stage reports, or when the run ends. A
stage that never started shows no duration at all.

When a run ends, stages that were never reached are labelled honestly:
SKIPPED when the run succeeded without them (``--no-chain``), NOT REACHED when
the run stopped early.
"""
from __future__ import annotations

import time
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

#: The stages main.run() reports, in order, with the glyph and the one-line
#: description of what that stage is doing while it runs.
STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("01", "FACE SCAN", "◎", "Detecting faces and generating the embedding"),
    ("02", "WEB DISCOVERY", "⌕", "Reverse image search via Google Lens"),
    ("03", "CANDIDATE RETRIEVAL", "▤", "Downloading candidate images"),
    ("04", "FACE MATCHING", "◉", "Comparing every candidate face"),
    ("05", "EVIDENCE", "❋", "Building the bundle and its fingerprint"),
    ("06", "BLOCKCHAIN", "⛓", "Anchoring on Polygon Amoy"),
    ("07", "VERIFICATION", "⛨", "Reading the anchor back and comparing"),
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


def _restyle(*widgets: QWidget) -> None:
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class StageRow(QWidget):
    """One stage: glyph, number, title, live note, badge and measured time."""

    def __init__(self, number: str, title: str, glyph: str, note: str,
                 last: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.number = number
        self.title = title
        self._glyph = glyph
        self._default_note = note
        self._state = StageState.PENDING

        self._icon = QLabel(glyph)
        self._icon.setObjectName("StageIcon")
        self._icon.setFixedSize(30, 30)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._connector = QFrame()
        self._connector.setObjectName("StageConnector")
        self._connector.setFixedWidth(1)
        self._connector.setVisible(not last)

        rail = QVBoxLayout()
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(0)
        rail.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignHCenter)
        rail.addWidget(self._connector, 1, Qt.AlignmentFlag.AlignHCenter)

        self._number = QLabel(number)
        self._number.setObjectName("StageNumber")
        self._title = QLabel(title)
        self._title.setObjectName("StageTitle")

        heading = QHBoxLayout()
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(9)
        heading.addWidget(self._number)
        heading.addWidget(self._title)
        heading.addStretch(1)

        self._note = QLabel(note)
        self._note.setObjectName("StageNote")
        self._note.setWordWrap(True)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        body.addLayout(heading)
        body.addWidget(self._note)

        self._badge = QLabel("")
        self._badge.setObjectName("StageBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._timing = QLabel("")
        self._timing.setObjectName("StageTiming")
        self._timing.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        trailing = QVBoxLayout()
        trailing.setContentsMargins(0, 0, 0, 0)
        trailing.setSpacing(2)
        trailing.addWidget(self._badge)
        trailing.addWidget(self._timing)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 5, 4, 5)
        layout.setSpacing(12)
        layout.addLayout(rail)
        layout.addLayout(body, 1)
        layout.addLayout(trailing)
        self._apply()

    @property
    def state(self) -> StageState:
        return self._state

    def set_state(self, state: StageState) -> None:
        self._state = state
        self._apply()

    def set_note(self, text: str) -> None:
        """Replace the note with something the pipeline actually reported."""
        self._note.setText(text)

    def reset_note(self) -> None:
        self._note.setText(self._default_note)

    def set_duration(self, seconds: float | None) -> None:
        self._timing.setText("" if seconds is None else f"{seconds:.1f}s")

    def _apply(self) -> None:
        icon = {
            StageState.PENDING: self._glyph,
            StageState.RUNNING: self._glyph,
            StageState.DONE: "✓",
            StageState.FAILED: "✗",
            StageState.SKIPPED: "–",
            StageState.NOT_REACHED: "–",
        }[self._state]
        self._icon.setText(icon)
        self._badge.setText("" if self._state is StageState.PENDING else self._state.value)
        for widget in (self._icon, self._number, self._title,
                       self._note, self._badge, self._connector):
            widget.setProperty("state", self._state.value)
        _restyle(self._icon, self._number, self._title,
                 self._note, self._badge, self._connector)


class StageList(QWidget):
    """Seven rows, driven entirely by real reporter events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, StageRow] = {}
        self._order: list[str] = [n for n, _, _, _ in STAGES]
        self._current: str | None = None
        self._started_at: dict[str, float] = {}
        self._durations: dict[str, float] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for index, (number, title, glyph, note) in enumerate(STAGES):
            row = StageRow(number, title, glyph, note, last=index == len(STAGES) - 1)
            self._rows[number] = row
            layout.addWidget(row)
        layout.addStretch(1)

    # -- queries --------------------------------------------------------

    def state_of(self, number: str) -> StageState:
        return self._rows[number].state

    def states(self) -> dict[str, StageState]:
        return {n: r.state for n, r in self._rows.items()}

    def duration_of(self, number: str) -> float | None:
        return self._durations.get(number)

    @property
    def current(self) -> str | None:
        return self._current

    # -- driven by the pipeline -----------------------------------------

    def reset(self) -> None:
        self._current = None
        self._started_at.clear()
        self._durations.clear()
        for row in self._rows.values():
            row.set_state(StageState.PENDING)
            row.reset_note()
            row.set_duration(None)

    def advance(self, number: str) -> None:
        """A stage reported that it started."""
        if number not in self._rows:
            return
        self._close_current(StageState.DONE)
        self._current = number
        self._started_at[number] = time.perf_counter()
        self._rows[number].set_state(StageState.RUNNING)

    def note(self, text: str) -> None:
        """Show the pipeline's own words under the running stage."""
        if self._current:
            self._rows[self._current].set_note(text)

    def mark_failed(self) -> None:
        if self._current:
            self._close_current(StageState.FAILED)
            self._rows[self._current].set_state(StageState.FAILED)

    def finish(self, exit_code: int) -> None:
        """Settle every row once the run is over."""
        succeeded = exit_code == 0
        if self._current and self._rows[self._current].state is StageState.RUNNING:
            self._close_current(StageState.DONE if succeeded else StageState.FAILED)

        started = self._order.index(self._current) if self._current else -1
        for index, number in enumerate(self._order):
            if index <= started:
                continue
            self._rows[number].set_state(
                StageState.SKIPPED if succeeded else StageState.NOT_REACHED)
            self._rows[number].set_note("not run" if succeeded else "not reached")

    def _close_current(self, state: StageState) -> None:
        """Stop the clock on the running stage and record what it measured."""
        if not self._current:
            return
        row = self._rows[self._current]
        if row.state is not StageState.RUNNING:
            return
        began = self._started_at.get(self._current)
        if began is not None:
            elapsed = time.perf_counter() - began
            self._durations[self._current] = elapsed
            row.set_duration(elapsed)
        row.set_state(state)


class StageBoard(QFrame):
    """The stage list plus a live counter of how many have completed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list = StageList()

        self._counter = QLabel("0 / 7")
        self._counter.setObjectName("StageTiming")

        head = QGridLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._counter, 0, 1, Qt.AlignmentFlag.AlignRight)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(head)
        layout.addWidget(self.list)

    def refresh_counter(self) -> None:
        done = sum(1 for s in self.list.states().values() if s is StageState.DONE)
        self._counter.setText(f"{done} / {len(STAGES)}")

    def counter_text(self) -> str:
        return self._counter.text()
