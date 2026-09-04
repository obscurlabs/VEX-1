"""Result presentation, populated from the real ``RunResult``.

Every value is read off the object ``main.run()`` emitted. When a field is
absent - no transaction was broadcast, the chain step was skipped - the panel
says so explicitly rather than leaving a blank that could read as a value.

On similarity: it is shown as a cosine value, never as a percentage or a
probability. 0.9534 is a distance between two embeddings; it is not "95%
likely to be this person", and presenting it that way would misstate what the
pipeline measured.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.pipeline import RunResult

#: Shown wherever the pipeline produced no value.
UNAVAILABLE = "not performed"
NONE_YET = "—"

DETAIL_FIELDS = (
    "investigation",
    "evidence SHA-256",
    "evidence bundle",
    "network",
    "contract",
    "transaction",
    "block",
    "on-chain hash",
    "verification",
    "faces in candidate",
    "runner-up",
    "threshold",
    "elapsed",
)


def _wrapping(widget: QWidget, maximum: int | None = None) -> None:
    """Let a label wrap inside its column instead of widening it.

    QLabel only wraps when the layout consults heightForWidth, which requires
    the size policy to advertise it. Without this a 64-character hash is
    simply clipped.
    """
    policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)
    widget.setMinimumWidth(1)
    if maximum is not None:
        widget.setMaximumWidth(maximum)


def _restyle(*widgets: QWidget) -> None:
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(13, 11, 13, 12)
    layout.setSpacing(9)
    caption = QLabel(title)
    caption.setObjectName("PanelTitle")
    layout.addWidget(caption)
    return frame, layout


class VerdictBanner(QFrame):
    """The headline outcome. Neutral until the pipeline decides."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Verdict")

        self._mark = QLabel("○")
        self._mark.setObjectName("VerdictMark")
        self._mark.setFixedWidth(38)
        self._mark.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._title = QLabel("NO INVESTIGATION YET")
        self._title.setObjectName("VerdictTitle")
        self._sub = QLabel("Select an image and start an investigation.")
        self._sub.setObjectName("VerdictSub")
        self._sub.setWordWrap(True)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        text.addWidget(self._title)
        text.addWidget(self._sub)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self._mark)
        layout.addLayout(text, 1)

    def show_state(self, state: str, mark: str, title: str, subtitle: str) -> None:
        self._mark.setText(mark)
        self._title.setText(title)
        self._sub.setText(subtitle)
        for widget in (self, self._title):
            widget.setProperty("state", state)
        _restyle(self, self._title)

    def title(self) -> str:
        return self._title.text()

    def subtitle(self) -> str:
        return self._sub.text()


class SimilarityScale(QWidget):
    """Where the measured similarity sits, with the threshold marked.

    The bar is a rendering of one number the pipeline produced against one
    configured threshold. It is not a confidence, a probability or a score out
    of anything.
    """

    RANGE = 1.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(6)
        self._track = QFrame(self)
        self._track.setObjectName("ScaleTrack")
        self._fill = QFrame(self)
        self._fill.setObjectName("ScaleFill")
        self._threshold = QFrame(self)
        self._threshold.setObjectName("ScaleThreshold")
        self._value: float | None = None
        self._threshold_value: float | None = None
        self._fill.setVisible(False)
        self._threshold.setVisible(False)

    def set_values(self, similarity: float | None, threshold: float | None) -> None:
        self._value = similarity
        self._threshold_value = threshold
        matched = (similarity is not None and threshold is not None
                   and similarity >= threshold)
        self._fill.setProperty("state", "match" if matched else "reject")
        _restyle(self._fill)
        self._fill.setVisible(similarity is not None)
        self._threshold.setVisible(threshold is not None)
        self._relayout()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        width = max(self.width(), 1)
        self._track.setGeometry(0, 0, width, 6)
        if self._value is not None:
            span = max(0.0, min(self._value / self.RANGE, 1.0))
            self._fill.setGeometry(0, 0, int(width * span), 6)
        if self._threshold_value is not None:
            at = max(0.0, min(self._threshold_value / self.RANGE, 1.0))
            self._threshold.setGeometry(int(width * at), -2, 2, 10)


class MatchSummary(QFrame):
    """Similarity and decision, side by side."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        caption = QLabel("MATCH SUMMARY")
        caption.setObjectName("PanelTitle")

        sim_caption = QLabel("COSINE SIMILARITY")
        sim_caption.setObjectName("MetricCaption")
        self._similarity = QLabel(NONE_YET)
        self._similarity.setObjectName("MetricValue")
        self._scale = SimilarityScale()
        self._scale_note = QLabel("")
        self._scale_note.setObjectName("MetricNote")

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        left.addWidget(sim_caption)
        left.addWidget(self._similarity)
        left.addWidget(self._scale)
        left.addWidget(self._scale_note)

        status_caption = QLabel("DECISION")
        status_caption.setObjectName("MetricCaption")
        self._status = QLabel(NONE_YET)
        self._status.setObjectName("MetricValue")
        self._status.setStyleSheet("font-size: 17px;")
        self._status_note = QLabel("")
        self._status_note.setObjectName("MetricNote")
        self._status_note.setWordWrap(True)
        for widget in (self._status, self._status_note):
            _wrapping(widget)
        self._similarity.setMinimumWidth(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        right.addWidget(status_caption)
        right.addWidget(self._status)
        right.addWidget(self._status_note)
        right.addStretch(1)

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(14)
        columns.addLayout(left, 5)
        columns.addLayout(right, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 11, 13, 12)
        layout.setSpacing(9)
        layout.addWidget(caption)
        layout.addLayout(columns)

    def reset(self) -> None:
        self._similarity.setText(NONE_YET)
        self._similarity.setProperty("state", "")
        self._status.setText(NONE_YET)
        self._scale_note.setText("")
        self._status_note.setText("")
        self._scale.set_values(None, None)
        _restyle(self._similarity)

    def show_match(self, match) -> None:
        similarity = match.best_similarity
        threshold = match.threshold
        matched = bool(match.status.is_match)

        self._similarity.setText(
            f"{similarity:.4f}" if similarity is not None else NONE_YET)
        self._similarity.setProperty("state", "match" if matched else "reject")
        self._status.setText(match.status.value.replace("_", " "))
        self._status.setProperty("state", "match" if matched else "reject")

        if threshold is not None:
            self._scale_note.setText(
                f"threshold {threshold:.2f} · cosine distance, not a probability")
        self._scale.set_values(similarity, threshold)

        faces = match.faces_detected
        if match.status.value == "MULTIPLE_FACE_MATCH":
            self._status_note.setText(
                f"one of {faces} faces in this image matches the target; "
                "the image as a whole does not")
        elif matched:
            self._status_note.setText("same person found")
        else:
            self._status_note.setText("below the configured threshold")
        _restyle(self._similarity, self._status)

    def similarity_text(self) -> str:
        return self._similarity.text()

    def status_text(self) -> str:
        return self._status.text()


class SourceCard(QFrame):
    """Where the match was found, with the image that was actually matched."""

    openRequested = Signal(str)

    THUMB = 96

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._url = ""

        caption = QLabel("MATCHED SOURCE")
        caption.setObjectName("PanelTitle")

        self._thumb = QLabel("—")
        self._thumb.setObjectName("SourceThumb")
        self._thumb.setFixedSize(self.THUMB, self.THUMB)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setScaledContents(False)

        self._domain = QLabel(NONE_YET)
        self._domain.setObjectName("SourceDomain")
        self._title = QLabel("")
        self._title.setObjectName("SourceTitle")
        self._title.setWordWrap(True)
        self._url_label = QLabel("")
        self._url_label.setObjectName("SourceUrl")
        self._url_label.setWordWrap(True)
        self._meta = QLabel("")
        self._meta.setObjectName("SourceMeta")
        self._meta.setWordWrap(True)
        for label in (self._domain, self._title, self._url_label, self._meta):
            _wrapping(label)
        self._flag = QLabel("")
        self._flag.setObjectName("SourceFlag")
        self._flag.setWordWrap(True)
        self._flag.setVisible(False)

        self._open = QPushButton("Open ↗")
        self._open.setObjectName("IconButton")
        self._open.setEnabled(False)
        self._open.clicked.connect(lambda: self.openRequested.emit(self._url))

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(caption)
        head.addStretch(1)
        head.addWidget(self._open)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(3)
        text.addWidget(self._domain)
        text.addWidget(self._title)
        text.addWidget(self._url_label)
        text.addWidget(self._meta)
        text.addStretch(1)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(11)
        body.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(text, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 11, 13, 12)
        layout.setSpacing(9)
        layout.addLayout(head)
        layout.addLayout(body)
        layout.addWidget(self._flag)

    def reset(self) -> None:
        self._url = ""
        self._thumb.clear()
        self._thumb.setText("—")
        self._domain.setText(NONE_YET)
        self._title.setText("")
        self._url_label.setText("")
        self._meta.setText("")
        self._flag.setVisible(False)
        self._open.setEnabled(False)

    def show_match(self, match, retrieved_at: str = "") -> None:
        candidate = match.candidate
        self._url = candidate.url
        self._domain.setText(candidate.source_domain or NONE_YET)
        self._title.setText(candidate.title or "")
        self._url_label.setText(candidate.url)
        self._open.setEnabled(bool(candidate.url))

        # The bytes that were actually downloaded and matched.
        retrieval = match.retrieval
        if retrieval is not None and retrieval.content:
            pixmap = QPixmap()
            if pixmap.loadFromData(retrieval.content):
                self._thumb.setPixmap(pixmap.scaled(
                    self.THUMB, self.THUMB,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation))

        bits = []
        if match.image_size:
            bits.append(f"{match.image_size[0]} × {match.image_size[1]}")
        if retrieval is not None and retrieval.bytes_downloaded:
            bits.append(f"{retrieval.bytes_downloaded / 1024:,.0f} KB")
        bits.append(f"search position {candidate.position}")
        if retrieved_at:
            bits.append(f"retrieved {retrieved_at}")
        self._meta.setText("  ·  ".join(bits))

        if match.identical_to_input:
            self._flag.setText(
                "⚠ byte-identical to the input image — this locates the source "
                "file, it is not independent corroboration")
            self._flag.setVisible(True)

    def domain_text(self) -> str:
        return self._domain.text()

    def has_thumbnail(self) -> bool:
        pixmap = self._thumb.pixmap()
        return pixmap is not None and not pixmap.isNull()


class DetailRow(QWidget):
    """One technical field, with a copy button when there is something to copy."""

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = key
        self._raw = ""

        label = QLabel(key)
        label.setObjectName("DetailKey")
        label.setFixedWidth(96)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._value = QLabel(NONE_YET)
        self._value.setObjectName("DetailValue")
        self._value.setWordWrap(True)
        self._value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # A 64-character hash has no spaces to wrap at, so its natural width
        # would push the whole column wider than its scroll viewport. Capping
        # the width forces Qt to break it instead.
        _wrapping(self._value, 236)

        self._copy = QPushButton("⧉")
        self._copy.setObjectName("IconButton")
        self._copy.setFixedWidth(24)
        self._copy.setToolTip("Copy")
        self._copy.setVisible(False)
        self._copy.clicked.connect(self._on_copy)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(9)
        layout.addWidget(label)
        layout.addWidget(self._value, 1)
        layout.addWidget(self._copy, 0, Qt.AlignmentFlag.AlignTop)

    @staticmethod
    def _for_display(text: str) -> str:
        """Group a bare hex digest into 8-character blocks.

        Display only - :attr:`_raw` keeps the exact value, and that is what the
        copy button puts on the clipboard.
        """
        prefix, body = ("0x", text[2:]) if text.startswith("0x") else ("", text)
        if len(body) >= 40 and all(c in "0123456789abcdefABCDEF" for c in body):
            grouped = " ".join(body[i:i + 8] for i in range(0, len(body), 8))
            return prefix + grouped
        return text

    def _on_copy(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None and self._raw:
            clipboard.setText(self._raw)

    def text(self) -> str:
        """The exact value, as the pipeline produced it."""
        return self._raw or self._value.text()

    def set_value(self, text: str | None, state: str = "", copyable: bool = False) -> None:
        if text is None:
            self._raw = ""
            self._value.setText(UNAVAILABLE)
            self._value.setProperty("state", "absent")
            self._copy.setVisible(False)
        else:
            self._raw = text
            self._value.setText(self._for_display(text))
            self._value.setProperty("state", state)
            self._copy.setVisible(copyable and bool(text))
        _restyle(self._value)

    def reset(self) -> None:
        self._raw = ""
        self._value.setText(NONE_YET)
        self._value.setProperty("state", "")
        self._copy.setVisible(False)
        _restyle(self._value)


class DetailsPanel(QFrame):
    """Technical fields, collapsed by default.

    Progressive disclosure: a judge sees the verdict, the similarity and the
    source first. Hashes, addresses and block numbers are one click away for
    anyone who wants to check them.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._toggle = QPushButton("▸  TECHNICAL DETAILS")
        self._toggle.setObjectName("Disclosure")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.clicked.connect(self.toggle)

        self._body = QWidget()
        grid = QVBoxLayout(self._body)
        grid.setContentsMargins(0, 6, 0, 0)
        grid.setSpacing(3)

        self._rows: dict[str, DetailRow] = {}
        for field in DETAIL_FIELDS:
            row = DetailRow(field)
            self._rows[field] = row
            grid.addWidget(row)
        self._body.setVisible(False)
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 9, 13, 10)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    # -- disclosure -----------------------------------------------------

    @property
    def expanded(self) -> bool:
        """Tracked explicitly: isVisible() is False whenever no parent is shown."""
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._toggle.setText(("▾  " if expanded else "▸  ") + "TECHNICAL DETAILS")

    # -- values ---------------------------------------------------------

    def value_of(self, field: str) -> str:
        return self._rows[field].text()

    def reset(self) -> None:
        for row in self._rows.values():
            row.reset()

    def set_value(self, field: str, text: str | None,
                  state: str = "", copyable: bool = False) -> None:
        self._rows[field].set_value(text, state, copyable)


class SourceRow(QFrame):
    """One independent source: its rank, its measured figures, its link.

    Every figure is read off the CandidateMatch the pipeline produced. A value
    the pipeline did not establish is shown as an em dash, never as a zero or
    a guess.
    """

    openRequested = Signal(str)

    def __init__(self, position: int, group, first: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SourceRow")
        self.setProperty("first", "true" if first else "false")

        match = group.representative
        candidate = match.candidate
        self._url = candidate.url or ""
        self._position = position
        rejected = not match.is_match

        self._rank = QLabel(f"#{position}")
        self._rank.setObjectName("SourceRank")
        self._rank.setFixedWidth(28)

        similarity = match.best_similarity
        self._score = QLabel(
            f"{similarity:.6f}" if similarity is not None else NONE_YET)
        self._score.setObjectName("SourceScore")
        self._score.setProperty("state", "reject" if rejected else "")

        self._status = QLabel(match.status.value)
        self._status.setObjectName("SourceStatus")
        self._status.setProperty("state", "reject" if rejected else "")

        self._open = QPushButton("Open ↗")
        self._open.setObjectName("IconButton")
        self._open.setEnabled(bool(self._url))
        self._open.setToolTip(self._url or "no URL for this candidate")
        self._open.clicked.connect(lambda: self.openRequested.emit(self._url))

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(self._rank)
        head.addWidget(self._score)
        head.addWidget(self._status)
        head.addStretch(1)
        head.addWidget(self._open)

        self._domain = QLabel(candidate.source_domain or NONE_YET)
        self._domain.setObjectName("SourceTitle")
        self._domain.setWordWrap(True)
        _wrapping(self._domain)

        self._url_label = QLabel(self._url)
        self._url_label.setObjectName("SourceUrl")
        self._url_label.setWordWrap(True)
        self._url_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        _wrapping(self._url_label)

        self._figures = QLabel(self.figures_text(group, match))
        self._figures.setObjectName("SourceFigures")
        self._figures.setWordWrap(True)
        _wrapping(self._figures)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(head)
        layout.addWidget(self._domain)
        layout.addWidget(self._url_label)
        layout.addWidget(self._figures)

        _restyle(self, self._score, self._status)

    @staticmethod
    def figures_text(group, match) -> str:
        """The measured numbers behind this rank, in the pipeline's own terms."""
        bits: list[str] = []

        if match.faces_detected:
            index = match.best_face_index
            which = f"face {index + 1}" if index is not None else "best face"
            bits.append(f"{which} of {match.faces_detected}")
        if match.threshold is not None:
            bits.append(f"threshold {match.threshold:.2f}")
        bits.append(
            f"runner-up {match.runner_up_similarity:.6f}"
            if match.runner_up_similarity is not None else "runner-up —")
        if match.image_size:
            bits.append(f"{match.image_size[0]} × {match.image_size[1]} px")

        retrieval = match.retrieval
        if retrieval is not None and retrieval.bytes_downloaded:
            bits.append(f"{retrieval.bytes_downloaded / 1024:,.0f} KB")

        bits.append(f"search position {match.candidate.position}")
        if group.duplicates:
            n = len(group.duplicates)
            plural = "s" if n != 1 else ""
            bits.append(f"+{n} duplicate{plural} grouped")
        if match.identical_to_input:
            bits.append("identical to input")
        return "  ·  ".join(bits)

    # -- inspection (used by tests) -------------------------------------

    def url(self) -> str:
        return self._url

    def url_text(self) -> str:
        return self._url_label.text()

    def figures(self) -> str:
        return self._figures.text()

    def rank_text(self) -> str:
        return self._rank.text()

    def score_text(self) -> str:
        return self._score.text()

    def can_open(self) -> bool:
        return self._open.isEnabled()

    def click_open(self) -> None:
        self._open.click()


class SourcesCard(QFrame):
    """Every independent source the run found, with its full page URL.

    The anchored match is shown above by :class:`SourceCard`; this lists all
    of them so a reader can see how many sources corroborated, open any of
    them, and read the figures behind each rank. URLs are never elided - a
    truncated URL reads as a whole one, which is worse than a long line.
    """

    openRequested = Signal(str)

    #: Mirrors main.TOP_MATCHES so the collapsed GUI list matches the CLI.
    MAX_SHOWN = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._rows: list[SourceRow] = []
        self._groups: list = []
        self._expanded = False

        self._caption = QLabel("INDEPENDENT SOURCES")
        self._caption.setObjectName("PanelTitle")

        self._toggle = QPushButton("")
        self._toggle.setObjectName("IconButton")
        self._toggle.setVisible(False)
        self._toggle.clicked.connect(self.toggle)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._caption)
        head.addStretch(1)
        head.addWidget(self._toggle)

        self._empty = QLabel(NONE_YET)
        self._empty.setObjectName("SourceMeta")

        self._list = QVBoxLayout()
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(0)

        self._note = QLabel("")
        self._note.setObjectName("SourceMeta")
        self._note.setWordWrap(True)
        self._note.setVisible(False)
        _wrapping(self._note)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 11, 13, 12)
        layout.setSpacing(9)
        layout.addLayout(head)
        layout.addWidget(self._empty)
        layout.addLayout(self._list)
        layout.addWidget(self._note)

    # -- state ----------------------------------------------------------

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._list.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

    def reset(self) -> None:
        self._clear_rows()
        self._groups = []
        self._expanded = False
        self._caption.setText("INDEPENDENT SOURCES")
        self._empty.setText(NONE_YET)
        self._empty.setVisible(True)
        self._toggle.setVisible(False)
        self._note.setVisible(False)

    def show_sources(self, groups) -> None:
        """Render the ranked groups the pipeline produced. Never invents one."""
        self.reset()
        self._groups = list(groups or [])
        if not self._groups:
            return

        self._empty.setVisible(False)
        self._caption.setText(f"INDEPENDENT SOURCES ({len(self._groups)})")
        self._toggle.setVisible(len(self._groups) > self.MAX_SHOWN)
        self._render()

    def toggle(self) -> None:
        """Flip between the top few and every source found."""
        self._expanded = not self._expanded
        self._render()

    def _render(self) -> None:
        self._clear_rows()
        total = len(self._groups)
        shown = self._groups if self._expanded else self._groups[: self.MAX_SHOWN]

        for offset, group in enumerate(shown):
            row = SourceRow(offset + 1, group, first=(offset == 0))
            row.openRequested.connect(self.openRequested)
            self._list.addWidget(row)
            self._rows.append(row)

        hidden = total - len(shown)
        if hidden > 0:
            plural = "s" if hidden != 1 else ""
            self._note.setText(
                f"… {hidden} further independent source{plural} not shown "
                f"— all {total} are in matching.json and the manifest")
            self._note.setVisible(True)
            self._toggle.setText(f"View all {total} ↓")
        else:
            self._note.setVisible(False)
            self._toggle.setText(f"Show top {self.MAX_SHOWN} ↑")

    # -- inspection (used by tests) -------------------------------------

    def source_count(self) -> int:
        return len(self._groups)

    def rows_shown(self) -> int:
        return len(self._rows)

    def is_expanded(self) -> bool:
        return self._expanded

    def urls_shown(self) -> list[str]:
        """Exactly the URL strings on screen, so a test can prove none is cut."""
        return [row.url_text() for row in self._rows]

    def openable_urls(self) -> list[str]:
        return [row.url() for row in self._rows if row.can_open()]

    def figures_shown(self) -> list[str]:
        return [row.figures() for row in self._rows]


class ResultPanel(QWidget):
    """The whole results column."""

    openSourceRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.verdict = VerdictBanner()
        self.summary = MatchSummary()
        self.source = SourceCard()
        self.sources = SourcesCard()
        self.details = DetailsPanel()
        self.source.openRequested.connect(self.openSourceRequested)
        # Every listed source opens through the same window-level handler.
        self.sources.openRequested.connect(self.openSourceRequested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.verdict)
        layout.addWidget(self.summary)
        layout.addWidget(self.source)
        layout.addWidget(self.sources)
        layout.addWidget(self.details)
        layout.addStretch(1)

    # -- inspection (used by tests) -------------------------------------

    def headline(self) -> str:
        return self.verdict.title()

    def value_of(self, field: str) -> str:
        return self.details.value_of(field)

    # -- state ----------------------------------------------------------

    def reset(self, message: str = "INVESTIGATION RUNNING") -> None:
        self.verdict.show_state("running", "◈", message,
                                "Stages will complete as the pipeline reports them.")
        self.summary.reset()
        self.source.reset()
        self.sources.reset()
        self.details.reset()

    def clear(self) -> None:
        self.verdict.show_state("", "○", "NO INVESTIGATION YET",
                                "Select an image and start an investigation.")
        self.summary.reset()
        self.source.reset()
        self.sources.reset()
        self.details.reset()

    def show_failure(self, headline: str, detail: str = "") -> None:
        self.verdict.show_state("failed", "✕", headline,
                                detail or "The run did not complete.")

    def show_result(self, result: RunResult, retrieved_at: str = "") -> None:
        """Populate from a real RunResult. Absent values are labelled absent."""
        self.details.set_value("investigation", result.investigation_id, copyable=True)
        self.details.set_value("evidence SHA-256", result.evidence_sha256, copyable=True)
        self.details.set_value("evidence bundle", result.bundle_path, copyable=True)
        self.details.set_value("elapsed", f"{result.elapsed_seconds:.1f}s")

        # Every independent source, with full URLs. Empty list -> empty card.
        self.sources.show_sources(result.ranked)

        match = result.match
        if match is not None:
            self.summary.show_match(match)
            self.source.show_match(match, retrieved_at)
            self.details.set_value("faces in candidate", str(match.faces_detected))
            self.details.set_value(
                "runner-up",
                f"{match.runner_up_similarity:.4f}"
                if match.runner_up_similarity is not None else None)
            self.details.set_value(
                "threshold",
                f"{match.threshold:.2f}" if match.threshold is not None else None)

        receipt = result.receipt
        if receipt is not None:
            self.details.set_value("transaction", receipt.tx_hash, copyable=True)
            self.details.set_value("block", str(receipt.block_number))
        else:
            # Either --no-chain, or the investigation was already anchored.
            self.details.set_value("transaction", None)
            self.details.set_value("block", None)

        check = result.verification
        if check is not None:
            self.details.set_value(
                "network", f"Polygon Amoy · chain {check.chain_id}"
                if check.chain_id else None)
            self.details.set_value("contract", check.contract_address, copyable=True)
            self.details.set_value("on-chain hash", check.on_chain_sha256, copyable=True)
            self.details.set_value(
                "verification", check.status.value,
                state="good" if check.verified else "")
        else:
            for field in ("network", "contract", "on-chain hash", "verification"):
                self.details.set_value(field, None)

        if result.verified:
            self.verdict.show_state(
                "verified", "✓", "BLOCKCHAIN VERIFIED",
                "The local fingerprint matches the one anchored on Polygon Amoy.")
        elif check is not None:
            self.verdict.show_state(
                "failed", "✕", check.status.value.replace("_", " "),
                "The local fingerprint does not match the anchored record.")
        else:
            self.verdict.show_state(
                "partial", "◈", "EVIDENCE FINGERPRINT READY",
                "The bundle was built and verified locally. Not anchored on chain.")
