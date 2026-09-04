"""Image input: drag-and-drop, click-to-browse, and a preview.

Validation goes through the pipeline's own ``src.vision.quality.load_image``,
so the GUI accepts exactly what the pipeline accepts - no second opinion about
what counts as a usable image.

The image is never altered. There is no crop, rotate, filter or enhance step:
the input's SHA-256 is what the evidence bundle anchors, so the bytes that are
searched must be the bytes the operator supplied.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.models import ImageStatus
from src.vision.quality import load_image

ACCEPTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def inspect_image(path: Path) -> tuple[bool, str, tuple[int, int] | None]:
    """Validate with the pipeline's loader. Returns (ok, message, size)."""
    if path.suffix.lower() not in ACCEPTED_SUFFIXES:
        return False, f"unsupported file type: {path.suffix or 'no extension'}", None
    status, img, err = load_image(path)
    if status is not ImageStatus.OK or img is None:
        return False, f"{status.value}: {err}", None
    return True, "", (img.shape[1], img.shape[0])


def _restyle(*widgets: QWidget) -> None:
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class DropZone(QFrame):
    """Accepts one image and reports it, or explains why it was rejected."""

    imageSelected = Signal(Path)
    imageRejected = Signal(str)
    imageCleared = Signal()

    PREVIEW_HEIGHT = 208

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._path: Path | None = None
        self._enabled = True

        self._preview = QLabel("⊕")
        self._preview.setObjectName("Preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(self.PREVIEW_HEIGHT)

        self._caption = QLabel("Drop a face image")
        self._caption.setObjectName("DropCaption")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setWordWrap(True)

        self._hint = QLabel("or click to browse  ·  JPG, PNG, WebP")
        self._hint.setObjectName("DropHint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._meta = QLabel("")
        self._meta.setObjectName("DropMeta")
        self._meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta.setVisible(False)

        self._error = QLabel("")
        self._error.setObjectName("DropError")
        self._error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._caption)
        layout.addWidget(self._hint)
        layout.addWidget(self._meta)
        layout.addWidget(self._error)

    # -- state ----------------------------------------------------------

    @property
    def path(self) -> Path | None:
        return self._path

    def set_enabled(self, enabled: bool) -> None:
        """Locked while a run is in flight."""
        self._enabled = enabled
        self.setAcceptDrops(enabled)
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled
                       else Qt.CursorShape.ArrowCursor)
        self.setProperty("locked", not enabled)
        _restyle(self)

    # -- interaction ----------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._enabled:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a face image", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp);;All files (*)",
        )
        if path:
            self.accept_path(Path(path))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._enabled and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("hover", True)
            _restyle(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hover", False)
        _restyle(self)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.setProperty("hover", False)
        _restyle(self)
        if not self._enabled:
            event.ignore()
            return

        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if not urls:
            self._reject("that drop contained no local file")
            event.ignore()
            return
        if len(urls) > 1:
            self._reject("drop a single image, not several")
            event.ignore()
            return

        event.acceptProposedAction()
        self.accept_path(Path(urls[0].toLocalFile()))

    # -- validation -----------------------------------------------------

    def accept_path(self, path: Path) -> bool:
        """Validate and adopt an image. Returns whether it was accepted."""
        ok, message, size = inspect_image(path)
        if not ok:
            self._reject(message)
            return False

        self._path = path
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self._preview.setPixmap(pixmap.scaled(
                max(self.width() - 40, 260), self.PREVIEW_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        width, height = size or (0, 0)
        kb = path.stat().st_size / 1024
        self._caption.setText(path.name)
        self._hint.setVisible(False)
        self._error.setVisible(False)
        self._meta.setText(f"{width} × {height}  ·  {kb:,.0f} KB  ·  {path.suffix.lstrip('.').upper()}")
        self._meta.setVisible(True)
        self.setProperty("state", "ready")
        _restyle(self)
        self.imageSelected.emit(path)
        return True

    def clear(self) -> None:
        """Drop the selected image and return to the empty state.

        Only forgets the selection - the file on disk is never touched, and no
        previous run's evidence is affected.
        """
        had_image = self._path is not None
        self._empty("⊕")
        self.setProperty("state", "")
        _restyle(self)
        if had_image:
            self.imageCleared.emit()

    def _reject(self, message: str) -> None:
        self._empty("⊘")
        self._error.setText(message)
        self._error.setVisible(True)
        self.setProperty("state", "rejected")
        _restyle(self)
        self.imageRejected.emit(message)

    def _empty(self, mark: str) -> None:
        """Reset every visual back to 'no image selected'."""
        self._path = None
        self._preview.clear()
        self._preview.setText(mark)
        self._caption.setText("Drop a face image")
        self._hint.setVisible(True)
        self._meta.setVisible(False)
        self._error.setVisible(False)


class FaceCard(QFrame):
    """What the pipeline's face scan actually found.

    Populated from the stage 01 events, never from a detection the GUI ran
    itself - the interface reports the pipeline's finding, it does not form a
    second opinion.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FaceCard")

        caption = QLabel("FACE SCAN")
        caption.setObjectName("FaceCaption")

        self._headline = QLabel("awaiting scan")
        self._headline.setObjectName("FaceDetail")
        self._detail = QLabel("")
        self._detail.setObjectName("FaceDetail")
        self._detail.setWordWrap(True)
        self._detail.setVisible(False)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(caption)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(3)
        layout.addLayout(top)
        layout.addWidget(self._headline)
        layout.addWidget(self._detail)

    def reset(self) -> None:
        self._headline.setObjectName("FaceDetail")
        self._headline.setText("awaiting scan")
        self._detail.setVisible(False)
        _restyle(self._headline)

    def report(self, headline: str, detail: str = "") -> None:
        """Show a finding the pipeline reported, verbatim."""
        self._headline.setObjectName("FaceHeadline")
        self._headline.setText(headline)
        if detail:
            self._detail.setText(detail)
            self._detail.setVisible(True)
        _restyle(self._headline)

    def headline(self) -> str:
        return self._headline.text()
