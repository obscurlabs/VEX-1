"""Image input: drag-and-drop, click-to-browse, and a preview.

Validation goes through the pipeline's own ``src.vision.quality.load_image``,
so the GUI accepts exactly what the pipeline accepts - no second opinion about
what counts as a usable image.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
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


class DropZone(QFrame):
    """Accepts one image and reports it, or explains why it was rejected."""

    imageSelected = Signal(Path)
    imageRejected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self._path: Path | None = None
        self._enabled = True

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setObjectName("Preview")
        self._preview.setMinimumHeight(150)

        self._caption = QLabel("Drop a face image here, or click to browse")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setObjectName("DropCaption")
        self._caption.setWordWrap(True)

        self._meta = QLabel("")
        self._meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta.setObjectName("DropMeta")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._caption)
        layout.addWidget(self._meta)

    # -- state ----------------------------------------------------------

    @property
    def path(self) -> Path | None:
        return self._path

    def set_enabled(self, enabled: bool) -> None:
        """Locked while a run is in flight."""
        self._enabled = enabled
        self.setAcceptDrops(enabled)
        self.setProperty("locked", not enabled)
        self.style().unpolish(self)
        self.style().polish(self)

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
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().unpolish(self)
        self.style().polish(self)
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
                self._preview.width() or 320, 190,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        self._caption.setText(path.name)
        width, height = size or (0, 0)
        kb = path.stat().st_size / 1024
        self._meta.setText(f"{width} x {height}  ·  {kb:,.0f} KB")
        self.setProperty("state", "ready")
        self.style().unpolish(self)
        self.style().polish(self)
        self.imageSelected.emit(path)
        return True

    def _reject(self, message: str) -> None:
        self._path = None
        self._preview.clear()
        self._caption.setText("Drop a face image here, or click to browse")
        self._meta.setText(message)
        self.setProperty("state", "rejected")
        self.style().unpolish(self)
        self.style().polish(self)
        self.imageRejected.emit(message)
