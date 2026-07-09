"""Raw image preview canvas."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from neopest.gui_styles import (
    COLOR_MUTED_TEXT,
    COLOR_WORKSPACE_DIVIDER,
    IMAGE_CANVAS_OBJECT_NAME,
)


class ImageCanvas(QWidget):
    """Central image workspace for raw image preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(IMAGE_CANVAS_OBJECT_NAME)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hint = "Import a leaf image to preview"
        self._status = "No sample selected"
        self._pixmap: QPixmap | None = None

    def clear_image(self) -> None:
        self._pixmap = None
        self._status = "No sample selected"
        self.update()

    def set_hint(self, text: str) -> None:
        self._hint = text
        self.update()

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def set_image_path(self, image_path: Path | None) -> None:
        if image_path is None or not image_path.is_file():
            self.clear_image()
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap = None
            self._status = "Unable to load image"
        else:
            self._pixmap = pixmap
            self._status = image_path.name
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        inset = 24
        preview_rect = self.rect().adjusted(inset, inset, -inset, -inset)

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                preview_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            target = preview_rect
            x = target.x() + (target.width() - scaled.width()) // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QPen(QColor(COLOR_WORKSPACE_DIVIDER), 1, Qt.PenStyle.DashLine))
            painter.drawRect(preview_rect)
            painter.setPen(QColor(COLOR_MUTED_TEXT))
            painter.drawText(
                preview_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"{self._hint}\n\n{self._status}",
            )

        painter.end()
        super().paintEvent(event)


def build_image_canvas_host(parent: QWidget | None = None) -> tuple[QWidget, ImageCanvas]:
    """Return a styled host widget and the canvas inside it."""
    host = QWidget(parent)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    canvas = ImageCanvas(host)
    layout.addWidget(canvas, stretch=1)
    return host, canvas
