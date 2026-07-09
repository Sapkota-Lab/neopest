"""Image preview canvas placeholder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from neopest.gui_styles import (
    COLOR_MUTED_TEXT,
    COLOR_WORKSPACE_DIVIDER,
    IMAGE_CANVAS_OBJECT_NAME,
)


class ImageCanvas(QWidget):
    """Central image workspace — displays a placeholder until images are loaded."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(IMAGE_CANVAS_OBJECT_NAME)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hint = "Import a leaf image to preview"
        self._status = "No sample selected"

    def set_hint(self, text: str) -> None:
        self._hint = text
        self.update()

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        inset = 24
        preview_rect = self.rect().adjusted(inset, inset, -inset, -inset)
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
