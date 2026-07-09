"""Raw image preview canvas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from neopest.gui_styles import (
    COLOR_ACCENT,
    COLOR_MUTED_TEXT,
    COLOR_WORKSPACE_DIVIDER,
    IMAGE_CANVAS_OBJECT_NAME,
)

CANVAS_INSET = 24
CALIBRATION_POINT_RADIUS = 5


@dataclass(frozen=True)
class ImageDrawGeometry:
    """Mapping between widget coordinates and source image pixels."""

    draw_x: int
    draw_y: int
    draw_width: int
    draw_height: int
    source_width: int
    source_height: int

    def widget_to_image(self, widget_x: float, widget_y: float) -> tuple[float, float] | None:
        if self.draw_width <= 0 or self.draw_height <= 0:
            return None
        if not (
            self.draw_x <= widget_x <= self.draw_x + self.draw_width
            and self.draw_y <= widget_y <= self.draw_y + self.draw_height
        ):
            return None

        relative_x = (widget_x - self.draw_x) / self.draw_width
        relative_y = (widget_y - self.draw_y) / self.draw_height
        image_x = relative_x * self.source_width
        image_y = relative_y * self.source_height
        return (
            max(0.0, min(float(self.source_width), image_x)),
            max(0.0, min(float(self.source_height), image_y)),
        )

    def image_to_widget(self, image_x: float, image_y: float) -> tuple[float, float]:
        relative_x = image_x / self.source_width if self.source_width else 0.0
        relative_y = image_y / self.source_height if self.source_height else 0.0
        widget_x = self.draw_x + relative_x * self.draw_width
        widget_y = self.draw_y + relative_y * self.draw_height
        return widget_x, widget_y


class ImageCanvas(QWidget):
    """Central image workspace for raw image preview and manual calibration."""

    calibration_points_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(IMAGE_CANVAS_OBJECT_NAME)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._hint = "Import a leaf image to preview"
        self._status = "No sample selected"
        self._pixmap: QPixmap | None = None
        self._geometry: ImageDrawGeometry | None = None
        self._calibration_mode = False
        self._draft_points: list[tuple[float, float]] = []
        self._saved_line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._leaf_mask_overlay: QPixmap | None = None

    def clear_image(self) -> None:
        self._pixmap = None
        self._geometry = None
        self._status = "No sample selected"
        self._calibration_mode = False
        self._draft_points = []
        self._saved_line = None
        self._leaf_mask_overlay = None
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
            self._geometry = None
            self._status = "Unable to load image"
        else:
            self._pixmap = pixmap
            self._status = image_path.name
        self._draft_points = []
        self._leaf_mask_overlay = None
        self.update()

    def set_leaf_mask_overlay(self, overlay: QPixmap | None) -> None:
        """Show or hide a pre-built leaf mask overlay pixmap aligned to the source image."""
        self._leaf_mask_overlay = overlay
        self.update()

    def clear_leaf_mask_overlay(self) -> None:
        self._leaf_mask_overlay = None
        self.update()

    def set_calibration_mode(self, enabled: bool) -> None:
        self._calibration_mode = enabled
        if enabled:
            self._draft_points = []
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def clear_draft_calibration(self) -> None:
        self._draft_points = []
        self.calibration_points_changed.emit()
        self.update()

    def get_draft_calibration_points(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if len(self._draft_points) != 2:
            return None
        return self._draft_points[0], self._draft_points[1]

    def set_saved_calibration_line(
        self,
        point_1: tuple[float, float] | None,
        point_2: tuple[float, float] | None,
    ) -> None:
        if point_1 is None or point_2 is None:
            self._saved_line = None
        else:
            self._saved_line = (point_1, point_2)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if (
            not self._calibration_mode
            or self._pixmap is None
            or self._pixmap.isNull()
            or self._geometry is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return

        image_point = self._geometry.widget_to_image(event.position().x(), event.position().y())
        if image_point is None:
            return

        if len(self._draft_points) >= 2:
            self._draft_points = []
        self._draft_points.append(image_point)
        self.calibration_points_changed.emit()
        self.update()
        super().mousePressEvent(event)

    def _compute_geometry(self, preview_rect) -> ImageDrawGeometry | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None

        scaled = self._pixmap.scaled(
            preview_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = preview_rect.x() + (preview_rect.width() - scaled.width()) // 2
        y = preview_rect.y() + (preview_rect.height() - scaled.height()) // 2
        return ImageDrawGeometry(
            draw_x=x,
            draw_y=y,
            draw_width=scaled.width(),
            draw_height=scaled.height(),
            source_width=self._pixmap.width(),
            source_height=self._pixmap.height(),
        )

    def _draw_calibration_overlay(self, painter: QPainter) -> None:
        if self._geometry is None:
            return

        line_pen = QPen(QColor(COLOR_ACCENT), 2, Qt.PenStyle.SolidLine)
        point_pen = QPen(QColor("#ffffff"), 1, Qt.PenStyle.SolidLine)
        point_brush = QColor(COLOR_ACCENT)

        def draw_line(
            point_1: tuple[float, float],
            point_2: tuple[float, float],
            *,
            dashed: bool = False,
        ) -> None:
            wx1, wy1 = self._geometry.image_to_widget(point_1[0], point_1[1])
            wx2, wy2 = self._geometry.image_to_widget(point_2[0], point_2[1])
            pen = QPen(line_pen)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(wx1, wy1), QPointF(wx2, wy2))
            for wx, wy in ((wx1, wy1), (wx2, wy2)):
                painter.setPen(point_pen)
                painter.setBrush(point_brush)
                painter.drawEllipse(
                    QPointF(wx, wy),
                    CALIBRATION_POINT_RADIUS,
                    CALIBRATION_POINT_RADIUS,
                )

        if self._saved_line is not None:
            draw_line(self._saved_line[0], self._saved_line[1])

        if self._calibration_mode and self._draft_points:
            if len(self._draft_points) == 1:
                wx, wy = self._geometry.image_to_widget(
                    self._draft_points[0][0],
                    self._draft_points[0][1],
                )
                painter.setPen(point_pen)
                painter.setBrush(point_brush)
                painter.drawEllipse(
                    QPointF(wx, wy),
                    CALIBRATION_POINT_RADIUS,
                    CALIBRATION_POINT_RADIUS,
                )
            elif len(self._draft_points) == 2:
                draw_line(self._draft_points[0], self._draft_points[1], dashed=True)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        preview_rect = self.rect().adjusted(
            CANVAS_INSET,
            CANVAS_INSET,
            -CANVAS_INSET,
            -CANVAS_INSET,
        )

        if self._pixmap is not None and not self._pixmap.isNull():
            self._geometry = self._compute_geometry(preview_rect)
            if self._geometry is not None:
                scaled = self._pixmap.scaled(
                    preview_rect.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(self._geometry.draw_x, self._geometry.draw_y, scaled)
                if self._leaf_mask_overlay is not None and not self._leaf_mask_overlay.isNull():
                    mask_scaled = self._leaf_mask_overlay.scaled(
                        self._geometry.draw_width,
                        self._geometry.draw_height,
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    painter.drawPixmap(self._geometry.draw_x, self._geometry.draw_y, mask_scaled)
                self._draw_calibration_overlay(painter)
        else:
            self._geometry = None
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
