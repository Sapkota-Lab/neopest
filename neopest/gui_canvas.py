"""Raw image preview canvas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from neopest.image_io import load_preview_pixmap
from neopest.mask_correction import (
    ADD_AREA,
    MaskStroke,
    interpolate_segment,
    stroke_interpolation_spacing,
)
from neopest.gui_styles import (
    COLOR_ACCENT,
    COLOR_MUTED_TEXT,
    COLOR_WORKSPACE_DIVIDER,
    IMAGE_CANVAS_OBJECT_NAME,
)

CANVAS_INSET = 24
CALIBRATION_POINT_RADIUS = 5
MAX_DISPLAY_SCALE = 16.0


@dataclass(frozen=True)
class ImageDrawGeometry:
    """Mapping between widget coordinates and source image pixels."""

    draw_x: float
    draw_y: float
    draw_width: float
    draw_height: float
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
    correction_stroke_preview = Signal(object)
    correction_stroke_finished = Signal(object)
    zoom_text_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(IMAGE_CANVAS_OBJECT_NAME)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._hint = "Import a leaf image to preview"
        self._pixmap: QPixmap | None = None
        self._source_token: tuple[str, int, int] | None = None
        self._geometry: ImageDrawGeometry | None = None
        self._absolute_scale: float | None = None
        self._view_center: tuple[float, float] | None = None
        self._base_cache: QPixmap | None = None
        self._base_cache_key: tuple | None = None
        self._overlay_cache: QPixmap | None = None
        self._space_pan = False
        self._panning = False
        self._pan_last: tuple[float, float] | None = None
        self._zoom_text = "—"
        self._calibration_mode = False
        self._draft_points: list[tuple[float, float]] = []
        self._saved_line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._leaf_mask_overlay: QImage | None = None
        self._correction_mode = False
        self._brush_operation = ADD_AREA
        self._brush_radius_px = 6.0
        self._stroke_active = False
        self._stroke_gap = False
        self._stroke_grabbed = False
        self._stroke_points: list[tuple[float, float]] = []
        self._active_operation = ADD_AREA
        self._active_radius_px = 6.0
        self._cursor_widget_pos: tuple[float, float] | None = None

    def clear_image(self) -> None:
        self._pixmap = None
        self._source_token = None
        self._geometry = None
        self._absolute_scale = None
        self._view_center = None
        self._calibration_mode = False
        self._draft_points = []
        self._saved_line = None
        self._leaf_mask_overlay = None
        self._clear_display_cache()
        self._cancel_active_stroke()
        self._panning = False
        self._pan_last = None
        self._correction_mode = False
        self._cursor_widget_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._publish_zoom_text("—")
        self.update()

    def set_hint(self, text: str) -> None:
        self._hint = text
        self.update()

    def set_image_path(self, image_path: Path | None) -> None:
        if image_path is None or not image_path.is_file():
            self.clear_image()
            return

        resolved = image_path.expanduser().resolve()
        try:
            stat = resolved.stat()
            token = (str(resolved), stat.st_mtime_ns, stat.st_size)
        except OSError:
            token = None
        same_image = (
            token is not None
            and token == self._source_token
            and self._pixmap is not None
            and not self._pixmap.isNull()
        )
        if not same_image:
            try:
                pixmap = load_preview_pixmap(resolved)
            except (OSError, ValueError):
                pixmap = QPixmap()
            if pixmap.isNull():
                self._pixmap = None
                self._geometry = None
                self._source_token = None
                self._hint = "Unable to load image"
            else:
                self._pixmap = pixmap
                self._source_token = token
            self._absolute_scale = None
            self._view_center = None
        self._draft_points = []
        self._leaf_mask_overlay = None
        self._clear_display_cache()
        self.cancel_active_stroke()
        self.update()

    def set_leaf_mask_overlay(self, overlay: QPixmap | QImage | None) -> None:
        """Show or hide a mask overlay aligned to the source image."""
        self.set_overlay_image(self._as_overlay_image(overlay))

    def set_overlay_image(self, overlay: QImage | None) -> None:
        """Show a mask overlay without copying it again."""
        self._leaf_mask_overlay = overlay
        self._overlay_cache = None
        self.update()

    def refresh_overlay_image(self) -> None:
        """Redraw the current overlay after an in-place pixel update."""
        self._overlay_cache = None
        self.update()

    def clear_leaf_mask_overlay(self) -> None:
        self._leaf_mask_overlay = None
        self._overlay_cache = None
        self.update()

    def fit_in_view(self) -> None:
        """Fit the whole image in the canvas. Choosing another sample does this."""
        self._absolute_scale = None
        self._view_center = None
        self._apply_view_change()

    def zoom_actual_size(self) -> None:
        """Show one image pixel per screen pixel, keeping the current center."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        self._absolute_scale = 1.0
        self._apply_view_change()

    def set_space_pan(self, held: bool) -> None:
        """Arm click-drag panning so a brush or calibration click is not recorded."""
        if self._stroke_active and held:
            return
        self._space_pan = held
        if not held:
            self._panning = False
            self._pan_last = None
        self.update()

    def zoom_percent(self) -> int | None:
        geometry = self._geometry if self._geometry is not None else self._ensure_geometry()
        if geometry is None or geometry.source_width <= 0:
            return None
        return int(round(100.0 * geometry.draw_width / geometry.source_width))

    def set_calibration_mode(self, enabled: bool) -> None:
        self._calibration_mode = enabled
        if enabled:
            self._draft_points = []
            self._cancel_active_stroke()
            self._correction_mode = False
            self._cursor_widget_pos = None
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

    def set_correction_mode(self, enabled: bool) -> None:
        """Turn brush editing on or off. Editing is separate from calibration."""
        self._cancel_active_stroke()
        self._correction_mode = enabled
        self._cursor_widget_pos = None
        if enabled:
            self._calibration_mode = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_brush(self, operation: str, radius_px: float) -> None:
        """Set the idle brush. An in-progress stroke keeps the radius it started with."""
        self._brush_operation = operation
        self._brush_radius_px = float(radius_px)
        self.update()

    def brush_footprint_widget_radii(self) -> tuple[float, float] | None:
        """Widget-space radii of the circular brush, matching the painted disk."""
        return self._brush_widget_radii()

    def cancel_active_stroke(self) -> None:
        """Drop an in-progress stroke without committing it."""
        had_points = bool(self._stroke_points)
        self._cancel_active_stroke()
        if had_points:
            self.correction_stroke_preview.emit(None)
        self.update()

    def _display_radius_px(self) -> float:
        if self._stroke_active:
            return self._active_radius_px
        return self._brush_radius_px

    def _display_operation(self) -> str:
        if self._stroke_active:
            return self._active_operation
        return self._brush_operation

    def _brush_widget_radii(self) -> tuple[float, float] | None:
        geometry = self._geometry
        if (
            geometry is None
            or geometry.source_width <= 0
            or geometry.source_height <= 0
            or geometry.draw_width <= 0
            or geometry.draw_height <= 0
        ):
            return None
        radius = self._display_radius_px()
        return (
            radius * geometry.draw_width / geometry.source_width,
            radius * geometry.draw_height / geometry.source_height,
        )

    def _cancel_active_stroke(self) -> None:
        self._stroke_active = False
        self._stroke_gap = False
        self._stroke_points = []
        if self._stroke_grabbed:
            self.releaseMouse()
            self._stroke_grabbed = False

    def _emit_stroke_preview(self) -> None:
        if not self._stroke_points:
            return
        self.correction_stroke_preview.emit(
            MaskStroke(
                operation=self._active_operation,
                radius_px=self._active_radius_px,
                points=tuple(self._stroke_points),
            )
        )

    def _append_stroke_point(self, image_point: tuple[float, float]) -> None:
        if self._stroke_gap or not self._stroke_points:
            self._stroke_points.append(image_point)
            self._stroke_gap = False
            return
        spacing = stroke_interpolation_spacing(self._active_radius_px)
        self._stroke_points.extend(
            interpolate_segment(self._stroke_points[-1], image_point, spacing)
        )

    def _begin_stroke(self, widget_x: float, widget_y: float) -> None:
        if self._geometry is None:
            return
        image_point = self._geometry.widget_to_image(widget_x, widget_y)
        if image_point is None:
            return
        self._stroke_active = True
        self._stroke_gap = False
        self._active_operation = self._brush_operation
        self._active_radius_px = self._brush_radius_px
        self._stroke_points = [image_point]
        if QGuiApplication.platformName() != "offscreen":
            self.grabMouse()
            self._stroke_grabbed = True
        self._emit_stroke_preview()
        self.update()

    def _update_stroke(self, widget_x: float, widget_y: float) -> None:
        if not self._stroke_active or self._geometry is None:
            return
        image_point = self._geometry.widget_to_image(widget_x, widget_y)
        if image_point is None:
            self._stroke_gap = True
            return
        self._append_stroke_point(image_point)
        self._emit_stroke_preview()

    def _finish_stroke(self) -> None:
        points = tuple(self._stroke_points)
        operation = self._active_operation
        radius = self._active_radius_px
        self._cancel_active_stroke()
        if not points:
            return
        self.correction_stroke_finished.emit(
            MaskStroke(operation=operation, radius_px=radius, points=points)
        )
        self.update()

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
        if self._begin_pan(event):
            event.accept()
            return
        if self._correction_mode and event.button() == Qt.MouseButton.LeftButton:
            self._begin_stroke(event.position().x(), event.position().y())
            event.accept()
            return
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        position = event.position()
        if self._panning and self._pan_last is not None:
            self._pan_by(position.x() - self._pan_last[0], position.y() - self._pan_last[1])
            self._pan_last = (position.x(), position.y())
            event.accept()
            return
        if not self._correction_mode:
            super().mouseMoveEvent(event)
            return
        self._cursor_widget_pos = (position.x(), position.y())
        over_image = (
            self._geometry is not None
            and self._geometry.widget_to_image(position.x(), position.y()) is not None
        )
        self.setCursor(
            Qt.CursorShape.BlankCursor if over_image else Qt.CursorShape.ArrowCursor
        )
        if self._stroke_active:
            self._update_stroke(position.x(), position.y())
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._panning and event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        }:
            self._panning = False
            self._pan_last = None
            event.accept()
            return
        if self._correction_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._stroke_active:
                self._finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if self._pixmap is None or self._pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return
        self._zoom_at(event.position().x(), event.position().y(), 1.1 ** (delta / 120.0))
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pan(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pan(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._correction_mode and not self._stroke_active:
            self._cursor_widget_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        super().leaveEvent(event)

    def _preview_rect(self) -> QRectF:
        rect = self.rect()
        return QRectF(
            CANVAS_INSET,
            CANVAS_INSET,
            max(1, rect.width() - 2 * CANVAS_INSET),
            max(1, rect.height() - 2 * CANVAS_INSET),
        )

    def _fit_scale(self, preview_rect: QRectF) -> float:
        assert self._pixmap is not None
        return min(
            preview_rect.width() / self._pixmap.width(),
            preview_rect.height() / self._pixmap.height(),
        )

    def _scale_for(self, fit_scale: float) -> float:
        if self._absolute_scale is None:
            return fit_scale
        lower = min(fit_scale, 1.0)
        upper = max(fit_scale, MAX_DISPLAY_SCALE)
        return min(max(self._absolute_scale, lower), upper)

    def _ensure_geometry(self) -> ImageDrawGeometry | None:
        self._geometry = self._compute_geometry(self._preview_rect())
        return self._geometry

    def _compute_geometry(self, preview_rect: QRectF) -> ImageDrawGeometry | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        source_width = self._pixmap.width()
        source_height = self._pixmap.height()
        if source_width <= 0 or source_height <= 0 or preview_rect.width() <= 0:
            return None
        fit_scale = self._fit_scale(preview_rect)
        scale = self._scale_for(fit_scale)
        if self._view_center is None:
            center_x = source_width / 2.0
            center_y = source_height / 2.0
        else:
            center_x, center_y = self._view_center
        view_center_x = preview_rect.x() + preview_rect.width() / 2.0
        view_center_y = preview_rect.y() + preview_rect.height() / 2.0
        return ImageDrawGeometry(
            draw_x=view_center_x - center_x * scale,
            draw_y=view_center_y - center_y * scale,
            draw_width=source_width * scale,
            draw_height=source_height * scale,
            source_width=source_width,
            source_height=source_height,
        )

    def _zoom_at(self, widget_x: float, widget_y: float, factor: float) -> None:
        geometry = self._ensure_geometry()
        if geometry is None or factor <= 0:
            return
        anchor = geometry.widget_to_image(widget_x, widget_y)
        fit_scale = self._fit_scale(self._preview_rect())
        old_scale = geometry.draw_width / geometry.source_width
        self._absolute_scale = min(
            max(old_scale * factor, min(fit_scale, 1.0)),
            max(fit_scale, MAX_DISPLAY_SCALE),
        )
        if anchor is None:
            self._remember_center(geometry)
        else:
            preview = self._preview_rect()
            view_center_x = preview.x() + preview.width() / 2.0
            view_center_y = preview.y() + preview.height() / 2.0
            scale = self._absolute_scale
            self._view_center = (
                self._clamp_center(
                    anchor[0] - (widget_x - view_center_x) / scale,
                    geometry.source_width,
                ),
                self._clamp_center(
                    anchor[1] - (widget_y - view_center_y) / scale,
                    geometry.source_height,
                ),
            )
        self._apply_view_change()

    def _pan_by(self, widget_dx: float, widget_dy: float) -> None:
        geometry = self._ensure_geometry()
        if geometry is None or geometry.draw_width <= 0:
            return
        scale = geometry.draw_width / geometry.source_width
        center_x, center_y = self._current_center(geometry)
        self._view_center = (
            self._clamp_center(center_x - widget_dx / scale, geometry.source_width),
            self._clamp_center(center_y - widget_dy / scale, geometry.source_height),
        )
        if self._absolute_scale is None:
            self._absolute_scale = scale
        self._apply_view_change()

    def _apply_view_change(self) -> None:
        self._clear_display_cache()
        self._geometry = self._compute_geometry(self._preview_rect())
        self.update()

    def _begin_pan(self, event: QMouseEvent) -> bool:
        if self._pixmap is None or self._pixmap.isNull():
            return False
        button = event.button()
        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and self._space_pan
        ):
            self._panning = True
            self._pan_last = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        return False

    def _current_center(self, geometry: ImageDrawGeometry) -> tuple[float, float]:
        if self._view_center is not None:
            return self._view_center
        return (geometry.source_width / 2.0, geometry.source_height / 2.0)

    def _remember_center(self, geometry: ImageDrawGeometry) -> None:
        self._view_center = self._current_center(geometry)

    @staticmethod
    def _clamp_center(value: float, source_size: int) -> float:
        return min(max(value, 0.0), float(source_size))

    def _clear_display_cache(self) -> None:
        self._base_cache = None
        self._base_cache_key = None
        self._overlay_cache = None

    def _publish_zoom_text(self, text: str) -> None:
        if text == self._zoom_text:
            return
        self._zoom_text = text
        self.zoom_text_changed.emit(text)

    def _as_overlay_image(self, overlay: QPixmap | QImage | None) -> QImage | None:
        if overlay is None or overlay.isNull():
            return None
        if isinstance(overlay, QImage):
            image = overlay
        else:
            image = overlay.toImage()
        if image.format() != QImage.Format.Format_RGBA8888:
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return image

    def _layer_pixmap(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        layer = QPixmap(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        layer.setDevicePixelRatio(ratio)
        layer.fill(Qt.GlobalColor.transparent)
        return layer

    def _view_cache_key(self, geometry: ImageDrawGeometry) -> tuple:
        return (
            round(geometry.draw_x, 3),
            round(geometry.draw_y, 3),
            round(geometry.draw_width, 3),
            round(geometry.draw_height, 3),
            self.width(),
            self.height(),
            round(self.devicePixelRatioF(), 3),
        )

    def _render_visible_layer(self, source: QPixmap | QImage, geometry: ImageDrawGeometry) -> QPixmap:
        layer = self._layer_pixmap()
        scale = geometry.draw_width / geometry.source_width if geometry.source_width else 1.0
        visible = QRectF(geometry.draw_x, geometry.draw_y, geometry.draw_width, geometry.draw_height)
        visible = visible.intersected(QRectF(self.rect()))
        if visible.isEmpty():
            return layer
        source_rect = QRectF(
            (visible.left() - geometry.draw_x) / scale,
            (visible.top() - geometry.draw_y) / scale,
            visible.width() / scale,
            visible.height() / scale,
        )
        painter = QPainter(layer)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 1.0)
        if isinstance(source, QImage):
            painter.drawImage(visible, source, source_rect)
        else:
            painter.drawPixmap(visible, source, source_rect)
        painter.end()
        return layer

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

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        if not self._correction_mode or self._cursor_widget_pos is None or self._geometry is None:
            return
        if (
            self._geometry.widget_to_image(
                self._cursor_widget_pos[0],
                self._cursor_widget_pos[1],
            )
            is None
        ):
            return
        radii = self._brush_widget_radii()
        if radii is None:
            return
        color = QColor("#ff5f3c") if self._display_operation() == ADD_AREA else QColor("#f4f4f4")
        pen = QPen(color, 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(
            QPointF(self._cursor_widget_pos[0], self._cursor_widget_pos[1]),
            radii[0],
            radii[1],
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        preview_rect = self._preview_rect()

        if self._pixmap is not None and not self._pixmap.isNull():
            self._geometry = self._compute_geometry(preview_rect)
            geometry = self._geometry
            if geometry is not None:
                key = self._view_cache_key(geometry)
                if self._base_cache is None or self._base_cache_key != key:
                    self._base_cache = self._render_visible_layer(self._pixmap, geometry)
                    self._base_cache_key = key
                    self._overlay_cache = None
                painter.drawPixmap(0, 0, self._base_cache)
                overlay = self._leaf_mask_overlay
                if overlay is not None and not overlay.isNull():
                    if self._overlay_cache is None:
                        self._overlay_cache = self._render_visible_layer(overlay, geometry)
                    painter.drawPixmap(0, 0, self._overlay_cache)
                self._draw_calibration_overlay(painter)
                self._draw_brush_cursor(painter)
                self._publish_zoom_text(f"{self.zoom_percent()}%")
        else:
            self._geometry = None
            painter.setPen(QPen(QColor(COLOR_WORKSPACE_DIVIDER), 1, Qt.PenStyle.DashLine))
            painter.drawRect(preview_rect.toRect())
            painter.setPen(QColor(COLOR_MUTED_TEXT))
            painter.drawText(
                preview_rect.toRect(),
                Qt.AlignmentFlag.AlignCenter,
                self._hint,
            )
            self._publish_zoom_text("—")

        painter.end()
        super().paintEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._clear_display_cache()
        super().resizeEvent(event)


def build_image_canvas_host(parent: QWidget | None = None) -> tuple[QWidget, ImageCanvas]:
    """Return a styled host widget and the canvas inside it."""
    host = QWidget(parent)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    toolbar = QHBoxLayout()
    toolbar.setContentsMargins(0, 0, 0, 0)
    toolbar.setSpacing(8)
    fit_button = QPushButton("Fit", host)
    fit_button.setObjectName("fitViewButton")
    actual_button = QPushButton("100%", host)
    actual_button.setObjectName("actualSizeButton")
    zoom_label = QLabel("—", host)
    zoom_label.setObjectName("zoomPercentLabel")
    zoom_label.setMinimumWidth(52)
    for button in (fit_button, actual_button):
        button.setFixedHeight(28)
        button.setMinimumWidth(64)
    toolbar.addWidget(fit_button)
    toolbar.addWidget(actual_button)
    toolbar.addWidget(zoom_label)
    toolbar.addStretch(1)

    canvas = ImageCanvas(host)
    fit_button.clicked.connect(canvas.fit_in_view)
    actual_button.clicked.connect(canvas.zoom_actual_size)
    canvas.zoom_text_changed.connect(zoom_label.setText)
    fit_button.setToolTip("Fit the whole image in the view.")
    actual_button.setToolTip("Show the image at one image pixel per screen pixel.")
    zoom_label.setToolTip("Current magnification. Brush size stays in image pixels.")

    layout.addLayout(toolbar)
    layout.addWidget(canvas, stretch=1)
    return host, canvas
