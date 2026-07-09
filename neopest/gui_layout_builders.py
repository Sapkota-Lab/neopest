"""Main window layout construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from neopest.explorer_tree import build_explorer_panel
from neopest.gui_canvas import ImageCanvas, build_image_canvas_host
from neopest.gui_styles import (
    CONTROL_ROW_SPACING,
    INSPECTOR_PANEL_MAX_WIDTH,
    INSPECTOR_PANEL_MIN_WIDTH,
    LEFT_PANEL_MIN_WIDTH,
    PANEL_MARGIN,
    apply_hint_style,
    apply_inspector_panel_style,
    apply_main_splitter_style,
    apply_panel_margins,
    apply_section_title_style,
    apply_workbench_controls_panel_style,
    apply_workspace_preview_panel_style,
    make_vertical_divider,
)
from neopest.metadata import SampleRecord, WorkspaceMetadata
from neopest.results import AnalysisStatus, CalibrationStatus

if TYPE_CHECKING:
    from neopest.gui import MainWindow


class SampleInspectorPanel(QWidget):
    """Right-side inspector showing workspace or sample metadata."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        apply_inspector_panel_style(self)
        self.setMinimumWidth(INSPECTOR_PANEL_MIN_WIDTH)
        self.setMaximumWidth(INSPECTOR_PANEL_MAX_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        apply_panel_margins(layout)

        title = QLabel("Inspector")
        apply_section_title_style(title)
        layout.addWidget(title)

        subtitle = QLabel("Sample details and calibration")
        apply_hint_style(subtitle)
        layout.addWidget(subtitle)
        layout.addWidget(make_vertical_divider())

        self._value_labels: dict[str, QLabel] = {}
        field_defs = [
            ("sample", "Sample"),
            ("sample_id", "Sample ID"),
            ("condition_group_id", "Condition Group ID"),
            ("raw_image_path", "Raw Image"),
            ("original_source_path", "Original Source"),
            ("dimensions", "Dimensions"),
            ("image_hash", "Image Hash"),
            ("calibration", "Calibration"),
            ("pixels_per_mm", "pixels_per_mm"),
            ("known_distance", "Known Distance"),
            ("known_distance_mm", "Known Distance (mm)"),
            ("calibration_points", "Calibration Points"),
            ("analysis", "Analysis"),
        ]
        for key, label_text in field_defs:
            row = QHBoxLayout()
            row.setSpacing(CONTROL_ROW_SPACING)
            label = QLabel(label_text)
            apply_hint_style(label)
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setWordWrap(True)
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(value)
            layout.addLayout(row)
            self._value_labels[key] = value

        layout.addWidget(make_vertical_divider())

        calibration_title = QLabel("Manual Calibration")
        apply_section_title_style(calibration_title)
        layout.addWidget(calibration_title)

        calibration_hint = QLabel(
            "Manual calibration assumes a 1-inch ruler interval. "
            "Click two ruler marks exactly 1 inch apart, then save."
        )
        apply_hint_style(calibration_hint)
        layout.addWidget(calibration_hint)

        self.set_manual_calibration_button = QPushButton("Set Manual Calibration")
        self.save_calibration_button = QPushButton("Save 1-Inch Calibration")
        self.cancel_calibration_button = QPushButton("Cancel Calibration")
        for button in (
            self.set_manual_calibration_button,
            self.save_calibration_button,
            self.cancel_calibration_button,
        ):
            layout.addWidget(button)

        layout.addStretch(1)
        self.set_calibration_controls_enabled(False)
        self.show_workspace_state(None)

    def set_calibration_controls_enabled(self, enabled: bool) -> None:
        self.set_manual_calibration_button.setEnabled(enabled)
        self.save_calibration_button.setEnabled(False)
        self.cancel_calibration_button.setEnabled(enabled)

    def set_save_calibration_enabled(self, enabled: bool) -> None:
        self.save_calibration_button.setEnabled(enabled)

    def show_workspace_state(self, workspace: WorkspaceMetadata | None) -> None:
        if workspace is None:
            self._value_labels["sample"].setText("No workspace open")
        else:
            self._value_labels["sample"].setText("No sample selected")
        for key in (
            "sample_id",
            "condition_group_id",
            "raw_image_path",
            "original_source_path",
            "dimensions",
            "image_hash",
            "pixels_per_mm",
            "known_distance",
            "known_distance_mm",
            "calibration_points",
        ):
            self._value_labels[key].setText("—")
        self._value_labels["calibration"].setText("—")
        self._value_labels["analysis"].setText("—")
        self.set_calibration_controls_enabled(False)

    def show_sample(self, sample: SampleRecord, workspace: WorkspaceMetadata) -> None:
        self._value_labels["sample"].setText(sample.display_name)
        self._value_labels["sample_id"].setText(sample.sample_id)
        self._value_labels["condition_group_id"].setText(sample.condition_group_id)
        self._value_labels["raw_image_path"].setText(sample.raw_image_path)
        self._value_labels["original_source_path"].setText(
            sample.original_source_path or "—"
        )

        if sample.width_px is not None and sample.height_px is not None:
            self._value_labels["dimensions"].setText(
                f"{sample.width_px} × {sample.height_px} px"
            )
        else:
            self._value_labels["dimensions"].setText("—")

        self._value_labels["image_hash"].setText(sample.image_hash or "—")
        self._value_labels["calibration"].setText(
            _format_calibration_status(sample.calibration_status)
        )
        self._value_labels["analysis"].setText(
            _format_analysis_status(sample.analysis_status)
        )

        if sample.calibration is not None:
            calibration = sample.calibration
            self._value_labels["pixels_per_mm"].setText(
                f"{calibration.pixels_per_mm:.4f}"
            )
            self._value_labels["known_distance"].setText(
                calibration.entered_distance_label()
            )
            self._value_labels["known_distance_mm"].setText(
                f"{calibration.known_distance_mm:g} mm"
            )
            self._value_labels["calibration_points"].setText(
                f"({calibration.point_1_x:.1f}, {calibration.point_1_y:.1f})"
                f" → ({calibration.point_2_x:.1f}, {calibration.point_2_y:.1f})"
            )
        else:
            self._value_labels["pixels_per_mm"].setText("—")
            self._value_labels["known_distance"].setText("—")
            self._value_labels["known_distance_mm"].setText("—")
            self._value_labels["calibration_points"].setText("—")

        group = workspace.get_condition_group(sample.condition_group_id)
        if group is not None:
            self._value_labels["condition_group_id"].setText(
                f"{group.display_name}\n({sample.condition_group_id})"
            )

        self.set_calibration_controls_enabled(True)
        self.set_save_calibration_enabled(False)


def _format_calibration_status(status: CalibrationStatus) -> str:
    if status is CalibrationStatus.UNCALIBRATED:
        return "Uncalibrated"
    if status is CalibrationStatus.MANUAL:
        return "Manual"
    if status is CalibrationStatus.CALIBRATED:
        return "Calibrated"
    return "Missing"


def _format_analysis_status(status: AnalysisStatus) -> str:
    if status is AnalysisStatus.NOT_RUN:
        return "Not Run"
    if status is AnalysisStatus.COMPLETE:
        return "Complete"
    if status is AnalysisStatus.PENDING:
        return "Pending"
    return "Failed"


def build_inspector_panel(parent: QWidget | None = None) -> SampleInspectorPanel:
    """Right-side inspector / contextual settings panel."""
    return SampleInspectorPanel(parent)


def build_workbench_controls(parent: QWidget | None = None) -> tuple[QWidget, QPushButton, QLabel]:
    """Lower status/control strip beneath the main workspace."""
    panel = QWidget(parent)
    apply_workbench_controls_panel_style(panel)
    panel.setFixedHeight(56)

    layout = QHBoxLayout(panel)
    layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN)
    layout.setSpacing(12)

    status = QLabel("Ready")
    apply_hint_style(status)
    layout.addWidget(status, stretch=1)

    run_button = QPushButton("Run Analysis")
    run_button.setEnabled(False)
    run_button.setToolTip("Analysis pipeline not yet implemented (Phase 2)")
    layout.addWidget(run_button)

    return panel, run_button, status


def build_main_layout(window: MainWindow) -> None:
    """Construct the Workbench shell and attach widgets to ``window``."""
    central = QWidget(window)
    window.setCentralWidget(central)

    root = QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    horizontal_splitter = QSplitter(Qt.Orientation.Horizontal, central)
    apply_main_splitter_style(horizontal_splitter)

    explorer_panel, explorer_tree = build_explorer_panel(horizontal_splitter)
    explorer_panel.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)
    window.explorer_panel = explorer_panel
    window.explorer_tree = explorer_tree

    center_column = QWidget(horizontal_splitter)
    apply_workspace_preview_panel_style(center_column)
    center_layout = QVBoxLayout(center_column)
    center_layout.setContentsMargins(0, 0, 0, 0)
    center_layout.setSpacing(0)

    canvas_host, image_canvas = build_image_canvas_host(center_column)
    center_layout.addWidget(canvas_host, stretch=1)
    window.image_canvas = image_canvas

    controls_panel, run_button, controls_status = build_workbench_controls(center_column)
    center_layout.addWidget(controls_panel)
    window.controls_panel = controls_panel
    window.run_analysis_button = run_button
    window.controls_status_label = controls_status

    inspector_panel = build_inspector_panel(horizontal_splitter)
    window.inspector_panel = inspector_panel

    horizontal_splitter.addWidget(explorer_panel)
    horizontal_splitter.addWidget(center_column)
    horizontal_splitter.addWidget(inspector_panel)
    horizontal_splitter.setStretchFactor(0, 1)
    horizontal_splitter.setStretchFactor(1, 4)
    horizontal_splitter.setStretchFactor(2, 1)
    horizontal_splitter.setSizes([LEFT_PANEL_MIN_WIDTH, 900, INSPECTOR_PANEL_MIN_WIDTH])

    root.addWidget(horizontal_splitter, stretch=1)

    status_bar = QStatusBar(window)
    status_bar.showMessage("NeoPest — strawberry leaf disease analysis workbench")
    window.setStatusBar(status_bar)
    window.status_bar = status_bar
