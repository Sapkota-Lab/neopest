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
    apply_inspector_action_button,
    apply_inspector_panel_style,
    apply_main_splitter_style,
    apply_measurement_value_style,
    apply_panel_margins,
    apply_primary_action_button,
    apply_section_title_style,
    apply_workbench_controls_panel_style,
    apply_workspace_preview_panel_style,
    make_vertical_divider,
)
from neopest.measurements import (
    CURRENT_LEAF_MASK_METHOD,
    CURRENT_LEAF_MASK_SETTINGS_VERSION,
    leaf_measurement_matches_provenance,
)
from neopest.metadata import SampleRecord, WorkspaceMetadata
from neopest.results import CalibrationStatus

if TYPE_CHECKING:
    from neopest.gui import MainWindow


def _format_calibration_status_label(status: CalibrationStatus) -> str:
    if status is CalibrationStatus.UNCALIBRATED:
        return "Not calibrated"
    if status in {CalibrationStatus.MANUAL, CalibrationStatus.CALIBRATED}:
        return "Ready"
    return "Missing"


def _format_leaf_mask_status_label(*, preview_status: str | None) -> str:
    if preview_status == "Preview Ready":
        return "Preview Ready"
    if preview_status == "Failed Preview":
        return "Failed"
    return "Not previewed"


def _format_total_leaf_area_display(
    sample: SampleRecord,
    *,
    preview_status: str | None,
    preview_succeeded_without_calibration: bool = False,
) -> str:
    """Format researcher-facing total leaf area text without performing area math."""
    if preview_succeeded_without_calibration:
        return "Calibration required"

    if sample.calibration is None:
        measurement = sample.leaf_measurement
        if measurement is not None and measurement.status == "stale":
            return "Calibration required"
        if preview_status == "Preview Ready":
            return "Calibration required"
        return "Not calculated"

    if not leaf_measurement_matches_provenance(
        sample.leaf_measurement,
        calibration=sample.calibration,
        image_hash=sample.image_hash,
        method=CURRENT_LEAF_MASK_METHOD,
        settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
    ):
        if preview_status == "Preview Ready":
            return "Recalculate by previewing mask"
        if sample.leaf_measurement is not None and sample.leaf_measurement.status == "stale":
            return "Recalculate by previewing mask"
        return "Not calculated"

    measurement = sample.leaf_measurement
    assert measurement is not None
    if measurement.leaf_area_in2 is None:
        return "Not calculated"
    return f"{measurement.leaf_area_in2:.2f} in²"


class SampleInspectorPanel(QWidget):
    """Right-side inspector showing researcher-facing sample information."""

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

        sample_title = QLabel("Sample")
        apply_section_title_style(sample_title)
        layout.addWidget(sample_title)

        self.sample_name_label = QLabel("No workspace open")
        self.sample_name_label.setWordWrap(True)
        layout.addWidget(self.sample_name_label)

        self.condition_group_label = QLabel("—")
        apply_hint_style(self.condition_group_label)
        self.condition_group_label.setWordWrap(True)
        layout.addWidget(self.condition_group_label)

        layout.addWidget(make_vertical_divider())

        measurement_title = QLabel("Measurement")
        apply_section_title_style(measurement_title)
        layout.addWidget(measurement_title)

        area_caption = QLabel("Total Leaf Area")
        apply_hint_style(area_caption)
        layout.addWidget(area_caption)

        self.total_leaf_area_label = QLabel("Not calculated")
        apply_measurement_value_style(self.total_leaf_area_label)
        self.total_leaf_area_label.setWordWrap(True)
        layout.addWidget(self.total_leaf_area_label)

        layout.addWidget(make_vertical_divider())

        status_title = QLabel("Status")
        apply_section_title_style(status_title)
        layout.addWidget(status_title)

        self.calibration_status_label = QLabel("Calibration: —")
        self.leaf_mask_status_label = QLabel("Whole Leaf Mask: —")
        self.warnings_label = QLabel("")
        apply_hint_style(self.warnings_label)
        self.warnings_label.setWordWrap(True)
        self.warnings_label.hide()
        for status_label in (
            self.calibration_status_label,
            self.leaf_mask_status_label,
        ):
            status_label.setWordWrap(True)
            layout.addWidget(status_label)
        layout.addWidget(self.warnings_label)

        layout.addWidget(make_vertical_divider())

        actions_title = QLabel("Actions")
        apply_section_title_style(actions_title)
        layout.addWidget(actions_title)

        actions_container = QWidget(self)
        actions_layout = QVBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(CONTROL_ROW_SPACING)
        actions_layout.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )

        self.set_manual_calibration_button = QPushButton("Set 1-Inch Calibration")
        self.save_calibration_button = QPushButton("Save 1-Inch Calibration")
        self.cancel_calibration_button = QPushButton("Cancel Calibration")
        self.preview_leaf_mask_button = QPushButton("Preview Leaf Mask")
        self.run_analysis_button = QPushButton("Run Analysis")

        apply_inspector_action_button(self.set_manual_calibration_button)
        apply_inspector_action_button(self.save_calibration_button)
        apply_inspector_action_button(self.cancel_calibration_button)
        apply_primary_action_button(self.preview_leaf_mask_button)
        apply_inspector_action_button(self.run_analysis_button)

        self.run_analysis_button.setEnabled(False)
        self.run_analysis_button.setToolTip(
            "Final disease analysis is not yet available."
        )

        for button in (
            self.set_manual_calibration_button,
            self.save_calibration_button,
            self.cancel_calibration_button,
            self.preview_leaf_mask_button,
            self.run_analysis_button,
        ):
            actions_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(actions_container)
        layout.addStretch(1)

        self.set_calibration_controls_enabled(False)
        self.set_preview_controls_enabled(False)
        self.show_workspace_state(None)

    def set_preview_controls_enabled(self, enabled: bool) -> None:
        self.preview_leaf_mask_button.setEnabled(enabled)

    def set_calibration_controls_enabled(self, enabled: bool) -> None:
        self.set_manual_calibration_button.setEnabled(enabled)
        self.save_calibration_button.setEnabled(False)
        self.cancel_calibration_button.setEnabled(enabled)

    def set_save_calibration_enabled(self, enabled: bool) -> None:
        self.save_calibration_button.setEnabled(enabled)

    def show_workspace_state(self, workspace: WorkspaceMetadata | None) -> None:
        if workspace is None:
            self.sample_name_label.setText("No workspace open")
        else:
            self.sample_name_label.setText("No sample selected")
        self.condition_group_label.setText("—")
        self.total_leaf_area_label.setText("Not calculated")
        self.calibration_status_label.setText("Calibration: —")
        self.leaf_mask_status_label.setText("Whole Leaf Mask: —")
        self.warnings_label.hide()
        self.warnings_label.setText("")
        self.set_calibration_controls_enabled(False)
        self.set_preview_controls_enabled(False)

    def show_leaf_mask_preview_fields(
        self,
        *,
        status: str | None = None,
        warnings: str | None = None,
    ) -> None:
        self.leaf_mask_status_label.setText(
            f"Whole Leaf Mask: {_format_leaf_mask_status_label(preview_status=status)}"
        )
        if warnings:
            self.warnings_label.setText(warnings)
            self.warnings_label.show()
        else:
            self.warnings_label.hide()
            self.warnings_label.setText("")

    def show_sample(
        self,
        sample: SampleRecord,
        workspace: WorkspaceMetadata,
        *,
        preview_status: str | None = None,
        preview_succeeded_without_calibration: bool = False,
        warnings: str | None = None,
    ) -> None:
        self.sample_name_label.setText(sample.display_name)
        group = workspace.get_condition_group(sample.condition_group_id)
        self.condition_group_label.setText(
            group.display_name if group is not None else "Ungrouped"
        )

        self.calibration_status_label.setText(
            f"Calibration: {_format_calibration_status_label(sample.calibration_status)}"
        )
        self.leaf_mask_status_label.setText(
            f"Whole Leaf Mask: {_format_leaf_mask_status_label(preview_status=preview_status)}"
        )
        self.total_leaf_area_label.setText(
            _format_total_leaf_area_display(
                sample,
                preview_status=preview_status,
                preview_succeeded_without_calibration=preview_succeeded_without_calibration,
            )
        )

        if warnings:
            self.warnings_label.setText(warnings)
            self.warnings_label.show()
        else:
            self.warnings_label.hide()
            self.warnings_label.setText("")

        self.set_calibration_controls_enabled(True)
        self.set_preview_controls_enabled(True)
        self.set_save_calibration_enabled(False)


def build_inspector_panel(parent: QWidget | None = None) -> SampleInspectorPanel:
    """Right-side inspector / contextual settings panel."""
    return SampleInspectorPanel(parent)


def build_workbench_controls(parent: QWidget | None = None) -> tuple[QWidget, QLabel]:
    """Compact transient status strip beneath the image canvas."""
    panel = QWidget(parent)
    apply_workbench_controls_panel_style(panel)
    panel.setFixedHeight(40)

    layout = QHBoxLayout(panel)
    layout.setContentsMargins(PANEL_MARGIN, 6, PANEL_MARGIN, 6)
    layout.setSpacing(12)

    status = QLabel("Ready")
    apply_hint_style(status)
    layout.addWidget(status, stretch=1)

    return panel, status


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

    controls_panel, controls_status = build_workbench_controls(center_column)
    center_layout.addWidget(controls_panel)
    window.controls_panel = controls_panel
    window.controls_status_label = controls_status

    inspector_panel = build_inspector_panel(horizontal_splitter)
    window.inspector_panel = inspector_panel
    window.run_analysis_button = inspector_panel.run_analysis_button

    horizontal_splitter.addWidget(explorer_panel)
    horizontal_splitter.addWidget(center_column)
    horizontal_splitter.addWidget(inspector_panel)
    horizontal_splitter.setStretchFactor(0, 1)
    horizontal_splitter.setStretchFactor(1, 4)
    horizontal_splitter.setStretchFactor(2, 1)
    horizontal_splitter.setSizes([LEFT_PANEL_MIN_WIDTH, 900, INSPECTOR_PANEL_MIN_WIDTH])

    root.addWidget(horizontal_splitter, stretch=1)

    status_bar = QStatusBar(window)
    status_bar.showMessage("Ready")
    window.setStatusBar(status_bar)
    window.status_bar = status_bar
