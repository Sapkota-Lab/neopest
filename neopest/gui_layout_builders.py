"""Main window layout construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
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
    apply_area_caption_style,
    apply_hint_style,
    apply_inspector_action_button,
    apply_inspector_panel_style,
    apply_main_splitter_style,
    apply_panel_margins,
    apply_primary_action_button,
    apply_sample_identity_style,
    apply_status_row_label_style,
    apply_status_row_value_style,
    apply_workspace_preview_panel_style,
)
from neopest.measurements import (
    candidate_disease_measurement_matches_provenance,
    format_area_in2_display,
    format_coverage_percent_display,
    phase5_measurement_row_placeholder,
    phase5_result_is_current,
    phase5_result_status_display,
)
from neopest.metadata import SampleRecord, WorkspaceMetadata
from neopest.results import CalibrationStatus

if TYPE_CHECKING:
    from neopest.gui import MainWindow

VIEW_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Raw Image", "raw_image"),
    ("Whole Leaf Mask", "whole_leaf"),
    ("Candidate Disease Mask", "disease"),
    ("Healthy Tissue Preview", "healthy_tissue"),
)


def _format_calibration_status_value(status: CalibrationStatus) -> str:
    if status is CalibrationStatus.UNCALIBRATED:
        return "Not calibrated"
    if status in {CalibrationStatus.MANUAL, CalibrationStatus.CALIBRATED}:
        return "Ready"
    return "Missing"


def _calculation_button_state(
    sample: SampleRecord,
    *,
    image_available: bool,
) -> tuple[str, bool, str]:
    if not image_available:
        return (
            "Calculate Results",
            False,
            "Source image is not available for calculation.",
        )
    if sample.calibration is None:
        return (
            "Calculate Results",
            False,
            "Manual calibration is required before calculating results.",
        )

    measurement = sample.candidate_disease_measurement
    if measurement is not None and measurement.status == "valid":
        if candidate_disease_measurement_matches_provenance(
            measurement,
            calibration=sample.calibration,
            image_hash=sample.image_hash,
        ):
            return ("Results Up to Date", False, "Persisted results match current inputs.")

    if measurement is not None and measurement.status == "stale":
        return ("Recalculate Results", True, "Scientific inputs changed; recalculate to refresh.")

    return ("Calculate Results", True, "Run explicit scientific calculation for this sample.")


class SampleInspectorPanel(QWidget):
    """Right-side result panel for the selected sample."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        apply_inspector_panel_style(self)
        self.setMinimumWidth(INSPECTOR_PANEL_MIN_WIDTH)
        self.setMaximumWidth(INSPECTOR_PANEL_MAX_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Sample inspector")

        layout = QVBoxLayout(self)
        apply_panel_margins(layout)
        layout.setSpacing(12)

        self.sample_name_label = QLabel("No workspace open")
        apply_sample_identity_style(self.sample_name_label)
        self.sample_name_label.setWordWrap(True)
        layout.addWidget(self.sample_name_label)

        self.condition_group_label = QLabel("—")
        apply_hint_style(self.condition_group_label)
        self.condition_group_label.setWordWrap(True)
        layout.addWidget(self.condition_group_label)

        layout.addSpacing(8)

        measurements_caption = QLabel("MEASUREMENTS")
        apply_area_caption_style(measurements_caption)
        layout.addWidget(measurements_caption)

        measurements_grid = QGridLayout()
        measurements_grid.setHorizontalSpacing(12)
        measurements_grid.setVerticalSpacing(6)
        measurements_grid.setColumnStretch(1, 1)

        total_leaf_caption = QLabel("Total leaf area")
        apply_status_row_label_style(total_leaf_caption)
        self.total_leaf_area_label = QLabel("Not calculated")
        apply_status_row_value_style(self.total_leaf_area_label)
        self.total_leaf_area_label.setWordWrap(True)
        self.total_leaf_area_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        candidate_disease_caption = QLabel("Candidate disease area")
        apply_status_row_label_style(candidate_disease_caption)
        self.candidate_disease_area_label = QLabel("Not calculated")
        apply_status_row_value_style(self.candidate_disease_area_label)
        self.candidate_disease_area_label.setWordWrap(True)
        self.candidate_disease_area_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        coverage_caption = QLabel("Candidate disease coverage")
        apply_status_row_label_style(coverage_caption)
        self.candidate_disease_coverage_label = QLabel("Not calculated")
        apply_status_row_value_style(self.candidate_disease_coverage_label)
        self.candidate_disease_coverage_label.setWordWrap(True)
        self.candidate_disease_coverage_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        non_candidate_caption = QLabel("Non-candidate leaf area")
        apply_status_row_label_style(non_candidate_caption)
        self.non_candidate_leaf_area_label = QLabel("Not calculated")
        apply_status_row_value_style(self.non_candidate_leaf_area_label)
        self.non_candidate_leaf_area_label.setWordWrap(True)
        self.non_candidate_leaf_area_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        measurements_grid.addWidget(total_leaf_caption, 0, 0)
        measurements_grid.addWidget(self.total_leaf_area_label, 0, 1)
        measurements_grid.addWidget(candidate_disease_caption, 1, 0)
        measurements_grid.addWidget(self.candidate_disease_area_label, 1, 1)
        measurements_grid.addWidget(coverage_caption, 2, 0)
        measurements_grid.addWidget(self.candidate_disease_coverage_label, 2, 1)
        measurements_grid.addWidget(non_candidate_caption, 3, 0)
        measurements_grid.addWidget(self.non_candidate_leaf_area_label, 3, 1)
        layout.addLayout(measurements_grid)

        layout.addSpacing(8)

        status_caption = QLabel("STATUS")
        apply_area_caption_style(status_caption)
        layout.addWidget(status_caption)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(12)
        status_grid.setVerticalSpacing(6)
        status_grid.setColumnStretch(1, 1)

        calibration_caption = QLabel("Calibration")
        apply_status_row_label_style(calibration_caption)
        self.calibration_status_value = QLabel("—")
        apply_status_row_value_style(self.calibration_status_value)
        self.calibration_status_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        result_status_caption = QLabel("Results")
        apply_status_row_label_style(result_status_caption)
        self.result_status_value = QLabel("—")
        apply_status_row_value_style(self.result_status_value)
        self.result_status_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        status_grid.addWidget(calibration_caption, 0, 0)
        status_grid.addWidget(self.calibration_status_value, 0, 1)
        status_grid.addWidget(result_status_caption, 1, 0)
        status_grid.addWidget(self.result_status_value, 1, 1)
        layout.addLayout(status_grid)

        self.warnings_label = QLabel("")
        apply_hint_style(self.warnings_label)
        self.warnings_label.setWordWrap(True)
        self.warnings_label.hide()
        layout.addWidget(self.warnings_label)

        layout.addSpacing(8)

        actions_container = QWidget(self)
        actions_layout = QVBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(CONTROL_ROW_SPACING)
        actions_layout.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )

        self.set_manual_calibration_button = QPushButton("Set 1-Inch Calibration")
        self.save_calibration_button = QPushButton("Save Calibration")
        self.cancel_calibration_button = QPushButton("Cancel Calibration")
        self.calculate_results_button = QPushButton("Calculate Results")
        self.run_analysis_button = QPushButton("Run Analysis")

        apply_inspector_action_button(self.set_manual_calibration_button)
        apply_inspector_action_button(self.save_calibration_button)
        apply_inspector_action_button(self.cancel_calibration_button)
        apply_primary_action_button(self.calculate_results_button)
        apply_inspector_action_button(self.run_analysis_button)

        self.calculate_results_button.setToolTip(
            "Explicitly calculate and persist total leaf area and candidate disease metrics."
        )
        self.run_analysis_button.setEnabled(False)
        self.run_analysis_button.setVisible(False)
        self.run_analysis_button.setToolTip(
            "Final disease analysis is not yet available."
        )
        self.save_calibration_button.setVisible(False)
        self.cancel_calibration_button.setVisible(False)

        for button in (
            self.set_manual_calibration_button,
            self.save_calibration_button,
            self.cancel_calibration_button,
            self.calculate_results_button,
        ):
            actions_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        view_caption = QLabel("View")
        apply_status_row_label_style(view_caption)
        actions_layout.addWidget(view_caption, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.view_mode_combo = QComboBox(self)
        for label, key in VIEW_MODE_OPTIONS:
            self.view_mode_combo.addItem(label, key)
        self.view_mode_combo.setToolTip(
            "Display-only preview. Changing View does not calculate or save results."
        )
        self.view_mode_combo.setEnabled(False)
        actions_layout.addWidget(self.view_mode_combo, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(actions_container)
        layout.addStretch(1)

        self.set_calibration_controls_enabled(False)
        self.set_view_controls_enabled(False)
        self.set_calculation_controls_enabled(False)
        self.set_calibration_in_progress(False)
        self.show_workspace_state(None)

    def set_view_controls_enabled(self, enabled: bool) -> None:
        self.view_mode_combo.setEnabled(enabled)

    def set_calculation_controls_enabled(self, enabled: bool) -> None:
        self.calculate_results_button.setEnabled(enabled)

    def get_view_mode_key(self) -> str:
        value = self.view_mode_combo.currentData()
        return str(value) if value is not None else "raw_image"

    def set_view_mode_key(self, mode_key: str) -> None:
        index = self.view_mode_combo.findData(mode_key)
        if index >= 0:
            self.view_mode_combo.setCurrentIndex(index)

    def set_calculation_button_state(
        self,
        *,
        label: str,
        enabled: bool,
        hint: str | None = None,
    ) -> None:
        self.calculate_results_button.setText(label)
        self.calculate_results_button.setEnabled(enabled)
        if enabled:
            self.calculate_results_button.setToolTip(
                "Explicitly calculate and persist total leaf area and candidate disease metrics."
            )
        elif hint:
            self.calculate_results_button.setToolTip(hint)
        else:
            self.calculate_results_button.setToolTip("")

    def set_calibration_controls_enabled(self, enabled: bool) -> None:
        self.set_manual_calibration_button.setEnabled(enabled)
        self.cancel_calibration_button.setEnabled(enabled)

    def set_calibration_in_progress(self, active: bool) -> None:
        self.set_manual_calibration_button.setVisible(not active)
        self.save_calibration_button.setVisible(active)
        self.cancel_calibration_button.setVisible(active)
        if active:
            self.save_calibration_button.setEnabled(False)
        else:
            self.save_calibration_button.setEnabled(False)

    def set_save_calibration_enabled(self, enabled: bool) -> None:
        if self.save_calibration_button.isVisible():
            self.save_calibration_button.setEnabled(enabled)

    def _set_measurement_placeholder(self, text: str) -> None:
        self.total_leaf_area_label.setText(text)
        self.candidate_disease_area_label.setText(text)
        self.candidate_disease_coverage_label.setText(text)
        self.non_candidate_leaf_area_label.setText(text)

    def _reset_measurement_labels(self) -> None:
        self._set_measurement_placeholder("Not calculated")

    def show_workspace_state(self, workspace: WorkspaceMetadata | None) -> None:
        if workspace is None:
            self.sample_name_label.setText("No workspace open")
        else:
            self.sample_name_label.setText("Select a sample")
        self.condition_group_label.setText("—")
        self._reset_measurement_labels()
        self.calibration_status_value.setText("—")
        self.result_status_value.setText("—")
        self.view_mode_combo.setCurrentIndex(0)
        self.view_mode_combo.setEnabled(False)
        self.warnings_label.hide()
        self.warnings_label.setText("")
        self.set_calibration_controls_enabled(False)
        self.set_view_controls_enabled(False)
        self.set_calculation_controls_enabled(False)
        self.set_calculation_button_state(label="Calculate Results", enabled=False, hint=None)
        self.set_calibration_in_progress(False)

    def show_preview_warnings(self, warnings: str | None) -> None:
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
        image_available: bool = True,
        warnings: str | None = None,
        calculation_failed: bool = False,
    ) -> None:
        self.sample_name_label.setText(sample.display_name)
        group = workspace.get_condition_group(sample.condition_group_id)
        self.condition_group_label.setText(
            group.display_name if group is not None else "Ungrouped"
        )

        self.calibration_status_value.setText(
            _format_calibration_status_value(sample.calibration_status)
        )
        self.result_status_value.setText(
            phase5_result_status_display(sample, calculation_failed=calculation_failed)
        )

        if phase5_result_is_current(sample):
            measurement = sample.candidate_disease_measurement
            assert measurement is not None
            self.total_leaf_area_label.setText(
                format_area_in2_display(measurement.total_leaf_area_in2)
            )
            self.candidate_disease_area_label.setText(
                format_area_in2_display(measurement.candidate_disease_area_in2)
            )
            self.candidate_disease_coverage_label.setText(
                format_coverage_percent_display(
                    measurement.candidate_disease_coverage_percent
                )
            )
            self.non_candidate_leaf_area_label.setText(
                format_area_in2_display(measurement.non_candidate_leaf_area_in2)
            )
        else:
            placeholder = phase5_measurement_row_placeholder(
                sample,
                calculation_failed=calculation_failed,
            )
            self._set_measurement_placeholder(placeholder)

        label, enabled, hint = _calculation_button_state(
            sample,
            image_available=image_available,
        )
        self.set_calculation_button_state(label=label, enabled=enabled, hint=hint)
        self.show_preview_warnings(warnings)

        self.set_calibration_controls_enabled(True)
        self.set_view_controls_enabled(True)
        self.set_save_calibration_enabled(False)


def build_inspector_panel(parent: QWidget | None = None) -> SampleInspectorPanel:
    """Right-side inspector / contextual settings panel."""
    return SampleInspectorPanel(parent)


def sample_image_available(sample: SampleRecord, workspace_root: Path) -> bool:
    """Return True when the managed source image file exists on disk."""
    return sample.raw_image_absolute(workspace_root).is_file()


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
