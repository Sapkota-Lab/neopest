"""Main window layout construction helpers."""

from __future__ import annotations

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
    apply_measurement_value_style,
    apply_panel_margins,
    apply_primary_action_button,
    apply_sample_identity_style,
    apply_status_row_label_style,
    apply_status_row_value_style,
    apply_workspace_preview_panel_style,
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


def _format_calibration_status_value(status: CalibrationStatus) -> str:
    if status is CalibrationStatus.UNCALIBRATED:
        return "Not calibrated"
    if status in {CalibrationStatus.MANUAL, CalibrationStatus.CALIBRATED}:
        return "Ready"
    return "Missing"


def _format_whole_leaf_status_value(*, preview_status: str | None) -> str:
    if preview_status == "Preview Ready":
        return "Ready"
    if preview_status == "Failed Preview":
        return "Failed"
    return "Not previewed"


def _format_disease_mask_status_value(*, preview_status: str | None) -> str:
    if preview_status == "Ready":
        return "Preview ready"
    if preview_status == "Failed":
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
            return "Recalculate leaf mask"
        if sample.leaf_measurement is not None and sample.leaf_measurement.status == "stale":
            return "Recalculate leaf mask"
        return "Not calculated"

    measurement = sample.leaf_measurement
    assert measurement is not None
    if measurement.leaf_area_in2 is None:
        return "Not calculated"
    return f"{measurement.leaf_area_in2:.2f} in²"


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

        area_caption = QLabel("TOTAL LEAF AREA")
        apply_area_caption_style(area_caption)
        layout.addWidget(area_caption)

        self.total_leaf_area_label = QLabel("Not calculated")
        apply_measurement_value_style(self.total_leaf_area_label)
        self.total_leaf_area_label.setWordWrap(True)
        layout.addWidget(self.total_leaf_area_label)

        layout.addSpacing(8)

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

        leaf_mask_caption = QLabel("Whole leaf")
        apply_status_row_label_style(leaf_mask_caption)
        self.whole_leaf_status_value = QLabel("—")
        apply_status_row_value_style(self.whole_leaf_status_value)
        self.whole_leaf_status_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        disease_mask_caption = QLabel("Candidate disease")
        apply_status_row_label_style(disease_mask_caption)
        self.disease_mask_status_value = QLabel("—")
        apply_status_row_value_style(self.disease_mask_status_value)
        self.disease_mask_status_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        status_grid.addWidget(calibration_caption, 0, 0)
        status_grid.addWidget(self.calibration_status_value, 0, 1)
        status_grid.addWidget(leaf_mask_caption, 1, 0)
        status_grid.addWidget(self.whole_leaf_status_value, 1, 1)
        status_grid.addWidget(disease_mask_caption, 2, 0)
        status_grid.addWidget(self.disease_mask_status_value, 2, 1)
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
        self.preview_whole_leaf_button = QPushButton("Preview Whole Leaf")
        self.preview_disease_mask_button = QPushButton("Preview Candidate Disease")
        self.run_analysis_button = QPushButton("Run Analysis")

        apply_inspector_action_button(self.set_manual_calibration_button)
        apply_inspector_action_button(self.save_calibration_button)
        apply_inspector_action_button(self.cancel_calibration_button)
        apply_primary_action_button(self.preview_whole_leaf_button)
        apply_inspector_action_button(self.preview_disease_mask_button)
        apply_inspector_action_button(self.run_analysis_button)

        self.preview_disease_mask_button.setToolTip(
            "Preview candidate brown/necrotic tissue only. Not confirmed disease."
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
            self.preview_whole_leaf_button,
            self.preview_disease_mask_button,
        ):
            actions_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        preview_mode_caption = QLabel("Preview mode")
        apply_status_row_label_style(preview_mode_caption)
        actions_layout.addWidget(preview_mode_caption, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.preview_mode_combo = QComboBox(self)
        self.preview_mode_combo.addItem("Raw Image", "raw_image")
        self.preview_mode_combo.addItem("Whole Leaf", "whole_leaf")
        self.preview_mode_combo.addItem("Candidate Disease", "disease")
        self.preview_mode_combo.addItem("Healthy Tissue Preview", "healthy_tissue")
        self.preview_mode_combo.setToolTip(
            "Candidate disease shows brown/necrotic preview pixels only. "
            "Healthy tissue preview shows leaf pixels not classified as candidates."
        )
        self.preview_mode_combo.setEnabled(False)
        actions_layout.addWidget(self.preview_mode_combo, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.preview_scope_hint = QLabel(
            "Disease output is a candidate preview only."
        )
        apply_hint_style(self.preview_scope_hint)
        self.preview_scope_hint.setWordWrap(True)
        self.preview_scope_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        actions_layout.addWidget(self.preview_scope_hint, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(actions_container)
        layout.addStretch(1)

        self.set_calibration_controls_enabled(False)
        self.set_preview_controls_enabled(False)
        self.set_calibration_in_progress(False)
        self.show_workspace_state(None)

    def set_preview_controls_enabled(self, enabled: bool) -> None:
        self.preview_whole_leaf_button.setEnabled(enabled)
        self.preview_disease_mask_button.setEnabled(enabled)
        if not enabled:
            self.preview_mode_combo.setEnabled(False)

    def set_disease_preview_controls_enabled(self, enabled: bool) -> None:
        self.preview_disease_mask_button.setEnabled(enabled)

    def set_preview_mode_enabled(self, enabled: bool) -> None:
        self.preview_mode_combo.setEnabled(enabled)

    def set_preview_mode_item_enabled(self, mode_key: str, enabled: bool) -> None:
        index = self.preview_mode_combo.findData(mode_key)
        if index >= 0:
            model = self.preview_mode_combo.model()
            item = model.item(index)
            if item is not None:
                item.setEnabled(enabled)

    def get_preview_mode_key(self) -> str:
        value = self.preview_mode_combo.currentData()
        return str(value) if value is not None else "raw_image"

    def set_preview_mode_key(self, mode_key: str) -> None:
        index = self.preview_mode_combo.findData(mode_key)
        if index >= 0:
            self.preview_mode_combo.setCurrentIndex(index)

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

    def show_workspace_state(self, workspace: WorkspaceMetadata | None) -> None:
        if workspace is None:
            self.sample_name_label.setText("No workspace open")
        else:
            self.sample_name_label.setText("Select a sample")
        self.condition_group_label.setText("—")
        self.total_leaf_area_label.setText("Not calculated")
        self.calibration_status_value.setText("—")
        self.whole_leaf_status_value.setText("—")
        self.disease_mask_status_value.setText("—")
        self.preview_mode_combo.setCurrentIndex(0)
        self.preview_mode_combo.setEnabled(False)
        for mode_key in ("raw_image", "whole_leaf", "disease", "healthy_tissue"):
            self.set_preview_mode_item_enabled(mode_key, mode_key == "raw_image")
        self.warnings_label.hide()
        self.warnings_label.setText("")
        self.set_calibration_controls_enabled(False)
        self.set_preview_controls_enabled(False)
        self.set_calibration_in_progress(False)

    def show_whole_leaf_preview_fields(
        self,
        *,
        status: str | None = None,
        warnings: str | None = None,
    ) -> None:
        self.whole_leaf_status_value.setText(
            _format_whole_leaf_status_value(preview_status=status)
        )
        if warnings:
            self.warnings_label.setText(warnings)
            self.warnings_label.show()
        else:
            self.warnings_label.hide()
            self.warnings_label.setText("")

    def show_disease_preview_fields(
        self,
        *,
        status: str | None = None,
        warnings: str | None = None,
    ) -> None:
        self.disease_mask_status_value.setText(
            _format_disease_mask_status_value(preview_status=status)
        )
        if warnings:
            self.warnings_label.setText(warnings)
            self.warnings_label.show()
        else:
            self.warnings_label.hide()
            self.warnings_label.setText("")

    def show_leaf_mask_preview_fields(
        self,
        *,
        status: str | None = None,
        warnings: str | None = None,
    ) -> None:
        """Backward-compatible alias for whole-leaf preview status updates."""
        self.show_whole_leaf_preview_fields(status=status, warnings=warnings)

    def show_sample(
        self,
        sample: SampleRecord,
        workspace: WorkspaceMetadata,
        *,
        preview_status: str | None = None,
        disease_preview_status: str | None = None,
        preview_succeeded_without_calibration: bool = False,
        warnings: str | None = None,
    ) -> None:
        self.sample_name_label.setText(sample.display_name)
        group = workspace.get_condition_group(sample.condition_group_id)
        self.condition_group_label.setText(
            group.display_name if group is not None else "Ungrouped"
        )

        self.calibration_status_value.setText(
            _format_calibration_status_value(sample.calibration_status)
        )
        self.whole_leaf_status_value.setText(
            _format_whole_leaf_status_value(preview_status=preview_status)
        )
        self.disease_mask_status_value.setText(
            _format_disease_mask_status_value(preview_status=disease_preview_status)
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
