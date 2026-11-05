"""NeoPest desktop GUI — MainWindow orchestration."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QKeyEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
)

from neopest import __version__
from neopest.calibration import (
    CURRENT_PIXEL_GRID,
    CalibrationError,
    build_one_inch_manual_calibration_record,
    calibration_matches_display_grid,
)
from neopest.desktop_services import open_path_in_file_manager
from neopest.gui_layout_builders import build_main_layout, sample_image_available
from neopest.gui_styles import build_application_stylesheet
from neopest.image_io import (
    DISEASE_MASK_OVERLAY_RGBA,
    HEALTHY_TISSUE_MASK_OVERLAY_RGBA,
    SUPPORTED_IMAGE_SUFFIXES,
    disease_mask_to_qimage,
    healthy_tissue_mask_to_qimage,
    leaf_mask_to_qimage,
    load_image_rgb_array,
    mask_to_qimage,
    write_overlay_region,
)
from neopest.calculation import (
    CalculationError,
    ExplicitCalculationOutput,
    run_explicit_calculation,
)
from neopest.mask_correction import (
    CorrectionHistory,
    MaskStroke,
    ADD_AREA,
    apply_corrections,
    build_disease_mask_correction_record,
    paint_disks,
    can_start_disease_mask_correction,
    correction_display_state,
    reconcile_correction_after_calculation,
)
from neopest.metadata import (
    WORKSPACE_JSON_FILENAME,
    DiseaseMaskCorrectionRecord,
    SampleRecord,
    WorkspaceMetadata,
)
from neopest.project_manager import (
    WorkspaceError,
    create_condition_group,
    ensure_default_workspace,
    get_default_workspace_root,
    import_images,
    initialize_workspace,
    load_workspace,
    move_sample_to_condition_group,
    remove_condition_group,
    remove_sample,
    rename_condition_group,
    rename_sample,
    save_sample_calibration,
    save_sample_calculation_results,
    save_sample_disease_mask_correction,
)
from neopest.workspace_compat import legacy_inherited_calibration_repair_message
from neopest.results import CalibrationStatus
from neopest.disease_segmentation import (
    DISEASE_PREVIEW_METHOD,
    DISEASE_SETTINGS_VERSION,
    DiseaseMaskPreview,
    DiseaseMaskPreviewStatus,
    enforce_mask_invariants,
    segment_disease_preview,
)
from neopest.segmentation import (
    LeafMaskPreview,
    LeafMaskPreviewStatus,
    PREVIEW_METHOD,
    SETTINGS_VERSION,
    segment_leaf_preview,
)


def _format_disease_mask_preview_status(
    preview: DiseaseMaskPreview | None,
) -> tuple[str | None, str | None]:
    """Translate disease preview results into plain inspector strings."""
    if preview is None:
        return None, None

    if preview.status is DiseaseMaskPreviewStatus.READY:
        status = "Ready"
    else:
        status = "Failed"

    if preview.warnings:
        warnings = "\n".join(preview.warnings)
    elif preview.status is DiseaseMaskPreviewStatus.FAILED:
        warnings = "Candidate disease preview could not be generated."
    else:
        warnings = None

    return status, warnings


def _format_leaf_mask_preview_status(
    preview: LeafMaskPreview | None,
) -> tuple[str | None, str | None]:
    """Translate preview results into plain inspector strings."""
    if preview is None:
        return None, None

    if preview.status is LeafMaskPreviewStatus.SUCCESS:
        status = "Preview Ready"
    else:
        status = "Failed Preview"

    if preview.warnings:
        warnings = "\n".join(preview.warnings)
    elif preview.status is LeafMaskPreviewStatus.FAILED:
        warnings = "Leaf mask preview could not be generated."
    else:
        warnings = None

    return status, warnings


def _format_status_bar_message(
    *,
    workspace: WorkspaceMetadata | None = None,
    sample: SampleRecord | None = None,
    transient: str | None = None,
) -> str:
    if transient:
        return transient
    if workspace is None:
        return "Ready"
    if sample is None:
        return (
            f"Ready | Workspace: {workspace.display_name} "
            f"| Samples: {len(workspace.samples)}"
        )
    calibration_text = (
        "Ready"
        if sample.calibration_status
        in {CalibrationStatus.MANUAL, CalibrationStatus.CALIBRATED}
        else "Not calibrated"
    )
    return f"Ready | Sample: {sample.display_name} | Calibration: {calibration_text}"


class MainWindow(QMainWindow):
    """Primary workbench window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"NeoPest {__version__}")
        self.resize(1280, 800)
        self.setMinimumSize(960, 640)
        self._workspace: WorkspaceMetadata | None = None
        self._selected_sample_id: str | None = None
        self._leaf_mask_previews: dict[str, LeafMaskPreview] = {}
        self._disease_mask_previews: dict[str, DiseaseMaskPreview] = {}
        self._preview_modes: dict[str, str] = {}
        self._calculation_failed_sample_ids: set[str] = set()
        self._correction_sample_id: str | None = None
        self._correction_history: CorrectionHistory | None = None
        self._correction_leaf_mask: np.ndarray | None = None
        self._correction_automatic_mask: np.ndarray | None = None
        self._correction_preview_stroke: MaskStroke | None = None
        self._correction_modified = False
        self._correction_loaded_state = "none"
        self._correction_committed_mask: np.ndarray | None = None
        self._correction_preview_mask: np.ndarray | None = None
        self._correction_live_points: tuple[tuple[float, float], ...] = ()
        self._correction_live_operation: str | None = None
        self._correction_live_radius: float | None = None
        self._correction_overlay_mode: str | None = None
        self._preview_image_hash: dict[str, str | None] = {}
        self._calculation_request_id = 0
        self._calculation_active = False
        self._calculation_job: _ExplicitCalculationJob | None = None
        self._correction_prepare_request_id = 0
        self._correction_prepare_active = False
        self._correction_prepare_job: _DiseaseMaskPreviewJob | None = None
        self._prepared_correction_previews: tuple[LeafMaskPreview, DiseaseMaskPreview] | None = None
        self._work_progress_owner: tuple[str, int, str] | None = None
        build_main_layout(self)
        _install_view_gesture_filter()
        self._build_menus()
        self._connect_signals()
        self._load_default_workspace_on_startup()
        self._refresh_ui()

    def _notify_legacy_inherited_calibration_repairs(
        self,
        workspace: WorkspaceMetadata,
    ) -> None:
        repair_count = workspace.legacy_inherited_calibration_repairs
        if repair_count <= 0:
            return
        QMessageBox.information(
            self,
            "Workspace Updated",
            legacy_inherited_calibration_repair_message(repair_count),
        )
        workspace.legacy_inherited_calibration_repairs = 0

    def _load_default_workspace_on_startup(self) -> None:
        try:
            self._workspace = ensure_default_workspace()
            self._selected_sample_id = None
            if self._workspace is not None:
                self._notify_legacy_inherited_calibration_repairs(self._workspace)
        except WorkspaceError as exc:
            QMessageBox.warning(
                self,
                "Default Workspace Error",
                f"NeoPest could not open the default workspace:\n{exc}",
            )
            self._workspace = None
            self._selected_sample_id = None
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Default Workspace Error",
                f"NeoPest could not create or open the default workspace:\n{exc}",
            )
            self._workspace = None
            self._selected_sample_id = None

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self._action_new_workspace = QAction("New Workspace…", self)
        self._action_new_workspace.triggered.connect(self._on_new_workspace)
        file_menu.addAction(self._action_new_workspace)

        self._action_open_workspace = QAction("Open Workspace…", self)
        self._action_open_workspace.triggered.connect(self._on_open_workspace)
        file_menu.addAction(self._action_open_workspace)

        self._action_open_default_workspace = QAction("Open Default Workspace", self)
        self._action_open_default_workspace.triggered.connect(self._on_open_default_workspace)
        file_menu.addAction(self._action_open_default_workspace)

        self._action_new_default_workspace = QAction("New Default Workspace", self)
        self._action_new_default_workspace.triggered.connect(self._on_new_default_workspace)
        file_menu.addAction(self._action_new_default_workspace)

        file_menu.addSeparator()

        self._action_import_images = QAction("Import Images…", self)
        self._action_import_images.triggered.connect(self._on_import_images)
        self._action_import_images.setEnabled(False)
        file_menu.addAction(self._action_import_images)

        self._action_create_condition_group = QAction("New Condition Group…", self)
        self._action_create_condition_group.triggered.connect(self._on_create_condition_group)
        self._action_create_condition_group.setEnabled(False)
        file_menu.addAction(self._action_create_condition_group)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _connect_signals(self) -> None:
        tree = self.explorer_tree
        tree.sample_selected.connect(self._on_sample_selected)
        tree.selection_cleared.connect(self._on_selection_cleared)
        tree.new_condition_group_requested.connect(self._on_create_condition_group)
        tree.import_images_requested.connect(self._on_import_images)
        tree.import_images_into_group_requested.connect(self._on_import_images_into_group)
        tree.open_workspace_folder_requested.connect(self._on_open_workspace_folder)
        tree.rename_condition_group_requested.connect(self._on_rename_condition_group)
        tree.remove_condition_group_requested.connect(self._on_remove_condition_group)
        tree.sample_preview_requested.connect(self._on_sample_selected)
        tree.rename_sample_requested.connect(self._on_rename_sample)
        tree.remove_sample_requested.connect(self._on_remove_sample)
        tree.reveal_sample_image_requested.connect(self._on_reveal_sample_image)
        tree.sample_move_requested.connect(self._on_sample_move_requested)
        tree.files_dropped_on_group.connect(self._on_files_dropped_on_group)
        self.image_canvas.calibration_points_changed.connect(
            self._on_calibration_points_changed
        )
        self.image_canvas.correction_stroke_preview.connect(self._on_correction_stroke_preview)
        self.image_canvas.correction_stroke_finished.connect(self._on_correction_stroke_finished)
        self.inspector_panel.correct_disease_mask_button.clicked.connect(
            self._on_correct_disease_mask
        )
        self.inspector_panel.add_area_button.toggled.connect(self._sync_canvas_brush)
        self.inspector_panel.remove_area_button.toggled.connect(self._sync_canvas_brush)
        self.inspector_panel.brush_size_slider.valueChanged.connect(self._sync_canvas_brush)
        self.inspector_panel.undo_correction_button.clicked.connect(self._on_undo_correction)
        self.inspector_panel.redo_correction_button.clicked.connect(self._on_redo_correction)
        self.inspector_panel.reset_corrections_button.clicked.connect(
            self._on_reset_corrections
        )
        self.inspector_panel.done_correction_button.clicked.connect(self._on_done_correction)
        self.inspector_panel.set_manual_calibration_button.clicked.connect(
            self._on_set_manual_calibration
        )
        self.inspector_panel.save_calibration_button.clicked.connect(
            self._on_save_calibration
        )
        self.inspector_panel.cancel_calibration_button.clicked.connect(
            self._on_cancel_calibration
        )
        self.inspector_panel.calculate_results_button.clicked.connect(
            self._on_calculate_results
        )
        self.inspector_panel.view_mode_combo.currentIndexChanged.connect(
            self._on_view_mode_changed
        )
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        undo_shortcut.activated.connect(self._on_undo_correction)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        redo_shortcut.activated.connect(self._on_redo_correction)

    def _set_workspace(self, workspace: WorkspaceMetadata | None) -> None:
        if not self._leave_correction_session(save=True):
            self._exit_correction_mode()
        self._calculation_request_id += 1
        self._correction_prepare_request_id += 1
        self._abandon_work_progress()
        self._workspace = workspace
        self._selected_sample_id = None
        self._leaf_mask_previews = {}
        self._disease_mask_previews = {}
        self._preview_image_hash = {}
        self._preview_modes = {}
        self._calculation_failed_sample_ids = set()
        self._refresh_ui()

    def _current_sample(self) -> SampleRecord | None:
        if self._workspace is None or self._selected_sample_id is None:
            return None
        return self._workspace.get_sample(self._selected_sample_id)

    def _update_status_bar(
        self,
        *,
        sample: SampleRecord | None = None,
        transient: str | None = None,
    ) -> None:
        self.status_bar.showMessage(
            _format_status_bar_message(
                workspace=self._workspace,
                sample=sample,
                transient=transient,
            )
        )

    def _begin_work_progress(self, owner: tuple[str, int, str], stage: str) -> None:
        """Show the status-bar indicator immediately for one request."""
        self._work_progress_owner = owner
        self.work_progress_label.setText(stage)
        self.status_bar.clearMessage()
        self._sync_work_progress_visibility()
        self.work_progress_bar.repaint()
        self.status_bar.repaint()
        app = QApplication.instance()
        if app is not None:
            app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _note_work_stage(self, owner: tuple[str, int, str], stage: str) -> None:
        """Update the label when this request is still the one on screen."""
        if self._work_progress_owner != owner:
            return
        kind, request_id, sample_id = owner
        if kind == "calculation" and request_id != self._calculation_request_id:
            return
        if kind == "correction" and request_id != self._correction_prepare_request_id:
            return
        if self._selected_sample_id != sample_id:
            return
        self.work_progress_label.setText(stage)
        self._sync_work_progress_visibility()

    def _clear_work_progress(self, owner: tuple[str, int, str]) -> None:
        """Hide the indicator only if ``owner`` is still the visible request."""
        if self._work_progress_owner != owner:
            return
        self._work_progress_owner = None
        self.work_progress_label.clear()
        self._sync_work_progress_visibility()

    def _abandon_work_progress(self) -> None:
        """Drop the indicator when the workspace cancels outstanding work."""
        self._work_progress_owner = None
        self.work_progress_label.clear()
        self._sync_work_progress_visibility()

    def _sync_work_progress_visibility(self) -> None:
        owner = self._work_progress_owner
        current = False
        if owner is not None and self._selected_sample_id == owner[2]:
            kind, request_id, _sample_id = owner
            if kind == "calculation":
                current = request_id == self._calculation_request_id
            elif kind == "correction":
                current = request_id == self._correction_prepare_request_id
        self.work_progress_label.setVisible(current)
        self.work_progress_bar.setVisible(current)
        if current:
            # Segmentation does not report a fraction, so the bar stays busy.
            self.work_progress_bar.setRange(0, 0)
        else:
            self.work_progress_bar.setRange(0, 1)
            self.work_progress_bar.setValue(0)

    def _refresh_ui(self) -> None:
        workspace_open = self._workspace is not None
        self._action_import_images.setEnabled(workspace_open)
        self._action_create_condition_group.setEnabled(workspace_open)

        if self._workspace is None:
            self.explorer_tree.refresh_placeholder()
            self.image_canvas.clear_image()
            self.image_canvas.set_hint("Open or create a workspace to begin")
            self.inspector_panel.show_workspace_state(None)
            self._update_status_bar()
            self.setWindowTitle(f"NeoPest {__version__}")
            return

        self.explorer_tree.refresh_workspace(self._workspace)
        self.inspector_panel.show_workspace_state(self._workspace)
        self.image_canvas.clear_image()
        self.image_canvas.set_hint("Select a sample to view its image")
        self._update_status_bar()
        self.setWindowTitle(
            f"NeoPest {__version__} — {self._workspace.display_name}"
        )

        if self._selected_sample_id:
            self.explorer_tree.select_sample(self._selected_sample_id)
            self._show_sample(self._selected_sample_id)

    def _on_new_workspace(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Folder for New Workspace",
        )
        if not folder:
            return

        root = Path(folder)
        display_name, ok = QInputDialog.getText(
            self,
            "Workspace Name",
            "Display name:",
            text=root.name,
        )
        if not ok:
            return

        try:
            workspace = initialize_workspace(root, display_name=display_name.strip() or root.name)
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Workspace Already Exists", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._set_workspace(workspace)

    def _on_open_default_workspace(self) -> None:
        root = get_default_workspace_root()
        try:
            workspace = load_workspace(root)
        except WorkspaceError:
            QMessageBox.information(
                self,
                "Default Workspace Not Found",
                f"No NeoPest workspace was found at:\n{root}\n\n"
                f"Use File → New Default Workspace to create it, or "
                f"File → New Workspace… to choose another location.",
            )
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._set_workspace(workspace)
        if workspace.legacy_inherited_calibration_repairs > 0:
            self._notify_legacy_inherited_calibration_repairs(workspace)

    def _on_new_default_workspace(self) -> None:
        root = get_default_workspace_root()
        metadata_path = root / WORKSPACE_JSON_FILENAME
        if metadata_path.is_file():
            QMessageBox.warning(
                self,
                "Default Workspace Already Exists",
                f"A NeoPest workspace already exists at:\n{metadata_path}\n\n"
                "Use File → Open Default Workspace to open it.",
            )
            return

        try:
            workspace = initialize_workspace(root, display_name="NeoPest")
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Workspace Already Exists", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._set_workspace(workspace)

    def _on_open_workspace(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Workspace Folder",
        )
        if not folder:
            return

        root = Path(folder)
        try:
            workspace = load_workspace(root)
        except WorkspaceError:
            QMessageBox.warning(
                self,
                "Not a NeoPest Workspace",
                f"The selected folder does not contain {WORKSPACE_JSON_FILENAME}.\n\n"
                "Choose an existing NeoPest workspace folder, or use "
                "File → New Workspace… to create one.",
            )
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._set_workspace(workspace)
        if workspace.legacy_inherited_calibration_repairs > 0:
            self._notify_legacy_inherited_calibration_repairs(workspace)

    def _on_create_condition_group(self) -> None:
        if self._workspace is None:
            return

        display_name, ok = QInputDialog.getText(
            self,
            "New Condition Group",
            "Condition group name:",
        )
        if not ok or not display_name.strip():
            return

        try:
            create_condition_group(self._workspace, display_name=display_name.strip())
        except WorkspaceError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._refresh_ui()

    def _import_paths_into_group(
        self,
        paths: list[Path | str],
        condition_group_id: str | None = None,
    ) -> None:
        """Import ``paths`` with the same validation and error dialogs as File → Import."""
        if self._workspace is None or not paths:
            return

        try:
            imported = import_images(
                self._workspace,
                [Path(path) for path in paths],
                condition_group_id=condition_group_id,
            )
        except WorkspaceError as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        if imported:
            self._selected_sample_id = imported[-1].sample_id
        self._refresh_ui()

    def _on_import_images(self, condition_group_id: str | None = None) -> None:
        if self._workspace is None:
            return

        suffix_filter = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_SUFFIXES))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Still Images",
            "",
            f"Images ({suffix_filter});;All Files (*)",
        )
        if not paths:
            return

        self._import_paths_into_group(paths, condition_group_id)

    def _on_import_images_into_group(self, condition_group_id: str) -> None:
        self._on_import_images(condition_group_id=condition_group_id)

    def _on_files_dropped_on_group(self, condition_group_id: str, paths: list[str]) -> None:
        self._import_paths_into_group(paths, condition_group_id)

    def _on_open_workspace_folder(self) -> None:
        if self._workspace is None:
            return
        if not open_path_in_file_manager(self._workspace.root_path):
            QMessageBox.warning(
                self,
                "Open Folder Failed",
                "NeoPest could not open the workspace folder.",
            )

    def _on_rename_condition_group(self, condition_group_id: str) -> None:
        if self._workspace is None:
            return
        group = self._workspace.get_condition_group(condition_group_id)
        if group is None:
            return

        display_name, ok = QInputDialog.getText(
            self,
            "Rename Condition Group",
            "Condition group name:",
            text=group.display_name,
        )
        if not ok or not display_name.strip():
            return

        try:
            rename_condition_group(
                self._workspace,
                condition_group_id,
                display_name=display_name.strip(),
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Rename Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._refresh_ui()

    def _on_remove_condition_group(self, condition_group_id: str) -> None:
        if self._workspace is None:
            return
        group = self._workspace.get_condition_group(condition_group_id)
        if group is None:
            return

        sample_count = len(self._workspace.samples_for_group(condition_group_id))
        if sample_count:
            message = (
                f"Remove condition group '{group.display_name}' and its "
                f"{sample_count} sample(s) from this workspace?\n\n"
                "Managed workspace image copies will be deleted. "
                "Original source files outside the workspace are not deleted."
            )
        else:
            message = (
                f"Remove empty condition group '{group.display_name}' "
                "from this workspace?"
            )

        answer = QMessageBox.question(
            self,
            "Remove Condition Group",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed_sample_ids = {
            sample.sample_id
            for sample in self._workspace.samples_for_group(condition_group_id)
        }
        if self._selected_sample_id in removed_sample_ids:
            self._selected_sample_id = None

        try:
            remove_condition_group(self._workspace, condition_group_id)
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Remove Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._leaf_mask_previews = {
            key: value
            for key, value in self._leaf_mask_previews.items()
            if key not in removed_sample_ids
        }
        self._disease_mask_previews = {
            key: value
            for key, value in self._disease_mask_previews.items()
            if key not in removed_sample_ids
        }
        self._preview_modes = {
            key: value
            for key, value in self._preview_modes.items()
            if key not in removed_sample_ids
        }
        self._refresh_ui()

    def _on_rename_sample(self, sample_id: str) -> None:
        if self._workspace is None:
            return
        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            return

        display_name, ok = QInputDialog.getText(
            self,
            "Rename Sample",
            "Sample name:",
            text=sample.display_name,
        )
        if not ok or not display_name.strip():
            return

        try:
            rename_sample(
                self._workspace,
                sample_id,
                display_name=display_name.strip(),
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Rename Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._refresh_ui()
        if self._selected_sample_id == sample_id:
            self._show_sample(sample_id)

    def _on_remove_sample(self, sample_id: str) -> None:
        if self._workspace is None:
            return
        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            return

        answer = QMessageBox.question(
            self,
            "Remove Sample",
            f"Remove sample '{sample.display_name}' from this workspace?\n\n"
            "The managed workspace image copy will be deleted. "
            "Original source files outside the workspace are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if self._selected_sample_id == sample_id:
            self._selected_sample_id = None

        try:
            remove_sample(self._workspace, sample_id)
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Remove Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._leaf_mask_previews.pop(sample_id, None)
        self._disease_mask_previews.pop(sample_id, None)
        self._preview_modes.pop(sample_id, None)
        self._refresh_ui()

    def _on_reveal_sample_image(self, sample_id: str) -> None:
        if self._workspace is None:
            return
        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            return

        image_path = sample.raw_image_absolute(self._workspace.root_path)
        if not open_path_in_file_manager(image_path):
            QMessageBox.warning(
                self,
                "Reveal Image Failed",
                "NeoPest could not reveal the managed workspace image file.",
            )

    def _on_sample_move_requested(
        self,
        sample_id: str,
        destination_condition_group_id: str,
    ) -> None:
        if self._workspace is None:
            return

        sample = self._workspace.get_sample(sample_id)
        destination_group = self._workspace.get_condition_group(
            destination_condition_group_id
        )
        if sample is None or destination_group is None:
            return

        if sample.condition_group_id == destination_condition_group_id:
            return

        try:
            updated = move_sample_to_condition_group(
                self._workspace,
                sample_id,
                destination_condition_group_id,
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Move Sample Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Move Sample Failed", str(exc))
            return

        self._selected_sample_id = sample_id
        self.explorer_tree.refresh_workspace(self._workspace)
        self.explorer_tree.expand_condition_group(destination_condition_group_id)
        self.explorer_tree.select_sample(sample_id)
        self._show_sample(sample_id)
        self._update_status_bar(
            sample=updated,
            transient=(
                f"Moved {updated.display_name} to {destination_group.display_name}"
            ),
        )

    def _on_sample_selected(self, sample_id: str) -> None:
        self._selected_sample_id = sample_id
        self._sync_work_progress_visibility()
        self._show_sample(sample_id)

    def _on_selection_cleared(self) -> None:
        if not self._leave_correction_session(save=True):
            return
        self._selected_sample_id = None
        self.image_canvas.set_calibration_mode(False)
        self.image_canvas.clear_leaf_mask_overlay()
        if self._workspace is None:
            self.inspector_panel.show_workspace_state(None)
        else:
            self.inspector_panel.show_workspace_state(self._workspace)
        self.image_canvas.clear_image()
        if self._workspace is not None:
            self.image_canvas.set_hint("Select a sample to view its image")
        self._sync_work_progress_visibility()
        self._update_status_bar()

    def _on_set_manual_calibration(self) -> None:
        if self._correction_sample_id is not None:
            return
        if self._workspace is None or self._selected_sample_id is None:
            QMessageBox.information(
                self,
                "No Sample Selected",
                "Select a sample with a raw image before setting manual calibration.",
            )
            return

        self.image_canvas.set_calibration_mode(True)
        self.inspector_panel.set_calibration_in_progress(True)
        self.inspector_panel.set_save_calibration_enabled(False)
        self._update_status_bar(
            transient="Manual calibration: click two ruler marks exactly 1 inch apart"
        )

    def _on_calibration_points_changed(self) -> None:
        points = self.image_canvas.get_draft_calibration_points()
        self.inspector_panel.set_save_calibration_enabled(points is not None)

    def _on_save_calibration(self) -> None:
        if self._workspace is None or self._selected_sample_id is None:
            QMessageBox.information(
                self,
                "No Sample Selected",
                "Select a sample before saving calibration.",
            )
            return

        points = self.image_canvas.get_draft_calibration_points()
        if points is None:
            QMessageBox.warning(
                self,
                "Calibration Points Required",
                "Click two ruler marks exactly 1 inch apart before saving calibration.",
            )
            return

        try:
            calibration = build_one_inch_manual_calibration_record(
                points[0],
                points[1],
            )
            sample = save_sample_calibration(
                self._workspace,
                self._selected_sample_id,
                calibration,
            )
        except CalibrationError as exc:
            QMessageBox.warning(self, "Invalid Calibration", str(exc))
            return
        except WorkspaceError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self.image_canvas.set_calibration_mode(False)
        self.inspector_panel.set_calibration_in_progress(False)
        self._apply_sample_calibration_display(sample)
        self._update_status_bar(sample=sample, transient="1-inch calibration saved")

    def _on_cancel_calibration(self) -> None:
        self.image_canvas.set_calibration_mode(False)
        self.image_canvas.clear_draft_calibration()
        self.inspector_panel.set_calibration_in_progress(False)
        if self._workspace is not None and self._selected_sample_id is not None:
            sample = self._workspace.get_sample(self._selected_sample_id)
            if sample is not None:
                self._apply_sample_calibration_display(sample)
                self._update_status_bar(sample=sample, transient="Calibration cancelled")
                return
        self.inspector_panel.set_save_calibration_enabled(False)
        self._update_status_bar(transient="Calibration cancelled")

    def _sample_image_available(self, sample: SampleRecord) -> bool:
        if self._workspace is None:
            return False
        return sample_image_available(sample, self._workspace.root_path)

    def _refresh_sample_inspector(
        self,
        sample: SampleRecord,
        *,
        warnings: str | None = None,
        calculation_failed: bool | None = None,
        correction_override: DiseaseMaskCorrectionRecord | None = None,
    ) -> None:
        if self._workspace is None:
            return
        if calculation_failed is None:
            calculation_failed = sample.sample_id in self._calculation_failed_sample_ids
        shown = sample
        if (
            correction_override is None
            and self._correction_sample_id == sample.sample_id
        ):
            correction_override = self._session_correction_record(sample)
        if correction_override is not None:
            shown = replace(sample, disease_mask_correction=correction_override)
        self.inspector_panel.show_sample(
            shown,
            self._workspace,
            image_available=self._sample_image_available(sample),
            warnings=warnings,
            calculation_failed=calculation_failed,
            calibration_grid_matches=self._calibration_grid_matches(sample),
        )
        if self._correction_sample_id == sample.sample_id:
            self._update_correction_history_buttons()

    def _invalidate_disease_preview(self, sample_id: str) -> None:
        self._disease_mask_previews.pop(sample_id, None)
        if self._preview_modes.get(sample_id) in {"disease", "healthy_tissue"}:
            self._preview_modes[sample_id] = "raw_image"

    def _update_view_mode_controls(self, sample_id: str | None) -> None:
        panel = self.inspector_panel
        if sample_id is None:
            panel.set_view_controls_enabled(False)
            panel.set_view_mode_key("raw_image")
            return

        panel.set_view_controls_enabled(True)
        mode = self._preview_modes.get(sample_id, "raw_image")
        panel.set_view_mode_key(mode)

    def _get_or_create_transient_leaf_mask_preview(
        self,
        sample: SampleRecord,
    ) -> LeafMaskPreview:
        """Return a cached or freshly generated whole-leaf mask without persisting area."""
        existing = self._leaf_mask_previews.get(sample.sample_id)
        if (
            existing is not None
            and existing.status is LeafMaskPreviewStatus.SUCCESS
            and existing.mask is not None
            and self._preview_source_matches(sample)
            and int(existing.settings_snapshot.get("settings_version", -1)) == SETTINGS_VERSION
        ):
            return existing

        self._disease_mask_previews.pop(sample.sample_id, None)
        image_path = sample.raw_image_absolute(self._workspace.root_path)  # type: ignore[union-attr]
        try:
            image_array = load_image_rgb_array(image_path)
            preview = segment_leaf_preview(image_array)
        except (OSError, ValueError) as exc:
            preview = LeafMaskPreview(
                mask=None,
                status=LeafMaskPreviewStatus.FAILED,
                warnings=[str(exc)],
            )

        previous = self._leaf_mask_previews.get(sample.sample_id)
        self._leaf_mask_previews[sample.sample_id] = preview
        self._remember_preview_source(sample)
        if previous is not None and (
            previous.status != preview.status
            or previous.mask is None
            or preview.mask is None
            or not np.array_equal(previous.mask, preview.mask)
            or previous.settings_snapshot != preview.settings_snapshot
        ):
            self._invalidate_disease_preview(sample.sample_id)

        return preview

    def _get_or_create_transient_disease_mask_preview(
        self,
        sample: SampleRecord,
    ) -> tuple[LeafMaskPreview, DiseaseMaskPreview]:
        """Return cached or freshly generated disease preview masks without persisting."""
        leaf_preview = self._get_or_create_transient_leaf_mask_preview(sample)
        if (
            leaf_preview.status is not LeafMaskPreviewStatus.SUCCESS
            or leaf_preview.mask is None
        ):
            disease_preview = DiseaseMaskPreview(
                disease_mask=None,
                healthy_tissue_mask=None,
                status=DiseaseMaskPreviewStatus.FAILED,
                warnings=["Whole-leaf preview is required."],
            )
            self._disease_mask_previews[sample.sample_id] = disease_preview
            return leaf_preview, disease_preview

        existing = self._disease_mask_previews.get(sample.sample_id)
        if (
            existing is not None
            and self._preview_source_matches(sample)
            and existing.status is DiseaseMaskPreviewStatus.READY
            and existing.disease_mask is not None
            and existing.healthy_tissue_mask is not None
            and existing.whole_leaf_method == (leaf_preview.method or PREVIEW_METHOD)
            and existing.whole_leaf_settings_version
            == int(leaf_preview.settings_snapshot.get("settings_version", SETTINGS_VERSION))
            and int(existing.settings_snapshot.get("settings_version", -1))
            == DISEASE_SETTINGS_VERSION
        ):
            return leaf_preview, existing

        image_path = sample.raw_image_absolute(self._workspace.root_path)  # type: ignore[union-attr]
        try:
            image_array = load_image_rgb_array(image_path)
            settings_version = int(
                leaf_preview.settings_snapshot.get("settings_version", SETTINGS_VERSION)
            )
            disease_preview = segment_disease_preview(
                image_array,
                leaf_preview.mask,
                whole_leaf_method=leaf_preview.method or PREVIEW_METHOD,
                whole_leaf_settings_version=settings_version,
            )
        except (OSError, ValueError) as exc:
            disease_preview = DiseaseMaskPreview(
                disease_mask=None,
                healthy_tissue_mask=None,
                status=DiseaseMaskPreviewStatus.FAILED,
                warnings=[str(exc)],
            )

        self._disease_mask_previews[sample.sample_id] = disease_preview
        return leaf_preview, disease_preview

    def _ensure_transient_previews_for_view_mode(
        self,
        sample: SampleRecord,
        mode: str,
    ) -> str | None:
        """Generate transient preview masks for a display-only view mode."""
        if mode == "raw_image":
            return None

        if mode == "whole_leaf":
            leaf_preview = self._get_or_create_transient_leaf_mask_preview(sample)
            _, leaf_warnings = _format_leaf_mask_preview_status(leaf_preview)
            return leaf_warnings

        if mode in {"disease", "healthy_tissue"}:
            leaf_preview, disease_preview = self._get_or_create_transient_disease_mask_preview(
                sample
            )
            _, leaf_warnings = _format_leaf_mask_preview_status(leaf_preview)
            _, disease_warnings = _format_disease_mask_preview_status(disease_preview)
            return disease_warnings or leaf_warnings

        return None

    def _calibration_grid_matches(self, sample: SampleRecord) -> bool:
        if self._workspace is None or sample.calibration is None:
            return True
        return calibration_matches_display_grid(
            sample.calibration,
            sample.raw_image_absolute(self._workspace.root_path),
        )

    def _apply_sample_calibration_display(self, sample: SampleRecord) -> None:
        if self._workspace is None:
            return
        if sample.calibration is not None and self._calibration_grid_matches(sample):
            calibration = sample.calibration
            self.image_canvas.set_saved_calibration_line(
                (calibration.point_1_x, calibration.point_1_y),
                (calibration.point_2_x, calibration.point_2_y),
            )
        else:
            self.image_canvas.set_saved_calibration_line(None, None)

        mode = self._preview_modes.get(sample.sample_id, "raw_image")
        warnings = self._ensure_transient_previews_for_view_mode(sample, mode)
        self._refresh_sample_inspector(sample, warnings=warnings)
        self._update_view_mode_controls(sample.sample_id)
        self._apply_active_preview_overlay(sample.sample_id)
        self.inspector_panel.set_save_calibration_enabled(False)
        self._update_status_bar(sample=sample)

    def _apply_active_preview_overlay(self, sample_id: str) -> None:
        if (
            self._correction_sample_id == sample_id
            and self._correction_committed_mask is not None
        ):
            self._sync_correction_overlay()
            return
        mode = self._preview_modes.get(sample_id, "raw_image")
        leaf_preview = self._leaf_mask_previews.get(sample_id)
        disease_preview = self._disease_mask_previews.get(sample_id)
        corrected = self._corrected_disease_mask_for_display(sample_id)

        if mode == "raw_image":
            self.image_canvas.clear_leaf_mask_overlay()
            return
        if mode == "whole_leaf":
            self._apply_leaf_mask_overlay(leaf_preview)
            return
        if mode == "disease":
            if corrected is not None:
                self.image_canvas.set_leaf_mask_overlay(
                    QPixmap.fromImage(disease_mask_to_qimage(corrected))
                )
                return
            self._apply_disease_mask_overlay(disease_preview)
            return
        if mode == "healthy_tissue":
            if corrected is not None:
                leaf_mask = self._leaf_mask_for_sample(sample_id)
                if leaf_mask is not None:
                    _disease, healthy = enforce_mask_invariants(leaf_mask, corrected)
                    self.image_canvas.set_leaf_mask_overlay(
                        QPixmap.fromImage(healthy_tissue_mask_to_qimage(healthy))
                    )
                    return
            self._apply_healthy_tissue_mask_overlay(disease_preview)
            return
        self.image_canvas.clear_leaf_mask_overlay()

    def _apply_leaf_mask_overlay(self, preview: LeafMaskPreview | None) -> None:
        if (
            preview is None
            or preview.status is not LeafMaskPreviewStatus.SUCCESS
            or preview.mask is None
        ):
            self.image_canvas.clear_leaf_mask_overlay()
            return

        overlay_image = leaf_mask_to_qimage(preview.mask)
        self.image_canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay_image))

    def _apply_disease_mask_overlay(self, preview: DiseaseMaskPreview | None) -> None:
        if (
            preview is None
            or preview.status is not DiseaseMaskPreviewStatus.READY
            or preview.disease_mask is None
        ):
            self.image_canvas.clear_leaf_mask_overlay()
            return

        overlay_image = disease_mask_to_qimage(preview.disease_mask)
        self.image_canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay_image))

    def _apply_healthy_tissue_mask_overlay(self, preview: DiseaseMaskPreview | None) -> None:
        if (
            preview is None
            or preview.status is not DiseaseMaskPreviewStatus.READY
            or preview.healthy_tissue_mask is None
        ):
            self.image_canvas.clear_leaf_mask_overlay()
            return

        overlay_image = healthy_tissue_mask_to_qimage(preview.healthy_tissue_mask)
        self.image_canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay_image))

    def _on_calculate_results(self) -> None:
        if (
            self._calculation_active
            or self._correction_prepare_active
            or self._correction_sample_id is not None
        ):
            return
        if self._workspace is None or self._selected_sample_id is None:
            QMessageBox.information(
                self,
                "No Sample Selected",
                "Select a sample before calculating results.",
            )
            return

        sample = self._workspace.get_sample(self._selected_sample_id)
        if sample is None:
            return

        if sample.calibration is None:
            self._update_status_bar(
                sample=sample,
                transient="Manual calibration is required before calculating results.",
            )
            return

        if not self._calibration_grid_matches(sample):
            self._update_status_bar(
                sample=sample,
                transient=(
                    "Saved calibration is in the previous image orientation. "
                    "Set the 1-inch calibration again."
                ),
            )
            return

        if not self._sample_image_available(sample):
            self._update_status_bar(
                sample=sample,
                transient="Source image is not available for calculation.",
            )
            return

        reusable = self._reusable_segmentation(sample)
        image_path = sample.raw_image_absolute(self._workspace.root_path)
        self._calculation_request_id += 1
        request_id = self._calculation_request_id
        leaf_preview, disease_preview = reusable if reusable is not None else (None, None)
        job = _ExplicitCalculationJob(
            None,
            image_path,
            sample.calibration,
            sample.image_hash,
            leaf_preview,
            disease_preview,
            request_id,
            sample.sample_id,
        )
        self._calculation_job = job
        self._calculation_active = True
        self.inspector_panel.calculate_results_button.setEnabled(False)
        progress_owner = ("calculation", request_id, sample.sample_id)
        self._begin_work_progress(
            progress_owner,
            "Updating measurements" if reusable is not None else "Preparing image",
        )
        thread = QThread()
        job.moveToThread(thread)
        thread.started.connect(job.run, Qt.ConnectionType.DirectConnection)
        job.stage_changed.connect(self._on_calculation_stage, Qt.ConnectionType.QueuedConnection)
        job.succeeded.connect(self._on_calculation_succeeded, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(self._on_calculation_failed, Qt.ConnectionType.QueuedConnection)
        job.succeeded.connect(thread.quit, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(thread.quit, Qt.ConnectionType.QueuedConnection)
        loop = QEventLoop(self)
        thread.finished.connect(loop.quit)
        thread.start()
        try:
            loop.exec()
            thread.wait()
        finally:
            self._calculation_active = False
            self._calculation_job = None
            self._clear_work_progress(progress_owner)
            if (
                self._workspace is not None
                and self._selected_sample_id is not None
                and self._correction_sample_id is None
            ):
                selected = self._workspace.get_sample(self._selected_sample_id)
                if selected is not None:
                    self._refresh_sample_inspector(selected)

    def _on_calculation_stage(self, stage: str) -> None:
        job = self._calculation_job
        if job is None:
            return
        self._note_work_stage(("calculation", job.request_id, job.sample_id), stage)

    def _on_calculation_succeeded(self, output: object) -> None:
        job = self._calculation_job
        if job is None or not isinstance(output, ExplicitCalculationOutput):
            return
        self._accept_calculation_output(job.sample_id, job.request_id, output)

    def _on_calculation_failed(self, exc: object) -> None:
        job = self._calculation_job
        if job is None or job.request_id != self._calculation_request_id:
            return
        self._clear_work_progress(("calculation", job.request_id, job.sample_id))
        if self._workspace is None:
            return
        sample = self._workspace.get_sample(job.sample_id)
        if isinstance(exc, CalculationError):
            if sample is not None:
                self._calculation_failed_sample_ids.add(sample.sample_id)
                if self._selected_sample_id == sample.sample_id:
                    self._refresh_sample_inspector(sample, calculation_failed=True)
                    self._update_status_bar(sample=sample, transient=str(exc))
            return
        if isinstance(exc, OSError):
            if sample is not None and self._selected_sample_id == sample.sample_id:
                self._refresh_sample_inspector(sample)
                self._update_status_bar(
                    sample=sample,
                    transient="Could not read the source image.",
                )
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return
        if self._selected_sample_id == job.sample_id:
            self._update_status_bar(sample=sample)
        QMessageBox.critical(self, "Calculation Error", str(exc))

    def _accept_calculation_output(
        self,
        sample_id: str,
        request_id: int,
        output: ExplicitCalculationOutput,
    ) -> None:
        """Store one calculation if it is still the latest request."""
        if request_id != self._calculation_request_id or self._workspace is None:
            return
        progress_owner = ("calculation", request_id, sample_id)
        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            self._clear_work_progress(progress_owner)
            return
        try:
            sample = save_sample_calculation_results(
                self._workspace,
                sample.sample_id,
                output.leaf_measurement,
                output.candidate_disease_measurement,
            )
            if (
                sample.calibration is not None
                and output.leaf_preview.mask is not None
                and output.disease_preview.disease_mask is not None
            ):
                update = reconcile_correction_after_calculation(
                    sample,
                    output.leaf_preview.mask,
                    output.disease_preview.disease_mask,
                    sample.calibration,
                )
                if update.changed:
                    sample = save_sample_disease_mask_correction(
                        self._workspace,
                        sample.sample_id,
                        update.record,
                    )
        except WorkspaceError as exc:
            self._clear_work_progress(progress_owner)
            if self._selected_sample_id == sample.sample_id:
                self._update_status_bar(sample=sample)
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return
        except OSError as exc:
            self._clear_work_progress(progress_owner)
            if self._selected_sample_id == sample.sample_id:
                self._refresh_sample_inspector(sample)
                self._update_status_bar(
                    sample=sample,
                    transient="Could not save calculation results to the workspace.",
                )
            QMessageBox.critical(self, "Workspace Error", str(exc))
            return

        self._calculation_failed_sample_ids.discard(sample.sample_id)
        self._leaf_mask_previews[sample.sample_id] = output.leaf_preview
        self._disease_mask_previews[sample.sample_id] = output.disease_preview
        self._remember_preview_source(sample)
        self._clear_work_progress(progress_owner)
        if self._selected_sample_id != sample.sample_id or self._correction_sample_id is not None:
            return
        self._refresh_sample_inspector(sample)
        self._update_view_mode_controls(sample.sample_id)
        self._apply_active_preview_overlay(sample.sample_id)
        self._update_status_bar(
            sample=sample,
            transient=f"Results calculated: {sample.display_name}",
        )

    def _preview_source_matches(self, sample: SampleRecord) -> bool:
        if not sample.image_hash:
            return False
        return self._preview_image_hash.get(sample.sample_id) == sample.image_hash

    def _remember_preview_source(self, sample: SampleRecord) -> None:
        self._preview_image_hash[sample.sample_id] = sample.image_hash

    def _reusable_segmentation(
        self,
        sample: SampleRecord,
    ) -> tuple[LeafMaskPreview, DiseaseMaskPreview] | None:
        """Return cached full-resolution masks when the image and settings still match."""
        if not self._preview_source_matches(sample):
            return None
        leaf = self._leaf_mask_previews.get(sample.sample_id)
        disease = self._disease_mask_previews.get(sample.sample_id)
        if (
            leaf is None
            or disease is None
            or leaf.status is not LeafMaskPreviewStatus.SUCCESS
            or leaf.mask is None
            or disease.status is not DiseaseMaskPreviewStatus.READY
            or disease.disease_mask is None
            or disease.healthy_tissue_mask is None
        ):
            return None
        if int(leaf.settings_snapshot.get("settings_version", -1)) != SETTINGS_VERSION:
            return None
        if int(disease.settings_snapshot.get("settings_version", -1)) != DISEASE_SETTINGS_VERSION:
            return None
        if disease.whole_leaf_method != (leaf.method or PREVIEW_METHOD):
            return None
        leaf_version = int(leaf.settings_snapshot.get("settings_version", SETTINGS_VERSION))
        if disease.whole_leaf_settings_version != leaf_version:
            return None
        return leaf, disease

    def _on_view_mode_changed(self) -> None:
        if self._workspace is None or self._selected_sample_id is None:
            return

        sample = self._workspace.get_sample(self._selected_sample_id)
        if sample is None:
            return

        mode = self.inspector_panel.get_view_mode_key()
        self._preview_modes[self._selected_sample_id] = mode
        warnings = self._ensure_transient_previews_for_view_mode(sample, mode)
        self._refresh_sample_inspector(sample, warnings=warnings)
        self._apply_active_preview_overlay(self._selected_sample_id)

    def _show_sample(self, sample_id: str) -> None:
        if self._workspace is None:
            return
        if (
            self._correction_sample_id is not None
            and self._correction_sample_id != sample_id
            and not self._leave_correction_session(save=True)
        ):
            return

        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            self._on_selection_cleared()
            return

        image_path = sample.raw_image_absolute(self._workspace.root_path)
        self.image_canvas.set_image_path(image_path)
        self._apply_sample_calibration_display(sample)
        self.image_canvas.set_calibration_mode(False)
        self.inspector_panel.set_calibration_in_progress(False)
        self._refresh_sample_inspector(sample)
        self._update_status_bar(sample=sample)

    def _leaf_mask_for_sample(self, sample_id: str) -> np.ndarray | None:
        if (
            self._correction_sample_id == sample_id
            and self._correction_leaf_mask is not None
        ):
            return self._correction_leaf_mask
        leaf_preview = self._leaf_mask_previews.get(sample_id)
        if leaf_preview is None:
            return None
        return leaf_preview.mask

    def _corrected_disease_mask_for_display(self, sample_id: str) -> np.ndarray | None:
        if (
            self._correction_sample_id == sample_id
            and self._correction_committed_mask is not None
        ):
            if self._correction_preview_mask is not None:
                return self._correction_preview_mask
            return self._correction_committed_mask
        if self._workspace is None:
            return None
        sample = self._workspace.get_sample(sample_id)
        if sample is None or sample.disease_mask_correction is None:
            return None
        if (
            correction_display_state(
                sample,
                calibration_grid_matches=self._calibration_grid_matches(sample),
            )
            != "applied"
        ):
            return None
        leaf_preview = self._leaf_mask_previews.get(sample_id)
        disease_preview = self._disease_mask_previews.get(sample_id)
        if (
            leaf_preview is None
            or leaf_preview.mask is None
            or disease_preview is None
            or disease_preview.disease_mask is None
        ):
            return None
        return apply_corrections(
            disease_preview.disease_mask,
            leaf_preview.mask,
            sample.disease_mask_correction.strokes,
        )

    def _session_correction_record(
        self,
        sample: SampleRecord,
    ) -> DiseaseMaskCorrectionRecord | None:
        if (
            self._correction_history is None
            or self._correction_leaf_mask is None
            or self._correction_automatic_mask is None
            or sample.calibration is None
            or sample.candidate_disease_measurement is None
        ):
            return None
        measurement = sample.candidate_disease_measurement
        strokes = list(self._correction_history.strokes)
        if self._correction_preview_stroke is not None:
            strokes.append(self._correction_preview_stroke)
        leaf_version = measurement.leaf_mask_settings_version
        disease_version = measurement.disease_mask_settings_version
        return build_disease_mask_correction_record(
            whole_leaf_mask=self._correction_leaf_mask,
            automatic_disease_mask=self._correction_automatic_mask,
            strokes=strokes,
            pixels_per_mm=sample.calibration.pixels_per_mm,
            source_image_hash=sample.image_hash,
            leaf_mask_method=measurement.leaf_mask_method or PREVIEW_METHOD,
            leaf_mask_settings_version=int(
                leaf_version if leaf_version is not None else SETTINGS_VERSION
            ),
            disease_mask_method=measurement.disease_mask_method or DISEASE_PREVIEW_METHOD,
            disease_mask_settings_version=int(
                disease_version
                if disease_version is not None
                else DISEASE_SETTINGS_VERSION
            ),
            pixel_grid=sample.calibration.pixel_grid or CURRENT_PIXEL_GRID,
        )

    def _sync_canvas_brush(self, *_args: object) -> None:
        self.image_canvas.set_brush(
            self.inspector_panel.brush_operation(),
            self.inspector_panel.brush_radius_px(),
        )

    def _update_correction_history_buttons(self) -> None:
        history = self._correction_history
        if history is None:
            return
        self.inspector_panel.set_correction_history_enabled(
            can_undo=history.can_undo,
            can_redo=history.can_redo,
            can_reset=history.can_reset,
        )

    def _refresh_correction_display(self) -> None:
        if self._workspace is None or self._correction_sample_id is None:
            return
        sample = self._workspace.get_sample(self._correction_sample_id)
        if sample is None:
            return
        self._refresh_sample_inspector(sample)
        self._apply_active_preview_overlay(sample.sample_id)

    def _commit_open_stroke(self) -> None:
        if self.image_canvas._stroke_active:
            self.image_canvas._finish_stroke()

    def _persist_correction_session(self) -> None:
        if (
            self._workspace is None
            or self._correction_sample_id is None
            or self._correction_history is None
            or not self._correction_modified
        ):
            return
        sample = self._workspace.get_sample(self._correction_sample_id)
        if (
            sample is None
            or sample.calibration is None
            or sample.candidate_disease_measurement is None
            or self._correction_leaf_mask is None
            or self._correction_automatic_mask is None
        ):
            return
        measurement = sample.candidate_disease_measurement
        strokes = self._correction_history.strokes
        if strokes:
            leaf_version = measurement.leaf_mask_settings_version
            disease_version = measurement.disease_mask_settings_version
            record = build_disease_mask_correction_record(
                whole_leaf_mask=self._correction_leaf_mask,
                automatic_disease_mask=self._correction_automatic_mask,
                strokes=strokes,
                pixels_per_mm=sample.calibration.pixels_per_mm,
                source_image_hash=sample.image_hash,
                leaf_mask_method=measurement.leaf_mask_method or PREVIEW_METHOD,
                leaf_mask_settings_version=int(
                    leaf_version if leaf_version is not None else SETTINGS_VERSION
                ),
                disease_mask_method=(
                    measurement.disease_mask_method or DISEASE_PREVIEW_METHOD
                ),
                disease_mask_settings_version=int(
                    disease_version
                    if disease_version is not None
                    else DISEASE_SETTINGS_VERSION
                ),
                pixel_grid=sample.calibration.pixel_grid or CURRENT_PIXEL_GRID,
            )
            save_sample_disease_mask_correction(
                self._workspace,
                sample.sample_id,
                record,
            )
        elif self._correction_loaded_state == "applied":
            save_sample_disease_mask_correction(
                self._workspace,
                sample.sample_id,
                None,
            )
        self._correction_modified = False

    def _exit_correction_mode(self) -> None:
        self._correction_sample_id = None
        self._correction_history = None
        self._correction_leaf_mask = None
        self._correction_automatic_mask = None
        self._correction_preview_stroke = None
        self._correction_modified = False
        self._correction_loaded_state = "none"
        self._correction_committed_mask = None
        self._correction_preview_mask = None
        self._correction_live_points = ()
        self._correction_live_operation = None
        self._correction_live_radius = None
        self._correction_overlay_mode = None
        self.image_canvas.set_correction_mode(False)
        self.inspector_panel.set_correction_in_progress(False)

    def _leave_correction_session(self, *, save: bool) -> bool:
        if self._correction_sample_id is None:
            return True
        if save:
            try:
                self._commit_open_stroke()
                self._persist_correction_session()
            except WorkspaceError as exc:
                QMessageBox.critical(self, "Workspace Error", str(exc))
                return False
            except OSError as exc:
                QMessageBox.critical(self, "Workspace Error", str(exc))
                return False
        self._exit_correction_mode()
        return True

    def _on_correction_stroke_preview(self, stroke: object) -> None:
        if self._correction_sample_id is None:
            return
        if not isinstance(stroke, MaskStroke):
            self._correction_preview_stroke = None
            self._correction_preview_mask = None
            self._correction_live_points = ()
            self._correction_live_operation = None
            self._correction_live_radius = None
            self._sync_correction_overlay()
            return
        self._correction_preview_stroke = stroke
        self._extend_live_correction(stroke)

    def _on_correction_stroke_finished(self, stroke: MaskStroke) -> None:
        if self._correction_history is None or self._correction_sample_id is None:
            return
        self._correction_preview_stroke = None
        self._correction_history.commit(stroke)
        self._correction_modified = True
        if (
            self._correction_preview_mask is not None
            and self._correction_live_points == stroke.points
            and self._correction_live_operation == stroke.operation
            and self._correction_live_radius == stroke.radius_px
        ):
            self._correction_committed_mask = self._correction_preview_mask
        else:
            self._recompute_committed_mask()
        self._correction_preview_mask = None
        self._correction_live_points = ()
        self._correction_live_operation = None
        self._correction_live_radius = None
        self._refresh_correction_display()

    def _on_correct_disease_mask(self) -> None:
        if self._workspace is None or self._selected_sample_id is None:
            return
        if (
            self._calculation_active
            or self._correction_prepare_active
            or self._correction_sample_id is not None
            or self.image_canvas._calibration_mode
        ):
            return
        sample = self._workspace.get_sample(self._selected_sample_id)
        if sample is None:
            return
        grid_matches = self._calibration_grid_matches(sample)
        if not can_start_disease_mask_correction(
            sample,
            image_available=self._sample_image_available(sample),
            calibration_grid_matches=grid_matches,
        ):
            return
        if self._reusable_segmentation(sample) is None:
            sample_id = sample.sample_id
            if not self._prepare_correction_masks(sample):
                return
            if self._selected_sample_id != sample_id or self._workspace is None:
                return
            sample = self._workspace.get_sample(sample_id)
            if sample is None:
                return
        leaf_preview, disease_preview = self._get_or_create_transient_disease_mask_preview(
            sample
        )
        if (
            leaf_preview.mask is None
            or disease_preview.disease_mask is None
            or disease_preview.status is not DiseaseMaskPreviewStatus.READY
        ):
            self._update_status_bar(
                sample=sample,
                transient="Candidate disease mask is not available to correct.",
            )
            return
        state = correction_display_state(
            sample,
            calibration_grid_matches=grid_matches,
        )
        strokes: tuple[MaskStroke, ...] = ()
        if state == "applied" and sample.disease_mask_correction is not None:
            strokes = sample.disease_mask_correction.strokes
        self._correction_sample_id = sample.sample_id
        self._correction_history = CorrectionHistory(strokes)
        self._correction_leaf_mask = np.array(leaf_preview.mask, dtype=bool, copy=True)
        self._correction_automatic_mask = np.array(
            disease_preview.disease_mask,
            dtype=bool,
            copy=True,
        )
        self._correction_preview_stroke = None
        self._correction_modified = False
        self._correction_loaded_state = state
        self._correction_committed_mask = apply_corrections(
            self._correction_automatic_mask,
            self._correction_leaf_mask,
            strokes,
        )
        self._correction_preview_mask = None
        self._correction_live_points = ()
        self._correction_live_operation = None
        self._correction_live_radius = None
        self.image_canvas.set_correction_mode(True)
        self.image_canvas.setFocus()
        self._sync_canvas_brush()
        self.inspector_panel.set_correction_in_progress(True)
        self._preview_modes[sample.sample_id] = "disease"
        self.inspector_panel.view_mode_combo.blockSignals(True)
        self.inspector_panel.set_view_mode_key("disease")
        self.inspector_panel.view_mode_combo.blockSignals(False)
        self._refresh_correction_display()
        self._update_status_bar(
            sample=sample,
            transient=(
                "Correcting candidate disease. Use View to hide the overlay and see the leaf."
            ),
        )

    def _on_undo_correction(self) -> None:
        if self._correction_history is None:
            return
        self.image_canvas.cancel_active_stroke()
        self._correction_preview_stroke = None
        if self._correction_history.undo():
            self._correction_modified = True
            self._recompute_committed_mask()
            self._refresh_correction_display()

    def _on_redo_correction(self) -> None:
        if self._correction_history is None:
            return
        self.image_canvas.cancel_active_stroke()
        self._correction_preview_stroke = None
        if self._correction_history.redo():
            self._correction_modified = True
            self._recompute_committed_mask()
            self._refresh_correction_display()

    def _on_reset_corrections(self) -> None:
        if self._correction_history is None:
            return
        self.image_canvas.cancel_active_stroke()
        self._correction_preview_stroke = None
        if self._correction_history.reset():
            self._correction_modified = True
            self._recompute_committed_mask()
            self._refresh_correction_display()

    def _recompute_committed_mask(self) -> None:
        if (
            self._correction_history is None
            or self._correction_automatic_mask is None
            or self._correction_leaf_mask is None
        ):
            return
        self._correction_committed_mask = apply_corrections(
            self._correction_automatic_mask,
            self._correction_leaf_mask,
            self._correction_history.strokes,
        )
        self._correction_preview_mask = None
        self._correction_live_points = ()
        self._correction_live_operation = None
        self._correction_live_radius = None

    def _extend_live_correction(self, stroke: MaskStroke) -> None:
        if (
            self._correction_committed_mask is None
            or self._correction_leaf_mask is None
        ):
            return
        prefix_matches = (
            self._correction_preview_mask is not None
            and self._correction_live_operation == stroke.operation
            and self._correction_live_radius == stroke.radius_px
            and stroke.points[: len(self._correction_live_points)] == self._correction_live_points
        )
        if prefix_matches and self._correction_preview_mask is not None:
            new_points = stroke.points[len(self._correction_live_points) :]
        else:
            self._correction_preview_mask = self._correction_committed_mask.copy()
            new_points = stroke.points
        self._correction_live_points = stroke.points
        self._correction_live_operation = stroke.operation
        self._correction_live_radius = stroke.radius_px
        if not new_points:
            return
        box = paint_disks(
            self._correction_preview_mask,
            self._correction_leaf_mask,
            new_points,
            stroke.radius_px,
            add=stroke.operation == ADD_AREA,
        )
        if box is None:
            return
        self._sync_correction_overlay(box)

    def _sync_correction_overlay(
        self,
        box: tuple[int, int, int, int] | None = None,
    ) -> None:
        sample_id = self._correction_sample_id
        if sample_id is None or self._correction_committed_mask is None:
            return
        mode = self._preview_modes.get(sample_id, "raw_image")
        if mode == "raw_image":
            self._correction_overlay_mode = None
            self.image_canvas.clear_leaf_mask_overlay()
            return
        if mode == "whole_leaf":
            self._correction_overlay_mode = None
            self._apply_leaf_mask_overlay(self._leaf_mask_previews.get(sample_id))
            return
        mask = (
            self._correction_preview_mask
            if self._correction_preview_mask is not None
            else self._correction_committed_mask
        )
        if mode == "disease":
            rgba = DISEASE_MASK_OVERLAY_RGBA
        elif mode == "healthy_tissue" and self._correction_leaf_mask is not None:
            rgba = HEALTHY_TISSUE_MASK_OVERLAY_RGBA
        else:
            self.image_canvas.clear_leaf_mask_overlay()
            return
        current = self.image_canvas._leaf_mask_overlay
        if (
            box is not None
            and current is not None
            and self._correction_overlay_mode == mode
            and isinstance(current, QImage)
            and current.format() == QImage.Format.Format_RGBA8888
            and current.width() == mask.shape[1]
            and current.height() == mask.shape[0]
        ):
            x0, y0, x1, y1 = box
            if mode == "disease":
                patch = mask[y0:y1, x0:x1]
            else:
                assert self._correction_leaf_mask is not None
                patch = self._correction_leaf_mask[y0:y1, x0:x1] & ~mask[y0:y1, x0:x1]
            write_overlay_region(current, patch, x0, y0, rgba)
            self.image_canvas.refresh_overlay_image()
            return
        if mode == "healthy_tissue":
            assert self._correction_leaf_mask is not None
            membership = self._correction_leaf_mask & ~mask
        else:
            membership = mask
        self._correction_overlay_mode = mode
        self.image_canvas.set_overlay_image(mask_to_qimage(membership, overlay_rgba=rgba))

    def _on_done_correction(self) -> None:
        sample_id = self._correction_sample_id
        if not self._leave_correction_session(save=True):
            return
        if (
            sample_id is not None
            and self._workspace is not None
            and self._workspace.get_sample(sample_id) is not None
        ):
            self._show_sample(sample_id)
            sample = self._workspace.get_sample(sample_id)
            self._update_status_bar(
                sample=sample,
                transient="Candidate disease corrections saved",
            )


    def _prepare_correction_masks(self, sample: SampleRecord) -> bool:
        """Segment off the UI thread when this session has no reusable masks.

        Returns True when masks for ``sample`` are cached and the sample is
        still selected. A newer request or a different sample does not open
        correction.
        """
        if self._workspace is None:
            return False
        self._correction_prepare_request_id += 1
        request_id = self._correction_prepare_request_id
        image_path = sample.raw_image_absolute(self._workspace.root_path)
        job = _DiseaseMaskPreviewJob(
            image_path,
            request_id,
            sample.sample_id,
            sample.image_hash,
        )
        self._correction_prepare_job = job
        self._correction_prepare_active = True
        self._prepared_correction_previews = None
        self.inspector_panel.correct_disease_mask_button.setEnabled(False)
        progress_owner = ("correction", request_id, sample.sample_id)
        self._begin_work_progress(progress_owner, "Preparing image")
        thread = QThread()
        job.moveToThread(thread)
        thread.started.connect(job.run, Qt.ConnectionType.DirectConnection)
        job.stage_changed.connect(
            self._on_correction_prepare_stage,
            Qt.ConnectionType.QueuedConnection,
        )
        job.succeeded.connect(
            self._on_correction_masks_prepared,
            Qt.ConnectionType.QueuedConnection,
        )
        job.failed.connect(
            self._on_correction_mask_prepare_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        job.succeeded.connect(thread.quit, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(thread.quit, Qt.ConnectionType.QueuedConnection)
        loop = QEventLoop(self)
        thread.finished.connect(loop.quit)
        thread.start()
        try:
            loop.exec()
            thread.wait()
        finally:
            self._correction_prepare_active = False
            self._correction_prepare_job = None
            self._clear_work_progress(progress_owner)
        if (
            request_id != self._correction_prepare_request_id
            or self._selected_sample_id != sample.sample_id
            or self._workspace is None
        ):
            self._restore_inspector_after_prepare()
            self._update_status_bar(sample=self._current_sample())
            return False
        prepared = self._prepared_correction_previews
        if (
            prepared is None
            or prepared[0].mask is None
            or prepared[0].status is not LeafMaskPreviewStatus.SUCCESS
            or prepared[1].disease_mask is None
            or prepared[1].status is not DiseaseMaskPreviewStatus.READY
        ):
            current = self._workspace.get_sample(sample.sample_id)
            if current is not None and self._selected_sample_id == current.sample_id:
                self._update_status_bar(
                    sample=current,
                    transient="Candidate disease mask is not available to correct.",
                )
                self._refresh_sample_inspector(current)
            return False
        if self._reusable_segmentation(sample) is None:
            self._restore_inspector_after_prepare()
            return False
        return True

    def _on_correction_prepare_stage(self, stage: str) -> None:
        job = self._correction_prepare_job
        if job is None:
            return
        self._note_work_stage(("correction", job.request_id, job.sample_id), stage)

    def _on_correction_masks_prepared(self, payload: object) -> None:
        job = self._correction_prepare_job
        if job is None or job.request_id != self._correction_prepare_request_id:
            return
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        leaf_preview, disease_preview = payload
        if not isinstance(leaf_preview, LeafMaskPreview) or not isinstance(
            disease_preview,
            DiseaseMaskPreview,
        ):
            return
        if self._workspace is None:
            return
        sample = self._workspace.get_sample(job.sample_id)
        if sample is None or sample.image_hash != job.image_hash:
            return
        self._leaf_mask_previews[sample.sample_id] = leaf_preview
        self._disease_mask_previews[sample.sample_id] = disease_preview
        self._remember_preview_source(sample)
        self._prepared_correction_previews = (leaf_preview, disease_preview)

    def _on_correction_mask_prepare_failed(self, exc: object) -> None:
        job = self._correction_prepare_job
        if job is None or job.request_id != self._correction_prepare_request_id:
            return
        self._clear_work_progress(("correction", job.request_id, job.sample_id))
        self._prepared_correction_previews = None
        if self._workspace is None or self._selected_sample_id != job.sample_id:
            return
        sample = self._workspace.get_sample(job.sample_id)
        if sample is None:
            return
        self._update_status_bar(
            sample=sample,
            transient="Candidate disease mask is not available to correct.",
        )
        QMessageBox.critical(self, "Correction Error", str(exc))

    def _restore_inspector_after_prepare(self) -> None:
        if self._workspace is None or self._selected_sample_id is None:
            return
        if self._correction_sample_id is not None:
            return
        sample = self._workspace.get_sample(self._selected_sample_id)
        if sample is not None:
            self._refresh_sample_inspector(sample)


class _DiseaseMaskPreviewJob(QObject):
    """Build leaf and candidate-disease masks without touching the workspace."""

    succeeded = Signal(object)
    failed = Signal(object)
    stage_changed = Signal(str)

    def __init__(
        self,
        image_path: Path,
        request_id: int,
        sample_id: str,
        image_hash: str | None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.request_id = request_id
        self.sample_id = sample_id
        self.image_hash = image_hash

    def run(self) -> None:
        try:
            self.stage_changed.emit("Preparing image")
            image_array = load_image_rgb_array(self.image_path)
            self.stage_changed.emit("Finding leaf")
            leaf_preview = segment_leaf_preview(image_array)
            if (
                leaf_preview.status is not LeafMaskPreviewStatus.SUCCESS
                or leaf_preview.mask is None
            ):
                disease_preview = DiseaseMaskPreview(
                    disease_mask=None,
                    healthy_tissue_mask=None,
                    status=DiseaseMaskPreviewStatus.FAILED,
                    warnings=["Whole-leaf preview is required."],
                )
            else:
                self.stage_changed.emit("Identifying candidate disease")
                settings_version = int(
                    leaf_preview.settings_snapshot.get("settings_version", SETTINGS_VERSION)
                )
                disease_preview = segment_disease_preview(
                    image_array,
                    leaf_preview.mask,
                    whole_leaf_method=leaf_preview.method or PREVIEW_METHOD,
                    whole_leaf_settings_version=settings_version,
                )
        except Exception as exc:
            self.failed.emit(exc)
            return
        self.succeeded.emit((leaf_preview, disease_preview))


class _ExplicitCalculationJob(QObject):
    """Run one explicit calculation off the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(object)
    stage_changed = Signal(str)

    def __init__(
        self,
        image_array: np.ndarray | None,
        image_path: Path | None,
        calibration: object,
        image_hash: str | None,
        leaf_preview: LeafMaskPreview | None,
        disease_preview: DiseaseMaskPreview | None,
        request_id: int,
        sample_id: str,
    ) -> None:
        super().__init__()
        self.image_array = image_array
        self.image_path = image_path
        self.calibration = calibration
        self.image_hash = image_hash
        self.leaf_preview = leaf_preview
        self.disease_preview = disease_preview
        self.request_id = request_id
        self.sample_id = sample_id

    def run(self) -> None:
        try:
            image = self.image_array
            needs_image = self.leaf_preview is None or self.disease_preview is None
            if needs_image and image is None:
                self.stage_changed.emit("Preparing image")
                if self.image_path is None:
                    raise OSError("Source image is not available for calculation.")
                image = load_image_rgb_array(self.image_path)
            output = run_explicit_calculation(
                image,
                self.calibration,  # type: ignore[arg-type]
                source_image_hash=self.image_hash,
                leaf_preview=self.leaf_preview,
                disease_preview=self.disease_preview,
                on_stage=self.stage_changed.emit,
            )
        except Exception as exc:
            self.failed.emit(exc)
            return
        self.succeeded.emit(output)


class _ViewGestureFilter(QObject):
    """Space-drag pans the image instead of painting or placing calibration points."""

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
            return False
        if not isinstance(event, QKeyEvent):
            return False
        if event.key() != Qt.Key.Key_Space or event.isAutoRepeat():
            return False
        window = QApplication.activeWindow()
        if not isinstance(window, MainWindow):
            return False
        if window.image_canvas._pixmap is None:
            return False
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)):
            return False
        window.image_canvas.set_space_pan(event.type() == QEvent.Type.KeyPress)
        return True


_VIEW_GESTURE_FILTER: _ViewGestureFilter | None = None


def _install_view_gesture_filter() -> None:
    global _VIEW_GESTURE_FILTER
    app = QApplication.instance()
    if app is None or _VIEW_GESTURE_FILTER is not None:
        return
    _VIEW_GESTURE_FILTER = _ViewGestureFilter()
    app.installEventFilter(_VIEW_GESTURE_FILTER)


def run_app(argv: list[str] | None = None) -> int:
    """Launch the NeoPest desktop application."""
    app = QApplication(argv or sys.argv)
    app.setApplicationName("NeoPest")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(build_application_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()
