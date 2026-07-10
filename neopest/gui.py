"""NeoPest desktop GUI — MainWindow orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from neopest import __version__
from neopest.calibration import (
    CalibrationError,
    build_one_inch_manual_calibration_record,
)
from neopest.desktop_services import open_path_in_file_manager
from neopest.gui_layout_builders import build_main_layout
from neopest.gui_styles import build_application_stylesheet
from neopest.image_io import SUPPORTED_IMAGE_SUFFIXES, leaf_mask_to_qimage, load_image_rgb_array
from neopest.measurements import (
    MeasurementError,
    build_leaf_measurement_record,
    calculate_leaf_area,
)
from neopest.metadata import WORKSPACE_JSON_FILENAME, SampleRecord, WorkspaceMetadata
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
    save_sample_leaf_measurement,
)
from neopest.results import CalibrationStatus
from neopest.segmentation import (
    LeafMaskPreview,
    LeafMaskPreviewStatus,
    PREVIEW_METHOD,
    SETTINGS_VERSION,
    segment_leaf_preview,
)


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
        build_main_layout(self)
        self._build_menus()
        self._connect_signals()
        self._load_default_workspace_on_startup()
        self._refresh_ui()

    def _load_default_workspace_on_startup(self) -> None:
        try:
            self._workspace = ensure_default_workspace()
            self._selected_sample_id = None
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
        self.image_canvas.calibration_points_changed.connect(
            self._on_calibration_points_changed
        )
        self.inspector_panel.set_manual_calibration_button.clicked.connect(
            self._on_set_manual_calibration
        )
        self.inspector_panel.save_calibration_button.clicked.connect(
            self._on_save_calibration
        )
        self.inspector_panel.cancel_calibration_button.clicked.connect(
            self._on_cancel_calibration
        )
        self.inspector_panel.preview_leaf_mask_button.clicked.connect(
            self._on_preview_leaf_mask
        )

    def _set_workspace(self, workspace: WorkspaceMetadata | None) -> None:
        self._workspace = workspace
        self._selected_sample_id = None
        self._leaf_mask_previews = {}
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

    def _on_import_images_into_group(self, condition_group_id: str) -> None:
        self._on_import_images(condition_group_id=condition_group_id)

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
        self._show_sample(sample_id)

    def _on_selection_cleared(self) -> None:
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
        self._update_status_bar()

    def _on_set_manual_calibration(self) -> None:
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

    def _apply_sample_calibration_display(self, sample: SampleRecord) -> None:
        if self._workspace is None:
            return
        if sample.calibration is not None:
            calibration = sample.calibration
            self.image_canvas.set_saved_calibration_line(
                (calibration.point_1_x, calibration.point_1_y),
                (calibration.point_2_x, calibration.point_2_y),
            )
        else:
            self.image_canvas.set_saved_calibration_line(None, None)

        preview = self._leaf_mask_previews.get(sample.sample_id)
        preview_status, warnings = _format_leaf_mask_preview_status(preview)
        self.inspector_panel.show_sample(
            sample,
            self._workspace,
            preview_status=preview_status,
            warnings=warnings,
        )
        self.inspector_panel.show_leaf_mask_preview_fields(
            status=preview_status,
            warnings=warnings,
        )
        self._apply_leaf_mask_overlay(preview)
        self.inspector_panel.set_save_calibration_enabled(False)
        self._update_status_bar(sample=sample)

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

    def _persist_leaf_area_from_preview(
        self,
        sample: SampleRecord,
        preview: LeafMaskPreview,
    ) -> SampleRecord:
        """Calculate and persist whole-leaf area when preview and calibration are valid."""
        if self._workspace is None:
            return sample
        if preview.mask is None or sample.calibration is None:
            return sample

        try:
            area = calculate_leaf_area(preview.mask, sample.calibration.pixels_per_mm)
        except MeasurementError:
            return sample

        settings_version = int(
            preview.settings_snapshot.get("settings_version", SETTINGS_VERSION)
        )
        measurement = build_leaf_measurement_record(
            area,
            method=preview.method or PREVIEW_METHOD,
            settings_version=settings_version,
            source_image_hash=sample.image_hash,
        )
        return save_sample_leaf_measurement(
            self._workspace,
            sample.sample_id,
            measurement,
        )

    def _on_preview_leaf_mask(self) -> None:
        if self._workspace is None or self._selected_sample_id is None:
            QMessageBox.information(
                self,
                "No Sample Selected",
                "Select a sample with a raw image before previewing a leaf mask.",
            )
            return

        sample = self._workspace.get_sample(self._selected_sample_id)
        if sample is None:
            return

        image_path = sample.raw_image_absolute(self._workspace.root_path)
        try:
            image_array = load_image_rgb_array(image_path)
            preview = segment_leaf_preview(image_array)
        except (OSError, ValueError) as exc:
            preview = LeafMaskPreview(
                mask=None,
                status=LeafMaskPreviewStatus.FAILED,
                warnings=[str(exc)],
            )

        self._leaf_mask_previews[sample.sample_id] = preview
        preview_status, warnings = _format_leaf_mask_preview_status(preview)
        preview_succeeded_without_calibration = False

        if preview.status is LeafMaskPreviewStatus.SUCCESS:
            if sample.calibration is not None:
                sample = self._persist_leaf_area_from_preview(sample, preview)
            else:
                preview_succeeded_without_calibration = True
            self._update_status_bar(
                sample=sample,
                transient=f"Whole Leaf Mask Preview ready: {sample.display_name}",
            )
        else:
            warning = preview.warnings[0] if preview.warnings else "Preview failed."
            self._update_status_bar(
                sample=sample,
                transient=f"Whole Leaf Mask Preview failed: {warning}",
            )

        self.inspector_panel.show_sample(
            sample,
            self._workspace,
            preview_status=preview_status,
            preview_succeeded_without_calibration=preview_succeeded_without_calibration,
            warnings=warnings,
        )
        self.inspector_panel.show_leaf_mask_preview_fields(
            status=preview_status,
            warnings=warnings,
        )
        self._apply_leaf_mask_overlay(preview)
        self._update_status_bar(sample=sample)

    def _show_sample(self, sample_id: str) -> None:
        if self._workspace is None:
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
        self._update_status_bar(sample=sample)


def run_app(argv: list[str] | None = None) -> int:
    """Launch the NeoPest desktop application."""
    app = QApplication(argv or sys.argv)
    app.setApplicationName("NeoPest")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(build_application_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()
