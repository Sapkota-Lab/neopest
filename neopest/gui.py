"""NeoPest desktop GUI — MainWindow orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from neopest import __version__
from neopest.gui_layout_builders import build_main_layout
from neopest.gui_styles import build_application_stylesheet
from neopest.image_io import SUPPORTED_IMAGE_SUFFIXES
from neopest.metadata import WORKSPACE_JSON_FILENAME, WorkspaceMetadata
from neopest.project_manager import (
    WorkspaceError,
    create_condition_group,
    get_default_workspace_root,
    import_images,
    initialize_workspace,
    load_workspace,
)


class MainWindow(QMainWindow):
    """Primary workbench window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"NeoPest {__version__}")
        self.resize(1280, 800)
        self.setMinimumSize(960, 640)
        self._workspace: WorkspaceMetadata | None = None
        self._selected_sample_id: str | None = None
        build_main_layout(self)
        self._build_menus()
        self._connect_signals()
        self._refresh_ui()

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
        self.explorer_tree.sample_selected.connect(self._on_sample_selected)
        self.explorer_tree.selection_cleared.connect(self._on_selection_cleared)

    def _set_workspace(self, workspace: WorkspaceMetadata | None) -> None:
        self._workspace = workspace
        self._selected_sample_id = None
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        workspace_open = self._workspace is not None
        self._action_import_images.setEnabled(workspace_open)
        self._action_create_condition_group.setEnabled(workspace_open)

        if self._workspace is None:
            self.explorer_tree.refresh_placeholder()
            self.image_canvas.clear_image()
            self.image_canvas.set_hint("Open or create a workspace to begin")
            self.inspector_panel.show_workspace_state(None)
            self.controls_status_label.setText("No workspace open")
            self.status_bar.showMessage("Open or create a workspace to begin")
            self.setWindowTitle(f"NeoPest {__version__}")
            return

        self.explorer_tree.refresh_workspace(self._workspace)
        self.inspector_panel.show_workspace_state(self._workspace)
        self.image_canvas.clear_image()
        self.image_canvas.set_hint("Select a sample to preview its raw image")
        self.controls_status_label.setText(
            f"Workspace: {self._workspace.display_name} "
            f"({len(self._workspace.samples)} sample(s))"
        )
        self.status_bar.showMessage(
            f"Workspace: {self._workspace.root_path} — "
            f"{len(self._workspace.condition_groups)} condition group(s), "
            f"{len(self._workspace.samples)} sample(s)"
        )
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

    def _on_import_images(self) -> None:
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
            imported = import_images(self._workspace, [Path(path) for path in paths])
        except WorkspaceError as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        if imported:
            self._selected_sample_id = imported[-1].sample_id
        self._refresh_ui()

    def _on_sample_selected(self, sample_id: str) -> None:
        self._selected_sample_id = sample_id
        self._show_sample(sample_id)

    def _on_selection_cleared(self) -> None:
        self._selected_sample_id = None
        if self._workspace is None:
            self.inspector_panel.show_workspace_state(None)
        else:
            self.inspector_panel.show_workspace_state(self._workspace)
        self.image_canvas.clear_image()
        if self._workspace is not None:
            self.image_canvas.set_hint("Select a sample to preview its raw image")

    def _show_sample(self, sample_id: str) -> None:
        if self._workspace is None:
            return

        sample = self._workspace.get_sample(sample_id)
        if sample is None:
            self._on_selection_cleared()
            return

        self.inspector_panel.show_sample(sample, self._workspace)
        image_path = sample.raw_image_absolute(self._workspace.root_path)
        self.image_canvas.set_image_path(image_path)
        self.controls_status_label.setText(f"Previewing: {sample.display_name}")


def run_app(argv: list[str] | None = None) -> int:
    """Launch the NeoPest desktop application."""
    app = QApplication(argv or sys.argv)
    app.setApplicationName("NeoPest")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(build_application_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()
