"""Tests for Phase 3D explorer and panel polish."""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from neopest.desktop_services import reveal_image_menu_label
from neopest.explorer_tree import ExplorerTreeWidget, build_explorer_panel
from neopest.gui import MainWindow
from neopest.gui_layout_builders import SampleInspectorPanel
from neopest.project_manager import (
    WorkspaceError,
    create_condition_group,
    import_images,
    initialize_workspace,
    remove_sample,
    rename_sample,
)
from tests.test_segmentation import _make_synthetic_leaf_image


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


def _main_window_without_startup_default() -> MainWindow:
    with patch("neopest.gui.ensure_default_workspace", side_effect=WorkspaceError("disabled")):
        with patch("neopest.gui.QMessageBox.warning"):
            return MainWindow()


class TestPanelSimplification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_explorer_heading_and_subtitle_removed(self) -> None:
        source = inspect.getsource(build_explorer_panel)
        self.assertNotIn('QLabel("Explorer")', source)
        self.assertNotIn("Workspace navigation", source)

    def test_inspector_heading_removed(self) -> None:
        source = inspect.getsource(SampleInspectorPanel.__init__)
        self.assertNotIn('QLabel("Inspector")', source)

    def test_center_bottom_workspace_strip_removed(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(hasattr(window, "controls_panel"))
        self.assertFalse(hasattr(window, "controls_status_label"))

    def test_single_canvas_empty_state_hint(self) -> None:
        window = _main_window_without_startup_default()
        window.image_canvas.set_hint("Select a sample to view its image")
        source = inspect.getsource(window.image_canvas.paintEvent)
        self.assertNotIn("self._status", source)

    def test_inspector_hides_internal_ids_and_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Polish")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]

            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(sample.sample_id)

            panel = window.inspector_panel
            visible_text = " ".join(
                [
                    panel.sample_name_label.text(),
                    panel.condition_group_label.text(),
                    panel.total_leaf_area_label.text(),
                    panel.calibration_status_value.text(),
                    panel.whole_leaf_status_value.text(),
                ]
            )
            self.assertNotIn(sample.sample_id, visible_text)
            self.assertNotIn(sample.raw_image_path, visible_text)


class TestExplorerTreePresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_tree_indentation_and_native_decoration_enabled(self) -> None:
        tree = ExplorerTreeWidget()
        self.assertEqual(tree.indentation(), 16)
        self.assertTrue(tree.rootIsDecorated())
        self.assertEqual(tree.styleSheet(), "")

    def test_hierarchy_labels_do_not_use_ascii_branch_characters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Tree")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            import_images(workspace, [image_path])

            tree = ExplorerTreeWidget()
            tree.refresh_workspace(workspace)
            texts = []
            for index in range(tree.topLevelItemCount()):
                root_item = tree.topLevelItem(index)
                if root_item is None:
                    continue
                texts.append(root_item.text(0))
                for group_index in range(root_item.childCount()):
                    group_item = root_item.child(group_index)
                    if group_item is None:
                        continue
                    texts.append(group_item.text(0))
                    for sample_index in range(group_item.childCount()):
                        sample_item = group_item.child(sample_index)
                        if sample_item is None:
                            continue
                        texts.append(sample_item.text(0))

            joined = "\n".join(texts)
            self.assertNotIn("├", joined)
            self.assertNotIn("└", joined)
            self.assertNotIn("->", joined)
            self.assertNotIn("|", joined)

    def test_keyboard_enter_selects_sample(self) -> None:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Keys")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])

            tree = ExplorerTreeWidget()
            tree.refresh_workspace(workspace)
            tree.select_sample(imported[0].sample_id)

            selected: list[str] = []
            tree.sample_preview_requested.connect(selected.append)
            event = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
            )
            tree.keyPressEvent(event)
            self.assertEqual(selected, [imported[0].sample_id])


class TestExplorerContextMenus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_workspace_menu_actions_are_workspace_specific(self) -> None:
        tree = ExplorerTreeWidget()
        menu = __import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu()
        tree._add_workspace_menu_actions(menu)
        labels = [action.text() for action in menu.actions()]
        self.assertEqual(labels, ["Open Workspace Folder"])

    def test_empty_space_menu_contains_new_condition_group_only(self) -> None:
        tree = ExplorerTreeWidget()
        menu = __import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu()
        tree._add_empty_space_menu_actions(menu)
        labels = [action.text() for action in menu.actions()]
        self.assertEqual(labels, ["New Condition Group"])

    def test_condition_group_menu_actions_are_group_specific(self) -> None:
        tree = ExplorerTreeWidget()
        menu = __import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu()
        tree._add_condition_group_menu_actions(menu, "group_1")
        labels = [action.text() for action in menu.actions()]
        self.assertEqual(
            labels,
            [
                "Import Image into Group",
                "Rename Condition Group",
                "Remove Condition Group",
            ],
        )

    def test_sample_menu_actions_are_sample_specific(self) -> None:
        tree = ExplorerTreeWidget()
        menu = __import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu()
        tree._add_sample_menu_actions(menu, "sample_1")
        labels = [action.text() for action in menu.actions()]
        self.assertEqual(
            labels,
            [
                "Select / Preview Sample",
                "Rename Sample",
                "Remove Sample",
                reveal_image_menu_label(),
            ],
        )

    def test_explorer_tree_does_not_import_project_manager(self) -> None:
        import neopest.explorer_tree as explorer_tree

        source = inspect.getsource(explorer_tree)
        self.assertNotIn("project_manager", source)

    def test_remove_sample_routes_through_project_manager(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Remove")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]
            managed_path = sample.raw_image_absolute(workspace.root_path)
            self.assertTrue(managed_path.is_file())

            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(sample.sample_id)

            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                window._on_remove_sample(sample.sample_id)

            self.assertIsNone(workspace.get_sample(sample.sample_id))
            self.assertFalse(managed_path.is_file())

    def test_open_folder_uses_desktop_services(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Folder")
            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)

            with patch("neopest.gui.open_path_in_file_manager", return_value=True) as opener:
                window._on_open_workspace_folder()
                opener.assert_called_once_with(workspace.root_path)


class TestExplorerDragAndDrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_sample(self):
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="Drag")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        tree = ExplorerTreeWidget()
        tree.refresh_workspace(workspace)
        return tmp, workspace, imported[0], tree

    def _node_items(self, tree: ExplorerTreeWidget):
        root = tree.topLevelItem(0)
        group_item = root.child(0)
        sample_item = None
        for index in range(group_item.childCount()):
            child = group_item.child(index)
            if tree.item_node_type(child) == "sample":
                sample_item = child
                break
        return root, group_item, sample_item

    def test_sample_items_are_draggable(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            _root, _group, sample_item = self._node_items(tree)
            self.assertIsNotNone(sample_item)
            self.assertTrue(sample_item.flags() & Qt.ItemFlag.ItemIsDragEnabled)
        finally:
            holder.cleanup()

    def test_workspace_and_group_nodes_are_not_draggable(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            root, group_item, _sample_item = self._node_items(tree)
            self.assertFalse(root.flags() & Qt.ItemFlag.ItemIsDragEnabled)
            self.assertFalse(group_item.flags() & Qt.ItemFlag.ItemIsDragEnabled)
        finally:
            holder.cleanup()

    def test_condition_groups_accept_sample_drops(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            _root, group_item, _sample_item = self._node_items(tree)
            self.assertTrue(group_item.flags() & Qt.ItemFlag.ItemIsDropEnabled)
            self.assertTrue(tree.accepts_sample_drop_on_item(group_item))
        finally:
            holder.cleanup()

    def test_samples_and_workspace_reject_sample_drops(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            root, _group, sample_item = self._node_items(tree)
            self.assertFalse(sample_item.flags() & Qt.ItemFlag.ItemIsDropEnabled)
            self.assertFalse(tree.accepts_sample_drop_on_item(sample_item))
            self.assertFalse(tree.accepts_sample_drop_on_item(root))
            self.assertFalse(tree.accepts_sample_drop_on_item(None))
        finally:
            holder.cleanup()

    def test_drop_event_emits_sample_move_requested(self) -> None:
        from PySide6.QtCore import QMimeData, QPoint
        from PySide6.QtGui import QDropEvent

        holder, workspace, sample, tree = self._workspace_with_sample()
        try:
            destination_group = create_condition_group(workspace, display_name="Target")
            tree.refresh_workspace(workspace)
            _root, _group, sample_item = self._node_items(tree)
            dest_item = None
            root = tree.topLevelItem(0)
            for index in range(root.childCount()):
                child = root.child(index)
                if tree.item_node_id(child) == destination_group.condition_group_id:
                    dest_item = child
                    break
            assert dest_item is not None

            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )

            mime = QMimeData()
            mime.setData(
                "application/x-neopest-sample-id",
                sample.sample_id.encode("utf-8"),
            )
            event = QDropEvent(
                QPoint(10, 10),
                Qt.DropAction.MoveAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            with patch.object(tree, "itemAt", return_value=dest_item):
                tree.dropEvent(event)
            self.assertEqual(
                moves,
                [(sample.sample_id, destination_group.condition_group_id)],
            )
        finally:
            holder.cleanup()


class TestExplorerMoveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_move_refreshes_selection_and_condition_group_label(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Move UI")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]
            destination_group = create_condition_group(workspace, display_name="Treatment B")

            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(sample.sample_id)

            window._on_sample_move_requested(sample.sample_id, destination_group.condition_group_id)

            self.assertEqual(window._selected_sample_id, sample.sample_id)
            self.assertEqual(
                window.inspector_panel.condition_group_label.text(),
                destination_group.display_name,
            )
            self.assertIsNotNone(window.image_canvas._pixmap)
            persisted = workspace.get_sample(sample.sample_id)
            assert persisted is not None
            self.assertEqual(persisted.condition_group_id, destination_group.condition_group_id)


class TestPlatformRevealWording(unittest.TestCase):
    def test_reveal_label_matches_platform_or_fallback(self) -> None:
        with patch("neopest.desktop_services.sys.platform", "darwin"):
            self.assertEqual(reveal_image_menu_label(), "Reveal Image in Finder")
        with patch("neopest.desktop_services.sys.platform", "win32"):
            self.assertEqual(reveal_image_menu_label(), "Show Image in File Explorer")
        with patch("neopest.desktop_services.sys.platform", "linux"):
            self.assertEqual(reveal_image_menu_label(), "Show Image in Folder")


class TestCalibrationActionVisibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_normal_mode_shows_set_and_hides_save_cancel(self) -> None:
        source = inspect.getsource(SampleInspectorPanel.set_calibration_in_progress)
        self.assertIn("setVisible(not active)", source)
        self.assertIn("save_calibration_button.setVisible(active)", source)

    def test_active_calibration_mode_shows_save_cancel(self) -> None:
        source = inspect.getsource(SampleInspectorPanel.set_calibration_in_progress)
        self.assertIn("cancel_calibration_button.setVisible(active)", source)

    def test_save_returns_to_normal_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            image = QApplication.instance()
            _ = image
            from PySide6.QtGui import QColor, QImage

            png = QImage(40, 40, QImage.Format.Format_RGB32)
            png.fill(QColor("green"))
            png.save(str(source), "PNG")

            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])

            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window._on_set_manual_calibration()
            window.image_canvas._draft_points = [(0.0, 0.0), (50.0, 0.0)]
            window._on_save_calibration()

            panel = window.inspector_panel
            self.assertTrue(panel.set_manual_calibration_button.isVisible())
            self.assertFalse(panel.save_calibration_button.isVisible())
            self.assertFalse(panel.cancel_calibration_button.isVisible())

    def test_cancel_returns_to_normal_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            from PySide6.QtGui import QColor, QImage

            png = QImage(40, 40, QImage.Format.Format_RGB32)
            png.fill(QColor("green"))
            png.save(str(source), "PNG")

            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])

            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window._on_set_manual_calibration()
            window._on_cancel_calibration()

            panel = window.inspector_panel
            self.assertTrue(panel.set_manual_calibration_button.isVisible())
            self.assertFalse(panel.save_calibration_button.isVisible())
            self.assertFalse(panel.cancel_calibration_button.isVisible())


class TestInspectorPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_total_leaf_area_caption_is_uppercase(self) -> None:
        source = inspect.getsource(SampleInspectorPanel.__init__)
        self.assertIn('QLabel("TOTAL LEAF AREA")', source)

    def test_run_analysis_not_visible(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(window.run_analysis_button.isVisible())
        self.assertFalse(window.run_analysis_button.isEnabled())

    def test_rename_sample_updates_display_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Rename")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])

            renamed = rename_sample(
                workspace,
                imported[0].sample_id,
                display_name="Leaf A",
            )
            self.assertEqual(renamed.display_name, "Leaf A")


if __name__ == "__main__":
    unittest.main()
