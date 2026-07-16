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

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QInputDialog,
    QMessageBox,
    QAbstractItemView,
    QTreeWidgetItem,
)

from neopest.desktop_services import reveal_image_menu_label
from neopest.explorer_tree import (
    MIME_SAMPLE_ID,
    ExplorerTreeWidget,
    build_explorer_panel,
    build_sample_drag_pixmap,
    sample_drag_hotspot,
)
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


def _destroy_explorer_tree(tree: ExplorerTreeWidget) -> None:
    tree.clear()
    tree.setParent(None)
    tree.deleteLater()


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
                    panel.result_status_value.text(),
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
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_workspace_and_group_nodes_are_not_draggable(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            root, group_item, _sample_item = self._node_items(tree)
            self.assertFalse(root.flags() & Qt.ItemFlag.ItemIsDragEnabled)
            self.assertFalse(group_item.flags() & Qt.ItemFlag.ItemIsDragEnabled)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_condition_groups_accept_sample_drops(self) -> None:
        holder, _workspace, _sample, tree = self._workspace_with_sample()
        try:
            _root, group_item, _sample_item = self._node_items(tree)
            self.assertTrue(group_item.flags() & Qt.ItemFlag.ItemIsDropEnabled)
            self.assertTrue(tree.accepts_sample_drop_on_item(group_item))
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_samples_and_workspace_reject_sample_drops(self) -> None:
        holder, workspace, sample, tree = self._workspace_with_sample()
        try:
            root, group_item, sample_item = self._node_items(tree)
            self.assertFalse(sample_item.flags() & Qt.ItemFlag.ItemIsDropEnabled)
            self.assertFalse(tree.accepts_sample_drop_on_item(root))
            self.assertFalse(tree.accepts_sample_drop_on_item(None))
            self.assertFalse(
                tree.accepts_sample_drop_on_item(
                    sample_item,
                    dragging_sample_id=sample.sample_id,
                )
            )
            self.assertFalse(
                tree.accepts_sample_drop_on_item(
                    group_item,
                    dragging_sample_id=sample.sample_id,
                )
            )
        finally:
            _destroy_explorer_tree(tree)
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
            _destroy_explorer_tree(tree)
            holder.cleanup()


class TestExplorerDragTargetResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _write_leaf_image(self, path: Path) -> None:
        cv2.imwrite(
            str(path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )

    def _multi_group_workspace(self):
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="DND")
        source_group = workspace.condition_groups[0]

        source_samples = []
        for index in range(3):
            image_path = Path(tmp.name) / f"leaf_a_{index}.png"
            self._write_leaf_image(image_path)
            source_samples.extend(import_images(workspace, [image_path]))

        destination_group = create_condition_group(workspace, display_name="Group B")
        dest_image = Path(tmp.name) / "leaf_b.png"
        self._write_leaf_image(dest_image)
        dest_samples = import_images(
            workspace,
            [dest_image],
            condition_group_id=destination_group.condition_group_id,
        )

        tree = ExplorerTreeWidget()
        tree.show()
        tree.resize(320, 480)
        tree.refresh_workspace(workspace)
        return (
            tmp,
            workspace,
            source_group,
            destination_group,
            source_samples,
            dest_samples[0],
            tree,
        )

    def _group_item(self, tree: ExplorerTreeWidget, group_id: str) -> QTreeWidgetItem:
        root = tree.topLevelItem(0)
        for index in range(root.childCount()):
            child = root.child(index)
            if tree.item_node_id(child) == group_id:
                return child
        raise AssertionError(f"Missing group item: {group_id}")

    def _sample_item(
        self,
        tree: ExplorerTreeWidget,
        sample_id: str,
    ) -> QTreeWidgetItem:
        root = tree.topLevelItem(0)
        match = tree._find_sample_item(root, sample_id)
        assert match is not None
        return match

    def _mime_for_sample(self, sample_id: str):
        from PySide6.QtCore import QMimeData

        mime = QMimeData()
        mime.setData(MIME_SAMPLE_ID, sample_id.encode("utf-8"))
        return mime

    def _drag_move_over_item(
        self,
        tree: ExplorerTreeWidget,
        item: QTreeWidgetItem | None,
        sample_id: str,
    ) -> None:
        tree.update_drag_hover_for_item(item, dragging_sample_id=sample_id)

    def _drop_on_item(
        self,
        tree: ExplorerTreeWidget,
        item: QTreeWidgetItem | None,
        sample_id: str,
    ) -> bool:
        return tree.drop_sample_on_item(item, dragging_sample_id=sample_id)

    def test_group_header_resolves_to_that_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_item = self._group_item(tree, dest_group.condition_group_id)
            group_id, highlight = tree.resolve_drag_destination_group(
                dest_item,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertEqual(group_id, dest_group.condition_group_id)
            self.assertIs(highlight, dest_item)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_sample_row_resolves_to_parent_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_sample_item = self._sample_item(tree, dest_sample.sample_id)
            group_id, highlight = tree.resolve_drag_destination_group(
                dest_sample_item,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertEqual(group_id, dest_group.condition_group_id)
            self.assertIs(
                highlight,
                self._group_item(tree, dest_group.condition_group_id),
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_drag_move_across_intervening_samples_updates_target(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dragging_id = source_samples[0].sample_id
            dest_group_item = self._group_item(tree, dest_group.condition_group_id)
            dest_sample_item = self._sample_item(tree, dest_sample.sample_id)
            intervening_sample_item = self._sample_item(tree, source_samples[2].sample_id)

            self._drag_move_over_item(tree, intervening_sample_item, dragging_id)
            self.assertIsNone(tree.drop_highlight_item())

            self._drag_move_over_item(tree, intervening_sample_item, dragging_id)
            self.assertIsNone(tree.drop_highlight_item())

            self._drag_move_over_item(tree, dest_group_item, dragging_id)
            self.assertIs(tree.drop_highlight_item(), dest_group_item)

            self._drag_move_over_item(tree, dest_sample_item, dragging_id)
            self.assertIs(tree.drop_highlight_item(), dest_group_item)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_farther_destination_group_is_accepted(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_item = self._group_item(tree, dest_group.condition_group_id)
            self.assertTrue(
                tree.accepts_sample_drop_on_item(
                    dest_item,
                    dragging_sample_id=source_samples[-1].sample_id,
                )
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_drop_on_group_header_moves_sample(self) -> None:
        holder, workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )
            dest_item = self._group_item(tree, dest_group.condition_group_id)
            accepted = self._drop_on_item(tree, dest_item, source_samples[0].sample_id)
            self.assertTrue(accepted)
            self.assertEqual(
                moves,
                [(source_samples[0].sample_id, dest_group.condition_group_id)],
            )
            self.assertEqual(
                workspace.get_sample(source_samples[0].sample_id).condition_group_id,
                source_samples[0].condition_group_id,
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_drop_on_child_sample_moves_into_parent_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )
            dest_sample_item = self._sample_item(tree, dest_sample.sample_id)
            accepted = self._drop_on_item(tree, dest_sample_item, source_samples[0].sample_id)
            self.assertTrue(accepted)
            self.assertEqual(
                moves,
                [(source_samples[0].sample_id, dest_group.condition_group_id)],
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_drop_within_source_group_is_rejected(self) -> None:
        holder, workspace, source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )
            source_group_item = self._group_item(tree, source_group.condition_group_id)
            before_group_id = workspace.get_sample(source_samples[0].sample_id).condition_group_id
            accepted = self._drop_on_item(
                tree,
                source_group_item,
                source_samples[0].sample_id,
            )
            self.assertFalse(accepted)
            self.assertEqual(moves, [])
            self.assertEqual(
                workspace.get_sample(source_samples[0].sample_id).condition_group_id,
                before_group_id,
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_workspace_root_is_not_accepted(self) -> None:
        holder, _workspace, _source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            root = tree.topLevelItem(0)
            group_id, highlight = tree.resolve_drag_destination_group(
                root,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertIsNone(group_id)
            self.assertIsNone(highlight)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_empty_explorer_space_is_not_accepted(self) -> None:
        holder, _workspace, _source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            self.assertFalse(
                tree.accepts_sample_drop_on_item(
                    None,
                    dragging_sample_id=source_samples[0].sample_id,
                )
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_invalid_drop_does_not_emit_move(self) -> None:
        holder, workspace, source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )
            source_group_item = self._group_item(tree, source_group.condition_group_id)
            self._drop_on_item(tree, source_group_item, source_samples[0].sample_id)
            self.assertEqual(moves, [])
            self.assertEqual(
                workspace.get_sample(source_samples[0].sample_id).condition_group_id,
                source_group.condition_group_id,
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_valid_drop_preserves_sample_metadata(self) -> None:
        from neopest.metadata import CalibrationRecord
        from neopest.project_manager import move_sample_to_condition_group

        holder, workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            sample = workspace.get_sample(source_samples[0].sample_id)
            assert sample is not None
            sample.calibration = CalibrationRecord(
                pixels_per_mm=3.937,
                known_distance_mm=25.4,
                point_1_x=10.0,
                point_1_y=20.0,
                point_2_x=110.0,
                point_2_y=20.0,
                known_distance_value=1.0,
                known_distance_unit="in",
            )
            before_id = sample.sample_id
            before_calibration = sample.calibration
            before_image = sample.raw_image_path
            before_hash = sample.image_hash

            move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                dest_group.condition_group_id,
                save=False,
            )

            updated = workspace.get_sample(before_id)
            assert updated is not None
            self.assertEqual(updated.sample_id, before_id)
            self.assertEqual(updated.calibration, before_calibration)
            self.assertNotEqual(updated.raw_image_path, before_image)
            self.assertEqual(updated.image_hash, before_hash)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_highlight_changes_with_hover_target(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_group_item = self._group_item(tree, dest_group.condition_group_id)
            dest_sample_item = self._sample_item(tree, dest_sample.sample_id)
            dragging_id = source_samples[0].sample_id

            self._drag_move_over_item(tree, dest_group_item, dragging_id)
            self.assertIs(tree.drop_highlight_item(), dest_group_item)

            self._drag_move_over_item(tree, dest_sample_item, dragging_id)
            self.assertIs(tree.drop_highlight_item(), dest_group_item)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_highlight_clears_on_drag_leave(self) -> None:
        from PySide6.QtGui import QDragLeaveEvent

        holder, _workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_group_item = self._group_item(tree, dest_group.condition_group_id)
            self._drag_move_over_item(tree, dest_group_item, source_samples[0].sample_id)
            self.assertIsNotNone(tree.drop_highlight_item())
            tree.dragLeaveEvent(QDragLeaveEvent())
            self.assertIsNone(tree.drop_highlight_item())
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_highlight_clears_after_successful_drop(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            dest_sample_item = self._sample_item(tree, dest_sample.sample_id)
            self._drag_move_over_item(tree, dest_sample_item, source_samples[0].sample_id)
            self._drop_on_item(tree, dest_sample_item, source_samples[0].sample_id)
            self.assertIsNone(tree.drop_highlight_item())
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_highlight_clears_after_rejected_drop(self) -> None:
        holder, workspace, source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            source_group_item = self._group_item(tree, source_group.condition_group_id)
            self._drag_move_over_item(tree, source_group_item, source_samples[0].sample_id)
            self._drop_on_item(tree, source_group_item, source_samples[0].sample_id)
            self.assertIsNone(tree.drop_highlight_item())
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_auto_scroll_enabled(self) -> None:
        holder, _workspace, _source_group, _dest_group, _source_samples, _dest_sample, tree = (
            self._multi_group_workspace()
        )
        try:
            self.assertTrue(tree.hasAutoScroll())
            self.assertGreaterEqual(tree.autoScrollMargin(), 16)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()


class TestExplorerDragCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_drag_pixmap_contains_sample_name(self) -> None:
        app = QApplication.instance()
        assert app is not None
        pixmap = build_sample_drag_pixmap(
            "Photo 3 - Brillance R1.jpg",
            app.palette(),
            device_pixel_ratio=2.0,
        )
        self.assertFalse(pixmap.isNull())
        logical_width = pixmap.width() / pixmap.devicePixelRatio()
        logical_height = pixmap.height() / pixmap.devicePixelRatio()
        self.assertGreater(logical_width, 0)
        self.assertLessEqual(logical_width, 280)
        self.assertLessEqual(logical_height, 80)

    def test_drag_pixmap_elides_long_names(self) -> None:
        app = QApplication.instance()
        assert app is not None
        long_name = "A" * 120 + ".jpg"
        pixmap = build_sample_drag_pixmap(long_name, app.palette())
        self.assertFalse(pixmap.isNull())
        self.assertLessEqual(pixmap.width() / pixmap.devicePixelRatio(), 280)

    def test_drag_hotspot_is_set(self) -> None:
        hotspot = sample_drag_hotspot()
        self.assertGreater(hotspot.x(), 0)
        self.assertGreater(hotspot.y(), 0)

    def test_start_drag_sets_custom_pixmap_for_sample(self) -> None:
        holder, _workspace, _source_group, _dest_group, source_samples, _dest_sample, tree = (
            TestExplorerDragTargetResolution()._multi_group_workspace()
        )
        try:
            sample_item = TestExplorerDragTargetResolution()._sample_item(
                tree,
                source_samples[0].sample_id,
            )
            tree.setCurrentItem(sample_item)
            with patch("neopest.explorer_tree.QDrag") as mock_drag_cls:
                mock_drag = mock_drag_cls.return_value
                mock_drag.exec.return_value = Qt.DropAction.IgnoreAction
                tree.startDrag(Qt.DropAction.MoveAction)
                mock_drag.setPixmap.assert_called_once()
                pixmap = mock_drag.setPixmap.call_args[0][0]
                self.assertFalse(pixmap.isNull())
                mock_drag.setHotSpot.assert_called_once()
                mock_drag.setMimeData.assert_called_once()
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_non_sample_does_not_start_drag(self) -> None:
        holder, _workspace, source_group, _dest_group, _source_samples, _dest_sample, tree = (
            TestExplorerDragTargetResolution()._multi_group_workspace()
        )
        try:
            group_item = TestExplorerDragTargetResolution()._group_item(
                tree,
                source_group.condition_group_id,
            )
            tree.setCurrentItem(group_item)
            with patch("neopest.explorer_tree.QDrag") as mock_drag_cls:
                tree.startDrag(Qt.DropAction.MoveAction)
                mock_drag_cls.assert_not_called()
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()


class TestExplorerDragPositionResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _helper(self) -> TestExplorerDragTargetResolution:
        return TestExplorerDragTargetResolution()

    def _position_on_item(
        self,
        tree: ExplorerTreeWidget,
        item: QTreeWidgetItem,
        *,
        region: str,
    ) -> QPoint:
        rect = tree.visualItemRect(item)
        if region == "right":
            return QPoint(min(tree.viewport().width() - 2, rect.right() + 24), rect.center().y())
        if region == "indent":
            return QPoint(max(1, rect.left() - 6), rect.center().y())
        if region == "gap":
            return QPoint(rect.center().x(), rect.top() - 2)
        return QPoint(rect.center().x(), rect.center().y())

    def test_blank_right_of_child_row_resolves_to_parent_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            dest_sample_item = self._helper()._sample_item(tree, dest_sample.sample_id)
            position = self._position_on_item(tree, dest_sample_item, region="right")
            with patch.object(tree, "itemAt", return_value=None):
                group_id, highlight = tree.resolve_drag_destination_group_at_position(
                    position,
                    dragging_sample_id=source_samples[0].sample_id,
                )
            self.assertEqual(group_id, dest_group.condition_group_id)
            self.assertIs(
                highlight,
                self._helper()._group_item(tree, dest_group.condition_group_id),
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_indentation_beside_child_row_resolves_to_parent_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            dest_sample_item = self._helper()._sample_item(tree, dest_sample.sample_id)
            position = self._position_on_item(tree, dest_sample_item, region="indent")
            with patch.object(tree, "itemAt", return_value=None):
                group_id, _highlight = tree.resolve_drag_destination_group_at_position(
                    position,
                    dragging_sample_id=source_samples[0].sample_id,
                )
            self.assertEqual(group_id, dest_group.condition_group_id)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_row_gap_resolves_to_nearest_group_row(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            from PySide6.QtCore import QModelIndex

            dest_group_item = self._helper()._group_item(tree, dest_group.condition_group_id)
            first_child = dest_group_item.child(0)
            rect_a = tree.visualItemRect(dest_group_item)
            rect_b = tree.visualItemRect(first_child)
            gap_y = (rect_a.bottom() + rect_b.top()) // 2
            position = QPoint(rect_a.center().x(), gap_y)
            with patch.object(tree, "itemAt", return_value=None):
                with patch.object(tree, "indexAt", return_value=QModelIndex()):
                    group_id, _highlight = tree.resolve_drag_destination_group_at_position(
                        position,
                        dragging_sample_id=source_samples[0].sample_id,
                    )
            self.assertEqual(group_id, dest_group.condition_group_id)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_hint_row_resolves_to_parent_group(self) -> None:
        holder, workspace, source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            empty_group = create_condition_group(workspace, display_name="Empty Group")
            tree.refresh_workspace(workspace)
            empty_group_item = self._helper()._group_item(tree, empty_group.condition_group_id)
            hint_item = empty_group_item.child(0)
            self.assertIn("no samples", hint_item.text(0).lower())
            position = self._position_on_item(tree, hint_item, region="center")
            group_id, highlight = tree.resolve_drag_destination_group_at_position(
                position,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertEqual(group_id, empty_group.condition_group_id)
            self.assertIs(highlight, empty_group_item)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_boundary_movement_does_not_stick_after_leaving_group(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            dest_group_item = self._helper()._group_item(tree, dest_group.condition_group_id)
            dest_position = self._position_on_item(tree, dest_group_item, region="center")
            root_item = tree.topLevelItem(0)
            root_position = self._position_on_item(tree, root_item, region="center")

            self.assertTrue(
                tree.update_drag_hover_at_position(
                    dest_position,
                    dragging_sample_id=source_samples[0].sample_id,
                )
            )
            self.assertIs(tree.drop_highlight_item(), dest_group_item)
            self.assertFalse(
                tree.update_drag_hover_at_position(
                    root_position,
                    dragging_sample_id=source_samples[0].sample_id,
                )
            )
            self.assertIsNone(tree.drop_highlight_item())
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_drop_re_resolves_from_final_position_not_stale_highlight(self) -> None:
        holder, workspace, source_group, dest_group, source_samples, _dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            moves: list[tuple[str, str]] = []
            tree.sample_move_requested.connect(
                lambda sample_id, group_id: moves.append((sample_id, group_id))
            )
            dest_group_item = self._helper()._group_item(tree, dest_group.condition_group_id)
            source_group_item = self._helper()._group_item(tree, source_group.condition_group_id)
            dest_position = self._position_on_item(tree, dest_group_item, region="center")
            source_position = self._position_on_item(tree, source_group_item, region="center")

            tree.update_drag_hover_at_position(
                dest_position,
                dragging_sample_id=source_samples[0].sample_id,
            )
            accepted = tree.drop_sample_at_position(
                source_position,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertFalse(accepted)
            self.assertEqual(moves, [])
            self.assertEqual(
                workspace.get_sample(source_samples[0].sample_id).condition_group_id,
                source_group.condition_group_id,
            )
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_auto_scroll_resolution_after_viewport_scroll(self) -> None:
        holder, _workspace, _source_group, dest_group, source_samples, dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            dest_group_item = self._helper()._group_item(tree, dest_group.condition_group_id)
            tree.scrollToItem(dest_group_item, QAbstractItemView.ScrollHint.PositionAtBottom)
            QApplication.processEvents()
            position = self._position_on_item(tree, dest_group_item, region="center")
            group_id, highlight = tree.resolve_drag_destination_group_at_position(
                position,
                dragging_sample_id=source_samples[0].sample_id,
            )
            self.assertEqual(group_id, dest_group.condition_group_id)
            self.assertIs(highlight, dest_group_item)
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_invalid_mime_rejected_on_drag_enter(self) -> None:
        from PySide6.QtCore import QMimeData
        from PySide6.QtGui import QDragEnterEvent

        holder, _workspace, _source_group, _dest_group, source_samples, _dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            mime = QMimeData()
            mime.setText("not-a-sample")
            event = QDragEnterEvent(
                QPoint(8, 8),
                Qt.DropAction.MoveAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            tree.dragEnterEvent(event)
            self.assertFalse(event.isAccepted())
        finally:
            _destroy_explorer_tree(tree)
            holder.cleanup()

    def test_valid_drop_preserves_both_measurement_records(self) -> None:
        from neopest.metadata import (
            CalibrationRecord,
            CandidateDiseaseMeasurementRecord,
            LeafMeasurementRecord,
        )
        from neopest.project_manager import move_sample_to_condition_group

        holder, workspace, _source_group, dest_group, source_samples, _dest_sample, tree = (
            self._helper()._multi_group_workspace()
        )
        try:
            sample = workspace.get_sample(source_samples[0].sample_id)
            assert sample is not None
            sample.calibration = CalibrationRecord(
                pixels_per_mm=3.937,
                known_distance_mm=25.4,
                point_1_x=10.0,
                point_1_y=20.0,
                point_2_x=110.0,
                point_2_y=20.0,
                known_distance_value=1.0,
                known_distance_unit="in",
            )
            sample.leaf_measurement = LeafMeasurementRecord(
                status="valid",
                leaf_area_pixels=1000,
                leaf_area_in2=1.5,
            )
            sample.candidate_disease_measurement = CandidateDiseaseMeasurementRecord(
                status="valid",
                candidate_disease_pixels=100,
                candidate_disease_area_in2=0.15,
                candidate_disease_coverage_percent=10.0,
                non_candidate_leaf_area_in2=1.35,
            )
            before_name = sample.display_name
            before_hash = sample.image_hash

            move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                dest_group.condition_group_id,
                save=False,
            )

            updated = workspace.get_sample(sample.sample_id)
            assert updated is not None
            self.assertEqual(updated.display_name, before_name)
            self.assertEqual(updated.image_hash, before_hash)
            self.assertEqual(updated.calibration, sample.calibration)
            self.assertEqual(updated.leaf_measurement, sample.leaf_measurement)
            self.assertEqual(
                updated.candidate_disease_measurement,
                sample.candidate_disease_measurement,
            )
        finally:
            _destroy_explorer_tree(tree)
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

    def test_measurements_caption_is_uppercase(self) -> None:
        source = inspect.getsource(SampleInspectorPanel.__init__)
        self.assertIn('QLabel("MEASUREMENTS")', source)

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
