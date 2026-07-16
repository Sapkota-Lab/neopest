"""Tests for display-only view mode GUI wiring without a visible window."""

from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from neopest.gui import MainWindow
from neopest.metadata import WORKSPACE_JSON_FILENAME
from neopest.project_manager import WorkspaceError, import_images, initialize_workspace
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


class TestGuiViewMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_view_dropdown_disabled_without_sample(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(window.inspector_panel.view_mode_combo.isEnabled())

    def test_view_dropdown_enabled_for_sample_with_image(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Preview Test")
            image_path = Path(tmp) / "leaf.png"
            image = _make_synthetic_leaf_image()
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            imported = import_images(workspace, [image_path])
            self.assertEqual(len(imported), 1)

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)

            self.assertTrue(window.inspector_panel.view_mode_combo.isEnabled())

    def test_view_whole_leaf_updates_overlay(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Preview Overlay")
            image_path = Path(tmp) / "leaf.png"
            image = _make_synthetic_leaf_image()
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            imported = import_images(workspace, [image_path])

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()

            self.assertIn(imported[0].sample_id, window._leaf_mask_previews)
            preview = window._leaf_mask_previews[imported[0].sample_id]
            self.assertEqual(preview.status.value, "success")
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

    def test_view_does_not_change_sample_analysis_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Status Test")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()

            sample = workspace.get_sample(imported[0].sample_id)
            assert sample is not None
            self.assertEqual(sample.analysis_status.value, "not_run")

    def test_view_failure_does_not_change_sample_analysis_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Failure Status")
            image_path = Path(tmp) / "no_blue.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(np.full((120, 120, 3), 245, dtype=np.uint8), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()

            sample = workspace.get_sample(imported[0].sample_id)
            assert sample is not None
            self.assertEqual(sample.analysis_status.value, "not_run")

    def test_switching_samples_clears_stale_overlay(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Overlay Switch")
            image_a = Path(tmp) / "leaf_a.png"
            image_b = Path(tmp) / "leaf_b.png"
            cv2.imwrite(
                str(image_a),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(image_b),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_a, image_b])
            sample_a, sample_b = imported[0], imported[1]

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_a.sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

            window._on_sample_selected(sample_b.sample_id)
            self.assertIsNone(window.image_canvas._leaf_mask_overlay)

    def test_workspace_switch_clears_preview_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "ws_a"
            root_b = Path(tmp) / "ws_b"
            workspace_a = initialize_workspace(root_a, display_name="Workspace A")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace_a, [image_path])

            window = _main_window_without_startup_default()
            window._set_workspace(workspace_a)
            window._on_sample_selected(imported[0].sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()
            self.assertIn(imported[0].sample_id, window._leaf_mask_previews)

            workspace_b = initialize_workspace(root_b, display_name="Workspace B")
            window._set_workspace(workspace_b)
            self.assertEqual(window._leaf_mask_previews, {})
            self.assertIsNone(window.image_canvas._leaf_mask_overlay)

    def test_view_does_not_persist_measurement(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Persistence")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            metadata_path = root / WORKSPACE_JSON_FILENAME
            before = json.loads(metadata_path.read_text(encoding="utf-8"))

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()

            after = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(before, after)
            self.assertNotIn("leaf_mask", json.dumps(after))

    def test_gui_layout_builders_do_not_import_segmentation(self) -> None:
        import neopest.gui_layout_builders as layout_builders

        source = inspect.getsource(layout_builders)
        self.assertNotIn("segmentation", source)

    def test_gui_layout_builders_do_not_perform_measurement_math(self) -> None:
        import neopest.gui_layout_builders as layout_builders

        source = inspect.getsource(layout_builders)
        self.assertNotIn("calculate_leaf_area", source)
        self.assertNotIn("pixels_per_inch", source)

    def test_run_analysis_button_stays_disabled_and_hidden(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(window.run_analysis_button.isEnabled())
        self.assertFalse(window.run_analysis_button.isVisible())


if __name__ == "__main__":
    unittest.main()
