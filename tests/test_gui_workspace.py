"""Tests for workspace open/new GUI behavior without a visible window."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from neopest.gui import MainWindow
from neopest.metadata import WORKSPACE_JSON_FILENAME
from neopest.project_manager import initialize_workspace


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


class TestGuiWorkspaceOpenNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_open_workspace_does_not_initialize_missing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "random_folder"
            folder.mkdir()

            window = MainWindow()
            with (
                patch.object(QFileDialog, "getExistingDirectory", return_value=str(folder)),
                patch.object(QMessageBox, "warning"),
            ):
                window._on_open_workspace()

            self.assertIsNone(window._workspace)
            self.assertFalse((folder / WORKSPACE_JSON_FILENAME).exists())

    def test_open_workspace_loads_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            created = initialize_workspace(root, display_name="GUI Open Test")

            window = MainWindow()
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)):
                window._on_open_workspace()

            self.assertIsNotNone(window._workspace)
            self.assertEqual(window._workspace.workspace_id, created.workspace_id)

    def test_open_default_workspace_does_not_create_missing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "Documents" / "NeoPest"

            window = MainWindow()
            with (
                patch(
                    "neopest.gui.get_default_workspace_root",
                    return_value=default_root,
                ),
                patch.object(QMessageBox, "information"),
            ):
                window._on_open_default_workspace()

            self.assertIsNone(window._workspace)
            self.assertFalse(default_root.exists())

    def test_open_default_workspace_loads_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "Documents" / "NeoPest"
            created = initialize_workspace(default_root, display_name="Default")

            window = MainWindow()
            with patch(
                "neopest.gui.get_default_workspace_root",
                return_value=default_root,
            ):
                window._on_open_default_workspace()

            self.assertIsNotNone(window._workspace)
            self.assertEqual(window._workspace.workspace_id, created.workspace_id)

    def test_new_default_workspace_initializes_missing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "Documents" / "NeoPest"

            window = MainWindow()
            with patch(
                "neopest.gui.get_default_workspace_root",
                return_value=default_root,
            ):
                window._on_new_default_workspace()

            self.assertIsNotNone(window._workspace)
            self.assertTrue((default_root / WORKSPACE_JSON_FILENAME).is_file())

    def test_new_default_workspace_refuses_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "Documents" / "NeoPest"
            initialize_workspace(default_root, display_name="Default")
            original_payload = (default_root / WORKSPACE_JSON_FILENAME).read_text(
                encoding="utf-8"
            )

            window = MainWindow()
            with (
                patch(
                    "neopest.gui.get_default_workspace_root",
                    return_value=default_root,
                ),
                patch.object(QMessageBox, "warning"),
            ):
                window._on_new_default_workspace()

            self.assertIsNone(window._workspace)
            self.assertEqual(
                (default_root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8"),
                original_payload,
            )

    def test_new_workspace_refuses_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            initialize_workspace(root, display_name="Existing")
            original_payload = (root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8")

            window = MainWindow()
            with (
                patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)),
                patch.object(QInputDialog, "getText", return_value=("Replacement", True)),
                patch.object(QMessageBox, "warning"),
            ):
                window._on_new_workspace()

            self.assertIsNone(window._workspace)
            self.assertEqual(
                (root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8"),
                original_payload,
            )


if __name__ == "__main__":
    unittest.main()
