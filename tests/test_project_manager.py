"""Tests for workspace import and persistence behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neopest.analysis import placeholder_analysis_result, run_leaf_analysis
from neopest.metadata import DEFAULT_CONDITION_GROUP_NAME, WORKSPACE_JSON_FILENAME
from neopest.project_manager import (
    WorkspaceError,
    create_condition_group,
    get_default_workspace_root,
    import_images,
    initialize_workspace,
    load_workspace,
)


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


def _write_test_png(path: Path, *, width: int = 32, height: int = 24) -> None:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("green"))
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"Failed to write test image: {path}")


class TestProjectManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_get_default_workspace_root_points_to_documents_neopest(self) -> None:
        root = get_default_workspace_root()
        self.assertEqual(root, Path.home() / "Documents" / "NeoPest")
        self.assertTrue(str(root).endswith("Documents/NeoPest"))

    def test_initialize_workspace_creates_default_condition_group(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Lab A")
            self.assertEqual(len(workspace.condition_groups), 1)
            self.assertEqual(
                workspace.condition_groups[0].display_name,
                DEFAULT_CONDITION_GROUP_NAME,
            )
            self.assertTrue((root / WORKSPACE_JSON_FILENAME).is_file())

    def test_initialize_workspace_refuses_to_overwrite_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            initialize_workspace(root, display_name="Original")
            original_payload = (root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8")

            with self.assertRaises(WorkspaceError):
                initialize_workspace(root, display_name="Replacement")

            self.assertEqual(
                (root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8"),
                original_payload,
            )

    def test_load_workspace_opens_existing_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            created = initialize_workspace(root, display_name="Existing")
            loaded = load_workspace(root)
            self.assertEqual(loaded.workspace_id, created.workspace_id)
            self.assertEqual(loaded.display_name, "Existing")

    def test_load_workspace_raises_when_metadata_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty_folder"
            root.mkdir()
            with self.assertRaises(WorkspaceError):
                load_workspace(root)
            self.assertFalse((root / WORKSPACE_JSON_FILENAME).exists())

    def test_load_workspace_does_not_create_workspace_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "not_a_workspace"
            root.mkdir()
            with self.assertRaises(WorkspaceError):
                load_workspace(root)
            self.assertFalse((root / WORKSPACE_JSON_FILENAME).exists())
            self.assertFalse((root / "raw_images").exists())

    def test_create_condition_group_adds_entry(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            group = create_condition_group(workspace, display_name="Treatment B")
            loaded = load_workspace(workspace.root_path)
            ids = [item.condition_group_id for item in loaded.condition_groups]
            self.assertIn(group.condition_group_id, ids)

    def test_import_images_creates_samples_with_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "sources"
            source_dir.mkdir()
            source_a = source_dir / "leaf_a.png"
            source_b = source_dir / "leaf_b.png"
            _write_test_png(source_a, width=40, height=30)
            _write_test_png(source_b, width=50, height=35)

            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source_a, source_b])
            self.assertEqual(len(imported), 2)

            loaded = load_workspace(workspace.root_path)
            self.assertEqual(len(loaded.samples), 2)
            sample = loaded.get_sample(imported[0].sample_id)
            self.assertIsNotNone(sample)
            self.assertTrue(sample.sample_id.startswith("sample_"))
            self.assertEqual(sample.condition_group_id, loaded.condition_groups[0].condition_group_id)
            self.assertTrue(sample.raw_image_path.startswith("raw_images/"))
            self.assertEqual(sample.original_source_path, str(source_a.resolve()))
            self.assertIsNotNone(sample.image_hash)
            self.assertTrue(sample.image_hash.startswith("sha256:"))

            absolute = sample.raw_image_absolute(loaded.root_path)
            self.assertTrue(absolute.is_file())

    def test_import_to_specific_condition_group(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)

            workspace = initialize_workspace(Path(tmp) / "ws")
            group = create_condition_group(workspace, display_name="Mutant Line")
            imported = import_images(workspace, [source], condition_group_id=group.condition_group_id)
            self.assertEqual(imported[0].condition_group_id, group.condition_group_id)

    def test_selecting_metadata_does_not_require_analysis(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            sample = load_workspace(workspace.root_path).get_sample(imported[0].sample_id)
            self.assertIsNotNone(sample)

            result = placeholder_analysis_result(sample.sample_id)
            self.assertTrue(result.is_placeholder)
            with self.assertRaises(NotImplementedError):
                run_leaf_analysis(sample.sample_id, sample.raw_image_absolute(workspace.root_path))


if __name__ == "__main__":
    unittest.main()
