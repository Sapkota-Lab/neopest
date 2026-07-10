"""Tests for workspace import and persistence behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neopest.analysis import placeholder_analysis_result, run_leaf_analysis
from neopest.metadata import DEFAULT_CONDITION_GROUP_NAME, WORKSPACE_JSON_FILENAME
from neopest.calibration import build_one_inch_manual_calibration_record
from neopest.measurements import (
    CURRENT_LEAF_MASK_METHOD,
    CURRENT_LEAF_MASK_SETTINGS_VERSION,
    build_leaf_measurement_record,
    calculate_leaf_area,
)
from neopest.project_manager import (
    RAW_IMAGES_DIR,
    WorkspaceError,
    create_condition_group,
    ensure_default_workspace,
    get_default_workspace_root,
    import_images,
    initialize_workspace,
    load_workspace,
    move_sample_to_condition_group,
    save_sample_calibration,
    save_sample_leaf_measurement,
)
from neopest.results import AnalysisStatus, CalibrationStatus
from neopest.segmentation import PREVIEW_METHOD, segment_leaf_preview
from tests.test_segmentation import _make_synthetic_leaf_image


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

    def test_ensure_default_workspace_creates_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "NeoPest"
            with patch(
                "neopest.project_manager.get_default_workspace_root",
                return_value=default_root,
            ):
                workspace = ensure_default_workspace(display_name="Test Default")
            self.assertEqual(workspace.root_path, default_root.resolve())
            self.assertTrue((default_root / WORKSPACE_JSON_FILENAME).is_file())

    def test_ensure_default_workspace_loads_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "NeoPest"
            with patch(
                "neopest.project_manager.get_default_workspace_root",
                return_value=default_root,
            ):
                created = ensure_default_workspace(display_name="Existing Default")
                loaded = ensure_default_workspace()
            self.assertEqual(loaded.workspace_id, created.workspace_id)

    def test_ensure_default_workspace_initializes_folder_without_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            default_root = Path(tmp) / "NeoPest"
            default_root.mkdir()
            with patch(
                "neopest.project_manager.get_default_workspace_root",
                return_value=default_root,
            ):
                workspace = ensure_default_workspace()
            self.assertTrue((default_root / WORKSPACE_JSON_FILENAME).is_file())
            self.assertEqual(workspace.root_path, default_root.resolve())

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


class TestSampleMove(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _import_leaf(self, workspace, name: str = "leaf.png"):
        image_path = workspace.root_path.parent / name
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        return import_images(workspace, [image_path])[0]

    def test_move_updates_group_and_managed_path(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            source_group = workspace.condition_groups[0]
            destination_group = create_condition_group(workspace, display_name="Treatment B")
            sample = self._import_leaf(workspace)
            source_path = sample.raw_image_absolute(workspace.root_path)
            original_path = sample.raw_image_path
            original_hash = sample.image_hash
            original_source = sample.original_source_path

            updated = move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                destination_group.condition_group_id,
            )

            self.assertEqual(updated.condition_group_id, destination_group.condition_group_id)
            self.assertEqual(updated.sample_id, sample.sample_id)
            self.assertEqual(updated.image_hash, original_hash)
            self.assertEqual(updated.original_source_path, original_source)
            self.assertNotEqual(updated.raw_image_path, original_path)
            self.assertFalse(source_path.is_file())
            destination_path = updated.raw_image_absolute(workspace.root_path)
            self.assertTrue(destination_path.is_file())
            self.assertIn(destination_group.condition_group_id, updated.raw_image_path)

            loaded = load_workspace(workspace.root_path)
            persisted = loaded.get_sample(sample.sample_id)
            assert persisted is not None
            self.assertEqual(persisted.condition_group_id, destination_group.condition_group_id)

    def test_move_preserves_calibration_and_measurement(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            destination_group = create_condition_group(workspace, display_name="Moved Here")
            sample = self._import_leaf(workspace)
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample.sample_id, calibration)
            sample = workspace.get_sample(sample.sample_id)
            assert sample is not None
            sample.analysis_status = AnalysisStatus.NOT_RUN

            preview = segment_leaf_preview(_make_synthetic_leaf_image())
            assert preview.mask is not None
            area = calculate_leaf_area(preview.mask, calibration.pixels_per_mm)
            measurement = build_leaf_measurement_record(
                area,
                method=PREVIEW_METHOD,
                settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
                source_image_hash=sample.image_hash,
            )
            save_sample_leaf_measurement(workspace, sample.sample_id, measurement)

            before = workspace.get_sample(sample.sample_id)
            assert before is not None
            assert before.calibration is not None
            assert before.leaf_measurement is not None

            updated = move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                destination_group.condition_group_id,
            )

            self.assertEqual(updated.calibration.to_dict(), before.calibration.to_dict())
            self.assertEqual(
                updated.leaf_measurement.to_dict(),
                before.leaf_measurement.to_dict(),
            )
            self.assertEqual(updated.calibration_status, CalibrationStatus.MANUAL)
            self.assertEqual(updated.analysis_status, AnalysisStatus.NOT_RUN)
            self.assertEqual(updated.leaf_measurement.status, "valid")

    def test_same_group_move_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            sample = self._import_leaf(workspace)
            source_path = sample.raw_image_absolute(workspace.root_path)
            original_path = sample.raw_image_path

            updated = move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                sample.condition_group_id,
            )

            self.assertIs(updated, sample)
            self.assertEqual(updated.raw_image_path, original_path)
            self.assertTrue(source_path.is_file())

    def test_collision_generates_unique_managed_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            destination_group = create_condition_group(workspace, display_name="Collision")
            sample = self._import_leaf(workspace, name="leaf_a.png")
            source_path = sample.raw_image_absolute(workspace.root_path)
            managed_name = source_path.name

            destination_dir = (
                workspace.root_path / RAW_IMAGES_DIR / destination_group.condition_group_id
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            blocker = destination_dir / managed_name
            blocker.write_bytes(b"existing managed file")

            updated = move_sample_to_condition_group(
                workspace,
                sample.sample_id,
                destination_group.condition_group_id,
            )

            destination_path = updated.raw_image_absolute(workspace.root_path)
            self.assertNotEqual(destination_path.name, managed_name)
            self.assertTrue(destination_path.name.startswith(Path(managed_name).stem))
            self.assertTrue(destination_path.is_file())
            self.assertTrue(blocker.is_file())
            self.assertFalse(source_path.is_file())

    def test_file_move_failure_leaves_metadata_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            destination_group = create_condition_group(workspace, display_name="Target")
            sample = self._import_leaf(workspace)
            original_group = sample.condition_group_id
            original_path = sample.raw_image_path

            with patch("neopest.project_manager.shutil.move", side_effect=OSError("move failed")):
                with self.assertRaises(WorkspaceError):
                    move_sample_to_condition_group(
                        workspace,
                        sample.sample_id,
                        destination_group.condition_group_id,
                    )

            persisted = workspace.get_sample(sample.sample_id)
            assert persisted is not None
            self.assertEqual(persisted.condition_group_id, original_group)
            self.assertEqual(persisted.raw_image_path, original_path)

    def test_metadata_save_failure_attempts_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            destination_group = create_condition_group(workspace, display_name="Target")
            sample = self._import_leaf(workspace)
            source_path = sample.raw_image_absolute(workspace.root_path)
            original_group = sample.condition_group_id
            original_path = sample.raw_image_path

            with patch(
                "neopest.project_manager.save_workspace",
                side_effect=OSError("save failed"),
            ):
                with self.assertRaises(WorkspaceError):
                    move_sample_to_condition_group(
                        workspace,
                        sample.sample_id,
                        destination_group.condition_group_id,
                    )

            persisted = workspace.get_sample(sample.sample_id)
            assert persisted is not None
            self.assertEqual(persisted.condition_group_id, original_group)
            self.assertEqual(persisted.raw_image_path, original_path)
            self.assertTrue(source_path.is_file())


if __name__ == "__main__":
    unittest.main()
