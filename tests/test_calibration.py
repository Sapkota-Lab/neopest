"""Tests for manual calibration math and persistence."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neopest.analysis import placeholder_analysis_result, run_leaf_analysis
from neopest.calibration import (
    CalibrationError,
    KNOWN_ONE_INCH_DISTANCE_UNIT,
    KNOWN_ONE_INCH_DISTANCE_VALUE,
    ONE_INCH_MM,
    build_manual_calibration_record,
    build_one_inch_manual_calibration_record,
    calculate_pixels_per_mm,
    convert_distance_to_mm,
    pixel_distance,
)
from neopest.metadata import (
    DISTANCE_UNIT_IN,
    DISTANCE_UNIT_MM,
    WORKSPACE_JSON_FILENAME,
    CalibrationRecord,
)
from neopest.project_manager import (
    import_images,
    initialize_workspace,
    load_workspace,
    save_sample_calibration,
)
from neopest.results import AnalysisStatus, CalibrationStatus


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


def _write_test_png(path: Path, *, width: int = 100, height: int = 80) -> None:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("green"))
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"Failed to write test image: {path}")


class TestCalibrationMath(unittest.TestCase):
    def test_one_inch_equals_25_4_mm(self) -> None:
        self.assertAlmostEqual(ONE_INCH_MM, 25.4)
        self.assertAlmostEqual(convert_distance_to_mm(1.0, DISTANCE_UNIT_IN), 25.4)

    def test_build_one_inch_record_uses_fixed_distance(self) -> None:
        record = build_one_inch_manual_calibration_record((0.0, 0.0), (254.0, 0.0))
        self.assertEqual(record.known_distance_value, KNOWN_ONE_INCH_DISTANCE_VALUE)
        self.assertEqual(record.known_distance_unit, KNOWN_ONE_INCH_DISTANCE_UNIT)
        self.assertAlmostEqual(record.known_distance_mm, ONE_INCH_MM)
        self.assertAlmostEqual(record.pixels_per_mm, 10.0)

    def test_pixels_per_mm_from_one_inch_points(self) -> None:
        pixels_per_mm = calculate_pixels_per_mm((0.0, 0.0), (100.0, 0.0), ONE_INCH_MM)
        self.assertAlmostEqual(pixels_per_mm, 100.0 / 25.4)

    def test_calculate_pixels_per_mm_with_known_distance(self) -> None:
        pixels_per_mm = calculate_pixels_per_mm((0.0, 0.0), (100.0, 0.0), 10.0)
        self.assertAlmostEqual(pixels_per_mm, 10.0)

    def test_pixel_distance_matches_hypot(self) -> None:
        self.assertAlmostEqual(pixel_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_zero_known_distance_raises(self) -> None:
        with self.assertRaises(CalibrationError):
            calculate_pixels_per_mm((0.0, 0.0), (10.0, 0.0), 0.0)

    def test_identical_points_raise(self) -> None:
        with self.assertRaises(CalibrationError):
            build_one_inch_manual_calibration_record((12.0, 8.0), (12.0, 8.0))


class TestCalibrationPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_one_inch_record_round_trip_dict(self) -> None:
        record = build_one_inch_manual_calibration_record((10.0, 20.0), (110.0, 20.0))
        restored = CalibrationRecord.from_dict(record.to_dict())
        self.assertAlmostEqual(restored.known_distance_mm, 25.4)
        self.assertEqual(restored.known_distance_value, 1.0)
        self.assertEqual(restored.known_distance_unit, DISTANCE_UNIT_IN)
        self.assertEqual(restored.entered_distance_label(), "1 in")

    def test_legacy_metadata_with_only_known_distance_mm_loads(self) -> None:
        legacy = {
            "calibration_method": "manual",
            "pixels_per_mm": 8.0,
            "known_distance_mm": 5.0,
            "point_1_x": 0.0,
            "point_1_y": 0.0,
            "point_2_x": 40.0,
            "point_2_y": 0.0,
            "calibrated_at": "2026-01-01T00:00:00+00:00",
            "notes": "",
        }
        record = CalibrationRecord.from_dict(legacy)
        self.assertEqual(record.known_distance_value, 5.0)
        self.assertEqual(record.known_distance_unit, DISTANCE_UNIT_MM)
        self.assertEqual(record.entered_distance_label(), "5 mm")

    def test_saved_one_inch_calibration_persists_through_workspace_save_load(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            sample_id = imported[0].sample_id

            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)

            loaded = load_workspace(workspace.root_path)
            sample = loaded.get_sample(sample_id)
            self.assertIsNotNone(sample)
            self.assertIsNotNone(sample.calibration)
            self.assertEqual(sample.calibration_status, CalibrationStatus.MANUAL)
            self.assertAlmostEqual(sample.calibration.known_distance_mm, 25.4)
            self.assertEqual(sample.calibration.known_distance_unit, DISTANCE_UNIT_IN)

            payload = json.loads((workspace.root_path / WORKSPACE_JSON_FILENAME).read_text())
            sample_payload = payload["samples"][0]["calibration"]
            self.assertEqual(sample_payload["known_distance_value"], 1.0)
            self.assertEqual(sample_payload["known_distance_unit"], DISTANCE_UNIT_IN)
            self.assertAlmostEqual(sample_payload["known_distance_mm"], 25.4)

    def test_existing_uncalibrated_sample_loads_safely(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])

            loaded = load_workspace(workspace.root_path)
            sample = loaded.get_sample(imported[0].sample_id)
            self.assertIsNotNone(sample)
            self.assertIsNone(sample.calibration)
            self.assertEqual(sample.calibration_status, CalibrationStatus.UNCALIBRATED)

    def test_manual_calibration_does_not_run_analysis(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (20.0, 0.0))
            sample = save_sample_calibration(workspace, imported[0].sample_id, calibration)

            self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)
            self.assertNotEqual(sample.analysis_status, AnalysisStatus.COMPLETE)
            result = placeholder_analysis_result(sample.sample_id)
            self.assertTrue(result.is_placeholder)
            with self.assertRaises(NotImplementedError):
                run_leaf_analysis(
                    sample.sample_id,
                    sample.raw_image_absolute(workspace.root_path),
                )

    def test_save_calibration_preserves_sample_identity_and_image_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source, width=120, height=90)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            original = imported[0]

            calibration = build_one_inch_manual_calibration_record((1.0, 2.0), (51.0, 2.0))
            updated = save_sample_calibration(workspace, original.sample_id, calibration)

            self.assertEqual(updated.sample_id, original.sample_id)
            self.assertEqual(updated.condition_group_id, original.condition_group_id)
            self.assertEqual(updated.raw_image_path, original.raw_image_path)
            self.assertEqual(updated.original_source_path, original.original_source_path)
            self.assertEqual(updated.image_hash, original.image_hash)
            self.assertEqual(updated.width_px, 120)
            self.assertEqual(updated.height_px, 90)
            self.assertEqual(updated.display_name, original.display_name)

    def test_save_calibration_does_not_modify_other_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            source_a = Path(tmp) / "leaf_a.png"
            source_b = Path(tmp) / "leaf_b.png"
            _write_test_png(source_a)
            _write_test_png(source_b)

            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source_a, source_b])
            sample_a_id = imported[0].sample_id
            sample_b_id = imported[1].sample_id
            sample_b_before = workspace.get_sample(sample_b_id)
            self.assertIsNotNone(sample_b_before)

            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (10.0, 0.0))
            save_sample_calibration(workspace, sample_a_id, calibration)

            loaded = load_workspace(workspace.root_path)
            sample_b_after = loaded.get_sample(sample_b_id)
            self.assertIsNotNone(sample_b_after)
            self.assertIsNone(sample_b_after.calibration)
            self.assertEqual(sample_b_after.calibration_status, CalibrationStatus.UNCALIBRATED)
            self.assertEqual(sample_b_after.analysis_status, sample_b_before.analysis_status)
            self.assertEqual(sample_b_after.raw_image_path, sample_b_before.raw_image_path)
            self.assertEqual(sample_b_after.image_hash, sample_b_before.image_hash)

    def test_save_calibration_preserves_existing_analysis_status(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            sample = workspace.get_sample(imported[0].sample_id)
            self.assertIsNotNone(sample)
            sample.analysis_status = AnalysisStatus.COMPLETE

            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (30.0, 0.0))
            updated = save_sample_calibration(workspace, sample.sample_id, calibration)

            self.assertEqual(updated.analysis_status, AnalysisStatus.COMPLETE)


if __name__ == "__main__":
    unittest.main()
