"""Tests for leaf measurement persistence and invalidation."""

from __future__ import annotations

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

from neopest.calibration import build_one_inch_manual_calibration_record
from neopest.gui import MainWindow
from neopest.measurements import (
    CURRENT_LEAF_MASK_METHOD,
    CURRENT_LEAF_MASK_SETTINGS_VERSION,
    build_leaf_measurement_record,
    calculate_leaf_area,
    leaf_measurement_matches_provenance,
)
from neopest.metadata import WORKSPACE_JSON_FILENAME, LeafMeasurementRecord
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
    load_workspace,
    mark_sample_leaf_measurement_stale,
    save_sample_calibration,
    save_sample_leaf_measurement,
)
from neopest.results import AnalysisStatus
from neopest.segmentation import PREVIEW_METHOD, segment_leaf_preview
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


class TestLeafMeasurementPersistence(unittest.TestCase):
    def test_measurement_persists_in_workspace_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Persist")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample.sample_id, calibration)

            preview = segment_leaf_preview(_make_synthetic_leaf_image())
            assert preview.mask is not None
            area = calculate_leaf_area(preview.mask, calibration.pixels_per_mm)
            record = build_leaf_measurement_record(
                area,
                method=PREVIEW_METHOD,
                settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
                source_image_hash=sample.image_hash,
            )
            save_sample_leaf_measurement(workspace, sample.sample_id, record)

            payload = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())
            measurement = payload["samples"][0]["leaf_measurement"]
            self.assertEqual(measurement["status"], "valid")
            self.assertIsInstance(measurement["leaf_area_in2"], float)
            self.assertIsInstance(measurement["leaf_area_pixels"], int)
            self.assertNotIn("in²", json.dumps(measurement))

    def test_workspace_without_measurement_loads_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Legacy")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(np.full((40, 40, 3), 128, dtype=np.uint8), cv2.COLOR_RGB2BGR),
            )
            import_images(workspace, [image_path])

            loaded = load_workspace(root)
            sample = loaded.samples[0]
            self.assertIsNone(sample.leaf_measurement)

    def test_calibration_change_marks_measurement_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Stale")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]

            calibration_a = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample.sample_id, calibration_a)
            preview = segment_leaf_preview(_make_synthetic_leaf_image())
            assert preview.mask is not None
            area = calculate_leaf_area(preview.mask, calibration_a.pixels_per_mm)
            record = build_leaf_measurement_record(
                area,
                source_image_hash=sample.image_hash,
            )
            save_sample_leaf_measurement(workspace, sample.sample_id, record)

            calibration_b = build_one_inch_manual_calibration_record((0.0, 0.0), (60.0, 0.0))
            updated = save_sample_calibration(workspace, sample.sample_id, calibration_b)
            self.assertIsNotNone(updated.leaf_measurement)
            assert updated.leaf_measurement is not None
            self.assertEqual(updated.leaf_measurement.status, "stale")
            self.assertEqual(updated.leaf_measurement.stale_reason, "calibration_changed")

    def test_provenance_mismatch_detected(self) -> None:
        record = LeafMeasurementRecord(
            status="valid",
            leaf_area_pixels=100,
            leaf_area_in2=1.0,
            leaf_area_mm2=10.0,
            pixels_per_mm_used=5.0,
            pixels_per_inch_used=127.0,
            leaf_mask_method=CURRENT_LEAF_MASK_METHOD,
            leaf_mask_settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
            source_image_hash="abc",
        )
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        self.assertFalse(
            leaf_measurement_matches_provenance(
                record,
                calibration=calibration,
                image_hash="different",
            )
        )

    def test_mark_stale_helper(self) -> None:
        record = LeafMeasurementRecord(status="valid", leaf_area_in2=2.5)
        sample_record = type("Sample", (), {"leaf_measurement": record})()
        mark_sample_leaf_measurement_stale(sample_record, "image_changed")  # type: ignore[arg-type]
        self.assertEqual(sample_record.leaf_measurement.status, "stale")


class TestAutomaticLeafAreaGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_calibrated_leaf(self) -> tuple[Path, object, str, TemporaryDirectory]:
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="Auto Area")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        sample_id = imported[0].sample_id
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        save_sample_calibration(workspace, sample_id, calibration)
        return root, workspace, sample_id, tmp

    def test_successful_preview_with_calibration_calculates_area(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_leaf()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_preview_leaf_mask()

            sample = workspace.get_sample(sample_id)
            assert sample is not None
            self.assertIsNotNone(sample.leaf_measurement)
            assert sample.leaf_measurement is not None
            self.assertEqual(sample.leaf_measurement.status, "valid")
            self.assertIsNotNone(sample.leaf_measurement.leaf_area_in2)
            self.assertIn("in²", window.inspector_panel.total_leaf_area_label.text())
        finally:
            tmp.cleanup()

    def test_successful_preview_without_calibration_does_not_fabricate_area(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="No Cal")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window._on_preview_leaf_mask()

            sample = workspace.get_sample(imported[0].sample_id)
            assert sample is not None
            self.assertIsNone(sample.leaf_measurement)
            self.assertEqual(
                window.inspector_panel.total_leaf_area_label.text(),
                "Calibration required",
            )

    def test_failed_preview_does_not_calculate_area(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Fail")
            image_path = Path(tmp) / "no_blue.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(np.full((120, 120, 3), 245, dtype=np.uint8), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, imported[0].sample_id, calibration)

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            window._on_preview_leaf_mask()

            sample = workspace.get_sample(imported[0].sample_id)
            assert sample is not None
            self.assertIsNone(sample.leaf_measurement)

    def test_automatic_measurement_does_not_change_analysis_status(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_leaf()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_preview_leaf_mask()

            sample = workspace.get_sample(sample_id)
            assert sample is not None
            self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)
        finally:
            tmp.cleanup()

    def test_workspace_reopen_restores_displayed_measurement(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_leaf()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_preview_leaf_mask()
            expected_text = window.inspector_panel.total_leaf_area_label.text()

            loaded = load_workspace(root)
            window._set_workspace(loaded)
            window._on_sample_selected(sample_id)
            self.assertEqual(
                window.inspector_panel.total_leaf_area_label.text(),
                expected_text,
            )
        finally:
            tmp.cleanup()

    def test_calibration_change_clears_current_area_display(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_leaf()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_preview_leaf_mask()
            self.assertIn("in²", window.inspector_panel.total_leaf_area_label.text())

            calibration_b = build_one_inch_manual_calibration_record((0.0, 0.0), (60.0, 0.0))
            sample = save_sample_calibration(workspace, sample_id, calibration_b)
            window._apply_sample_calibration_display(sample)
            self.assertNotIn("in²", window.inspector_panel.total_leaf_area_label.text())
            self.assertIn(
                window.inspector_panel.total_leaf_area_label.text(),
                {"Recalculate leaf mask", "Not calculated"},
            )
        finally:
            tmp.cleanup()


class TestInspectorUiCleanup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_inspector_does_not_show_internal_ids_or_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="UI")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample = imported[0]

            window = _main_window_without_startup_default()
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
            self.assertNotIn(sample.condition_group_id, visible_text)
            self.assertNotIn(sample.raw_image_path, visible_text)
            self.assertNotIn(sample.image_hash or "", visible_text)

    def test_run_analysis_stays_disabled_and_hidden_in_inspector(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(window.run_analysis_button.isEnabled())
        self.assertFalse(window.run_analysis_button.isVisible())

    def test_status_bar_uses_neutral_styling(self) -> None:
        from neopest.gui_styles import COLOR_STATUS_BAR_BACKGROUND, build_application_stylesheet

        stylesheet = build_application_stylesheet()
        self.assertIn(COLOR_STATUS_BAR_BACKGROUND, stylesheet)
        self.assertNotIn("#007acc", stylesheet)


if __name__ == "__main__":
    unittest.main()
