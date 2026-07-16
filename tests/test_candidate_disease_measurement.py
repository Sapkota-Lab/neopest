"""Tests for Phase 5 candidate-disease measurement math and persistence."""

from __future__ import annotations

import copy
import inspect
import json
import math
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

from neopest.calculation import CalculationError, run_explicit_calculation
from neopest.calibration import build_one_inch_manual_calibration_record
from neopest.disease_segmentation import DISEASE_PREVIEW_METHOD, DISEASE_SETTINGS_VERSION
from neopest.gui import MainWindow
from neopest.measurements import (
    CURRENT_DISEASE_MASK_METHOD,
    CURRENT_DISEASE_MASK_SETTINGS_VERSION,
    CURRENT_LEAF_MASK_METHOD,
    CURRENT_LEAF_MASK_SETTINGS_VERSION,
    MeasurementError,
    build_candidate_disease_measurement_record,
    build_leaf_measurement_record,
    calculate_areas_from_pixel_counts,
    calculate_candidate_disease_measurement,
    calculate_leaf_area,
    candidate_disease_measurement_matches_provenance,
)
from neopest.metadata import (
    MM2_PER_IN2,
    WORKSPACE_JSON_FILENAME,
    CandidateDiseaseMeasurementRecord,
    LeafMeasurementRecord,
)
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
    mark_sample_candidate_disease_measurement_stale,
    mark_sample_measurements_stale,
    move_sample_to_condition_group,
    rename_sample,
    save_sample_calculation_results,
    save_sample_calibration,
    save_sample_leaf_measurement,
)
from neopest.segmentation import PREVIEW_METHOD, SETTINGS_VERSION, segment_leaf_preview
from tests.test_segmentation import _make_synthetic_leaf_image


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


def _main_window_without_startup_default() -> MainWindow:
    with patch("neopest.gui.ensure_default_workspace", side_effect=WorkspaceError("disabled")):
        with patch("neopest.gui.QMessageBox.warning"):
            with patch("neopest.gui.QMessageBox.critical"):
                return MainWindow()


class TestCandidateDiseaseMeasurementMath(unittest.TestCase):
    def test_total_leaf_area_from_pixels(self) -> None:
        result = calculate_areas_from_pixel_counts(400, 40, 10.0)
        self.assertAlmostEqual(result.total_leaf_area_mm2, 400 / 100.0)
        self.assertAlmostEqual(result.total_leaf_area_in2, result.total_leaf_area_mm2 / MM2_PER_IN2)

    def test_candidate_disease_area_from_pixels(self) -> None:
        result = calculate_areas_from_pixel_counts(400, 40, 10.0)
        self.assertAlmostEqual(result.candidate_disease_area_mm2, 40 / 100.0)
        self.assertAlmostEqual(
            result.candidate_disease_area_in2,
            result.candidate_disease_area_mm2 / MM2_PER_IN2,
        )

    def test_candidate_disease_coverage_from_pixel_counts(self) -> None:
        result = calculate_areas_from_pixel_counts(400, 40, 10.0)
        self.assertAlmostEqual(result.candidate_disease_coverage_percent, 10.0)

    def test_non_candidate_area_is_total_minus_candidate(self) -> None:
        result = calculate_areas_from_pixel_counts(400, 40, 10.0)
        self.assertAlmostEqual(
            result.non_candidate_leaf_area_mm2,
            result.total_leaf_area_mm2 - result.candidate_disease_area_mm2,
        )
        self.assertAlmostEqual(
            result.non_candidate_leaf_area_in2,
            result.non_candidate_leaf_area_mm2 / MM2_PER_IN2,
        )

    def test_candidate_pixels_cannot_exceed_leaf_pixels(self) -> None:
        with self.assertRaises(MeasurementError):
            calculate_areas_from_pixel_counts(100, 101, 5.0)

    def test_zero_candidate_pixels_is_valid(self) -> None:
        result = calculate_areas_from_pixel_counts(500, 0, 8.0)
        self.assertEqual(result.candidate_disease_pixels, 0)
        self.assertEqual(result.candidate_disease_area_mm2, 0.0)
        self.assertEqual(result.candidate_disease_coverage_percent, 0.0)

    def test_empty_leaf_mask_fails(self) -> None:
        with self.assertRaises(MeasurementError):
            calculate_areas_from_pixel_counts(0, 0, 5.0)

    def test_invalid_calibration_fails(self) -> None:
        with self.assertRaises(MeasurementError):
            calculate_areas_from_pixel_counts(100, 10, 0.0)

    def test_inputs_are_not_mutated(self) -> None:
        leaf = np.zeros((20, 20), dtype=bool)
        leaf[2:18, 2:18] = True
        disease = np.zeros((20, 20), dtype=bool)
        disease[5:8, 5:8] = True
        leaf_copy = leaf.copy()
        disease_copy = disease.copy()
        calculate_candidate_disease_measurement(leaf, disease, 5.0)
        self.assertTrue(np.array_equal(leaf, leaf_copy))
        self.assertTrue(np.array_equal(disease, disease_copy))

    def test_full_precision_retained_internally(self) -> None:
        result = calculate_areas_from_pixel_counts(3, 1, 7.0)
        self.assertTrue(math.isfinite(result.total_leaf_area_mm2))
        self.assertNotEqual(round(result.total_leaf_area_mm2, 6), round(result.total_leaf_area_mm2, 2))


class TestCandidateDiseaseMeasurementProvenance(unittest.TestCase):
    def test_pixel_counts_and_calibration_persisted(self) -> None:
        measurement = calculate_areas_from_pixel_counts(200, 20, 6.0)
        record = build_candidate_disease_measurement_record(
            measurement,
            source_image_hash="sha256:test",
        )
        self.assertEqual(record.total_leaf_pixels, 200)
        self.assertEqual(record.candidate_disease_pixels, 20)
        self.assertAlmostEqual(record.pixels_per_mm_used, 6.0)

    def test_algorithm_versions_persisted(self) -> None:
        measurement = calculate_areas_from_pixel_counts(200, 20, 6.0)
        record = build_candidate_disease_measurement_record(measurement)
        self.assertEqual(record.leaf_mask_method, PREVIEW_METHOD)
        self.assertEqual(record.leaf_mask_settings_version, SETTINGS_VERSION)
        self.assertEqual(record.disease_mask_method, DISEASE_PREVIEW_METHOD)
        self.assertEqual(record.disease_mask_settings_version, DISEASE_SETTINGS_VERSION)

    def test_explicit_calculation_produces_matching_provenance(self) -> None:
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        output = run_explicit_calculation(
            _make_synthetic_leaf_image(),
            calibration,
            source_image_hash="sha256:phase5",
        )
        self.assertEqual(
            output.leaf_measurement.calculated_at,
            output.candidate_disease_measurement.calculated_at,
        )
        self.assertEqual(
            output.leaf_measurement.source_image_hash,
            output.candidate_disease_measurement.source_image_hash,
        )
        self.assertEqual(
            output.leaf_measurement.leaf_area_pixels,
            output.candidate_disease_measurement.total_leaf_pixels,
        )


class TestExplicitCalculationPersistence(unittest.TestCase):
    def _workspace_with_sample(self) -> tuple[Path, object, str]:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="Calc")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        sample_id = imported[0].sample_id
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        save_sample_calibration(workspace, sample_id, calibration)
        return root, workspace, sample_id

    def test_save_calculation_results_persists_both_records(self) -> None:
        root, workspace, sample_id = self._workspace_with_sample()
        sample = workspace.get_sample(sample_id)
        assert sample is not None
        output = run_explicit_calculation(
            _make_synthetic_leaf_image(),
            sample.calibration,  # type: ignore[arg-type]
            source_image_hash=sample.image_hash,
        )
        save_sample_calculation_results(
            workspace,
            sample_id,
            output.leaf_measurement,
            output.candidate_disease_measurement,
        )
        payload = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())
        sample_payload = payload["samples"][0]
        self.assertEqual(sample_payload["leaf_measurement"]["status"], "valid")
        self.assertEqual(sample_payload["candidate_disease_measurement"]["status"], "valid")
        self.assertEqual(
            sample_payload["leaf_measurement"]["leaf_area_pixels"],
            sample_payload["candidate_disease_measurement"]["total_leaf_pixels"],
        )

    def test_save_failure_rolls_back_in_memory_results(self) -> None:
        root, workspace, sample_id = self._workspace_with_sample()
        sample = workspace.get_sample(sample_id)
        assert sample is not None
        previous_leaf = LeafMeasurementRecord(status="valid", leaf_area_in2=1.0)
        previous_candidate = CandidateDiseaseMeasurementRecord(
            status="valid",
            total_leaf_area_in2=1.0,
        )
        sample.leaf_measurement = previous_leaf
        sample.candidate_disease_measurement = previous_candidate
        output = run_explicit_calculation(
            _make_synthetic_leaf_image(),
            sample.calibration,  # type: ignore[arg-type]
            source_image_hash=sample.image_hash,
        )
        with patch("neopest.project_manager.save_workspace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_sample_calculation_results(
                    workspace,
                    sample_id,
                    output.leaf_measurement,
                    output.candidate_disease_measurement,
                )
        self.assertIs(sample.leaf_measurement, previous_leaf)
        self.assertIs(sample.candidate_disease_measurement, previous_candidate)


class TestMeasurementStaleness(unittest.TestCase):
    def test_calibration_change_marks_candidate_result_stale(self) -> None:
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
            output = run_explicit_calculation(
                _make_synthetic_leaf_image(),
                calibration_a,
                source_image_hash=sample.image_hash,
            )
            save_sample_calculation_results(
                workspace,
                sample.sample_id,
                output.leaf_measurement,
                output.candidate_disease_measurement,
            )
            calibration_b = build_one_inch_manual_calibration_record((0.0, 0.0), (60.0, 0.0))
            updated = save_sample_calibration(workspace, sample.sample_id, calibration_b)
            assert updated.candidate_disease_measurement is not None
            self.assertEqual(updated.candidate_disease_measurement.status, "stale")
            self.assertEqual(updated.candidate_disease_measurement.stale_reason, "calibration_changed")

    def test_sample_rename_does_not_mark_results_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Rename")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)
            output = run_explicit_calculation(
                _make_synthetic_leaf_image(),
                calibration,
                source_image_hash=imported[0].image_hash,
            )
            save_sample_calculation_results(
                workspace,
                sample_id,
                output.leaf_measurement,
                output.candidate_disease_measurement,
            )
            rename_sample(workspace, sample_id, display_name="Renamed Leaf")
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            assert sample.candidate_disease_measurement is not None
            self.assertEqual(sample.candidate_disease_measurement.status, "valid")

    def test_move_between_groups_does_not_mark_results_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Move")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)
            output = run_explicit_calculation(
                _make_synthetic_leaf_image(),
                calibration,
                source_image_hash=imported[0].image_hash,
            )
            save_sample_calculation_results(
                workspace,
                sample_id,
                output.leaf_measurement,
                output.candidate_disease_measurement,
            )
            destination_group_id = workspace.condition_groups[0].condition_group_id
            move_sample_to_condition_group(workspace, sample_id, destination_group_id)
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            assert sample.candidate_disease_measurement is not None
            self.assertEqual(sample.candidate_disease_measurement.status, "valid")

    def test_provenance_mismatch_detected_for_leaf_and_disease_versions(self) -> None:
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        record = CandidateDiseaseMeasurementRecord(
            status="valid",
            pixels_per_mm_used=calibration.pixels_per_mm,
            leaf_mask_method=CURRENT_LEAF_MASK_METHOD,
            leaf_mask_settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
            disease_mask_method=CURRENT_DISEASE_MASK_METHOD,
            disease_mask_settings_version=1,
            source_image_hash="sha256:test",
        )
        self.assertFalse(
            candidate_disease_measurement_matches_provenance(
                record,
                calibration=calibration,
                image_hash="sha256:test",
            )
        )

    def test_mark_stale_helpers(self) -> None:
        record = CandidateDiseaseMeasurementRecord(status="valid", total_leaf_pixels=10)
        sample_record = type(
            "Sample",
            (),
            {"candidate_disease_measurement": record},
        )()
        mark_sample_candidate_disease_measurement_stale(sample_record, "image_changed")  # type: ignore[arg-type]
        self.assertEqual(sample_record.candidate_disease_measurement.status, "stale")

        leaf = LeafMeasurementRecord(status="valid", leaf_area_in2=1.0)
        candidate = CandidateDiseaseMeasurementRecord(status="valid", total_leaf_area_in2=1.0)
        combined = type(
            "Sample",
            (),
            {
                "leaf_measurement": leaf,
                "candidate_disease_measurement": candidate,
            },
        )()
        mark_sample_measurements_stale(combined, "calibration_removed")  # type: ignore[arg-type]
        self.assertEqual(combined.leaf_measurement.status, "stale")
        self.assertEqual(combined.candidate_disease_measurement.status, "stale")


class TestPhase5GuiWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_calibrated_sample(self) -> tuple[Path, object, str, TemporaryDirectory]:
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="GUI")
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

    def test_no_duplicate_preview_buttons_or_hint_labels_remain(self) -> None:
        window = _main_window_without_startup_default()
        panel = window.inspector_panel
        self.assertFalse(hasattr(panel, "preview_whole_leaf_button"))
        self.assertFalse(hasattr(panel, "preview_disease_mask_button"))
        self.assertFalse(hasattr(panel, "calculation_hint_label"))
        self.assertFalse(hasattr(panel, "preview_scope_hint"))
        self.assertTrue(hasattr(panel, "view_mode_combo"))

    def test_view_dropdown_does_not_call_calculation_handler(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            with patch.object(window, "_on_calculate_results") as calculate:
                window.inspector_panel.set_view_mode_key("whole_leaf")
                window._on_view_mode_changed()
                window.inspector_panel.set_view_mode_key("disease")
                window._on_view_mode_changed()
                window.inspector_panel.set_view_mode_key("healthy_tissue")
                window._on_view_mode_changed()
                window.inspector_panel.set_view_mode_key("raw_image")
                window._on_view_mode_changed()
            calculate.assert_not_called()
        finally:
            tmp.cleanup()

    def test_view_modes_do_not_persist_measurements(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            metadata_path = root / WORKSPACE_JSON_FILENAME
            before_bytes = metadata_path.read_bytes()
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            for mode in ("whole_leaf", "disease", "healthy_tissue", "raw_image"):
                window.inspector_panel.set_view_mode_key(mode)
                window._on_view_mode_changed()
            after_bytes = metadata_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            self.assertIsNone(sample.candidate_disease_measurement)
        finally:
            tmp.cleanup()

    def test_explicit_calculation_creates_current_results(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            assert sample.candidate_disease_measurement is not None
            self.assertEqual(sample.candidate_disease_measurement.status, "valid")
            self.assertIn("in²", window.inspector_panel.total_leaf_area_label.text())
            self.assertIn("%", window.inspector_panel.candidate_disease_coverage_label.text())
            self.assertEqual(window.inspector_panel.result_status_value.text(), "Current")
            self.assertEqual(
                window.inspector_panel.calculate_results_button.text(),
                "Results Up to Date",
            )
            self.assertFalse(window.inspector_panel.calculate_results_button.isEnabled())
        finally:
            tmp.cleanup()

    def test_stale_results_enable_recalculate(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            calibration_b = build_one_inch_manual_calibration_record((0.0, 0.0), (60.0, 0.0))
            sample = save_sample_calibration(workspace, sample_id, calibration_b)
            window._refresh_sample_inspector(sample)
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), "Recalculation required")
            self.assertEqual(panel.total_leaf_area_label.text(), "Recalculation required")
            self.assertNotIn("in²", panel.total_leaf_area_label.text())
            self.assertEqual(
                panel.calculate_results_button.text(),
                "Recalculate Results",
            )
            self.assertTrue(panel.calculate_results_button.isEnabled())
        finally:
            tmp.cleanup()

    def test_missing_calibration_disables_calculation_with_explanation(self) -> None:
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
            self.assertFalse(window.inspector_panel.calculate_results_button.isEnabled())
            self.assertEqual(
                window.inspector_panel.result_status_value.text(),
                "Calibration required",
            )
            self.assertEqual(
                window.inspector_panel.total_leaf_area_label.text(),
                "Calibration required",
            )

    def test_failed_calculation_does_not_overwrite_valid_result(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            before = copy.deepcopy(workspace.get_sample(sample_id))
            with patch(
                "neopest.gui.run_explicit_calculation",
                side_effect=CalculationError("Whole-leaf segmentation failed."),
            ):
                window._on_calculate_results()
            after = workspace.get_sample(sample_id)
            assert before is not None and after is not None
            self.assertEqual(
                after.candidate_disease_measurement.to_dict(),
                before.candidate_disease_measurement.to_dict(),  # type: ignore[union-attr]
            )
        finally:
            tmp.cleanup()

    def test_explicit_calculation_regenerates_fresh_masks(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()
            cached_leaf = window._leaf_mask_previews[sample_id]
            with patch("neopest.calculation.segment_leaf_preview") as segment_leaf:
                segment_leaf.return_value = cached_leaf
                window._on_calculate_results()
            segment_leaf.assert_called()
        finally:
            tmp.cleanup()

    def test_scientific_modules_do_not_import_pyside6(self) -> None:
        import neopest.calculation as calculation
        import neopest.measurements as measurements

        for module in (calculation, measurements):
            source = inspect.getsource(module)
            self.assertNotIn("PySide6", source)

    def test_gui_does_not_contain_area_formulas(self) -> None:
        import neopest.gui as gui_module
        import neopest.gui_layout_builders as layout_builders

        for module in (gui_module, layout_builders):
            source = inspect.getsource(module)
            self.assertNotIn("calculate_leaf_area", source)
            self.assertNotIn("calculate_candidate_disease_measurement", source)
            self.assertNotIn("calculate_areas_from_pixel_counts", source)


class TestExplicitCalculationFailures(unittest.TestCase):
    def test_empty_leaf_mask_raises_calculation_error(self) -> None:
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        blank = np.full((80, 80, 3), 245, dtype=np.uint8)
        with self.assertRaises(CalculationError):
            run_explicit_calculation(blank, calibration)


if __name__ == "__main__":
    unittest.main()
