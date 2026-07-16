"""Phase 5 stabilization tests for display, UI, and transactional behavior."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
from neopest.gui import MainWindow
from neopest.gui_layout_builders import VIEW_MODE_OPTIONS
from neopest.image_io import load_image_rgb_array
from neopest.measurements import (
    CALIBRATION_REQUIRED_DISPLAY,
    CALCULATION_FAILED_DISPLAY,
    CURRENT_RESULT_DISPLAY,
    DISPLAY_AREA_DECIMAL_PLACES,
    DISPLAY_COVERAGE_DECIMAL_PLACES,
    NOT_CALCULATED_DISPLAY,
    RECALCULATION_REQUIRED_DISPLAY,
    build_leaf_measurement_record,
    calculate_leaf_area,
    calculation_records_are_consistent,
    format_area_in2_display,
    format_coverage_percent_display,
    phase5_measurement_row_placeholder,
    phase5_result_status_display,
)
from neopest.metadata import WORKSPACE_JSON_FILENAME, CandidateDiseaseMeasurementRecord, LeafMeasurementRecord
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
    save_sample_calculation_results,
    save_sample_calibration,
    save_sample_leaf_measurement,
)
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
            with patch("neopest.gui.QMessageBox.critical"):
                return MainWindow()


class TestDisplayFormatting(unittest.TestCase):
    def test_area_display_uses_three_decimal_places(self) -> None:
        self.assertEqual(format_area_in2_display(1.68), "1.680 in²")
        self.assertEqual(format_area_in2_display(0.012), "0.012 in²")
        self.assertEqual(format_area_in2_display(0.0), "0.000 in²")

    def test_coverage_display_uses_three_decimal_places(self) -> None:
        self.assertEqual(format_coverage_percent_display(0.714), "0.714%")
        self.assertEqual(format_coverage_percent_display(0.0), "0.000%")

    def test_formatting_does_not_mutate_input_value(self) -> None:
        value = 1.23456789
        original = value
        format_area_in2_display(value)
        format_coverage_percent_display(value)
        self.assertEqual(value, original)

    def test_display_constants_match_decimal_policy(self) -> None:
        self.assertEqual(DISPLAY_AREA_DECIMAL_PLACES, 3)
        self.assertEqual(DISPLAY_COVERAGE_DECIMAL_PLACES, 3)

    def test_none_values_show_not_calculated(self) -> None:
        self.assertEqual(format_area_in2_display(None), NOT_CALCULATED_DISPLAY)
        self.assertEqual(format_coverage_percent_display(None), NOT_CALCULATED_DISPLAY)


class TestMeasurementRecordConsistency(unittest.TestCase):
    def test_explicit_calculation_records_remain_aligned(self) -> None:
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        output = run_explicit_calculation(
            _make_synthetic_leaf_image(),
            calibration,
            source_image_hash="sha256:consistency",
        )
        self.assertTrue(
            calculation_records_are_consistent(
                output.leaf_measurement,
                output.candidate_disease_measurement,
            )
        )

    def test_mismatched_timestamps_fail_consistency_check(self) -> None:
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        output = run_explicit_calculation(
            _make_synthetic_leaf_image(),
            calibration,
            source_image_hash="sha256:consistency",
        )
        diverged = CandidateDiseaseMeasurementRecord(
            status="valid",
            total_leaf_pixels=output.candidate_disease_measurement.total_leaf_pixels,
            total_leaf_area_mm2=output.candidate_disease_measurement.total_leaf_area_mm2,
            total_leaf_area_in2=output.candidate_disease_measurement.total_leaf_area_in2,
            pixels_per_mm_used=output.candidate_disease_measurement.pixels_per_mm_used,
            source_image_hash=output.candidate_disease_measurement.source_image_hash,
            leaf_mask_method=output.candidate_disease_measurement.leaf_mask_method,
            leaf_mask_settings_version=output.candidate_disease_measurement.leaf_mask_settings_version,
            calculated_at="different-timestamp",
        )
        self.assertFalse(
            calculation_records_are_consistent(
                output.leaf_measurement,
                diverged,
            )
        )


class TestTransactionalSaveBehavior(unittest.TestCase):
    def test_save_failure_restores_both_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Txn")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)
            previous_leaf = LeafMeasurementRecord(status="valid", leaf_area_in2=1.0)
            previous_candidate = CandidateDiseaseMeasurementRecord(
                status="valid",
                total_leaf_area_in2=1.0,
            )
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            sample.leaf_measurement = previous_leaf
            sample.candidate_disease_measurement = previous_candidate
            output = run_explicit_calculation(
                _make_synthetic_leaf_image(),
                calibration,
                source_image_hash=imported[0].image_hash,
            )
            before_bytes = (root / WORKSPACE_JSON_FILENAME).read_bytes()
            with patch("neopest.project_manager.save_workspace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_sample_calculation_results(
                        workspace,
                        sample_id,
                        output.leaf_measurement,
                        output.candidate_disease_measurement,
                    )
            after_bytes = (root / WORKSPACE_JSON_FILENAME).read_bytes()
            self.assertEqual(before_bytes, after_bytes)
            self.assertIs(sample.leaf_measurement, previous_leaf)
            self.assertIs(sample.candidate_disease_measurement, previous_candidate)

    def test_gui_refresh_after_save_failure_shows_prior_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="GuiTxn")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()

            sample = workspace.get_sample(sample_id)
            assert sample is not None
            prior_text = window.inspector_panel.total_leaf_area_label.text()
            self.assertEqual(window.inspector_panel.result_status_value.text(), CURRENT_RESULT_DISPLAY)

            with patch(
                "neopest.gui.save_sample_calculation_results",
                side_effect=OSError("disk full"),
            ):
                with patch("neopest.gui.QMessageBox.critical"):
                    window._on_calculate_results()

            self.assertEqual(window.inspector_panel.total_leaf_area_label.text(), prior_text)
            self.assertEqual(window.inspector_panel.result_status_value.text(), CURRENT_RESULT_DISPLAY)


class TestPhase5InspectorDisplayStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_sample(self, *, calibrated: bool = True) -> tuple[object, str, TemporaryDirectory]:
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="States")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        sample_id = imported[0].sample_id
        if calibrated:
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, sample_id, calibration)
        return workspace, sample_id, tmp

    def test_current_state_shows_formatted_values(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), CURRENT_RESULT_DISPLAY)
            self.assertRegex(panel.total_leaf_area_label.text(), r"^\d+\.\d{3} in²$")
            self.assertRegex(panel.candidate_disease_coverage_label.text(), r"^\d+\.\d{3}%$")
        finally:
            tmp.cleanup()

    def test_not_calculated_state(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), NOT_CALCULATED_DISPLAY)
            self.assertEqual(panel.total_leaf_area_label.text(), NOT_CALCULATED_DISPLAY)
        finally:
            tmp.cleanup()

    def test_calibration_required_state(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample(calibrated=False)
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), CALIBRATION_REQUIRED_DISPLAY)
            self.assertEqual(panel.total_leaf_area_label.text(), CALIBRATION_REQUIRED_DISPLAY)
            self.assertFalse(panel.calculate_results_button.isEnabled())
        finally:
            tmp.cleanup()

    def test_stale_state_hides_numeric_values(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            prior_area_text = window.inspector_panel.total_leaf_area_label.text()
            calibration_b = build_one_inch_manual_calibration_record((0.0, 0.0), (60.0, 0.0))
            sample = save_sample_calibration(workspace, sample_id, calibration_b)
            window._refresh_sample_inspector(sample)
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), RECALCULATION_REQUIRED_DISPLAY)
            self.assertEqual(panel.total_leaf_area_label.text(), RECALCULATION_REQUIRED_DISPLAY)
            self.assertNotEqual(panel.total_leaf_area_label.text(), prior_area_text)
            self.assertNotIn("in²", panel.total_leaf_area_label.text())
        finally:
            tmp.cleanup()

    def test_calculation_failed_state_without_valid_record(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            with patch(
                "neopest.gui.run_explicit_calculation",
                side_effect=CalculationError("Whole-leaf segmentation failed."),
            ):
                window._on_calculate_results()
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), CALCULATION_FAILED_DISPLAY)
            self.assertEqual(panel.total_leaf_area_label.text(), CALCULATION_FAILED_DISPLAY)
        finally:
            tmp.cleanup()

    def test_legacy_leaf_only_record_shows_not_calculated(self) -> None:
        workspace, sample_id, tmp = self._workspace_with_sample()
        try:
            preview = segment_leaf_preview(_make_synthetic_leaf_image())
            assert preview.mask is not None
            sample = workspace.get_sample(sample_id)
            assert sample is not None and sample.calibration is not None
            area = calculate_leaf_area(preview.mask, sample.calibration.pixels_per_mm)
            record = build_leaf_measurement_record(
                area,
                method=PREVIEW_METHOD,
                source_image_hash=sample.image_hash,
            )
            save_sample_leaf_measurement(workspace, sample_id, record)
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            panel = window.inspector_panel
            self.assertEqual(panel.result_status_value.text(), NOT_CALCULATED_DISPLAY)
            self.assertEqual(panel.total_leaf_area_label.text(), NOT_CALCULATED_DISPLAY)
        finally:
            tmp.cleanup()


class TestViewDropdownStabilization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_view_dropdown_has_four_options(self) -> None:
        window = _main_window_without_startup_default()
        combo = window.inspector_panel.view_mode_combo
        self.assertEqual(combo.count(), 4)
        labels = [combo.itemText(index) for index in range(combo.count())]
        keys = [combo.itemData(index) for index in range(combo.count())]
        self.assertEqual(labels, [label for label, _key in VIEW_MODE_OPTIONS])
        self.assertEqual(keys, [key for _label, key in VIEW_MODE_OPTIONS])

    def test_each_view_option_updates_display_transiently(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="View")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            panel = window.inspector_panel

            panel.set_view_mode_key("raw_image")
            window._on_view_mode_changed()
            self.assertIsNone(window.image_canvas._leaf_mask_overlay)

            panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

            panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

            panel.set_view_mode_key("healthy_tissue")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

    def test_view_changes_do_not_persist_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="ViewPersist")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            metadata_path = root / WORKSPACE_JSON_FILENAME
            before_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            for mode in ("whole_leaf", "disease", "healthy_tissue", "raw_image"):
                window.inspector_panel.set_view_mode_key(mode)
                window._on_view_mode_changed()

            after_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            self.assertEqual(before_hash, after_hash)

    def test_view_changes_do_not_invoke_calculation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="ViewCalc")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
            save_sample_calibration(workspace, imported[0].sample_id, calibration)

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(imported[0].sample_id)
            with patch.object(window, "_on_calculate_results") as calculate:
                for mode in ("whole_leaf", "disease", "healthy_tissue", "raw_image"):
                    window.inspector_panel.set_view_mode_key(mode)
                    window._on_view_mode_changed()
            calculate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
