"""Tests for display-only disease view mode GUI wiring without a visible window."""

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

from neopest.calibration import build_one_inch_manual_calibration_record
from neopest.disease_segmentation import DiseaseMaskPreviewStatus
from neopest.gui import MainWindow
from neopest.measurements import PREVIEW_METHOD, build_leaf_measurement_record, calculate_leaf_area
from neopest.metadata import WORKSPACE_JSON_FILENAME, LeafMeasurementRecord
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
    save_sample_calibration,
    save_sample_leaf_measurement,
)
from neopest.segmentation import LeafMaskPreview, LeafMaskPreviewStatus, segment_leaf_preview
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


class TestGuiDiseaseViewMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_sample(self) -> tuple[Path, object, str]:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="Disease Preview")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        return root, workspace, imported[0].sample_id

    def test_view_dropdown_enabled_for_sample_with_image(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        self.assertTrue(window.inspector_panel.view_mode_combo.isEnabled())

    def test_disease_view_generates_whole_leaf_when_absent(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()

        self.assertIn(sample_id, window._leaf_mask_previews)
        self.assertIn(sample_id, window._disease_mask_previews)
        disease = window._disease_mask_previews[sample_id]
        self.assertEqual(disease.status, DiseaseMaskPreviewStatus.READY)

    def test_disease_view_reuses_existing_whole_leaf_preview(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("whole_leaf")
        window._on_view_mode_changed()
        leaf_before = window._leaf_mask_previews[sample_id]
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()
        leaf_after = window._leaf_mask_previews[sample_id]
        self.assertIs(leaf_before, leaf_after)

    def test_disease_view_updates_overlay(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()
        self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

    def test_disease_view_does_not_change_analysis_status(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()

        sample = workspace.get_sample(sample_id)
        assert sample is not None
        self.assertEqual(sample.analysis_status.value, "not_run")

    def test_disease_view_does_not_persist_to_workspace_json(self) -> None:
        root, workspace, sample_id = self._workspace_with_sample()
        metadata_path = root / WORKSPACE_JSON_FILENAME
        before = json.loads(metadata_path.read_text(encoding="utf-8"))

        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()

        after = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(before, after)
        self.assertNotIn("disease_mask", json.dumps(after))

    def test_disease_view_does_not_change_persisted_leaf_measurement(self) -> None:
        root, workspace, sample_id = self._workspace_with_sample()
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        save_sample_calibration(workspace, sample_id, calibration)
        preview = segment_leaf_preview(_make_synthetic_leaf_image())
        assert preview.mask is not None
        area = calculate_leaf_area(preview.mask, calibration.pixels_per_mm)
        record = build_leaf_measurement_record(
            area,
            method=PREVIEW_METHOD,
            source_image_hash=workspace.get_sample(sample_id).image_hash,  # type: ignore[union-attr]
        )
        save_sample_leaf_measurement(workspace, sample_id, record)
        before_bytes = (root / WORKSPACE_JSON_FILENAME).read_bytes()

        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()

        after_bytes = (root / WORKSPACE_JSON_FILENAME).read_bytes()
        self.assertEqual(before_bytes, after_bytes)

    def test_switching_samples_clears_stale_disease_overlay(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Switch")
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
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

            window._on_sample_selected(sample_b.sample_id)
            self.assertIsNone(window.image_canvas._leaf_mask_overlay)

    def test_disease_view_failure_shows_warning(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        with patch(
            "neopest.gui.segment_leaf_preview",
            return_value=LeafMaskPreview(
                mask=None,
                status=LeafMaskPreviewStatus.FAILED,
                warnings=["Synthetic leaf failure."],
            ),
        ):
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
        disease = window._disease_mask_previews[sample_id]
        self.assertEqual(disease.status, DiseaseMaskPreviewStatus.FAILED)
        self.assertIn(
            "Whole-leaf preview is required.",
            window.inspector_panel.warnings_label.text(),
        )

    def test_view_mode_disease_shows_overlay_after_preview(self) -> None:
        _, workspace, sample_id = self._workspace_with_sample()
        window = _main_window_without_startup_default()
        window._set_workspace(workspace)
        window._on_sample_selected(sample_id)
        window.inspector_panel.set_view_mode_key("disease")
        window._on_view_mode_changed()
        self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

    def test_calibration_line_draws_above_overlay(self) -> None:
        import neopest.gui_canvas as gui_canvas

        source = inspect.getsource(gui_canvas.ImageCanvas.paintEvent)
        calibration_index = source.index("_draw_calibration_overlay")
        overlay_index = source.index("_leaf_mask_overlay")
        self.assertGreater(calibration_index, overlay_index)

    def test_run_analysis_button_stays_disabled_and_hidden(self) -> None:
        window = _main_window_without_startup_default()
        self.assertFalse(window.run_analysis_button.isEnabled())
        self.assertFalse(window.run_analysis_button.isVisible())


class TestDiseaseViewLeafMeasurementNonPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _workspace_with_calibrated_sample(
        self,
        *,
        leaf_measurement: LeafMeasurementRecord | None = None,
    ) -> tuple[Path, object, str, TemporaryDirectory]:
        tmp = TemporaryDirectory()
        root = Path(tmp.name) / "ws"
        workspace = initialize_workspace(root, display_name="Non Persist")
        image_path = Path(tmp.name) / "leaf.png"
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
        )
        imported = import_images(workspace, [image_path])
        sample_id = imported[0].sample_id
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        save_sample_calibration(workspace, sample_id, calibration)
        if leaf_measurement is not None:
            save_sample_leaf_measurement(workspace, sample_id, leaf_measurement)
        return root, workspace, sample_id, tmp

    def test_disease_view_generates_transient_whole_leaf_mask(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            self.assertNotIn(sample_id, window._leaf_mask_previews)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            preview = window._leaf_mask_previews[sample_id]
            self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
            self.assertIsNotNone(preview.mask)
        finally:
            tmp.cleanup()

    def test_disease_view_does_not_create_leaf_measurement(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            self.assertIsNone(sample.leaf_measurement)
        finally:
            tmp.cleanup()

    def test_disease_view_does_not_refresh_stale_leaf_measurement(self) -> None:
        stale = LeafMeasurementRecord(
            status="stale",
            leaf_area_pixels=1000,
            leaf_area_in2=1.11,
            leaf_area_mm2=700.0,
            pixels_per_mm_used=9.0,
            pixels_per_inch_used=228.6,
            leaf_mask_method=PREVIEW_METHOD,
            leaf_mask_settings_version=2,
            stale_reason="legacy_inherited_calibration_removed",
        )
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample(
            leaf_measurement=stale,
        )
        try:
            before = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())["samples"][0][
                "leaf_measurement"
            ]
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            after = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())["samples"][0][
                "leaf_measurement"
            ]
            self.assertEqual(after, before)
            self.assertEqual(after["status"], "stale")
        finally:
            tmp.cleanup()

    def test_disease_view_does_not_modify_valid_leaf_measurement(self) -> None:
        preview = segment_leaf_preview(_make_synthetic_leaf_image())
        assert preview.mask is not None
        calibration = build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0))
        area = calculate_leaf_area(preview.mask, calibration.pixels_per_mm)
        valid = build_leaf_measurement_record(
            area,
            method=PREVIEW_METHOD,
            source_image_hash="sha256:test",
        )
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample(
            leaf_measurement=valid,
        )
        try:
            before = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())["samples"][0][
                "leaf_measurement"
            ]
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            after = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())["samples"][0][
                "leaf_measurement"
            ]
            self.assertEqual(after, before)
            self.assertEqual(after["status"], "valid")
        finally:
            tmp.cleanup()

    def test_disease_view_does_not_call_workspace_persistence(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            with patch("neopest.gui.save_sample_calculation_results") as save_results:
                with patch("neopest.project_manager.save_workspace") as save_workspace:
                    window.inspector_panel.set_view_mode_key("disease")
                    window._on_view_mode_changed()
            save_results.assert_not_called()
            save_workspace.assert_not_called()
        finally:
            tmp.cleanup()

    def test_workspace_json_byte_for_byte_unchanged_after_disease_view(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample(
            leaf_measurement=LeafMeasurementRecord(
                status="stale",
                leaf_area_in2=0.91,
                stale_reason="legacy_inherited_calibration_removed",
            ),
        )
        try:
            metadata_path = root / WORKSPACE_JSON_FILENAME
            before_bytes = metadata_path.read_bytes()
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            after_bytes = metadata_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)
        finally:
            tmp.cleanup()

    def test_healthy_tissue_view_mode_does_not_persist_measurements(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            metadata_path = root / WORKSPACE_JSON_FILENAME
            before_bytes = metadata_path.read_bytes()
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("healthy_tissue")
            window._on_view_mode_changed()
            after_bytes = metadata_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)
        finally:
            tmp.cleanup()

    def test_view_does_not_persist_measurements(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("whole_leaf")
            window._on_view_mode_changed()
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            self.assertIsNone(sample.leaf_measurement)
            payload = json.loads((root / WORKSPACE_JSON_FILENAME).read_text())
            self.assertIsNone(payload["samples"][0]["leaf_measurement"])
        finally:
            tmp.cleanup()

    def test_disease_masks_remain_in_memory_only(self) -> None:
        root, workspace, sample_id, tmp = self._workspace_with_calibrated_sample()
        try:
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            disease = window._disease_mask_previews[sample_id]
            self.assertEqual(disease.status, DiseaseMaskPreviewStatus.READY)
            self.assertNotIn("disease_mask", (root / WORKSPACE_JSON_FILENAME).read_text())
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
