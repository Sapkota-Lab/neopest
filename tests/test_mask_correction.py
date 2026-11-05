"""Tests for manual candidate-disease mask correction."""

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

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from neopest.calibration import CURRENT_PIXEL_GRID, build_one_inch_manual_calibration_record
from neopest.disease_segmentation import DISEASE_SETTINGS_VERSION
from neopest.gui import MainWindow
from neopest.gui_canvas import ImageCanvas
from neopest.mask_correction import (
    ADD_AREA,
    REMOVE_AREA,
    RESEARCHER_CORRECTED_LABEL,
    REVIEW_REQUIRED_LABEL,
    CorrectionHistory,
    MaskStroke,
    apply_corrections,
    correction_display_state,
    interpolate_segment,
    result_status_for_sample,
    stroke_interpolation_spacing,
)
from neopest.metadata import WORKSPACE_JSON_FILENAME, SampleRecord
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
    load_workspace,
    save_sample_calibration,
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
            with patch("neopest.gui.QMessageBox.critical"):
                return MainWindow()


def _leaf_square() -> tuple[np.ndarray, np.ndarray]:
    leaf = np.zeros((20, 20), dtype=bool)
    leaf[2:18, 2:18] = True
    disease = np.zeros((20, 20), dtype=bool)
    disease[8:12, 8:12] = True
    return leaf, disease


class TestMaskCorrectionOperations(unittest.TestCase):
    def test_add_and_remove_are_clipped_to_the_leaf(self) -> None:
        leaf, disease = _leaf_square()
        original = disease.copy()
        added = apply_corrections(
            disease,
            leaf,
            [MaskStroke(ADD_AREA, 3.0, ((0.5, 0.5), (5.5, 5.5)))],
        )
        self.assertFalse(added[0, 0])
        self.assertTrue(added[5, 5])
        self.assertTrue(added[2, 2])
        self.assertTrue(np.array_equal(disease, original))
        self.assertTrue(np.array_equal(leaf, _leaf_square()[0]))

        removed = apply_corrections(
            disease,
            leaf,
            [MaskStroke(REMOVE_AREA, 1.5, ((10.5, 10.5),))],
        )
        self.assertFalse(removed[10, 10])
        self.assertTrue(removed[8, 8])
        self.assertFalse(removed[0, 0])

    def test_most_recent_stroke_wins_where_brushes_overlap(self) -> None:
        leaf, _disease = _leaf_square()
        blank = np.zeros(leaf.shape, dtype=bool)
        add_stroke = MaskStroke(ADD_AREA, 3.0, ((10.5, 10.5),))
        remove_stroke = MaskStroke(REMOVE_AREA, 1.0, ((10.5, 10.5),))

        removed_last = apply_corrections(blank, leaf, [add_stroke, remove_stroke])
        self.assertFalse(removed_last[10, 10])
        self.assertTrue(removed_last[12, 10])

        added_last = apply_corrections(blank, leaf, [remove_stroke, add_stroke])
        self.assertTrue(added_last[10, 10])

    def test_fast_stroke_interpolation_leaves_no_gap(self) -> None:
        start = (10.0, 10.0)
        end = (10.0, 30.0)
        spacing = stroke_interpolation_spacing(2.0)
        filled = interpolate_segment(start, end, spacing)
        self.assertGreater(len(filled), 2)
        leaf = np.ones((40, 40), dtype=bool)
        covered = apply_corrections(
            np.zeros((40, 40), dtype=bool),
            leaf,
            [MaskStroke(ADD_AREA, 2.0, (start, *filled))],
        )
        self.assertTrue(covered[20, 10])
        gapped = apply_corrections(
            np.zeros((40, 40), dtype=bool),
            leaf,
            [MaskStroke(ADD_AREA, 2.0, (start, end))],
        )
        self.assertFalse(gapped[20, 10])

    def test_undo_redo_and_reset_are_one_step_each(self) -> None:
        first = MaskStroke(ADD_AREA, 2.0, ((4.5, 4.5),))
        second = MaskStroke(REMOVE_AREA, 2.0, ((8.5, 8.5),))
        history = CorrectionHistory()
        history.commit(first)
        history.commit(second)
        self.assertTrue(history.undo())
        self.assertEqual(history.strokes, (first,))
        self.assertTrue(history.redo())
        self.assertEqual(history.strokes, (first, second))
        self.assertTrue(history.reset())
        self.assertEqual(history.strokes, ())
        self.assertTrue(history.undo())
        self.assertEqual(history.strokes, (first, second))
        self.assertFalse(CorrectionHistory().reset())


class TestCorrectionCanvasMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_image_coordinates_survive_canvas_resize(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.png"
            image = np.full((120, 160, 3), 40, dtype=np.uint8)
            cv2.imwrite(str(path), image)
            canvas = ImageCanvas()
            canvas.resize(640, 480)
            canvas.show()
            QApplication.processEvents()
            canvas.set_image_path(path)
            canvas.set_correction_mode(True)
            canvas.set_brush(ADD_AREA, 8.0)
            canvas.repaint()
            QApplication.processEvents()
            geometry = canvas._geometry
            self.assertIsNotNone(geometry)
            assert geometry is not None
            image_point = (40.0, 30.0)
            widget_point = geometry.image_to_widget(*image_point)
            radii_before = canvas.brush_footprint_widget_radii()
            self.assertIsNotNone(radii_before)
            assert radii_before is not None
            self.assertAlmostEqual(
                radii_before[0],
                8.0 * geometry.draw_width / geometry.source_width,
            )
            self.assertAlmostEqual(
                radii_before[1],
                8.0 * geometry.draw_height / geometry.source_height,
            )

            finished: list[MaskStroke] = []
            canvas.correction_stroke_finished.connect(finished.append)
            press = QPoint(int(widget_point[0]), int(widget_point[1]))
            release = QPoint(int(widget_point[0]) + 12, int(widget_point[1]))
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, press)
            QTest.mouseMove(canvas, release)
            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                release,
            )
            self.assertEqual(len(finished), 1)
            stored = finished[0]
            self.assertGreaterEqual(len(stored.points), 2)
            self.assertEqual(stored.radius_px, 8.0)

            canvas.resize(960, 700)
            canvas.repaint()
            QApplication.processEvents()
            resized = canvas._geometry
            self.assertIsNotNone(resized)
            assert resized is not None
            radii_after = canvas.brush_footprint_widget_radii()
            self.assertIsNotNone(radii_after)
            assert radii_after is not None
            self.assertNotAlmostEqual(radii_before[0], radii_after[0])
            for point in stored.points:
                mapped = resized.image_to_widget(point[0], point[1])
                restored = resized.widget_to_image(*mapped)
                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertAlmostEqual(restored[0], point[0], places=4)
                self.assertAlmostEqual(restored[1], point[1], places=4)

            leaf = np.ones((120, 160), dtype=bool)
            blank = np.zeros((120, 160), dtype=bool)
            from_small = apply_corrections(blank, leaf, [stored])
            from_large = apply_corrections(blank, leaf, [stored])
            self.assertTrue(np.array_equal(from_small, from_large))
            self.assertGreater(int(np.count_nonzero(from_small)), 0)
            canvas.close()


class TestCorrectionPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_missing_correction_defaults_to_absent(self) -> None:
        sample = SampleRecord.create(
            condition_group_id="condition_group_test",
            display_name="leaf.png",
            raw_image_path="raw_images/leaf.png",
        )
        payload = sample.to_dict()
        self.assertNotIn("disease_mask_correction", payload)
        self.assertIsNone(SampleRecord.from_dict(payload).disease_mask_correction)

    def test_save_reload_keeps_corrections_and_automatic_result(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Correct")
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
            window.show()
            QApplication.processEvents()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            sample = workspace.get_sample(sample_id)
            assert sample is not None and sample.candidate_disease_measurement is not None
            automatic_pixels = sample.candidate_disease_measurement.candidate_disease_pixels
            automatic_version = sample.candidate_disease_measurement.disease_mask_settings_version
            self.assertEqual(automatic_version, DISEASE_SETTINGS_VERSION)
            self.assertTrue(window.inspector_panel.correct_disease_mask_button.isEnabled())
            self.assertEqual(window.inspector_panel.result_status_value.text(), "Current")

            window._on_correct_disease_mask()
            self.assertFalse(window.inspector_panel.set_manual_calibration_button.isVisible())
            self.assertTrue(window.inspector_panel.add_area_button.isVisible())
            self.assertTrue(window.inspector_panel.done_correction_button.isVisible())
            assert window._correction_leaf_mask is not None
            assert window._correction_automatic_mask is not None
            candidates = np.argwhere(
                window._correction_leaf_mask & ~window._correction_automatic_mask
            )
            self.assertGreater(len(candidates), 0)
            y_px, x_px = (int(candidates[0][0]), int(candidates[0][1]))
            stroke = MaskStroke(ADD_AREA, 4.0, ((x_px + 0.5, y_px + 0.5),))
            window._on_correction_stroke_finished(stroke)
            self.assertEqual(window.inspector_panel.result_status_value.text(), RESEARCHER_CORRECTED_LABEL)
            self.assertTrue(window.inspector_panel.undo_correction_button.isEnabled())
            window._on_undo_correction()
            self.assertEqual(window._correction_history.strokes, ())  # type: ignore[union-attr]
            window._on_redo_correction()
            window._on_reset_corrections()
            self.assertEqual(window._correction_history.strokes, ())  # type: ignore[union-attr]
            window._on_undo_correction()
            self.assertEqual(window._correction_history.strokes, (stroke,))  # type: ignore[union-attr]

            window.inspector_panel.set_view_mode_key("raw_image")
            window._on_view_mode_changed()
            self.assertIsNone(window.image_canvas._leaf_mask_overlay)
            self.assertTrue(window.image_canvas._correction_mode)
            window.inspector_panel.set_view_mode_key("disease")
            window._on_view_mode_changed()
            self.assertIsNotNone(window.image_canvas._leaf_mask_overlay)

            window._on_done_correction()
            saved = workspace.get_sample(sample_id)
            assert saved is not None
            assert saved.candidate_disease_measurement is not None
            assert saved.disease_mask_correction is not None
            self.assertEqual(
                saved.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )
            self.assertEqual(
                saved.candidate_disease_measurement.disease_mask_settings_version,
                automatic_version,
            )
            self.assertGreater(
                saved.disease_mask_correction.corrected_candidate_disease_pixels or 0,
                automatic_pixels or 0,
            )
            self.assertEqual(
                saved.disease_mask_correction.result_label,
                RESEARCHER_CORRECTED_LABEL,
            )
            self.assertEqual(
                saved.disease_mask_correction.disease_mask_settings_version,
                automatic_version,
            )
            self.assertEqual(saved.disease_mask_correction.pixel_grid, CURRENT_PIXEL_GRID)
            self.assertEqual(
                window.inspector_panel.result_status_value.text(),
                RESEARCHER_CORRECTED_LABEL,
            )
            self.assertTrue(window.inspector_panel.automatic_result_label.text().startswith("Automatic"))
            payload = json.loads((root / WORKSPACE_JSON_FILENAME).read_text(encoding="utf-8"))
            stored = payload["samples"][0]
            self.assertEqual(
                stored["candidate_disease_measurement"]["candidate_disease_pixels"],
                automatic_pixels,
            )
            self.assertEqual(
                stored["disease_mask_correction"]["result_label"],
                RESEARCHER_CORRECTED_LABEL,
            )
            self.assertNotIn("corrected_candidate_disease_pixels", stored["candidate_disease_measurement"])

            reloaded = load_workspace(root)
            reloaded_sample = reloaded.get_sample(sample_id)
            assert reloaded_sample is not None
            assert reloaded_sample.disease_mask_correction is not None
            assert reloaded_sample.candidate_disease_measurement is not None
            self.assertEqual(
                reloaded_sample.disease_mask_correction.strokes,
                saved.disease_mask_correction.strokes,
            )
            self.assertEqual(
                reloaded_sample.disease_mask_correction.corrected_candidate_disease_pixels,
                saved.disease_mask_correction.corrected_candidate_disease_pixels,
            )
            self.assertEqual(
                reloaded_sample.disease_mask_correction.corrected_candidate_disease_area_in2,
                saved.disease_mask_correction.corrected_candidate_disease_area_in2,
            )
            self.assertEqual(
                reloaded_sample.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )

            window._on_correct_disease_mask()
            window._on_reset_corrections()
            window._on_done_correction()
            cleared = workspace.get_sample(sample_id)
            assert cleared is not None
            self.assertIsNone(cleared.disease_mask_correction)
            self.assertEqual(window.inspector_panel.result_status_value.text(), "Current")
            self.assertEqual(
                cleared.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )
            window.close()

    def test_changed_segmentation_provenance_keeps_strokes_for_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Review")
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
            window._on_correct_disease_mask()
            assert window._correction_leaf_mask is not None
            ys, xs = np.nonzero(window._correction_leaf_mask)
            stroke = MaskStroke(ADD_AREA, 2.0, ((float(xs[0]) + 0.5, float(ys[0]) + 0.5),))
            window._on_correction_stroke_finished(stroke)
            window._on_done_correction()
            sample = workspace.get_sample(sample_id)
            assert sample is not None and sample.disease_mask_correction is not None
            sample.disease_mask_correction.disease_mask_settings_version = (
                DISEASE_SETTINGS_VERSION + 1
            )
            self.assertEqual(
                correction_display_state(sample),
                "review",
            )
            self.assertEqual(sample.disease_mask_correction.strokes, (stroke,))
            self.assertEqual(result_status_for_sample(sample), REVIEW_REQUIRED_LABEL)
            window._refresh_sample_inspector(sample)
            self.assertEqual(
                window.inspector_panel.result_status_value.text(),
                REVIEW_REQUIRED_LABEL,
            )
            self.assertIn(
                "require review",
                window.inspector_panel.warnings_label.text(),
            )
            self.assertNotEqual(
                window.inspector_panel.result_status_value.text(),
                RESEARCHER_CORRECTED_LABEL,
            )
            window.close()

    def test_calibration_scale_change_recalculates_without_dropping_strokes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Scale")
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
            window._on_correct_disease_mask()
            assert window._correction_leaf_mask is not None
            ys, xs = np.nonzero(window._correction_leaf_mask)
            stroke = MaskStroke(REMOVE_AREA, 6.0, ((float(xs[len(xs) // 2]) + 0.5, float(ys[len(ys) // 2]) + 0.5),))
            window._on_correction_stroke_finished(stroke)
            window._on_done_correction()
            sample = workspace.get_sample(sample_id)
            assert sample is not None
            assert sample.disease_mask_correction is not None
            assert sample.candidate_disease_measurement is not None
            before_area = sample.disease_mask_correction.corrected_candidate_disease_area_mm2
            before_pixels = sample.disease_mask_correction.corrected_candidate_disease_pixels
            before_ppm = sample.disease_mask_correction.pixels_per_mm_used
            automatic_pixels = sample.candidate_disease_measurement.candidate_disease_pixels
            assert before_area is not None and before_ppm is not None and before_pixels is not None

            wider = build_one_inch_manual_calibration_record((0.0, 0.0), (200.0, 0.0))
            updated = save_sample_calibration(workspace, sample_id, wider)
            assert updated.disease_mask_correction is not None
            assert updated.candidate_disease_measurement is not None
            self.assertEqual(updated.disease_mask_correction.strokes, (stroke,))
            self.assertEqual(updated.disease_mask_correction.status, "valid")
            self.assertEqual(
                updated.disease_mask_correction.corrected_candidate_disease_pixels,
                before_pixels,
            )
            assert updated.disease_mask_correction.pixels_per_mm_used is not None
            assert updated.disease_mask_correction.corrected_candidate_disease_area_mm2 is not None
            expected_area = before_area * (before_ppm**2) / (
                updated.disease_mask_correction.pixels_per_mm_used**2
            )
            self.assertAlmostEqual(
                updated.disease_mask_correction.corrected_candidate_disease_area_mm2,
                expected_area,
            )
            self.assertEqual(updated.candidate_disease_measurement.status, "stale")
            self.assertEqual(
                updated.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )
            self.assertEqual(
                updated.candidate_disease_measurement.disease_mask_settings_version,
                DISEASE_SETTINGS_VERSION,
            )
            self.assertEqual(correction_display_state(updated), "applied")
            self.assertEqual(result_status_for_sample(updated), RESEARCHER_CORRECTED_LABEL)

            mismatched = build_one_inch_manual_calibration_record((0.0, 0.0), (80.0, 0.0))
            mismatched.pixel_grid = "legacy-grid"
            reviewed = save_sample_calibration(workspace, sample_id, mismatched)
            assert reviewed.disease_mask_correction is not None
            self.assertEqual(reviewed.disease_mask_correction.strokes, (stroke,))
            self.assertEqual(reviewed.disease_mask_correction.status, "review_required")
            self.assertEqual(reviewed.disease_mask_correction.review_reason, "pixel_grid_changed")
            self.assertEqual(
                reviewed.disease_mask_correction.corrected_candidate_disease_pixels,
                before_pixels,
            )
            self.assertEqual(correction_display_state(reviewed), "review")
            window.close()


if __name__ == "__main__":
    unittest.main()
