"""Zoom, pan, and calculation-cache checks for mask correction."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from neopest.calculation import run_explicit_calculation
from neopest.calibration import build_one_inch_manual_calibration_record
from neopest.gui import MainWindow
from neopest.gui_canvas import build_image_canvas_host
from neopest.image_io import (
    DISEASE_MASK_OVERLAY_RGBA,
    disease_mask_to_qimage,
    load_image_rgb_array,
    mask_to_qimage,
    write_overlay_region,
)
from neopest.mask_correction import ADD_AREA, MaskStroke, apply_corrections, paint_disks
from neopest.project_manager import (
    WorkspaceError,
    import_images,
    initialize_workspace,
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


def _wheel(canvas, x: float, y: float, notches: int = 1) -> None:
    local = QPointF(x, y)
    event = QWheelEvent(
        local,
        local,
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas, event)


class TestOverlayAndStrokeUpdates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_region_write_matches_a_full_overlay(self) -> None:
        mask = np.zeros((40, 50), dtype=bool)
        original = mask_to_qimage(mask, overlay_rgba=DISEASE_MASK_OVERLAY_RGBA)
        mask[10:18, 12:22] = True
        patched = mask_to_qimage(np.zeros_like(mask), overlay_rgba=DISEASE_MASK_OVERLAY_RGBA)
        write_overlay_region(patched, mask[8:20, 10:24], 10, 8, DISEASE_MASK_OVERLAY_RGBA)
        full = mask_to_qimage(mask, overlay_rgba=DISEASE_MASK_OVERLAY_RGBA)
        for image in (original, patched, full):
            self.assertEqual(image.format(), QImage.Format.Format_RGBA8888)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                self.assertEqual(patched.pixel(x, y), full.pixel(x, y))

    def test_incremental_disks_match_a_replayed_stroke(self) -> None:
        leaf = np.ones((30, 40), dtype=bool)
        leaf[:, :2] = False
        automatic = np.zeros_like(leaf)
        points = ((5.5, 8.5), (9.5, 8.5), (14.5, 12.5))
        replayed = apply_corrections(
            automatic,
            leaf,
            [MaskStroke(ADD_AREA, 3.0, points)],
        )
        working = automatic.copy()
        box = paint_disks(working, leaf, points[:1], 3.0, add=True)
        self.assertIsNotNone(box)
        paint_disks(working, leaf, points[1:], 3.0, add=True)
        self.assertTrue(np.array_equal(working, replayed))
        self.assertFalse(bool(working[8, 0]))
        self.assertEqual(int(automatic.sum()), 0)


class TestCanvasZoomAndPan(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def _canvas(self, tmp: str):
        path = Path(tmp) / "leaf.png"
        image = np.full((140, 180, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(path), image)
        host, canvas = build_image_canvas_host()
        host.resize(640, 480)
        host.show()
        QApplication.processEvents()
        canvas.set_image_path(path)
        canvas.repaint()
        QApplication.processEvents()
        return host, canvas, image

    def test_zoom_stays_anchored_and_fit_and_actual_size_are_explicit(self) -> None:
        with TemporaryDirectory() as tmp:
            host, canvas, _image = self._canvas(tmp)
            geometry = canvas._geometry
            assert geometry is not None
            fitted = canvas.zoom_percent()
            self.assertIsNotNone(fitted)
            anchor_widget = geometry.image_to_widget(70, 50)
            anchor_image = geometry.widget_to_image(*anchor_widget)
            assert anchor_image is not None
            _wheel(canvas, anchor_widget[0], anchor_widget[1], notches=2)
            canvas.repaint()
            QApplication.processEvents()
            zoomed = canvas._geometry
            assert zoomed is not None
            self.assertGreater(canvas.zoom_percent() or 0, fitted or 0)
            remapped = zoomed.image_to_widget(*anchor_image)
            self.assertAlmostEqual(remapped[0], anchor_widget[0], delta=1.0)
            self.assertAlmostEqual(remapped[1], anchor_widget[1], delta=1.0)

            host.findChild(QPushButton, "actualSizeButton").click()
            canvas.repaint()
            QApplication.processEvents()
            actual = canvas._geometry
            assert actual is not None
            self.assertAlmostEqual(actual.draw_width / actual.source_width, 1.0, places=4)
            self.assertEqual(canvas.zoom_percent(), 100)
            self.assertEqual(host.findChild(QLabel, "zoomPercentLabel").text(), "100%")
            canvas.set_brush("add", 8.0)
            radii = canvas.brush_footprint_widget_radii()
            assert radii is not None
            self.assertAlmostEqual(radii[0], 8.0, places=4)
            self.assertAlmostEqual(radii[1], 8.0, places=4)

            canvas.resize(900, 620)
            canvas.repaint()
            QApplication.processEvents()
            self.assertEqual(canvas.zoom_percent(), 100)

            host.findChild(QPushButton, "fitViewButton").click()
            canvas.repaint()
            QApplication.processEvents()
            self.assertEqual(canvas.zoom_percent(), fitted)
            host.close()

    def test_pan_and_zoom_do_not_paint_or_place_calibration_points(self) -> None:
        with TemporaryDirectory() as tmp:
            _host, canvas, _image = self._canvas(tmp)
            strokes: list[MaskStroke] = []
            canvas.correction_stroke_finished.connect(strokes.append)
            canvas.set_correction_mode(True)
            canvas.set_brush("add", 6.0)
            canvas.set_space_pan(True)
            geometry = canvas._geometry
            assert geometry is not None
            origin = geometry.image_to_widget(40, 40)
            press = QPoint(int(origin[0]), int(origin[1]))
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, press)
            QTest.mouseMove(canvas, press + QPoint(24, 10))
            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                press + QPoint(24, 10),
            )
            self.assertEqual(strokes, [])
            moved = canvas._geometry
            assert moved is not None
            self.assertNotAlmostEqual(
                moved.image_to_widget(40, 40)[0],
                origin[0],
                delta=2.0,
            )

            canvas.set_space_pan(False)
            canvas.set_calibration_mode(True)
            canvas.set_space_pan(True)
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, press)
            _wheel(canvas, origin[0], origin[1])
            self.assertEqual(canvas.get_draft_calibration_points(), None)
            self.assertEqual(canvas._draft_points, [])
            canvas.close()

    def test_overlay_and_brush_follow_zoom_and_pan(self) -> None:
        with TemporaryDirectory() as tmp:
            host, canvas, _image = self._canvas(tmp)
            leaf = np.ones((140, 180), dtype=bool)
            blank = np.zeros_like(leaf)
            center = (90.0, 70.0)
            mask = apply_corrections(blank, leaf, [MaskStroke(ADD_AREA, 10.0, (center,))])
            canvas.set_leaf_mask_overlay(QPixmap.fromImage(disease_mask_to_qimage(mask)))
            canvas.set_correction_mode(True)
            canvas.set_brush("add", 10.0)

            def center_is_marked() -> None:
                canvas.repaint()
                QApplication.processEvents()
                geometry = canvas._geometry
                assert geometry is not None
                marked = geometry.image_to_widget(*center)
                outside = geometry.image_to_widget(12, 12)
                grabbed = canvas.grab().toImage()
                marked_color = grabbed.pixelColor(int(marked[0]), int(marked[1]))
                outside_color = grabbed.pixelColor(int(outside[0]), int(outside[1]))
                self.assertGreater(marked_color.red(), outside_color.red() + 20)
                radii = canvas.brush_footprint_widget_radii()
                assert radii is not None
                scale = geometry.draw_width / geometry.source_width
                self.assertAlmostEqual(radii[0], 10.0 * scale, places=3)
                self.assertAlmostEqual(radii[1], 10.0 * scale, places=3)

            center_is_marked()
            canvas.zoom_actual_size()
            center_is_marked()
            canvas._pan_by(36, -18)
            center_is_marked()
            stored_before = (center,)
            widget = canvas._geometry.image_to_widget(*center)
            finished: list[MaskStroke] = []
            canvas.correction_stroke_finished.connect(finished.append)
            point = QPoint(int(round(widget[0])), int(round(widget[1])))
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
            self.assertEqual(len(finished), 1)
            self.assertAlmostEqual(finished[0].points[0][0], stored_before[0][0], delta=1.5)
            self.assertAlmostEqual(finished[0].points[0][1], stored_before[0][1], delta=1.5)
            self.assertEqual(finished[0].radius_px, 10.0)
            host.close()


class TestCalculationReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_second_calculation_reuses_masks_and_keeps_measurements(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Cache")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            save_sample_calibration(
                workspace,
                sample_id,
                build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0)),
            )
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            sample = workspace.get_sample(sample_id)
            assert sample is not None and sample.candidate_disease_measurement is not None
            leaf_mask = window._leaf_mask_previews[sample_id].mask.copy()
            disease_mask = window._disease_mask_previews[sample_id].disease_mask.copy()
            pixels = sample.candidate_disease_measurement.candidate_disease_pixels
            area = sample.candidate_disease_measurement.candidate_disease_area_in2
            fresh = run_explicit_calculation(
                load_image_rgb_array(image_path),
                sample.calibration,
                source_image_hash=sample.image_hash,
            )
            self.assertTrue(np.array_equal(fresh.leaf_preview.mask, leaf_mask))
            self.assertTrue(np.array_equal(fresh.disease_preview.disease_mask, disease_mask))
            self.assertEqual(fresh.candidate_disease_measurement.candidate_disease_pixels, pixels)

            with patch(
                "neopest.calculation.segment_leaf_preview",
                side_effect=AssertionError("leaf segmentation repeated"),
            ):
                with patch(
                    "neopest.calculation.segment_disease_preview",
                    side_effect=AssertionError("disease segmentation repeated"),
                ):
                    window._on_calculate_results()

            again = workspace.get_sample(sample_id)
            assert again is not None and again.candidate_disease_measurement is not None
            self.assertEqual(again.candidate_disease_measurement.candidate_disease_pixels, pixels)
            self.assertEqual(again.candidate_disease_measurement.candidate_disease_area_in2, area)
            self.assertTrue(np.array_equal(window._leaf_mask_previews[sample_id].mask, leaf_mask))
            self.assertTrue(
                np.array_equal(window._disease_mask_previews[sample_id].disease_mask, disease_mask)
            )

            again.candidate_disease_measurement.candidate_disease_pixels = -1
            window._calculation_request_id += 1
            window._accept_calculation_output(
                sample_id,
                window._calculation_request_id - 1,
                fresh,
            )
            stale = workspace.get_sample(sample_id)
            assert stale is not None and stale.candidate_disease_measurement is not None
            self.assertEqual(stale.candidate_disease_measurement.candidate_disease_pixels, -1)
            window.close()

    def test_first_correction_open_segments_once_and_ignores_a_stale_prepare(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Prepare")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            save_sample_calibration(
                workspace,
                sample_id,
                build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0)),
            )
            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            window._on_calculate_results()
            sample = workspace.get_sample(sample_id)
            assert sample is not None and sample.candidate_disease_measurement is not None
            automatic_pixels = sample.candidate_disease_measurement.candidate_disease_pixels
            disease_mask = window._disease_mask_previews[sample_id].disease_mask.copy()
            messages: list[str] = []
            stages: list[str] = []
            original_show = window.status_bar.showMessage
            original_note = window._note_work_stage

            def capture(text: str, timeout: int = 0) -> None:
                messages.append(text)
                original_show(text, timeout)

            def capture_stage(owner: tuple[str, int, str], stage: str) -> None:
                original_note(owner, stage)
                if window._work_progress_owner != owner:
                    return
                kind, request_id, _sample_id = owner
                if kind == "correction" and request_id != window._correction_prepare_request_id:
                    return
                stages.append(stage)
                self.assertTrue(window.work_progress_bar.isVisible())
                self.assertEqual(window.work_progress_bar.minimum(), 0)
                self.assertEqual(window.work_progress_bar.maximum(), 0)
                self.assertEqual(window.work_progress_label.text(), stage)

            window.status_bar.showMessage = capture  # type: ignore[method-assign]
            window._note_work_stage = capture_stage  # type: ignore[method-assign]
            window._on_correct_disease_mask()
            self.assertFalse(
                any(text == "Preparing candidate disease mask…" for text in messages)
            )
            self.assertEqual(stages, [])
            self.assertFalse(window.work_progress_bar.isVisible())
            assert window._correction_automatic_mask is not None
            self.assertTrue(np.array_equal(window._correction_automatic_mask, disease_mask))
            window._on_done_correction()

            window._leaf_mask_previews.clear()
            window._disease_mask_previews.clear()
            window._preview_image_hash.clear()
            messages.clear()
            stages.clear()
            window._on_correct_disease_mask()
            self.assertEqual(
                stages,
                ["Preparing image", "Finding leaf", "Identifying candidate disease"],
            )
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertIn(
                "Correcting candidate disease. Use View to hide the overlay and see the leaf.",
                messages,
            )
            assert window._correction_automatic_mask is not None
            self.assertTrue(np.array_equal(window._correction_automatic_mask, disease_mask))
            saved = workspace.get_sample(sample_id)
            assert saved is not None and saved.candidate_disease_measurement is not None
            self.assertEqual(
                saved.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )
            window._on_done_correction()

            with patch(
                "neopest.gui.segment_leaf_preview",
                side_effect=AssertionError("leaf segmentation repeated"),
            ):
                with patch(
                    "neopest.gui.segment_disease_preview",
                    side_effect=AssertionError("disease segmentation repeated"),
                ):
                    window._on_correct_disease_mask()
            assert window._correction_automatic_mask is not None
            self.assertTrue(np.array_equal(window._correction_automatic_mask, disease_mask))
            window._on_done_correction()

            window._leaf_mask_previews.clear()
            window._disease_mask_previews.clear()
            window._preview_image_hash.clear()
            from neopest.segmentation import segment_leaf_preview as real_leaf

            def stale_prepare(image: np.ndarray) -> object:
                window._correction_prepare_request_id += 1
                return real_leaf(image)

            with patch("neopest.gui.segment_leaf_preview", side_effect=stale_prepare):
                window._on_correct_disease_mask()
            self.assertIsNone(window._correction_sample_id)
            self.assertFalse(window.image_canvas._correction_mode)
            self.assertTrue(window.inspector_panel.correct_disease_mask_button.isEnabled())
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertIsNone(window._work_progress_owner)
            saved = workspace.get_sample(sample_id)
            assert saved is not None and saved.candidate_disease_measurement is not None
            self.assertEqual(
                saved.candidate_disease_measurement.candidate_disease_pixels,
                automatic_pixels,
            )
            window.close()

    def test_progress_follows_the_current_request_only(self) -> None:
        from neopest.calculation import CalculationError

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Progress")
            image_path = Path(tmp) / "leaf.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(_make_synthetic_leaf_image(), cv2.COLOR_RGB2BGR),
            )
            imported = import_images(workspace, [image_path])
            sample_id = imported[0].sample_id
            save_sample_calibration(
                workspace,
                sample_id,
                build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0)),
            )
            window = _main_window_without_startup_default()
            window.show()
            window._set_workspace(workspace)
            window._on_sample_selected(sample_id)
            seen: dict[str, object] = {}
            original_note = window._note_work_stage

            def capture_stage(owner: tuple[str, int, str], stage: str) -> None:
                original_note(owner, stage)
                if stage != "Finding leaf" or "visible" in seen:
                    return
                before = window._calculation_request_id
                window._on_calculate_results()
                seen["visible"] = window.work_progress_bar.isVisible()
                seen["indeterminate"] = (
                    window.work_progress_bar.minimum() == window.work_progress_bar.maximum() == 0
                )
                seen["stage"] = window.work_progress_label.text()
                seen["blocked"] = window._calculation_request_id == before
                seen["button_enabled"] = window.inspector_panel.calculate_results_button.isEnabled()

            window._note_work_stage = capture_stage  # type: ignore[method-assign]
            window._on_calculate_results()
            self.assertTrue(seen["visible"])
            self.assertTrue(seen["indeterminate"])
            self.assertEqual(seen["stage"], "Finding leaf")
            self.assertTrue(seen["blocked"])
            self.assertFalse(seen["button_enabled"])
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertEqual(window.work_progress_bar.maximum(), 1)
            self.assertIn("Results calculated", window.status_bar.currentMessage())

            reused: list[str] = []

            def capture_reused(owner: tuple[str, int, str], stage: str) -> None:
                original_note(owner, stage)
                reused.append(stage)

            window._note_work_stage = capture_reused  # type: ignore[method-assign]
            window._on_calculate_results()
            self.assertEqual(reused, ["Updating measurements"])
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertIn("Results calculated", window.status_bar.currentMessage())

            current = ("calculation", window._calculation_request_id + 1, sample_id)
            window._calculation_request_id += 1
            window._begin_work_progress(current, "Finding leaf")
            window._note_work_stage(("calculation", current[1] - 1, sample_id), "Preparing image")
            window._clear_work_progress(("calculation", current[1] - 1, sample_id))
            self.assertTrue(window.work_progress_bar.isVisible())
            self.assertEqual(window.work_progress_label.text(), "Finding leaf")
            window._selected_sample_id = "other-sample"
            window._sync_work_progress_visibility()
            self.assertFalse(window.work_progress_bar.isVisible())
            window._note_work_stage(current, "Updating measurements")
            window._clear_work_progress(("calculation", current[1], "other-sample"))
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertEqual(window.work_progress_label.text(), "Finding leaf")
            self.assertEqual(window._work_progress_owner, current)
            window._selected_sample_id = sample_id
            window._sync_work_progress_visibility()
            self.assertTrue(window.work_progress_bar.isVisible())
            self.assertEqual(window.work_progress_label.text(), "Finding leaf")
            window._clear_work_progress(current)
            self.assertIsNone(window._work_progress_owner)
            self.assertFalse(window.work_progress_bar.isVisible())

            def fail_leaf(_image: np.ndarray) -> None:
                raise CalculationError("Leaf segmentation failed.")

            with patch("neopest.calculation.segment_leaf_preview", side_effect=fail_leaf):
                window._leaf_mask_previews.clear()
                window._disease_mask_previews.clear()
                window._preview_image_hash.clear()
                window._on_calculate_results()
            self.assertFalse(window.work_progress_bar.isVisible())
            self.assertIsNone(window._work_progress_owner)
            self.assertIn("Leaf segmentation failed.", window.status_bar.currentMessage())
            self.assertFalse(window._calculation_active)
            window.close()

    def test_decode_cache_reloads_when_the_file_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            first = np.zeros((12, 16, 3), dtype=np.uint8)
            first[..., 1] = 20
            cv2.imwrite(str(path), cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
            loaded = load_image_rgb_array(path)
            self.assertIs(load_image_rgb_array(path), loaded)
            second = np.zeros_like(first)
            second[..., 0] = 90
            cv2.imwrite(str(path), cv2.cvtColor(second, cv2.COLOR_RGB2BGR))
            os.utime(path, None)
            reloaded = load_image_rgb_array(path)
            self.assertIsNot(reloaded, loaded)
            self.assertEqual(int(reloaded[0, 0, 0]), 90)
            self.assertEqual(int(loaded[0, 0, 0]), 0)
