"""Tests for image probing and hashing helpers."""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtGui import QColor, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import QApplication

from neopest.gui_canvas import ImageCanvas
from neopest.gui_styles import COLOR_ACCENT
from neopest.image_io import (
    SUPPORTED_IMAGE_SUFFIXES,
    compute_image_hash,
    is_supported_image,
    leaf_mask_to_qimage,
    load_image_info,
    load_image_rgb_array,
    load_preview_pixmap,
    qimage_to_rgb_uint8,
)
from neopest.project_manager import WorkspaceError, import_images, initialize_workspace

REAL_HEIC_FILES = (
    Path("/Users/danielkeetam/Downloads/IMG_7823.HEIC"),
    Path("/Users/danielkeetam/Downloads/lp_image.HEIC"),
)


def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication(sys.argv)
    return app


def _write_test_png(path: Path, *, width: int = 64, height: int = 48) -> None:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("brown"))
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"Failed to write test image: {path}")


class TestImageIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_is_supported_image_suffixes(self) -> None:
        self.assertTrue(is_supported_image(Path("leaf.png")))
        self.assertTrue(is_supported_image(Path("leaf.JPG")))
        self.assertTrue(is_supported_image(Path("leaf.HEIC")))
        self.assertTrue(is_supported_image(Path("leaf.heif")))
        self.assertFalse(is_supported_image(Path("clip.avi")))
        self.assertIn(".heic", SUPPORTED_IMAGE_SUFFIXES)
        self.assertIn(".heif", SUPPORTED_IMAGE_SUFFIXES)

    def test_load_image_info_reads_dimensions(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "leaf.png"
            _write_test_png(image_path, width=120, height=80)
            info = load_image_info(image_path)
            self.assertEqual(info.width_px, 120)
            self.assertEqual(info.height_px, 80)

    def test_compute_image_hash_matches_sha256(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "leaf.png"
            _write_test_png(image_path)
            expected = hashlib.sha256(image_path.read_bytes()).hexdigest()
            self.assertEqual(compute_image_hash(image_path), f"sha256:{expected}")


def _padded_rgb_image() -> QImage:
    """Build a 1-pixel-wide RGB image so Qt's row stride includes padding."""
    image = QImage(1, 2, QImage.Format.Format_RGB888)
    if image.bytesPerLine() <= 3:
        raise RuntimeError("Expected Qt to pad a 1-pixel RGB row")
    image.setPixelColor(0, 0, QColor(11, 22, 33))
    image.setPixelColor(0, 1, QColor(44, 55, 66))
    return image


class _RecordingHeifReader:
    """QImageReader stand-in that records autoTransform and returns a fixed image."""

    image = QImage()
    auto_transform_values: list[bool] = []

    def __init__(self, path: str) -> None:
        self.path = path

    @staticmethod
    def supportedImageFormats() -> list[bytes]:
        return [b"heic", b"heif", b"png", b"jpeg"]

    def setAutoTransform(self, enabled: bool) -> None:
        _RecordingHeifReader.auto_transform_values.append(enabled)

    def read(self) -> QImage:
        return _RecordingHeifReader.image

    def size(self):  # noqa: ANN201 - Qt stand-in
        from PySide6.QtCore import QSize

        image = _RecordingHeifReader.image
        return QSize(image.width(), image.height())


class TestHeifLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_qimage_conversion_drops_row_padding(self) -> None:
        image = _padded_rgb_image()
        array = qimage_to_rgb_uint8(image)
        self.assertEqual(array.shape, (2, 1, 3))
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(array[0, 0].tolist(), [11, 22, 33])
        self.assertEqual(array[1, 0].tolist(), [44, 55, 66])
        self.assertTrue(array.flags.c_contiguous)

    def test_heif_reader_applies_autotransform_once(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.heic"
            source.write_bytes(b"placeholder")
            _RecordingHeifReader.image = _padded_rgb_image()
            _RecordingHeifReader.auto_transform_values = []
            with patch("neopest.image_io.QImageReader", _RecordingHeifReader):
                info = load_image_info(source)
                array = load_image_rgb_array(source)
            self.assertEqual(_RecordingHeifReader.auto_transform_values, [True, True])
            self.assertEqual((info.width_px, info.height_px), (1, 2))
            self.assertEqual(array.shape, (2, 1, 3))

    def test_jpeg_loading_still_uses_opencv(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "leaf.jpg"
            _write_test_png(image_path.with_suffix(".png"))
            png = QImage(str(image_path.with_suffix(".png")))
            self.assertTrue(png.save(str(image_path), "JPG"))
            with patch("neopest.image_io.cv2.imread", wraps=__import__("cv2").imread) as imread:
                loaded = load_image_rgb_array(image_path)
            imread.assert_called_once()
            self.assertEqual(loaded.shape, (48, 64, 3))
            self.assertEqual(loaded.dtype, np.uint8)

    def test_missing_heic_plugin_reports_through_import(self) -> None:
        from tests.test_explorer_polish import _main_window_without_startup_default

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaf.heic"
            source.write_bytes(b"not-a-heic-file")
            original = source.read_bytes()
            workspace = initialize_workspace(Path(tmp) / "ws", display_name="HEIC")
            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            try:
                with patch.object(
                    QImageReader,
                    "supportedImageFormats",
                    return_value=[b"png", b"jpeg"],
                ):
                    with patch("neopest.gui.QMessageBox.critical") as critical:
                        window._import_paths_into_group([source], None)
                self.assertEqual(workspace.samples, [])
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(critical.call_args.args[1], "Import Error")
                self.assertIn("Qt HEIC image plugin is missing", critical.call_args.args[2])
            finally:
                window.close()

    def test_corrupt_heic_import_keeps_the_original_file(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "broken.HEIC"
            source.write_bytes(b"this is not a heif container")
            original = source.read_bytes()
            workspace = initialize_workspace(Path(tmp) / "ws", display_name="HEIC")
            with self.assertRaises(WorkspaceError) as caught:
                import_images(workspace, [source])
            self.assertIn("Unable to decode HEIC/HEIF image", str(caught.exception))
            self.assertEqual(workspace.samples, [])
            self.assertEqual(source.read_bytes(), original)


class TestRealHeicPreviewAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_real_files_match_preview_calibration_and_mask(self) -> None:
        missing = [path for path in REAL_HEIC_FILES if not path.is_file()]
        if missing:
            self.skipTest(f"Real HEIC files are not available: {missing}")

        accent = QColor(COLOR_ACCENT)
        for path in REAL_HEIC_FILES:
            original = path.read_bytes()
            info = load_image_info(path)
            array = load_image_rgb_array(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(array.ndim, 3)
            self.assertEqual(array.shape[2], 3)
            self.assertEqual((info.width_px, info.height_px), (array.shape[1], array.shape[0]))

            pixmap = load_preview_pixmap(path)
            self.assertFalse(pixmap.isNull())
            self.assertEqual((pixmap.width(), pixmap.height()), (array.shape[1], array.shape[0]))
            oriented_reader = QImageReader(str(path))
            oriented_reader.setAutoTransform(True)
            oriented = oriented_reader.read().convertToFormat(QImage.Format.Format_RGB888)
            stored_reader = QImageReader(str(path))
            stored_reader.setAutoTransform(False)
            stored = stored_reader.read()
            self.assertNotEqual(
                (stored.width(), stored.height()),
                (oriented.width(), oriented.height()),
            )
            self.assertEqual((oriented.width(), oriented.height()), (array.shape[1], array.shape[0]))
            preview = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
            for x, y in (
                (0, 0),
                (pixmap.width() - 1, 0),
                (0, pixmap.height() - 1),
                (pixmap.width() // 2, pixmap.height() // 3),
            ):
                color = preview.pixelColor(x, y)
                oriented_color = oriented.pixelColor(x, y)
                self.assertEqual(
                    array[y, x].tolist(),
                    [color.red(), color.green(), color.blue()],
                    msg=f"{path.name} preview pixel {(x, y)}",
                )
                self.assertEqual(
                    array[y, x].tolist(),
                    [oriented_color.red(), oriented_color.green(), oriented_color.blue()],
                    msg=f"{path.name} oriented pixel {(x, y)}",
                )
            stored_rgb = qimage_to_rgb_uint8(stored)
            best_score = -1
            landmark = (0, 0)
            for y in range(0, stored_rgb.shape[0], 64):
                for x in range(0, stored_rgb.shape[1], 64):
                    pixel = stored_rgb[y, x].astype(int)
                    score = int(pixel.max() - pixel.min())
                    if score > best_score:
                        best_score = score
                        landmark = (x, y)
            landmark_x, landmark_y = landmark
            landmark_color = stored_rgb[landmark_y, landmark_x].tolist()
            stored_height, stored_width = stored_rgb.shape[:2]
            clockwise = (stored_height - 1 - landmark_y, landmark_x)
            self.assertGreater(best_score, 40, msg=path.name)
            self.assertEqual(
                array[clockwise[1], clockwise[0]].tolist(),
                landmark_color,
                msg=f"{path.name} landmark did not land at the clockwise pixel",
            )
            if (landmark_x, landmark_y) != clockwise:
                self.assertNotEqual(
                    array[landmark_y, landmark_x].tolist(),
                    landmark_color,
                    msg=f"{path.name} landmark still sits on the stored pixel",
                )

            mask = np.zeros(array.shape[:2], dtype=bool)
            block = (
                slice(array.shape[0] // 4, array.shape[0] // 2),
                slice(array.shape[1] // 4, array.shape[1] // 2),
            )
            mask[block] = True
            overlay = leaf_mask_to_qimage(mask)
            self.assertEqual((overlay.width(), overlay.height()), (array.shape[1], array.shape[0]))

            point_1 = (array.shape[1] * 0.25, array.shape[0] * 0.25)
            point_2 = (array.shape[1] * 0.75, array.shape[0] * 0.75)
            canvas = ImageCanvas()
            canvas.resize(640, 480)
            canvas.show()
            QApplication.processEvents()
            try:
                canvas.set_image_path(path)
                canvas.set_saved_calibration_line(point_1, point_2)
                canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay))
                canvas.repaint()
                QApplication.processEvents()
                geometry = canvas._geometry
                self.assertIsNotNone(geometry)
                assert geometry is not None
                self.assertEqual(geometry.source_width, array.shape[1])
                self.assertEqual(geometry.source_height, array.shape[0])
                self.assertGreater(geometry.draw_width, 0)
                self.assertGreater(geometry.draw_height, 0)
                self.assertEqual(canvas._saved_line, (point_1, point_2))
                assert canvas._leaf_mask_overlay is not None
                self.assertEqual(
                    (canvas._leaf_mask_overlay.width(), canvas._leaf_mask_overlay.height()),
                    (geometry.source_width, geometry.source_height),
                )
                for point in (point_1, point_2):
                    widget_point = geometry.image_to_widget(point[0], point[1])
                    restored = geometry.widget_to_image(*widget_point)
                    self.assertIsNotNone(restored)
                    assert restored is not None
                    self.assertAlmostEqual(restored[0], point[0], places=5)
                    self.assertAlmostEqual(restored[1], point[1], places=5)

                canvas.set_leaf_mask_overlay(None)
                canvas.repaint()
                bare = canvas.grab().toImage()
                canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay))
                canvas.repaint()
                covered = canvas.grab().toImage()

                marker = geometry.image_to_widget(point_1[0], point_1[1])
                marker_color = covered.pixelColor(int(marker[0]), int(marker[1]))
                self.assertLess(abs(marker_color.red() - accent.red()), 40)
                self.assertLess(abs(marker_color.green() - accent.green()), 40)
                self.assertLess(abs(marker_color.blue() - accent.blue()), 40)

                mask_point = geometry.image_to_widget(
                    array.shape[1] * 0.30,
                    array.shape[0] * 0.45,
                )
                mask_x = int(mask_point[0])
                mask_y = int(mask_point[1])
                self.assertGreaterEqual(mask_x, geometry.draw_x)
                self.assertLess(mask_x, geometry.draw_x + geometry.draw_width)
                self.assertGreaterEqual(mask_y, geometry.draw_y)
                self.assertLess(mask_y, geometry.draw_y + geometry.draw_height)
                bare_color = bare.pixelColor(mask_x, mask_y)
                covered_color = covered.pixelColor(mask_x, mask_y)
                self.assertNotEqual(
                    (bare_color.red(), bare_color.green(), bare_color.blue()),
                    (covered_color.red(), covered_color.green(), covered_color.blue()),
                )
            finally:
                canvas.close()
                canvas.deleteLater()


def _inject_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    tiff = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 1)
    tiff += struct.pack("<HHII", 0x0112, 3, 1, orientation) + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:2] + segment + jpeg[2:]


def _write_landmark_jpeg(path: Path, orientation: int | None) -> tuple[int, int]:
    """Write a JPEG whose top-left block is red and whose other corners differ."""
    height, width = 80, 120
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[0:24, 0:24] = (220, 20, 20)
    image[0:24, width - 24 : width] = (20, 220, 20)
    image[height - 24 : height, 0:24] = (20, 20, 220)
    image[height - 24 : height, width - 24 : width] = (220, 220, 20)
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    ):
        raise RuntimeError(f"Failed to write {path}")
    if orientation is not None:
        path.write_bytes(_inject_exif_orientation(path.read_bytes(), orientation))
    return width, height


def _oriented_landmark(orientation: int | None, x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Return where a stored pixel lands after one EXIF orientation."""
    if orientation in {None, 1}:
        return x, y
    if orientation == 2:
        return width - 1 - x, y
    if orientation == 5:
        return y, x
    if orientation == 6:
        return height - 1 - y, x
    if orientation == 7:
        return height - 1 - y, width - 1 - x
    raise AssertionError(f"unexpected orientation {orientation}")


class TestOrientationAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qt_app()

    def test_preview_overlay_and_array_share_jpeg_landmarks(self) -> None:
        for orientation in (None, 2, 6, 7):
            with self.subTest(orientation=orientation):
                with TemporaryDirectory() as tmp:
                    path = Path(tmp) / "leaf.jpg"
                    width, height = _write_landmark_jpeg(path, orientation)
                    original = path.read_bytes()
                    array = load_image_rgb_array(path)
                    info = load_image_info(path)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(
                        (info.width_px, info.height_px),
                        (array.shape[1], array.shape[0]),
                    )

                    stored = cv2.cvtColor(
                        cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                        cv2.COLOR_BGR2RGB,
                    )
                    self.assertGreater(int(stored[12, 12, 0]), 180)
                    self.assertLess(int(stored[12, 12, 1]), 60)

                    landmark_x, landmark_y = _oriented_landmark(orientation, 12, 12, width, height)
                    patch = array[landmark_y - 4 : landmark_y + 4, landmark_x - 4 : landmark_x + 4]
                    self.assertGreater(float(patch[:, :, 0].mean()), 180)
                    self.assertLess(float(patch[:, :, 1].mean()), 60)
                    if orientation not in {None, 1}:
                        self.assertFalse(
                            float(array[12, 12, 0]) > 180 and float(array[12, 12, 1]) < 60
                        )

                    preview = load_preview_pixmap(path).toImage().convertToFormat(
                        QImage.Format.Format_RGB888
                    )
                    color = preview.pixelColor(landmark_x, landmark_y)
                    self.assertEqual(
                        array[landmark_y, landmark_x].tolist(),
                        [color.red(), color.green(), color.blue()],
                    )

                    mask = np.zeros(array.shape[:2], dtype=bool)
                    mask[
                        landmark_y - 8 : landmark_y + 8,
                        landmark_x - 8 : landmark_x + 8,
                    ] = True
                    overlay = leaf_mask_to_qimage(mask)
                    canvas = ImageCanvas()
                    canvas.resize(640, 480)
                    canvas.show()
                    QApplication.processEvents()
                    try:
                        canvas.set_image_path(path)
                        canvas.repaint()
                        QApplication.processEvents()
                        geometry = canvas._geometry
                        self.assertIsNotNone(geometry)
                        assert geometry is not None
                        self.assertEqual(geometry.source_width, array.shape[1])
                        self.assertEqual(geometry.source_height, array.shape[0])
                        bare = canvas.grab().toImage()
                        canvas.set_leaf_mask_overlay(QPixmap.fromImage(overlay))
                        canvas.repaint()
                        QApplication.processEvents()
                        covered = canvas.grab().toImage()

                        mark = geometry.image_to_widget(landmark_x, landmark_y)
                        mark_x, mark_y = int(mark[0]), int(mark[1])
                        bare_color = bare.pixelColor(mark_x, mark_y)
                        covered_color = covered.pixelColor(mark_x, mark_y)
                        self.assertNotEqual(
                            (bare_color.red(), bare_color.green(), bare_color.blue()),
                            (covered_color.red(), covered_color.green(), covered_color.blue()),
                        )

                        center = geometry.image_to_widget(
                            array.shape[1] / 2,
                            array.shape[0] / 2,
                        )
                        center_x, center_y = int(center[0]), int(center[1])
                        bare_center = bare.pixelColor(center_x, center_y)
                        covered_center = covered.pixelColor(center_x, center_y)
                        self.assertEqual(
                            (bare_center.red(), bare_center.green(), bare_center.blue()),
                            (
                                covered_center.red(),
                                covered_center.green(),
                                covered_center.blue(),
                            ),
                        )
                    finally:
                        canvas.close()
                        canvas.deleteLater()

    def test_legacy_calibration_on_rotated_jpeg_is_not_reused(self) -> None:
        from neopest.calibration import build_one_inch_manual_calibration_record
        from neopest.disease_segmentation import DISEASE_PREVIEW_METHOD, DISEASE_SETTINGS_VERSION
        from neopest.metadata import (
            WORKSPACE_JSON_FILENAME,
            CalibrationRecord,
            CandidateDiseaseMeasurementRecord,
        )
        from neopest.project_manager import save_sample_calibration, save_workspace
        from neopest.results import CalibrationStatus
        from neopest.segmentation import PREVIEW_METHOD, SETTINGS_VERSION
        from tests.test_explorer_polish import _main_window_without_startup_default

        with TemporaryDirectory() as tmp:
            rotated = Path(tmp) / "rotated.jpg"
            plain = Path(tmp) / "plain.jpg"
            _write_landmark_jpeg(rotated, 6)
            _write_landmark_jpeg(plain, None)
            workspace = initialize_workspace(Path(tmp) / "ws", display_name="Orientation")
            imported = import_images(workspace, [rotated, plain])
            rotated_sample = next(sample for sample in imported if sample.display_name == "rotated.jpg")
            plain_sample = next(sample for sample in imported if sample.display_name == "plain.jpg")

            legacy = build_one_inch_manual_calibration_record((10.0, 20.0), (110.0, 20.0))
            legacy.pixel_grid = None
            rotated_sample.calibration = legacy
            rotated_sample.calibration_status = CalibrationStatus.MANUAL
            rotated_sample.candidate_disease_measurement = CandidateDiseaseMeasurementRecord(
                status="valid",
                total_leaf_area_in2=1.25,
                candidate_disease_area_in2=0.25,
                candidate_disease_coverage_percent=20.0,
                non_candidate_leaf_area_in2=1.0,
                pixels_per_mm_used=legacy.pixels_per_mm,
                source_image_hash=rotated_sample.image_hash,
                leaf_mask_method=PREVIEW_METHOD,
                leaf_mask_settings_version=SETTINGS_VERSION,
                disease_mask_method=DISEASE_PREVIEW_METHOD,
                disease_mask_settings_version=DISEASE_SETTINGS_VERSION,
            )
            plain_legacy = CalibrationRecord(
                pixels_per_mm=legacy.pixels_per_mm,
                known_distance_mm=legacy.known_distance_mm,
                known_distance_value=legacy.known_distance_value,
                known_distance_unit=legacy.known_distance_unit,
                point_1_x=10.0,
                point_1_y=20.0,
                point_2_x=110.0,
                point_2_y=20.0,
                calibrated_at=legacy.calibrated_at,
            )
            plain_sample.calibration = plain_legacy
            plain_sample.calibration_status = CalibrationStatus.MANUAL
            save_workspace(workspace)
            metadata_path = workspace.root_path / WORKSPACE_JSON_FILENAME
            before = metadata_path.read_bytes()
            self.assertNotIn(b'"pixel_grid"', before)

            window = _main_window_without_startup_default()
            window._set_workspace(workspace)
            try:
                window._on_sample_selected(rotated_sample.sample_id)
                self.assertIsNone(window.image_canvas._saved_line)
                panel = window.inspector_panel
                self.assertEqual(panel.calibration_status_value.text(), "Recalibration required")
                self.assertEqual(panel.result_status_value.text(), "Recalculation required")
                self.assertEqual(panel.total_leaf_area_label.text(), "Recalculation required")
                self.assertFalse(panel.calculate_results_button.isEnabled())
                self.assertIn(
                    "previous image orientation",
                    panel.calculate_results_button.toolTip(),
                )
                window._on_calculate_results()
                self.assertEqual(metadata_path.read_bytes(), before)
                reloaded = workspace.get_sample(rotated_sample.sample_id)
                assert reloaded is not None and reloaded.calibration is not None
                self.assertEqual(reloaded.calibration.point_1_x, 10.0)
                self.assertEqual(reloaded.calibration.point_1_y, 20.0)
                self.assertEqual(reloaded.calibration.point_2_x, 110.0)
                self.assertEqual(reloaded.calibration.point_2_y, 20.0)
                self.assertIsNone(reloaded.calibration.pixel_grid)
                assert reloaded.candidate_disease_measurement is not None
                self.assertEqual(reloaded.candidate_disease_measurement.total_leaf_area_in2, 1.25)

                window._on_sample_selected(plain_sample.sample_id)
                self.assertEqual(
                    window.image_canvas._saved_line,
                    ((10.0, 20.0), (110.0, 20.0)),
                )
                self.assertNotEqual(
                    window.inspector_panel.calibration_status_value.text(),
                    "Recalibration required",
                )

                current = build_one_inch_manual_calibration_record((4.0, 8.0), (84.0, 8.0))
                save_sample_calibration(workspace, rotated_sample.sample_id, current)
                window._on_sample_selected(rotated_sample.sample_id)
                self.assertEqual(current.pixel_grid, "oriented-v1")
                self.assertEqual(
                    window.image_canvas._saved_line,
                    ((4.0, 8.0), (84.0, 8.0)),
                )
                self.assertTrue(window.inspector_panel.calculate_results_button.isEnabled())
                saved = metadata_path.read_text()
                self.assertIn('"pixel_grid": "oriented-v1"', saved)
                self.assertIn("10.0", saved)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
