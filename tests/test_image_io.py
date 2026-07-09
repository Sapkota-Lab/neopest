"""Tests for image probing and hashing helpers."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from neopest.image_io import compute_image_hash, is_supported_image, load_image_info


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
        self.assertFalse(is_supported_image(Path("clip.avi")))

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


if __name__ == "__main__":
    unittest.main()
