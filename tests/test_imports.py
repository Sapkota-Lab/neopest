"""Smoke tests that core modules import cleanly."""

from __future__ import annotations

import unittest


class TestImports(unittest.TestCase):
    def test_package_import(self) -> None:
        import neopest

        self.assertEqual(neopest.__version__, "0.4.0")

    def test_core_modules_import(self) -> None:
        modules = [
            "neopest.analysis",
            "neopest.app",
            "neopest.calibration",
            "neopest.export",
            "neopest.explorer_tree",
            "neopest.gui",
            "neopest.gui_canvas",
            "neopest.gui_layout_builders",
            "neopest.gui_styles",
            "neopest.image_io",
            "neopest.metadata",
            "neopest.project_manager",
            "neopest.results",
            "neopest.segmentation",
        ]
        for name in modules:
            with self.subTest(module=name):
                __import__(name)


if __name__ == "__main__":
    unittest.main()
