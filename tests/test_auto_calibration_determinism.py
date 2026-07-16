"""Determinism and isolation tests for the research prototype."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from research.auto_calibration.config import DEFAULT_CONFIG
from research.auto_calibration.pipeline import (
    list_supported_images,
    run_auto_calibration_proposal,
    run_batch_auto_calibration_proposals,
)
from research.auto_calibration.visualization import render_proposal_overlay


RESEARCH_PACKAGE = Path(__file__).resolve().parents[1] / "research" / "auto_calibration"
FORBIDDEN_PREFIXES = (
    "neopest.gui",
    "neopest.app",
    "neopest.gui_canvas",
    "neopest.gui_layout_builders",
    "neopest.gui_styles",
)


def _write_test_image(path: Path, *, width: int = 80, height: int = 60) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (40, 120, 40)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test image: {path}")


class TestAutoCalibrationDeterminism(unittest.TestCase):
    def test_repeated_pipeline_execution_is_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "leaf_sample.png"
            _write_test_image(image_path)

            first = run_auto_calibration_proposal(
                image_path,
                config=DEFAULT_CONFIG,
                capture_runtime=False,
            )
            second = run_auto_calibration_proposal(
                image_path,
                config=DEFAULT_CONFIG,
                capture_runtime=False,
            )
            self.assertEqual(first.to_canonical_dict(), second.to_canonical_dict())

    def test_batch_ordering_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_test_image(root / "b_image.png")
            _write_test_image(root / "A_image.png")
            _write_test_image(root / "c_image.png")

            listed = [path.name for path in list_supported_images(root)]
            self.assertEqual(listed, ["A_image.png", "b_image.png", "c_image.png"])

            batch = run_batch_auto_calibration_proposals(root, capture_runtime=False)
            self.assertEqual(
                [result.source_image for result in batch],
                [str(root / name) for name in listed],
            )

    def test_visualization_does_not_mutate_canonical_result(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "overlay_sample.png"
            _write_test_image(image_path)
            result = run_auto_calibration_proposal(image_path, capture_runtime=False)
            canonical_before = result.to_canonical_dict()

            render_proposal_overlay(
                image_path,
                result,
                diagnostic_geometry={
                    "reference_endpoints_px": [[10, 20], [60, 22]],
                    "ruler_roi": {"x": 5, "y": 5, "width": 70, "height": 20},
                },
            )
            self.assertEqual(result.to_canonical_dict(), canonical_before)


class TestResearchIsolation(unittest.TestCase):
    def test_no_neopest_gui_imports(self) -> None:
        offenders: list[str] = []
        for path in sorted(RESEARCH_PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(FORBIDDEN_PREFIXES):
                            offenders.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(FORBIDDEN_PREFIXES):
                        offenders.append(f"{path.name}: from {node.module}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
