"""Tests for auto-calibration research data models."""

from __future__ import annotations

import json
import math
import unittest

from research.auto_calibration.config import MM_PER_INCH
from research.auto_calibration.models import (
    AutoCalibrationResult,
    ComponentScores,
    FailureReason,
    ResultStatus,
    pixels_per_inch_from_pixels_per_mm,
    pixels_per_mm_from_one_inch_line,
    pixels_per_mm_from_pixels_per_inch,
)


class TestScaleMath(unittest.TestCase):
    def test_euclidean_one_inch_reference(self) -> None:
        ppm = pixels_per_mm_from_one_inch_line((0.0, 0.0), (254.0, 0.0))
        self.assertAlmostEqual(ppm, 10.0)

    def test_pixels_per_inch_and_mm_conversion(self) -> None:
        ppm = 10.0
        ppi = pixels_per_inch_from_pixels_per_mm(ppm)
        self.assertAlmostEqual(ppi, ppm * MM_PER_INCH)
        self.assertAlmostEqual(pixels_per_mm_from_pixels_per_inch(ppi), ppm)

    def test_failure_reason_serialization_round_trip(self) -> None:
        for reason in FailureReason:
            payload = {"failure_reason": reason.value}
            restored = FailureReason(payload["failure_reason"])
            self.assertEqual(restored, reason)


class TestAutoCalibrationResultSerialization(unittest.TestCase):
    def test_stable_json_round_trip(self) -> None:
        original = AutoCalibrationResult(
            schema_version="1.0",
            algorithm_name="neopest_assisted_autocal",
            algorithm_version="0.1.0-prototype",
            source_image="/tmp/sample.jpg",
            image_width_px=640,
            image_height_px=480,
            status=ResultStatus.NO_SUGGESTION,
            failure_reason=FailureReason.NO_RULER_CANDIDATE,
            proposed_endpoints_px=None,
            proposed_pixels_per_inch=None,
            proposed_pixels_per_mm=None,
            overall_confidence=None,
            component_scores=ComponentScores(),
            diagnostics={"warning_codes": ["NOT_IMPLEMENTED"]},
        )
        text_a = original.to_json(include_runtime=False)
        text_b = original.to_json(include_runtime=False)
        self.assertEqual(text_a, text_b)

        restored = AutoCalibrationResult.from_dict(json.loads(text_a))
        self.assertEqual(restored.to_canonical_dict(), original.to_canonical_dict())

    def test_canonical_dict_excludes_runtime(self) -> None:
        result = AutoCalibrationResult(
            schema_version="1.0",
            algorithm_name="neopest_assisted_autocal",
            algorithm_version="0.1.0-prototype",
            source_image="image.png",
            image_width_px=10,
            image_height_px=10,
            status=ResultStatus.NO_SUGGESTION,
            failure_reason=FailureReason.NO_RULER_CANDIDATE,
            proposed_endpoints_px=None,
            proposed_pixels_per_inch=None,
            proposed_pixels_per_mm=None,
            overall_confidence=None,
        )
        result.runtime.elapsed_seconds = 1.23
        canonical = result.to_canonical_dict()
        self.assertNotIn("runtime", canonical)


if __name__ == "__main__":
    unittest.main()
