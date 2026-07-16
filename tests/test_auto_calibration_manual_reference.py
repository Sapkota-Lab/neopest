"""Tests for manual reference annotations."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.auto_calibration.manual_reference import (
    ManualReferenceError,
    build_manual_reference,
    validate_manual_reference,
)


class TestManualReference(unittest.TestCase):
    def test_build_and_serialize_reference(self) -> None:
        reference = build_manual_reference(
            source_image="sample.jpg",
            endpoint_a_px=(100.0, 200.0),
            endpoint_b_px=(550.0, 205.0),
            annotator="tester",
            annotated_at="2026-07-16T12:00:00+00:00",
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            reference.write_json(path)
            restored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(restored["source_image"], "sample.jpg")
            self.assertAlmostEqual(
                restored["reference_pixels_per_mm"],
                math.hypot(450.0, 5.0) / 25.4,
            )

    def test_reject_identical_endpoints(self) -> None:
        with self.assertRaises(ManualReferenceError):
            build_manual_reference(
                source_image="sample.jpg",
                endpoint_a_px=(1.0, 2.0),
                endpoint_b_px=(1.0, 2.0),
            )

    def test_reject_non_finite_coordinates(self) -> None:
        with self.assertRaises(ManualReferenceError):
            build_manual_reference(
                source_image="sample.jpg",
                endpoint_a_px=(math.nan, 2.0),
                endpoint_b_px=(3.0, 4.0),
            )

    def test_reject_malformed_image_dimensions(self) -> None:
        reference = build_manual_reference(
            source_image="sample.jpg",
            endpoint_a_px=(10.0, 10.0),
            endpoint_b_px=(60.0, 10.0),
            annotated_at="2026-07-16T12:00:00+00:00",
        )
        with self.assertRaises(ManualReferenceError):
            validate_manual_reference(reference, image_width_px=0)
        with self.assertRaises(ManualReferenceError):
            validate_manual_reference(reference, image_width_px=20, image_height_px=5)

    def test_measurement_determinism_ignores_annotation_timestamp(self) -> None:
        kwargs = {
            "source_image": "sample.jpg",
            "endpoint_a_px": (100.0, 200.0),
            "endpoint_b_px": (550.0, 205.0),
        }
        first = build_manual_reference(**kwargs, annotated_at="2026-01-01T00:00:00+00:00")
        second = build_manual_reference(**kwargs, annotated_at="2026-12-31T23:59:59+00:00")
        self.assertEqual(first.to_measurement_dict(), second.to_measurement_dict())
        self.assertNotEqual(first.to_json(), second.to_json())

    def test_explicit_timestamp_yields_stable_reference_json(self) -> None:
        reference = build_manual_reference(
            source_image="sample.jpg",
            endpoint_a_px=(0.0, 0.0),
            endpoint_b_px=(254.0, 0.0),
            annotated_at="2026-07-16T12:00:00+00:00",
        )
        self.assertEqual(reference.to_json(), reference.to_json())


if __name__ == "__main__":
    unittest.main()
