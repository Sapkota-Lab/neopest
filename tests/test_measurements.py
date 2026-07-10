"""Tests for whole-leaf area measurement math."""

from __future__ import annotations

import math
import unittest

import numpy as np

from neopest.measurements import MeasurementError, calculate_leaf_area
from neopest.metadata import MM_PER_INCH


class TestLeafAreaMeasurement(unittest.TestCase):
    def test_known_mask_and_calibration_returns_expected_values(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True
        pixels_per_mm = 10.0

        result = calculate_leaf_area(mask, pixels_per_mm)

        self.assertEqual(result.leaf_area_pixels, 400)
        self.assertAlmostEqual(result.pixels_per_mm_used, 10.0)
        self.assertAlmostEqual(result.pixels_per_inch_used, 10.0 * MM_PER_INCH)
        expected_in2 = 400 / ((10.0 * MM_PER_INCH) ** 2)
        expected_mm2 = 400 / (10.0**2)
        self.assertAlmostEqual(result.leaf_area_in2, expected_in2)
        self.assertAlmostEqual(result.leaf_area_mm2, expected_mm2)

    def test_zero_pixels_per_mm_raises(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, 0.0)

    def test_negative_pixels_per_mm_raises(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, -1.0)

    def test_non_finite_pixels_per_mm_raises(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, float("nan"))
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, float("inf"))

    def test_empty_mask_raises(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, 5.0)

    def test_none_mask_raises(self) -> None:
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(None, 5.0)  # type: ignore[arg-type]

    def test_invalid_mask_shape_raises(self) -> None:
        mask = np.zeros((10, 10, 3), dtype=bool)
        with self.assertRaises(MeasurementError):
            calculate_leaf_area(mask, 5.0)

    def test_pixels_per_inch_formula(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        pixels_per_mm = 2.5
        result = calculate_leaf_area(mask, pixels_per_mm)
        self.assertAlmostEqual(
            result.pixels_per_inch_used,
            pixels_per_mm * MM_PER_INCH,
        )
        self.assertAlmostEqual(
            result.leaf_area_in2,
            4 / (result.pixels_per_inch_used**2),
        )
        self.assertTrue(math.isfinite(result.leaf_area_in2))


if __name__ == "__main__":
    unittest.main()
