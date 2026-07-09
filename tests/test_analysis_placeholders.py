"""Tests for placeholder analysis and result types."""

from __future__ import annotations

import unittest
from pathlib import Path

from neopest.analysis import placeholder_analysis_result, run_leaf_analysis
from neopest.calibration import calibrate_from_ruler
from neopest.export import export_results_csv
from neopest.results import AnalysisStatus, CalibrationStatus, LeafAnalysisResult
from neopest.segmentation import segment_leaf_and_disease


class TestAnalysisPlaceholders(unittest.TestCase):
    def test_leaf_analysis_result_fields(self) -> None:
        result = LeafAnalysisResult(
            sample_id="sample_abc123",
            leaf_area_mm2=None,
            disease_area_mm2=None,
            disease_fraction=None,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            analysis_status=AnalysisStatus.NOT_RUN,
        )
        self.assertTrue(result.is_placeholder)
        self.assertEqual(result.sample_id, "sample_abc123")

    def test_placeholder_analysis_result(self) -> None:
        result = placeholder_analysis_result("sample_test")
        self.assertEqual(result.sample_id, "sample_test")
        self.assertIsNone(result.leaf_area_mm2)
        self.assertEqual(result.calibration_status, CalibrationStatus.UNCALIBRATED)
        self.assertEqual(result.analysis_status, AnalysisStatus.NOT_RUN)

    def test_run_leaf_analysis_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            run_leaf_analysis("sample_x", Path("/tmp/leaf.jpg"))

    def test_calibration_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            calibrate_from_ruler()

    def test_segmentation_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            segment_leaf_and_disease()

    def test_export_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            export_results_csv([], Path("/tmp/out.csv"))


if __name__ == "__main__":
    unittest.main()
