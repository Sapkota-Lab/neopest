"""Tests for evaluation metrics and coverage handling."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.auto_calibration.evaluation import (
    NonSuccessEntry,
    compare_successful_proposal,
    compute_aggregate_evaluation,
    evaluate_result_sets,
)
from research.auto_calibration.manual_reference import build_manual_reference
from research.auto_calibration.models import (
    AutoCalibrationResult,
    FailureReason,
    ResultStatus,
)


def _reference() -> object:
    return build_manual_reference(
        source_image="sample.jpg",
        endpoint_a_px=(0.0, 0.0),
        endpoint_b_px=(254.0, 0.0),
        annotated_at="2026-07-16T12:00:00+00:00",
    )


def _success_result(*, ppi_scale: float = 1.0) -> AutoCalibrationResult:
    ref_ppm = 10.0
    ref_ppi = ref_ppm * 25.4
    proposed_ppm = ref_ppm * ppi_scale
    proposed_ppi = ref_ppi * ppi_scale
    return AutoCalibrationResult(
        schema_version="1.0",
        algorithm_name="neopest_assisted_autocal",
        algorithm_version="0.1.0-prototype",
        source_image="sample.jpg",
        image_width_px=640,
        image_height_px=480,
        status=ResultStatus.SUCCESS,
        failure_reason=FailureReason.NONE,
        proposed_endpoints_px=((0.0, 0.0), (254.0, 0.0)),
        proposed_pixels_per_inch=proposed_ppi,
        proposed_pixels_per_mm=proposed_ppm,
        overall_confidence=0.9,
    )


class TestEvaluation(unittest.TestCase):
    def test_single_image_comparison(self) -> None:
        comparison = compare_successful_proposal(_reference(), _success_result(ppi_scale=1.1))
        self.assertAlmostEqual(comparison.reference_pixels_per_inch, 254.0)
        self.assertAlmostEqual(comparison.signed_percentage_error, 10.0)
        self.assertAlmostEqual(comparison.absolute_percentage_error, 10.0)
        self.assertAlmostEqual(comparison.signed_pixels_per_inch_error, 25.4)

    def test_aggregate_coverage_counts_rejections(self) -> None:
        comparisons = [compare_successful_proposal(_reference(), _success_result())]
        non_success = [
            NonSuccessEntry(
                source_image="rejected.jpg",
                status=ResultStatus.NO_SUGGESTION.value,
                failure_reason=FailureReason.NO_RULER_CANDIDATE.value,
            )
        ]
        aggregate = compute_aggregate_evaluation(comparisons, non_success)
        self.assertEqual(aggregate.number_of_reference_images, 2)
        self.assertEqual(aggregate.number_of_successful_proposals, 1)
        self.assertEqual(aggregate.number_of_no_suggestion_results, 1)
        self.assertAlmostEqual(aggregate.proposal_coverage_rate, 0.5)
        self.assertAlmostEqual(aggregate.mean_signed_percentage_error, 0.0)

    def test_rejected_proposals_not_counted_as_zero_error(self) -> None:
        aggregate = compute_aggregate_evaluation(
            comparisons=[],
            non_success_entries=[
                NonSuccessEntry(
                    source_image="sample.jpg",
                    status=ResultStatus.NO_SUGGESTION.value,
                    failure_reason=FailureReason.NO_RULER_CANDIDATE.value,
                )
            ],
        )
        self.assertIsNone(aggregate.mean_signed_percentage_error)
        self.assertIsNone(aggregate.rmse_pixels_per_inch)
        self.assertEqual(aggregate.proposal_coverage_rate, 0.0)

    def test_evaluate_directories(self) -> None:
        reference = _reference()
        success = _success_result()
        rejection = AutoCalibrationResult(
            schema_version="1.0",
            algorithm_name="neopest_assisted_autocal",
            algorithm_version="0.1.0-prototype",
            source_image="rejected.jpg",
            image_width_px=100,
            image_height_px=100,
            status=ResultStatus.NO_SUGGESTION,
            failure_reason=FailureReason.NO_RULER_CANDIDATE,
            proposed_endpoints_px=None,
            proposed_pixels_per_inch=None,
            proposed_pixels_per_mm=None,
            overall_confidence=None,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            references_dir = root / "references"
            results_dir = root / "results"
            references_dir.mkdir()
            results_dir.mkdir()

            reference.write_json(references_dir / "sample.json")
            success.write_json(results_dir / "sample.json")
            rejection.write_json(results_dir / "rejected.json")
            build_manual_reference(
                source_image="rejected.jpg",
                endpoint_a_px=(0.0, 0.0),
                endpoint_b_px=(100.0, 0.0),
                annotated_at="2026-07-16T12:00:00+00:00",
            ).write_json(references_dir / "rejected.json")

            report = evaluate_result_sets(results_dir, references_dir)
            self.assertEqual(len(report.comparisons), 1)
            self.assertEqual(len(report.non_success_entries), 1)
            assert report.aggregate is not None
            self.assertAlmostEqual(report.aggregate.proposal_coverage_rate, 0.5)

            output_path = root / "evaluation.json"
            report.write_json(output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("aggregate", payload)


if __name__ == "__main__":
    unittest.main()
