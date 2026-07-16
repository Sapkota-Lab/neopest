"""Comparison of proposed auto-calibration output against manual references."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.auto_calibration.manual_reference import ManualReference
from research.auto_calibration.models import AutoCalibrationResult, ResultStatus


@dataclass
class ProposalComparison:
    """Per-image comparison when a successful proposal exists."""

    source_image: str
    status: str
    reference_pixels_per_inch: float
    reference_pixels_per_mm: float
    proposed_pixels_per_inch: float
    proposed_pixels_per_mm: float
    signed_pixels_per_inch_error: float
    absolute_pixels_per_inch_error: float
    signed_percentage_error: float
    absolute_percentage_error: float
    signed_pixels_per_mm_error: float
    absolute_pixels_per_mm_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "status": self.status,
            "reference_pixels_per_inch": self.reference_pixels_per_inch,
            "reference_pixels_per_mm": self.reference_pixels_per_mm,
            "proposed_pixels_per_inch": self.proposed_pixels_per_inch,
            "proposed_pixels_per_mm": self.proposed_pixels_per_mm,
            "signed_pixels_per_inch_error": self.signed_pixels_per_inch_error,
            "absolute_pixels_per_inch_error": self.absolute_pixels_per_inch_error,
            "signed_percentage_error": self.signed_percentage_error,
            "absolute_percentage_error": self.absolute_percentage_error,
            "signed_pixels_per_mm_error": self.signed_pixels_per_mm_error,
            "absolute_pixels_per_mm_error": self.absolute_pixels_per_mm_error,
        }


@dataclass
class NonSuccessEntry:
    """Recorded outcome for references without a successful proposal."""

    source_image: str
    status: str
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_image": self.source_image,
            "status": self.status,
        }
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        return payload


@dataclass
class AggregateEvaluation:
    """Coverage-aware aggregate metrics across a reference set."""

    number_of_reference_images: int
    number_of_successful_proposals: int
    number_of_no_suggestion_results: int
    number_of_errors: int
    proposal_coverage_rate: float | None
    mean_signed_percentage_error: float | None
    mean_absolute_percentage_error: float | None
    median_absolute_percentage_error: float | None
    rmse_pixels_per_inch: float | None
    maximum_absolute_percentage_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number_of_reference_images": self.number_of_reference_images,
            "number_of_successful_proposals": self.number_of_successful_proposals,
            "number_of_no_suggestion_results": self.number_of_no_suggestion_results,
            "number_of_errors": self.number_of_errors,
            "proposal_coverage_rate": self.proposal_coverage_rate,
            "mean_signed_percentage_error": self.mean_signed_percentage_error,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "median_absolute_percentage_error": self.median_absolute_percentage_error,
            "rmse_pixels_per_inch": self.rmse_pixels_per_inch,
            "maximum_absolute_percentage_error": self.maximum_absolute_percentage_error,
        }


@dataclass
class EvaluationReport:
    """Full evaluation output for a results/reference directory pair."""

    comparisons: list[ProposalComparison] = field(default_factory=list)
    non_success_entries: list[NonSuccessEntry] = field(default_factory=list)
    aggregate: AggregateEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [item.to_dict() for item in self.comparisons],
            "non_success_entries": [item.to_dict() for item in self.non_success_entries],
            "aggregate": None if self.aggregate is None else self.aggregate.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")


def compare_successful_proposal(
    reference: ManualReference,
    result: AutoCalibrationResult,
) -> ProposalComparison:
    """Compare a successful proposal against a manual reference."""
    if result.status != ResultStatus.SUCCESS:
        raise ValueError("compare_successful_proposal requires status SUCCESS.")
    if result.proposed_pixels_per_inch is None or result.proposed_pixels_per_mm is None:
        raise ValueError("Successful proposal must include scale values.")

    ref_ppi = reference.reference_pixels_per_inch
    ref_ppm = reference.reference_pixels_per_mm
    prop_ppi = result.proposed_pixels_per_inch
    prop_ppm = result.proposed_pixels_per_mm

    signed_ppi_error = prop_ppi - ref_ppi
    abs_ppi_error = abs(signed_ppi_error)
    signed_pct = (signed_ppi_error / ref_ppi) * 100.0
    abs_pct = abs(signed_pct)

    signed_ppm_error = prop_ppm - ref_ppm
    abs_ppm_error = abs(signed_ppm_error)

    return ProposalComparison(
        source_image=reference.source_image,
        status=result.status.value,
        reference_pixels_per_inch=ref_ppi,
        reference_pixels_per_mm=ref_ppm,
        proposed_pixels_per_inch=prop_ppi,
        proposed_pixels_per_mm=prop_ppm,
        signed_pixels_per_inch_error=signed_ppi_error,
        absolute_pixels_per_inch_error=abs_ppi_error,
        signed_percentage_error=signed_pct,
        absolute_percentage_error=abs_pct,
        signed_pixels_per_mm_error=signed_ppm_error,
        absolute_pixels_per_mm_error=abs_ppm_error,
    )


def compute_aggregate_evaluation(
    comparisons: list[ProposalComparison],
    non_success_entries: list[NonSuccessEntry],
) -> AggregateEvaluation:
    """Compute coverage-aware aggregate metrics."""
    total_references = len(comparisons) + len(non_success_entries)
    no_suggestion = sum(
        1 for entry in non_success_entries if entry.status == ResultStatus.NO_SUGGESTION.value
    )
    errors = sum(
        1 for entry in non_success_entries if entry.status == ResultStatus.ERROR.value
    )
    successes = len(comparisons)

    coverage: float | None
    if total_references == 0:
        coverage = None
    else:
        coverage = successes / total_references

    if not comparisons:
        return AggregateEvaluation(
            number_of_reference_images=total_references,
            number_of_successful_proposals=successes,
            number_of_no_suggestion_results=no_suggestion,
            number_of_errors=errors,
            proposal_coverage_rate=coverage,
            mean_signed_percentage_error=None,
            mean_absolute_percentage_error=None,
            median_absolute_percentage_error=None,
            rmse_pixels_per_inch=None,
            maximum_absolute_percentage_error=None,
        )

    signed_pcts = [item.signed_percentage_error for item in comparisons]
    abs_pcts = [item.absolute_percentage_error for item in comparisons]
    ppi_errors = [item.signed_pixels_per_inch_error for item in comparisons]

    return AggregateEvaluation(
        number_of_reference_images=total_references,
        number_of_successful_proposals=successes,
        number_of_no_suggestion_results=no_suggestion,
        number_of_errors=errors,
        proposal_coverage_rate=coverage,
        mean_signed_percentage_error=statistics.fmean(signed_pcts),
        mean_absolute_percentage_error=statistics.fmean(abs_pcts),
        median_absolute_percentage_error=statistics.median(abs_pcts),
        rmse_pixels_per_inch=math.sqrt(statistics.fmean(e * e for e in ppi_errors)),
        maximum_absolute_percentage_error=max(abs_pcts),
    )


def _stem_key(path: Path) -> str:
    return path.stem.lower()


def evaluate_result_sets(
    results_dir: Path | str,
    references_dir: Path | str,
) -> EvaluationReport:
    """Match result and reference JSON files by basename and evaluate."""
    results_path = Path(results_dir)
    references_path = Path(references_dir)

    reference_files = sorted(
        references_path.glob("*.json"),
        key=lambda path: _stem_key(path),
    )
    result_files = {
        _stem_key(path): path for path in results_path.glob("*.json")
    }

    comparisons: list[ProposalComparison] = []
    non_success: list[NonSuccessEntry] = []

    for reference_file in reference_files:
        reference = ManualReference.from_json_file(reference_file)
        result_file = result_files.get(_stem_key(reference_file))
        if result_file is None:
            non_success.append(
                NonSuccessEntry(
                    source_image=reference.source_image,
                    status="MISSING_RESULT",
                )
            )
            continue

        result = AutoCalibrationResult.from_json_file(result_file)
        if result.status == ResultStatus.SUCCESS:
            comparisons.append(compare_successful_proposal(reference, result))
        else:
            non_success.append(
                NonSuccessEntry(
                    source_image=reference.source_image,
                    status=result.status.value,
                    failure_reason=result.failure_reason.value,
                )
            )

    aggregate = compute_aggregate_evaluation(comparisons, non_success)
    return EvaluationReport(
        comparisons=comparisons,
        non_success_entries=non_success,
        aggregate=aggregate,
    )
