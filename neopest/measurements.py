"""Pure leaf and candidate-disease measurement — no GUI dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neopest.disease_segmentation import (
    DISEASE_PREVIEW_METHOD,
    DISEASE_SETTINGS_VERSION,
)
from neopest.metadata import (
    MM2_PER_IN2,
    MM_PER_INCH,
    CALCULATION_METHOD_EXPLICIT_V1,
    CalibrationRecord,
    CandidateDiseaseMeasurementRecord,
    LeafMeasurementRecord,
    SampleRecord,
)
from neopest.segmentation import PREVIEW_METHOD, SETTINGS_VERSION

CURRENT_LEAF_MASK_METHOD = PREVIEW_METHOD
CURRENT_LEAF_MASK_SETTINGS_VERSION = SETTINGS_VERSION
CURRENT_DISEASE_MASK_METHOD = DISEASE_PREVIEW_METHOD
CURRENT_DISEASE_MASK_SETTINGS_VERSION = DISEASE_SETTINGS_VERSION


class MeasurementError(ValueError):
    """Raised when area cannot be calculated from the given inputs."""


@dataclass(frozen=True)
class LeafAreaMeasurement:
    """Numeric whole-leaf area derived from a binary mask and scale calibration."""

    leaf_area_pixels: int
    leaf_area_in2: float
    leaf_area_mm2: float
    pixels_per_inch_used: float
    pixels_per_mm_used: float


@dataclass(frozen=True)
class CandidateDiseaseMeasurement:
    """Numeric leaf and candidate-disease areas from pixel counts and calibration."""

    total_leaf_pixels: int
    candidate_disease_pixels: int
    total_leaf_area_mm2: float
    total_leaf_area_in2: float
    candidate_disease_area_mm2: float
    candidate_disease_area_in2: float
    candidate_disease_coverage_percent: float
    non_candidate_leaf_area_mm2: float
    non_candidate_leaf_area_in2: float
    pixels_per_inch_used: float
    pixels_per_mm_used: float


def _validate_pixels_per_mm(pixels_per_mm: float) -> float:
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise MeasurementError("pixels_per_mm must be a finite value greater than zero.")
    return float(pixels_per_mm)


def _validate_mask(mask: np.ndarray, *, label: str) -> np.ndarray:
    if mask is None:
        raise MeasurementError(f"{label} is required for measurement.")
    if mask.ndim != 2:
        raise MeasurementError(
            f"Expected a 2-D boolean {label}; got shape {getattr(mask, 'shape', None)!r}."
        )
    return mask


def calculate_areas_from_pixel_counts(
    total_leaf_pixels: int,
    candidate_disease_pixels: int,
    pixels_per_mm: float,
) -> CandidateDiseaseMeasurement:
    """Calculate leaf and candidate-disease areas from raw pixel counts."""
    pixels_per_mm = _validate_pixels_per_mm(pixels_per_mm)
    if total_leaf_pixels <= 0:
        raise MeasurementError("Whole-leaf mask contains no pixels.")
    if candidate_disease_pixels < 0:
        raise MeasurementError("Candidate disease pixel count cannot be negative.")
    if candidate_disease_pixels > total_leaf_pixels:
        raise MeasurementError("Candidate disease pixels exceed whole-leaf pixels.")

    pixels_per_inch = pixels_per_mm * MM_PER_INCH
    total_leaf_area_mm2 = total_leaf_pixels / (pixels_per_mm**2)
    candidate_disease_area_mm2 = candidate_disease_pixels / (pixels_per_mm**2)
    non_candidate_pixels = total_leaf_pixels - candidate_disease_pixels
    non_candidate_leaf_area_mm2 = non_candidate_pixels / (pixels_per_mm**2)
    candidate_disease_coverage_percent = (
        (candidate_disease_pixels / total_leaf_pixels) * 100.0
    )

    return CandidateDiseaseMeasurement(
        total_leaf_pixels=total_leaf_pixels,
        candidate_disease_pixels=candidate_disease_pixels,
        total_leaf_area_mm2=total_leaf_area_mm2,
        total_leaf_area_in2=total_leaf_area_mm2 / MM2_PER_IN2,
        candidate_disease_area_mm2=candidate_disease_area_mm2,
        candidate_disease_area_in2=candidate_disease_area_mm2 / MM2_PER_IN2,
        candidate_disease_coverage_percent=candidate_disease_coverage_percent,
        non_candidate_leaf_area_mm2=non_candidate_leaf_area_mm2,
        non_candidate_leaf_area_in2=non_candidate_leaf_area_mm2 / MM2_PER_IN2,
        pixels_per_inch_used=pixels_per_inch,
        pixels_per_mm_used=pixels_per_mm,
    )


def calculate_candidate_disease_measurement(
    whole_leaf_mask: np.ndarray,
    candidate_disease_mask: np.ndarray,
    pixels_per_mm: float,
) -> CandidateDiseaseMeasurement:
    """Calculate leaf and candidate-disease areas from aligned boolean masks."""
    whole_leaf_mask = _validate_mask(whole_leaf_mask, label="whole-leaf mask")
    candidate_disease_mask = _validate_mask(
        candidate_disease_mask,
        label="candidate-disease mask",
    )
    if whole_leaf_mask.shape != candidate_disease_mask.shape:
        raise MeasurementError("Whole-leaf and candidate-disease masks must share dimensions.")

    total_leaf_pixels = int(np.count_nonzero(whole_leaf_mask))
    confined_disease = candidate_disease_mask.astype(bool) & whole_leaf_mask.astype(bool)
    candidate_disease_pixels = int(np.count_nonzero(confined_disease))
    return calculate_areas_from_pixel_counts(
        total_leaf_pixels,
        candidate_disease_pixels,
        pixels_per_mm,
    )


def calculate_leaf_area(
    mask: np.ndarray,
    pixels_per_mm: float,
) -> LeafAreaMeasurement:
    """Calculate whole-leaf area from a boolean mask and pixels-per-mm calibration."""
    mask = _validate_mask(mask, label="mask")
    pixels_per_mm = _validate_pixels_per_mm(pixels_per_mm)

    leaf_area_pixels = int(np.count_nonzero(mask))
    if leaf_area_pixels <= 0:
        raise MeasurementError("Mask contains no leaf pixels.")

    pixels_per_inch = pixels_per_mm * MM_PER_INCH
    leaf_area_in2 = leaf_area_pixels / (pixels_per_inch**2)
    leaf_area_mm2 = leaf_area_pixels / (pixels_per_mm**2)

    return LeafAreaMeasurement(
        leaf_area_pixels=leaf_area_pixels,
        leaf_area_in2=leaf_area_in2,
        leaf_area_mm2=leaf_area_mm2,
        pixels_per_inch_used=pixels_per_inch,
        pixels_per_mm_used=pixels_per_mm,
    )


def build_leaf_measurement_record(
    area: LeafAreaMeasurement,
    *,
    method: str = PREVIEW_METHOD,
    settings_version: int = SETTINGS_VERSION,
    source_image_hash: str | None = None,
    calculated_at: str | None = None,
) -> LeafMeasurementRecord:
    """Wrap a successful area calculation in persisted sample metadata."""
    from neopest.metadata import _utc_now_iso

    return LeafMeasurementRecord(
        status="valid",
        leaf_area_pixels=area.leaf_area_pixels,
        leaf_area_in2=area.leaf_area_in2,
        leaf_area_mm2=area.leaf_area_mm2,
        pixels_per_mm_used=area.pixels_per_mm_used,
        pixels_per_inch_used=area.pixels_per_inch_used,
        leaf_mask_method=method,
        leaf_mask_settings_version=settings_version,
        calculated_at=calculated_at or _utc_now_iso(),
        source_image_hash=source_image_hash,
        stale_reason=None,
    )


def build_candidate_disease_measurement_record(
    measurement: CandidateDiseaseMeasurement,
    *,
    leaf_method: str = PREVIEW_METHOD,
    leaf_settings_version: int = SETTINGS_VERSION,
    disease_method: str = DISEASE_PREVIEW_METHOD,
    disease_settings_version: int = DISEASE_SETTINGS_VERSION,
    calculation_method: str = CALCULATION_METHOD_EXPLICIT_V1,
    source_image_hash: str | None = None,
    calculated_at: str | None = None,
) -> CandidateDiseaseMeasurementRecord:
    """Wrap a successful combined calculation in persisted sample metadata."""
    from neopest.metadata import _utc_now_iso

    return CandidateDiseaseMeasurementRecord(
        status="valid",
        total_leaf_pixels=measurement.total_leaf_pixels,
        candidate_disease_pixels=measurement.candidate_disease_pixels,
        total_leaf_area_mm2=measurement.total_leaf_area_mm2,
        total_leaf_area_in2=measurement.total_leaf_area_in2,
        candidate_disease_area_mm2=measurement.candidate_disease_area_mm2,
        candidate_disease_area_in2=measurement.candidate_disease_area_in2,
        candidate_disease_coverage_percent=measurement.candidate_disease_coverage_percent,
        non_candidate_leaf_area_mm2=measurement.non_candidate_leaf_area_mm2,
        non_candidate_leaf_area_in2=measurement.non_candidate_leaf_area_in2,
        pixels_per_mm_used=measurement.pixels_per_mm_used,
        source_image_hash=source_image_hash,
        leaf_mask_method=leaf_method,
        leaf_mask_settings_version=leaf_settings_version,
        disease_mask_method=disease_method,
        disease_mask_settings_version=disease_settings_version,
        calculation_method=calculation_method,
        calculated_at=calculated_at or _utc_now_iso(),
        stale_reason=None,
    )


def leaf_measurement_matches_provenance(
    measurement: LeafMeasurementRecord | None,
    *,
    calibration: CalibrationRecord | None,
    image_hash: str | None,
    method: str = PREVIEW_METHOD,
    settings_version: int = SETTINGS_VERSION,
) -> bool:
    """Return True when a stored measurement matches current scientific inputs."""
    if measurement is None or measurement.status != "valid":
        return False
    if calibration is None:
        return False
    if measurement.pixels_per_mm_used is None:
        return False
    if not math.isclose(
        measurement.pixels_per_mm_used,
        calibration.pixels_per_mm,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    if measurement.source_image_hash is not None and image_hash is not None:
        if measurement.source_image_hash != image_hash:
            return False
    if measurement.leaf_mask_method != method:
        return False
    if measurement.leaf_mask_settings_version != settings_version:
        return False
    return True


def candidate_disease_measurement_matches_provenance(
    measurement: CandidateDiseaseMeasurementRecord | None,
    *,
    calibration: CalibrationRecord | None,
    image_hash: str | None,
    leaf_method: str = PREVIEW_METHOD,
    leaf_settings_version: int = SETTINGS_VERSION,
    disease_method: str = DISEASE_PREVIEW_METHOD,
    disease_settings_version: int = DISEASE_SETTINGS_VERSION,
) -> bool:
    """Return True when a stored Phase 5 result matches current scientific inputs."""
    if measurement is None or measurement.status != "valid":
        return False
    if calibration is None:
        return False
    if measurement.pixels_per_mm_used is None:
        return False
    if not math.isclose(
        measurement.pixels_per_mm_used,
        calibration.pixels_per_mm,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    if measurement.source_image_hash is not None and image_hash is not None:
        if measurement.source_image_hash != image_hash:
            return False
    if measurement.leaf_mask_method != leaf_method:
        return False
    if measurement.leaf_mask_settings_version != leaf_settings_version:
        return False
    if measurement.disease_mask_method != disease_method:
        return False
    if measurement.disease_mask_settings_version != disease_settings_version:
        return False
    return True


DISPLAY_AREA_DECIMAL_PLACES = 3
DISPLAY_COVERAGE_DECIMAL_PLACES = 3

NOT_CALCULATED_DISPLAY = "Not calculated"
CALIBRATION_REQUIRED_DISPLAY = "Calibration required"
RECALCULATION_REQUIRED_DISPLAY = "Recalculation required"
CALCULATION_FAILED_DISPLAY = "Calculation failed"
CURRENT_RESULT_DISPLAY = "Current"


def format_area_in2_display(value: float | None) -> str:
    """Format a square-inch area for researcher-facing display only."""
    if value is None:
        return NOT_CALCULATED_DISPLAY
    return f"{value:.{DISPLAY_AREA_DECIMAL_PLACES}f} in²"


def format_coverage_percent_display(value: float | None) -> str:
    """Format candidate-disease coverage for researcher-facing display only."""
    if value is None:
        return NOT_CALCULATED_DISPLAY
    return f"{value:.{DISPLAY_COVERAGE_DECIMAL_PLACES}f}%"


def phase5_result_is_current(
    sample: SampleRecord,
    *,
    calibration_grid_matches: bool = True,
) -> bool:
    """Return True when the Phase 5 combined result is current and displayable."""
    if not calibration_grid_matches:
        return False
    return candidate_disease_measurement_matches_provenance(
        sample.candidate_disease_measurement,
        calibration=sample.calibration,
        image_hash=sample.image_hash,
        leaf_method=CURRENT_LEAF_MASK_METHOD,
        leaf_settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
        disease_method=CURRENT_DISEASE_MASK_METHOD,
        disease_settings_version=CURRENT_DISEASE_MASK_SETTINGS_VERSION,
    )


def phase5_result_status_display(
    sample: SampleRecord,
    *,
    calculation_failed: bool = False,
    calibration_grid_matches: bool = True,
) -> str:
    """Return the researcher-facing Phase 5 result status label."""
    if phase5_result_is_current(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    ):
        return CURRENT_RESULT_DISPLAY
    if calculation_failed and sample.candidate_disease_measurement is None:
        return CALCULATION_FAILED_DISPLAY
    if not calibration_grid_matches:
        if sample.candidate_disease_measurement is not None:
            return RECALCULATION_REQUIRED_DISPLAY
        return CALIBRATION_REQUIRED_DISPLAY
    if sample.calibration is None:
        measurement = sample.candidate_disease_measurement
        if measurement is not None and measurement.status == "stale":
            return RECALCULATION_REQUIRED_DISPLAY
        return CALIBRATION_REQUIRED_DISPLAY
    measurement = sample.candidate_disease_measurement
    if measurement is None:
        if calculation_failed:
            return CALCULATION_FAILED_DISPLAY
        return NOT_CALCULATED_DISPLAY
    if measurement.status == "stale":
        return RECALCULATION_REQUIRED_DISPLAY
    if not candidate_disease_measurement_matches_provenance(
        measurement,
        calibration=sample.calibration,
        image_hash=sample.image_hash,
        leaf_method=CURRENT_LEAF_MASK_METHOD,
        leaf_settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
        disease_method=CURRENT_DISEASE_MASK_METHOD,
        disease_settings_version=CURRENT_DISEASE_MASK_SETTINGS_VERSION,
    ):
        return RECALCULATION_REQUIRED_DISPLAY
    return NOT_CALCULATED_DISPLAY


def phase5_measurement_row_placeholder(
    sample: SampleRecord,
    *,
    calculation_failed: bool = False,
    calibration_grid_matches: bool = True,
) -> str:
    """Return the non-numeric placeholder shown for all measurement rows."""
    if phase5_result_is_current(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    ):
        raise ValueError("current results must use formatted numeric values")
    if calculation_failed and sample.candidate_disease_measurement is None:
        return CALCULATION_FAILED_DISPLAY
    if not calibration_grid_matches:
        if sample.candidate_disease_measurement is not None:
            return RECALCULATION_REQUIRED_DISPLAY
        return CALIBRATION_REQUIRED_DISPLAY
    if sample.calibration is None:
        return CALIBRATION_REQUIRED_DISPLAY
    measurement = sample.candidate_disease_measurement
    if measurement is None:
        if calculation_failed:
            return CALCULATION_FAILED_DISPLAY
        return NOT_CALCULATED_DISPLAY
    if measurement.status == "stale":
        return RECALCULATION_REQUIRED_DISPLAY
    if not candidate_disease_measurement_matches_provenance(
        measurement,
        calibration=sample.calibration,
        image_hash=sample.image_hash,
        leaf_method=CURRENT_LEAF_MASK_METHOD,
        leaf_settings_version=CURRENT_LEAF_MASK_SETTINGS_VERSION,
        disease_method=CURRENT_DISEASE_MASK_METHOD,
        disease_settings_version=CURRENT_DISEASE_MASK_SETTINGS_VERSION,
    ):
        return RECALCULATION_REQUIRED_DISPLAY
    return NOT_CALCULATED_DISPLAY


def calculation_records_are_consistent(
    leaf_measurement: LeafMeasurementRecord | None,
    candidate_disease_measurement: CandidateDiseaseMeasurementRecord | None,
) -> bool:
    """Return True when Phase 3C and Phase 5 records share one calculation run."""
    if leaf_measurement is None or candidate_disease_measurement is None:
        return False
    if leaf_measurement.status != "valid" or candidate_disease_measurement.status != "valid":
        return False
    if leaf_measurement.leaf_area_pixels != candidate_disease_measurement.total_leaf_pixels:
        return False
    if leaf_measurement.pixels_per_mm_used is None:
        return False
    if candidate_disease_measurement.pixels_per_mm_used is None:
        return False
    if not math.isclose(
        leaf_measurement.pixels_per_mm_used,
        candidate_disease_measurement.pixels_per_mm_used,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    if leaf_measurement.source_image_hash != candidate_disease_measurement.source_image_hash:
        return False
    if leaf_measurement.leaf_mask_method != candidate_disease_measurement.leaf_mask_method:
        return False
    if (
        leaf_measurement.leaf_mask_settings_version
        != candidate_disease_measurement.leaf_mask_settings_version
    ):
        return False
    if leaf_measurement.calculated_at != candidate_disease_measurement.calculated_at:
        return False
    if leaf_measurement.leaf_area_mm2 is None or candidate_disease_measurement.total_leaf_area_mm2 is None:
        return False
    if not math.isclose(
        leaf_measurement.leaf_area_mm2,
        candidate_disease_measurement.total_leaf_area_mm2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if leaf_measurement.leaf_area_in2 is None or candidate_disease_measurement.total_leaf_area_in2 is None:
        return False
    return math.isclose(
        leaf_measurement.leaf_area_in2,
        candidate_disease_measurement.total_leaf_area_in2,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
