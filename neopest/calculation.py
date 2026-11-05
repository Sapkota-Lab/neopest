"""Explicit scientific calculation orchestration — no GUI dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from neopest.disease_segmentation import (
    DISEASE_PREVIEW_METHOD,
    DISEASE_SETTINGS_VERSION,
    DiseaseMaskPreview,
    DiseaseMaskPreviewStatus,
    enforce_mask_invariants,
    segment_disease_preview,
)
from neopest.measurements import (
    MeasurementError,
    build_candidate_disease_measurement_record,
    build_leaf_measurement_record,
    calculate_candidate_disease_measurement,
    calculate_leaf_area,
)
from neopest.metadata import CalibrationRecord, CandidateDiseaseMeasurementRecord, LeafMeasurementRecord
from neopest.segmentation import (
    PREVIEW_METHOD,
    SETTINGS_VERSION,
    LeafMaskPreview,
    LeafMaskPreviewStatus,
    segment_leaf_preview,
)


class CalculationError(Exception):
    """Raised when explicit scientific calculation cannot complete safely."""


@dataclass(frozen=True)
class ExplicitCalculationOutput:
    """Coherent explicit calculation result with shared provenance."""

    leaf_measurement: LeafMeasurementRecord
    candidate_disease_measurement: CandidateDiseaseMeasurementRecord
    leaf_preview: LeafMaskPreview
    disease_preview: DiseaseMaskPreview


def run_explicit_calculation(
    image_array: np.ndarray | None,
    calibration: CalibrationRecord,
    *,
    source_image_hash: str | None = None,
    leaf_preview: LeafMaskPreview | None = None,
    disease_preview: DiseaseMaskPreview | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ExplicitCalculationOutput:
    """Run a fresh leaf and candidate-disease calculation from one source image.

    Stored previews are reused only when the caller supplies masks that already
    match the current segmentation. Measurements are still calculated from those
    full-resolution masks.
    """
    if calibration is None:
        raise CalculationError("Manual calibration is required before calculating results.")

    if leaf_preview is None:
        if image_array is None:
            raise CalculationError("Source image is required.")
        if on_stage is not None:
            on_stage("Finding leaf")
        leaf_preview = segment_leaf_preview(image_array)
    if leaf_preview.status is not LeafMaskPreviewStatus.SUCCESS or leaf_preview.mask is None:
        message = (
            leaf_preview.warnings[0]
            if leaf_preview.warnings
            else "Whole-leaf segmentation failed."
        )
        raise CalculationError(message)

    leaf_mask = leaf_preview.mask
    if not np.any(leaf_mask):
        raise CalculationError("Whole-leaf mask is empty.")

    leaf_settings_version = int(
        leaf_preview.settings_snapshot.get("settings_version", SETTINGS_VERSION)
    )
    leaf_method = leaf_preview.method or PREVIEW_METHOD

    if disease_preview is None:
        if image_array is None:
            raise CalculationError("Source image is required.")
        if on_stage is not None:
            on_stage("Identifying candidate disease")
        disease_preview = segment_disease_preview(
            image_array,
            leaf_mask,
            whole_leaf_method=leaf_method,
            whole_leaf_settings_version=leaf_settings_version,
        )
    if (
        disease_preview.status is not DiseaseMaskPreviewStatus.READY
        or disease_preview.disease_mask is None
        or disease_preview.healthy_tissue_mask is None
    ):
        message = (
            disease_preview.warnings[0]
            if disease_preview.warnings
            else "Candidate-disease segmentation failed."
        )
        raise CalculationError(message)

    disease_mask, _healthy_mask = enforce_mask_invariants(
        leaf_mask,
        disease_preview.disease_mask,
    )
    disease_settings_version = int(
        disease_preview.settings_snapshot.get(
            "settings_version",
            DISEASE_SETTINGS_VERSION,
        )
    )
    disease_method = disease_preview.method or DISEASE_PREVIEW_METHOD

    if on_stage is not None:
        on_stage("Updating measurements")
    try:
        combined = calculate_candidate_disease_measurement(
            leaf_mask,
            disease_mask,
            calibration.pixels_per_mm,
        )
        leaf_area = calculate_leaf_area(leaf_mask, calibration.pixels_per_mm)
    except MeasurementError as exc:
        raise CalculationError(str(exc)) from exc

    if combined.total_leaf_pixels != leaf_area.leaf_area_pixels:
        raise CalculationError(
            "Leaf pixel counts disagree between combined and whole-leaf calculations."
        )

    calculated_at = None
    leaf_record = build_leaf_measurement_record(
        leaf_area,
        method=leaf_method,
        settings_version=leaf_settings_version,
        source_image_hash=source_image_hash,
        calculated_at=calculated_at,
    )
    calculated_at = leaf_record.calculated_at
    disease_record = build_candidate_disease_measurement_record(
        combined,
        leaf_method=leaf_method,
        leaf_settings_version=leaf_settings_version,
        disease_method=disease_method,
        disease_settings_version=disease_settings_version,
        source_image_hash=source_image_hash,
        calculated_at=calculated_at,
    )

    refreshed_disease_preview = DiseaseMaskPreview(
        disease_mask=disease_mask,
        healthy_tissue_mask=_healthy_mask,
        status=DiseaseMaskPreviewStatus.READY,
        warnings=list(disease_preview.warnings),
        method=disease_method,
        settings_snapshot=dict(disease_preview.settings_snapshot),
        whole_leaf_method=leaf_method,
        whole_leaf_settings_version=leaf_settings_version,
    )

    return ExplicitCalculationOutput(
        leaf_measurement=leaf_record,
        candidate_disease_measurement=disease_record,
        leaf_preview=leaf_preview,
        disease_preview=refreshed_disease_preview,
    )
