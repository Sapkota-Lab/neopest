"""Pure whole-leaf area measurement — no GUI dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neopest.metadata import MM_PER_INCH, CalibrationRecord, LeafMeasurementRecord
from neopest.segmentation import PREVIEW_METHOD, SETTINGS_VERSION

CURRENT_LEAF_MASK_METHOD = PREVIEW_METHOD
CURRENT_LEAF_MASK_SETTINGS_VERSION = SETTINGS_VERSION


class MeasurementError(ValueError):
    """Raised when leaf area cannot be calculated from the given inputs."""


@dataclass(frozen=True)
class LeafAreaMeasurement:
    """Numeric whole-leaf area derived from a binary mask and scale calibration."""

    leaf_area_pixels: int
    leaf_area_in2: float
    leaf_area_mm2: float
    pixels_per_inch_used: float
    pixels_per_mm_used: float


def calculate_leaf_area(
    mask: np.ndarray,
    pixels_per_mm: float,
) -> LeafAreaMeasurement:
    """Calculate whole-leaf area from a boolean mask and pixels-per-mm calibration."""
    if mask is None:
        raise MeasurementError("Mask is required for leaf area calculation.")

    if mask.ndim != 2:
        raise MeasurementError(
            f"Expected a 2-D boolean mask; got shape {getattr(mask, 'shape', None)!r}."
        )

    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise MeasurementError("pixels_per_mm must be a finite value greater than zero.")

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
