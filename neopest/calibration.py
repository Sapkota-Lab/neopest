"""Manual scale calibration math and record construction."""

from __future__ import annotations

import math
from typing import Sequence

from neopest.metadata import (
    CALIBRATION_METHOD_MANUAL,
    DISTANCE_UNIT_CM,
    DISTANCE_UNIT_IN,
    DISTANCE_UNIT_MM,
    MM_PER_INCH,
    SUPPORTED_DISTANCE_UNITS,
    CalibrationRecord,
)
from neopest.results import CalibrationStatus

Point = tuple[float, float]

ONE_INCH_MM = MM_PER_INCH
KNOWN_ONE_INCH_DISTANCE_VALUE = 1.0
KNOWN_ONE_INCH_DISTANCE_UNIT = DISTANCE_UNIT_IN


class CalibrationError(Exception):
    """Raised when calibration input is invalid."""


def convert_distance_to_mm(value: float, unit: str) -> float:
    """Convert a known physical distance to millimeters."""
    if value <= 0:
        raise CalibrationError("Known distance must be greater than zero.")

    normalized = unit.strip().lower()
    if normalized == DISTANCE_UNIT_MM:
        return value
    if normalized == DISTANCE_UNIT_CM:
        return value * 10.0
    if normalized in {DISTANCE_UNIT_IN, "inch", "inches"}:
        return value * MM_PER_INCH

    supported = ", ".join(SUPPORTED_DISTANCE_UNITS)
    raise CalibrationError(f"Unsupported distance unit {unit!r}. Use: {supported}.")


def pixel_distance(point_1: Point, point_2: Point) -> float:
    """Return the Euclidean distance between two image points in pixels."""
    return math.hypot(point_2[0] - point_1[0], point_2[1] - point_1[1])


def calculate_pixels_per_mm(
    point_1: Point,
    point_2: Point,
    known_distance_mm: float,
) -> float:
    """Compute scale as pixels per millimeter from two points and a known distance."""
    if known_distance_mm <= 0:
        raise CalibrationError("Known distance must be greater than zero millimeters.")

    distance_px = pixel_distance(point_1, point_2)
    if distance_px <= 0:
        raise CalibrationError("Calibration points must not be identical.")

    return distance_px / known_distance_mm


def build_manual_calibration_record(
    point_1: Point,
    point_2: Point,
    known_distance_value: float,
    known_distance_unit: str,
    *,
    notes: str = "",
) -> CalibrationRecord:
    """Build a manual calibration record from two image points and a known distance."""
    known_distance_mm = convert_distance_to_mm(known_distance_value, known_distance_unit)
    pixels_per_mm = calculate_pixels_per_mm(point_1, point_2, known_distance_mm)
    return CalibrationRecord(
        pixels_per_mm=pixels_per_mm,
        known_distance_mm=known_distance_mm,
        known_distance_value=known_distance_value,
        known_distance_unit=known_distance_unit.strip().lower(),
        point_1_x=point_1[0],
        point_1_y=point_1[1],
        point_2_x=point_2[0],
        point_2_y=point_2[1],
        calibration_method=CALIBRATION_METHOD_MANUAL,
        notes=notes,
    )


def build_one_inch_manual_calibration_record(
    point_1: Point,
    point_2: Point,
    *,
    notes: str = "",
) -> CalibrationRecord:
    """Build a manual calibration record for two ruler points exactly 1 inch apart."""
    return build_manual_calibration_record(
        point_1,
        point_2,
        KNOWN_ONE_INCH_DISTANCE_VALUE,
        KNOWN_ONE_INCH_DISTANCE_UNIT,
        notes=notes,
    )


def calibration_status_for_record(record: CalibrationRecord | None) -> CalibrationStatus:
    """Return the sample calibration status implied by a calibration record."""
    if record is None:
        return CalibrationStatus.UNCALIBRATED
    return CalibrationStatus.MANUAL


def calibrate_from_ruler(*_args, **_kwargs) -> None:
    """Automatic ruler detection is deferred to a future phase."""
    raise NotImplementedError(
        "Automatic ruler calibration is not implemented. Use manual calibration."
    )


def validate_calibration_points(points: Sequence[Point]) -> None:
    """Ensure exactly two distinct calibration points were provided."""
    if len(points) != 2:
        raise CalibrationError("Manual calibration requires exactly two image points.")
