"""Ruler-based scale calibration (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass

from neopest.results import CalibrationStatus


@dataclass(frozen=True)
class CalibrationResult:
    """Microns-per-pixel scale derived from a visible ruler."""

    microns_per_pixel: float | None
    status: CalibrationStatus
    message: str = ""


def calibrate_from_ruler(*_args, **_kwargs) -> CalibrationResult:
    """Estimate scale from a ruler visible beside the petri dish."""
    raise NotImplementedError("Ruler calibration is not implemented in Phase 0.")
