"""Analysis result types for NeoPest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CalibrationStatus(str, Enum):
    """Scale calibration state for a sample."""

    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"
    FAILED = "failed"


class AnalysisStatus(str, Enum):
    """High-level analysis pipeline state."""

    NOT_RUN = "not_run"
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class LeafAnalysisResult:
    """Quantitative leaf disease measurements for one sample image."""

    sample_id: str
    leaf_area_mm2: float | None
    disease_area_mm2: float | None
    disease_fraction: float | None
    calibration_status: CalibrationStatus
    analysis_status: AnalysisStatus

    @property
    def is_placeholder(self) -> bool:
        """True when no real analysis has been performed."""
        return self.analysis_status is AnalysisStatus.NOT_RUN
