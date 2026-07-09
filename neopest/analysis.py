"""Pure analysis orchestration — no GUI dependencies."""

from __future__ import annotations

from pathlib import Path

from neopest.results import AnalysisStatus, CalibrationStatus, LeafAnalysisResult


def placeholder_analysis_result(sample_id: str) -> LeafAnalysisResult:
    """Return an explicit placeholder result (no scientific computation)."""
    return LeafAnalysisResult(
        sample_id=sample_id,
        leaf_area_mm2=None,
        disease_area_mm2=None,
        disease_fraction=None,
        calibration_status=CalibrationStatus.UNCALIBRATED,
        analysis_status=AnalysisStatus.NOT_RUN,
    )


def run_leaf_analysis(
    sample_id: str,
    image_path: Path,
    *,
    workspace_root: Path | None = None,
) -> LeafAnalysisResult:
    """Run the full leaf disease analysis pipeline for one sample."""
    _ = image_path, workspace_root
    raise NotImplementedError(
        "Leaf disease analysis is not implemented in Phase 0. "
        f"Use placeholder_analysis_result() for sample {sample_id!r}."
    )
