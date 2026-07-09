"""Result export helpers (scaffold)."""

from __future__ import annotations

from pathlib import Path

from neopest.results import LeafAnalysisResult


def export_results_csv(results: list[LeafAnalysisResult], destination: Path) -> Path:
    """Write analysis results to a CSV file."""
    _ = results
    raise NotImplementedError("CSV export is not implemented in Phase 0.")
