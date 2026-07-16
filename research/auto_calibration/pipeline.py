"""Placeholder auto-calibration research pipeline (AC-0)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from research.auto_calibration.config import AutoCalibrationConfig, DEFAULT_CONFIG
from research.auto_calibration.models import (
    AutoCalibrationResult,
    ComponentScores,
    FailureReason,
    ResultStatus,
    RuntimeMetadata,
)


class PipelineError(Exception):
    """Raised when the pipeline cannot process an image."""


def _read_image_dimensions(image_path: Path) -> tuple[np.ndarray, int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"Unsupported or unreadable image: {image_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise PipelineError(f"Invalid image dimensions for: {image_path}")
    return image, width, height


def run_auto_calibration_proposal(
    image_path: Path | str,
    *,
    config: AutoCalibrationConfig = DEFAULT_CONFIG,
    capture_runtime: bool = True,
) -> AutoCalibrationResult:
    """Run the current research pipeline and return a calibration *proposal*.

    The AC-0 placeholder does not detect rulers. It fails closed with
    ``NO_SUGGESTION`` and an explicit failure reason.
    """
    path = Path(image_path)
    if path.suffix.lower() not in config.supported_image_extensions:
        return _build_unsupported_image_result(path, config)

    started = time.perf_counter()
    try:
        _image, width, height = _read_image_dimensions(path)
    except PipelineError as exc:
        return _build_error_result(path, config, message=str(exc))

    diagnostics = {
        "pipeline_stage": "ac0_placeholder",
        "message": (
            "Ruler localization is not implemented in AC-0. "
            "No calibration proposal was generated."
        ),
        "warning_codes": ["NOT_IMPLEMENTED"],
        "ruler_roi": None,
        "ruler_axis": None,
        "homography": None,
        "raw_tick_candidates": [],
        "accepted_tick_candidates": [],
        "rejected_tick_candidates": [],
        "candidate_inch_intervals": [],
        "residuals": [],
    }

    elapsed = time.perf_counter() - started if capture_runtime else None
    return AutoCalibrationResult(
        schema_version=config.schema_version,
        algorithm_name=config.algorithm_name,
        algorithm_version=config.algorithm_version,
        source_image=str(path),
        image_width_px=width,
        image_height_px=height,
        status=ResultStatus.NO_SUGGESTION,
        failure_reason=FailureReason.NO_RULER_CANDIDATE,
        proposed_endpoints_px=None,
        proposed_pixels_per_inch=None,
        proposed_pixels_per_mm=None,
        overall_confidence=None,
        component_scores=ComponentScores(),
        diagnostics=diagnostics,
        runtime=RuntimeMetadata(elapsed_seconds=elapsed),
    )


def list_supported_images(
    directory: Path | str,
    *,
    config: AutoCalibrationConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """Return supported images in deterministic sorted order."""
    root = Path(directory)
    if not root.is_dir():
        raise PipelineError(f"Image directory does not exist: {root}")

    extensions = {ext.lower() for ext in config.supported_image_extensions}
    images = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(images, key=lambda item: item.name.lower())


def run_batch_auto_calibration_proposals(
    directory: Path | str,
    *,
    config: AutoCalibrationConfig = DEFAULT_CONFIG,
    capture_runtime: bool = True,
) -> list[AutoCalibrationResult]:
    """Run the placeholder pipeline on all supported images in sorted order."""
    results: list[AutoCalibrationResult] = []
    for image_path in list_supported_images(directory, config=config):
        results.append(
            run_auto_calibration_proposal(
                image_path,
                config=config,
                capture_runtime=capture_runtime,
            )
        )
    return results


def _build_unsupported_image_result(
    path: Path,
    config: AutoCalibrationConfig,
) -> AutoCalibrationResult:
    return AutoCalibrationResult(
        schema_version=config.schema_version,
        algorithm_name=config.algorithm_name,
        algorithm_version=config.algorithm_version,
        source_image=str(path),
        image_width_px=0,
        image_height_px=0,
        status=ResultStatus.ERROR,
        failure_reason=FailureReason.UNSUPPORTED_IMAGE,
        proposed_endpoints_px=None,
        proposed_pixels_per_inch=None,
        proposed_pixels_per_mm=None,
        overall_confidence=None,
        component_scores=ComponentScores(),
        diagnostics={
            "message": f"Unsupported file extension: {path.suffix}",
            "warning_codes": [],
        },
        runtime=RuntimeMetadata(),
    )


def _build_error_result(
    path: Path,
    config: AutoCalibrationConfig,
    *,
    message: str,
) -> AutoCalibrationResult:
    return AutoCalibrationResult(
        schema_version=config.schema_version,
        algorithm_name=config.algorithm_name,
        algorithm_version=config.algorithm_version,
        source_image=str(path),
        image_width_px=0,
        image_height_px=0,
        status=ResultStatus.ERROR,
        failure_reason=FailureReason.INTERNAL_ERROR,
        proposed_endpoints_px=None,
        proposed_pixels_per_inch=None,
        proposed_pixels_per_mm=None,
        overall_confidence=None,
        component_scores=ComponentScores(),
        diagnostics={"message": message, "warning_codes": []},
        runtime=RuntimeMetadata(),
    )
