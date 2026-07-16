"""Diagnostic overlay rendering for auto-calibration research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research.auto_calibration.models import AutoCalibrationResult, ResultStatus


def render_proposal_overlay(
    image_path: Path | str,
    result: AutoCalibrationResult,
    *,
    diagnostic_geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a diagnostic overlay without affecting numerical pipeline output."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image for overlay: {image_path}")

    overlay = image.copy()
    geometry = diagnostic_geometry or {}

    _draw_optional_segment(
        overlay,
        geometry.get("reference_endpoints_px"),
        color=(0, 255, 255),
        label="reference",
    )
    _draw_optional_segment(
        overlay,
        geometry.get("proposal_endpoints_px"),
        color=(0, 200, 0),
        label="proposal",
    )
    _draw_optional_points(
        overlay,
        geometry.get("tick_candidates"),
        color=(255, 128, 0),
    )
    _draw_optional_points(
        overlay,
        geometry.get("rejected_tick_candidates"),
        color=(0, 0, 255),
    )

    roi = geometry.get("ruler_roi")
    if isinstance(roi, dict):
        x = int(roi.get("x", 0))
        y = int(roi.get("y", 0))
        w = int(roi.get("width", 0))
        h = int(roi.get("height", 0))
        if w > 0 and h > 0:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 2)

    summary_lines = [
        f"status: {result.status.value}",
        f"failure_reason: {result.failure_reason.value}",
    ]
    warning_codes = result.diagnostics.get("warning_codes") or []
    if warning_codes:
        summary_lines.append(f"warnings: {', '.join(str(code) for code in warning_codes)}")

    _draw_text_block(overlay, summary_lines)
    return overlay


def write_proposal_overlay(
    image_path: Path | str,
    result: AutoCalibrationResult,
    output_path: Path | str,
    *,
    diagnostic_geometry: dict[str, Any] | None = None,
) -> Path:
    """Write an overlay image to disk."""
    overlay = render_proposal_overlay(
        image_path,
        result,
        diagnostic_geometry=diagnostic_geometry,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), overlay):
        raise RuntimeError(f"Failed to write overlay image: {target}")
    return target


def _draw_optional_segment(
    image: np.ndarray,
    segment: Any,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if not isinstance(segment, (list, tuple)) or len(segment) != 2:
        return
    try:
        p1 = (int(round(float(segment[0][0]))), int(round(float(segment[0][1]))))
        p2 = (int(round(float(segment[1][0]))), int(round(float(segment[1][1]))))
    except (TypeError, ValueError, IndexError):
        return
    cv2.line(image, p1, p2, color, 2, lineType=cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        p1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_optional_points(
    image: np.ndarray,
    points: Any,
    *,
    color: tuple[int, int, int],
) -> None:
    if not isinstance(points, list):
        return
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(image, center, 4, color, -1, lineType=cv2.LINE_AA)


def _draw_text_block(image: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        y += 22
