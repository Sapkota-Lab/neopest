"""Typed, JSON-serializable data contracts for auto-calibration research."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from research.auto_calibration.config import MM_PER_INCH

Point = tuple[float, float]


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_SUGGESTION = "NO_SUGGESTION"
    ERROR = "ERROR"


class FailureReason(str, Enum):
    NONE = "NONE"
    NO_RULER_CANDIDATE = "NO_RULER_CANDIDATE"
    MULTIPLE_AMBIGUOUS_RULERS = "MULTIPLE_AMBIGUOUS_RULERS"
    LOW_RULER_CONFIDENCE = "LOW_RULER_CONFIDENCE"
    INSUFFICIENT_TICKS = "INSUFFICIENT_TICKS"
    INCONSISTENT_TICK_SPACING = "INCONSISTENT_TICK_SPACING"
    RULER_TRUNCATED = "RULER_TRUNCATED"
    EXCESSIVE_PERSPECTIVE = "EXCESSIVE_PERSPECTIVE"
    LOW_OVERALL_CONFIDENCE = "LOW_OVERALL_CONFIDENCE"
    UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


COMPONENT_SCORE_KEYS = (
    "ruler_detection",
    "perspective_quality",
    "tick_detection",
    "spacing_consistency",
    "interval_agreement",
    "endpoint_quality",
)


def pixel_distance(point_a: Point, point_b: Point) -> float:
    """Euclidean distance between two image points in pixels."""
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def pixels_per_mm_from_one_inch_line(endpoint_a: Point, endpoint_b: Point) -> float:
    """Derive pixels-per-millimeter from a 1-inch reference segment."""
    distance_px = pixel_distance(endpoint_a, endpoint_b)
    if distance_px <= 0.0 or not math.isfinite(distance_px):
        raise ValueError("Reference endpoints must define a positive finite segment.")
    return distance_px / MM_PER_INCH


def pixels_per_inch_from_pixels_per_mm(pixels_per_mm: float) -> float:
    """Convert pixels-per-millimeter to pixels-per-inch."""
    if pixels_per_mm <= 0.0 or not math.isfinite(pixels_per_mm):
        raise ValueError("pixels_per_mm must be a positive finite number.")
    return pixels_per_mm * MM_PER_INCH


def pixels_per_mm_from_pixels_per_inch(pixels_per_inch: float) -> float:
    """Convert pixels-per-inch to pixels-per-millimeter."""
    if pixels_per_inch <= 0.0 or not math.isfinite(pixels_per_inch):
        raise ValueError("pixels_per_inch must be a positive finite number.")
    return pixels_per_inch / MM_PER_INCH


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _normalize_point(point: SequenceLike) -> Point:
    if len(point) != 2:
        raise ValueError("Point must have exactly two coordinates.")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Point coordinates must be finite.")
    return (x, y)


# Avoid importing from typing for Sequence in older patterns - use Any for from_dict
SequenceLike = tuple[float, float] | list[float]


@dataclass
class ComponentScores:
    """Per-stage confidence components.

    Unavailable components are stored as ``null`` in JSON. Overall confidence is
    not scientifically meaningful until a validated combination rule is defined
    in a later research phase.
    """

    ruler_detection: float | None = None
    perspective_quality: float | None = None
    tick_detection: float | None = None
    spacing_consistency: float | None = None
    interval_agreement: float | None = None
    endpoint_quality: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "ruler_detection": _json_float(self.ruler_detection),
            "perspective_quality": _json_float(self.perspective_quality),
            "tick_detection": _json_float(self.tick_detection),
            "spacing_consistency": _json_float(self.spacing_consistency),
            "interval_agreement": _json_float(self.interval_agreement),
            "endpoint_quality": _json_float(self.endpoint_quality),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ComponentScores:
        if not data:
            return cls()
        return cls(
            ruler_detection=_optional_score(data.get("ruler_detection")),
            perspective_quality=_optional_score(data.get("perspective_quality")),
            tick_detection=_optional_score(data.get("tick_detection")),
            spacing_consistency=_optional_score(data.get("spacing_consistency")),
            interval_agreement=_optional_score(data.get("interval_agreement")),
            endpoint_quality=_optional_score(data.get("endpoint_quality")),
        )


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return score


@dataclass
class RuntimeMetadata:
    """Non-canonical runtime information excluded from determinism comparisons."""

    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "elapsed_seconds": _json_float(self.elapsed_seconds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuntimeMetadata:
        if not data:
            return cls()
        elapsed = data.get("elapsed_seconds")
        return cls(elapsed_seconds=None if elapsed is None else float(elapsed))


@dataclass
class AutoCalibrationResult:
    """Proposal output from the research pipeline (not an applied calibration)."""

    schema_version: str
    algorithm_name: str
    algorithm_version: str
    source_image: str
    image_width_px: int
    image_height_px: int
    status: ResultStatus
    failure_reason: FailureReason
    proposed_endpoints_px: tuple[Point, Point] | None
    proposed_pixels_per_inch: float | None
    proposed_pixels_per_mm: float | None
    overall_confidence: float | None
    component_scores: ComponentScores = field(default_factory=ComponentScores)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeMetadata = field(default_factory=RuntimeMetadata)

    def to_dict(self, *, include_runtime: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_version": self.algorithm_version,
            "source_image": self.source_image,
            "image_width_px": int(self.image_width_px),
            "image_height_px": int(self.image_height_px),
            "status": self.status.value,
            "failure_reason": self.failure_reason.value,
            "proposed_endpoints_px": (
                None
                if self.proposed_endpoints_px is None
                else [
                    list(self.proposed_endpoints_px[0]),
                    list(self.proposed_endpoints_px[1]),
                ]
            ),
            "proposed_pixels_per_inch": _json_float(self.proposed_pixels_per_inch),
            "proposed_pixels_per_mm": _json_float(self.proposed_pixels_per_mm),
            "overall_confidence": _json_float(self.overall_confidence),
            "component_scores": self.component_scores.to_dict(),
            "diagnostics": _sanitize_json_value(self.diagnostics),
        }
        if include_runtime:
            payload["runtime"] = self.runtime.to_dict()
        return payload

    def to_canonical_dict(self) -> dict[str, Any]:
        """Deterministic payload for equality comparisons (excludes runtime)."""
        return self.to_dict(include_runtime=False)

    def to_json(self, *, include_runtime: bool = True, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_runtime=include_runtime),
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoCalibrationResult:
        endpoints_raw = data.get("proposed_endpoints_px")
        endpoints: tuple[Point, Point] | None
        if endpoints_raw is None:
            endpoints = None
        else:
            endpoints = (
                _normalize_point(endpoints_raw[0]),
                _normalize_point(endpoints_raw[1]),
            )

        return cls(
            schema_version=str(data["schema_version"]),
            algorithm_name=str(data["algorithm_name"]),
            algorithm_version=str(data["algorithm_version"]),
            source_image=str(data["source_image"]),
            image_width_px=int(data["image_width_px"]),
            image_height_px=int(data["image_height_px"]),
            status=ResultStatus(str(data["status"])),
            failure_reason=FailureReason(str(data["failure_reason"])),
            proposed_endpoints_px=endpoints,
            proposed_pixels_per_inch=_optional_positive_float(
                data.get("proposed_pixels_per_inch")
            ),
            proposed_pixels_per_mm=_optional_positive_float(
                data.get("proposed_pixels_per_mm")
            ),
            overall_confidence=_optional_score(data.get("overall_confidence")),
            component_scores=ComponentScores.from_dict(data.get("component_scores")),
            diagnostics=dict(data.get("diagnostics") or {}),
            runtime=RuntimeMetadata.from_dict(data.get("runtime")),
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> AutoCalibrationResult:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def write_json(self, path: Path | str, *, include_runtime: bool = True) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.to_json(include_runtime=include_runtime),
            encoding="utf-8",
        )


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _sanitize_json_value(value: Any) -> Any:
    """Recursively coerce values to JSON-compatible primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _json_float(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _sanitize_json_value(value.tolist())
    return str(value)
