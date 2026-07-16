"""Manual one-inch reference annotations for prototype evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.auto_calibration.models import (
    Point,
    pixel_distance,
    pixels_per_inch_from_pixels_per_mm,
    pixels_per_mm_from_one_inch_line,
)

MANUAL_REFERENCE_SCHEMA_VERSION = "1.0"


class ManualReferenceError(Exception):
    """Raised when a manual reference is invalid."""


@dataclass
class ManualReference:
    """Ground-truth one-inch segment for research evaluation (not production calibration)."""

    schema_version: str
    source_image: str
    endpoint_a_px: Point
    endpoint_b_px: Point
    reference_pixels_per_inch: float
    reference_pixels_per_mm: float
    annotation_method: str
    annotator: str
    annotated_at: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_image": self.source_image,
            "endpoint_a_px": [float(self.endpoint_a_px[0]), float(self.endpoint_a_px[1])],
            "endpoint_b_px": [float(self.endpoint_b_px[0]), float(self.endpoint_b_px[1])],
            "reference_pixels_per_inch": float(self.reference_pixels_per_inch),
            "reference_pixels_per_mm": float(self.reference_pixels_per_mm),
            "annotation_method": self.annotation_method,
            "annotator": self.annotator,
            "annotated_at": self.annotated_at,
            "notes": self.notes,
        }

    def to_measurement_dict(self) -> dict[str, Any]:
        """Scientific measurement fields for comparison (excludes provenance metadata).

        Use this (not full ``to_dict``) when testing measurement determinism.
        Fields such as ``annotated_at`` are provenance only and may vary between
        otherwise equivalent references.
        """
        return {
            "source_image": self.source_image,
            "endpoint_a_px": [float(self.endpoint_a_px[0]), float(self.endpoint_a_px[1])],
            "endpoint_b_px": [float(self.endpoint_b_px[0]), float(self.endpoint_b_px[1])],
            "reference_pixels_per_inch": float(self.reference_pixels_per_inch),
            "reference_pixels_per_mm": float(self.reference_pixels_per_mm),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManualReference:
        endpoint_a = _parse_point(data["endpoint_a_px"], label="endpoint_a_px")
        endpoint_b = _parse_point(data["endpoint_b_px"], label="endpoint_b_px")
        reference = cls(
            schema_version=str(data.get("schema_version", MANUAL_REFERENCE_SCHEMA_VERSION)),
            source_image=str(data["source_image"]),
            endpoint_a_px=endpoint_a,
            endpoint_b_px=endpoint_b,
            reference_pixels_per_inch=float(data["reference_pixels_per_inch"]),
            reference_pixels_per_mm=float(data["reference_pixels_per_mm"]),
            annotation_method=str(data.get("annotation_method", "manual_one_inch_line")),
            annotator=str(data.get("annotator", "anonymous")),
            annotated_at=str(data.get("annotated_at", "")),
            notes=str(data.get("notes", "")),
        )
        validate_manual_reference(reference)
        return reference

    @classmethod
    def from_json_file(cls, path: Path | str) -> ManualReference:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def write_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")


def _parse_point(raw: Any, *, label: str) -> Point:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ManualReferenceError(f"{label} must be a two-element coordinate list.")
    x, y = float(raw[0]), float(raw[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ManualReferenceError(f"{label} coordinates must be finite.")
    return (x, y)


def validate_manual_reference(
    reference: ManualReference,
    *,
    image_width_px: int | None = None,
    image_height_px: int | None = None,
) -> None:
    """Reject invalid manual references."""
    if pixel_distance(reference.endpoint_a_px, reference.endpoint_b_px) <= 0.0:
        raise ManualReferenceError("Reference endpoints must not be identical.")

    if reference.reference_pixels_per_mm <= 0.0 or not math.isfinite(
        reference.reference_pixels_per_mm
    ):
        raise ManualReferenceError("reference_pixels_per_mm must be positive and finite.")

    if reference.reference_pixels_per_inch <= 0.0 or not math.isfinite(
        reference.reference_pixels_per_inch
    ):
        raise ManualReferenceError(
            "reference_pixels_per_inch must be positive and finite."
        )

    expected_ppm = pixels_per_mm_from_one_inch_line(
        reference.endpoint_a_px,
        reference.endpoint_b_px,
    )
    expected_ppi = pixels_per_inch_from_pixels_per_mm(expected_ppm)
    tolerance = 1e-6
    if abs(reference.reference_pixels_per_mm - expected_ppm) > tolerance:
        raise ManualReferenceError(
            "reference_pixels_per_mm is inconsistent with endpoint geometry."
        )
    if abs(reference.reference_pixels_per_inch - expected_ppi) > tolerance:
        raise ManualReferenceError(
            "reference_pixels_per_inch is inconsistent with endpoint geometry."
        )

    if image_width_px is not None:
        if image_width_px <= 0:
            raise ManualReferenceError("image_width_px must be positive when supplied.")
        for label, point in (
            ("endpoint_a_px", reference.endpoint_a_px),
            ("endpoint_b_px", reference.endpoint_b_px),
        ):
            if point[0] < 0 or point[0] >= image_width_px:
                raise ManualReferenceError(
                    f"{label} x-coordinate is outside image width {image_width_px}."
                )

    if image_height_px is not None:
        if image_height_px <= 0:
            raise ManualReferenceError("image_height_px must be positive when supplied.")
        for label, point in (
            ("endpoint_a_px", reference.endpoint_a_px),
            ("endpoint_b_px", reference.endpoint_b_px),
        ):
            if point[1] < 0 or point[1] >= image_height_px:
                raise ManualReferenceError(
                    f"{label} y-coordinate is outside image height {image_height_px}."
                )


def build_manual_reference(
    *,
    source_image: str,
    endpoint_a_px: Point,
    endpoint_b_px: Point,
    annotation_method: str = "manual_one_inch_line",
    annotator: str = "anonymous",
    annotated_at: str | None = None,
    notes: str = "",
    image_width_px: int | None = None,
    image_height_px: int | None = None,
) -> ManualReference:
    """Create a validated manual reference from explicit 1-inch endpoints.

    ``annotated_at`` defaults to the current UTC time for human provenance.
    Pass an explicit ISO-8601 timestamp (for example via CLI ``--annotated-at``)
    when byte-identical fixture JSON is required.
    """
    for label, point in (
        ("endpoint_a_px", endpoint_a_px),
        ("endpoint_b_px", endpoint_b_px),
    ):
        if not math.isfinite(point[0]) or not math.isfinite(point[1]):
            raise ManualReferenceError(f"{label} coordinates must be finite.")

    if pixel_distance(endpoint_a_px, endpoint_b_px) <= 0.0:
        raise ManualReferenceError("Reference endpoints must not be identical.")

    try:
        reference_pixels_per_mm = pixels_per_mm_from_one_inch_line(
            endpoint_a_px,
            endpoint_b_px,
        )
    except ValueError as exc:
        raise ManualReferenceError(str(exc)) from exc
    reference_pixels_per_inch = pixels_per_inch_from_pixels_per_mm(
        reference_pixels_per_mm
    )
    timestamp = annotated_at or datetime.now(timezone.utc).isoformat()
    reference = ManualReference(
        schema_version=MANUAL_REFERENCE_SCHEMA_VERSION,
        source_image=source_image,
        endpoint_a_px=endpoint_a_px,
        endpoint_b_px=endpoint_b_px,
        reference_pixels_per_inch=reference_pixels_per_inch,
        reference_pixels_per_mm=reference_pixels_per_mm,
        annotation_method=annotation_method,
        annotator=annotator,
        annotated_at=timestamp,
        notes=notes,
    )
    validate_manual_reference(
        reference,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )
    return reference
