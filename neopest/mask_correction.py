"""Binary candidate-disease mask corrections. No GUI dependencies.

Strokes are stored separately from the automatic disease mask. Each stroke is
painted in order, clipped to the whole-leaf mask, so the latest stroke wins
where brushes overlap. Removing pixels excludes them from candidate disease.
It does not record confirmed healthy tissue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neopest.calibration import CURRENT_PIXEL_GRID
from neopest.measurements import (
    CURRENT_DISEASE_MASK_METHOD,
    CURRENT_DISEASE_MASK_SETTINGS_VERSION,
    CURRENT_LEAF_MASK_METHOD,
    CURRENT_LEAF_MASK_SETTINGS_VERSION,
    MeasurementError,
    calculate_areas_from_pixel_counts,
    calculate_candidate_disease_measurement,
    candidate_disease_measurement_matches_provenance,
    format_area_in2_display,
    format_coverage_percent_display,
    phase5_result_is_current,
    phase5_result_status_display,
)
from neopest.metadata import (
    DISEASE_MASK_CORRECTION_STATUS_REVIEW,
    DISEASE_MASK_CORRECTION_STATUS_VALID,
    MASK_STROKE_ADD,
    MASK_STROKE_REMOVE,
    RESEARCHER_CORRECTED_RESULT_LABEL,
    CalibrationRecord,
    DiseaseMaskCorrectionRecord,
    MaskStroke,
    SampleRecord,
    _utc_now_iso,
)

ADD_AREA = MASK_STROKE_ADD
REMOVE_AREA = MASK_STROKE_REMOVE
RESEARCHER_CORRECTED_LABEL = RESEARCHER_CORRECTED_RESULT_LABEL
REVIEW_REQUIRED_LABEL = "Review required"
CORRECTION_REVIEW_MESSAGE = (
    "Saved corrections require review and were not applied to this mask."
)


class MaskCorrectionError(ValueError):
    """Raised when a correction cannot be applied to the given masks."""


@dataclass(frozen=True)
class CorrectionPersistenceUpdate:
    """Correction record to store after an automatic recalculation."""

    record: DiseaseMaskCorrectionRecord | None
    changed: bool


def stroke_interpolation_spacing(radius_px: float) -> float:
    """Return the image-pixel gap that still keeps a circular brush solid."""
    radius = float(radius_px)
    if not math.isfinite(radius) or radius <= 0:
        return 1.0
    return min(1.0, max(radius, 0.25))


def interpolate_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    spacing_px: float,
) -> list[tuple[float, float]]:
    """Return points from just after ``start`` through ``end``.

    Spacing is in image pixels. The start point is omitted so callers can
    append segments without duplicating the shared endpoint.
    """
    spacing = max(float(spacing_px), 1e-6)
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance == 0:
        return []
    steps = int(math.ceil(distance / spacing))
    return [
        (
            start[0] + (end[0] - start[0]) * (step / steps),
            start[1] + (end[1] - start[1]) * (step / steps),
        )
        for step in range(1, steps + 1)
    ]


def _as_bool_mask(mask: np.ndarray, *, label: str) -> np.ndarray:
    if mask is None or getattr(mask, "ndim", None) != 2:
        raise MaskCorrectionError(f"Expected a 2-D {label}.")
    return np.asarray(mask, dtype=bool)


def _paint_disk(
    mask: np.ndarray,
    leaf: np.ndarray,
    center_x: float,
    center_y: float,
    radius_px: float,
    *,
    add: bool,
) -> tuple[int, int, int, int] | None:
    """Set or clear leaf pixels whose centers fall inside the disk."""
    radius = float(radius_px)
    if not math.isfinite(radius) or radius <= 0:
        return None
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        return None
    height, width = mask.shape
    x0 = max(0, math.floor(center_x - radius))
    x1 = min(width, math.ceil(center_x + radius) + 1)
    y0 = max(0, math.floor(center_y - radius))
    y1 = min(height, math.ceil(center_y + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    inside = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius * radius
    region = inside & leaf[y0:y1, x0:x1]
    target = mask[y0:y1, x0:x1]
    if add:
        target[region] = True
    else:
        target[region] = False
    return (x0, y0, x1, y1)


def paint_disks(
    mask: np.ndarray,
    whole_leaf_mask: np.ndarray,
    points: Sequence[tuple[float, float]],
    radius_px: float,
    *,
    add: bool,
) -> tuple[int, int, int, int] | None:
    """Paint disks into ``mask`` in place and return the changed bounds.

    ``mask`` must already be a boolean array. Pixels outside the whole-leaf
    mask are left unchanged. The bounds are ``(x0, y0, x1, y1)`` in image
    pixels, exclusive on the far edges, or ``None`` when nothing is painted.
    """
    if getattr(mask, "dtype", None) != np.bool_:
        raise MaskCorrectionError("paint_disks writes into a boolean mask.")
    leaf = _as_bool_mask(whole_leaf_mask, label="whole-leaf mask")
    if mask.shape != leaf.shape:
        raise MaskCorrectionError(
            "Candidate-disease and whole-leaf masks must share dimensions."
        )
    bounds: tuple[int, int, int, int] | None = None
    for point in points:
        box = _paint_disk(mask, leaf, point[0], point[1], radius_px, add=add)
        if box is None:
            continue
        if bounds is None:
            bounds = box
            continue
        bounds = (
            min(bounds[0], box[0]),
            min(bounds[1], box[1]),
            max(bounds[2], box[2]),
            max(bounds[3], box[3]),
        )
    return bounds


def apply_stroke(
    disease_mask: np.ndarray,
    whole_leaf_mask: np.ndarray,
    stroke: MaskStroke,
) -> np.ndarray:
    """Return ``disease_mask`` with one stroke applied inside the leaf."""
    base = _as_bool_mask(disease_mask, label="candidate-disease mask")
    leaf = _as_bool_mask(whole_leaf_mask, label="whole-leaf mask")
    if base.shape != leaf.shape:
        raise MaskCorrectionError(
            "Candidate-disease and whole-leaf masks must share dimensions."
        )
    if stroke.operation not in {ADD_AREA, REMOVE_AREA}:
        raise MaskCorrectionError(f"Unknown stroke operation: {stroke.operation}")
    corrected = base.copy()
    corrected &= leaf
    add = stroke.operation == ADD_AREA
    for point in stroke.points:
        _paint_disk(corrected, leaf, point[0], point[1], stroke.radius_px, add=add)
    return corrected


def apply_corrections(
    automatic_disease_mask: np.ndarray,
    whole_leaf_mask: np.ndarray,
    strokes: Sequence[MaskStroke],
) -> np.ndarray:
    """Replay strokes onto a copy of the automatic mask.

    Later strokes overwrite earlier ones. Pixels outside the whole-leaf mask
    are never added or removed, and the leaf mask itself is not modified.
    """
    corrected = _as_bool_mask(automatic_disease_mask, label="candidate-disease mask")
    leaf = _as_bool_mask(whole_leaf_mask, label="whole-leaf mask")
    if corrected.shape != leaf.shape:
        raise MaskCorrectionError(
            "Candidate-disease and whole-leaf masks must share dimensions."
        )
    working = corrected.copy()
    working &= leaf
    for stroke in strokes:
        if stroke.operation not in {ADD_AREA, REMOVE_AREA}:
            raise MaskCorrectionError(f"Unknown stroke operation: {stroke.operation}")
        add = stroke.operation == ADD_AREA
        for point in stroke.points:
            _paint_disk(working, leaf, point[0], point[1], stroke.radius_px, add=add)
    return working


class CorrectionHistory:
    """Undo stack of complete strokes. Reset is one undoable step."""

    def __init__(self, strokes: Sequence[MaskStroke] | None = None) -> None:
        self.strokes: tuple[MaskStroke, ...] = tuple(strokes or ())
        self._past: list[tuple[MaskStroke, ...]] = []
        self._future: list[tuple[MaskStroke, ...]] = []

    def commit(self, stroke: MaskStroke) -> None:
        self._past.append(self.strokes)
        self._future.clear()
        self.strokes = (*self.strokes, stroke)

    def undo(self) -> bool:
        if not self._past:
            return False
        self._future.append(self.strokes)
        self.strokes = self._past.pop()
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        self._past.append(self.strokes)
        self.strokes = self._future.pop()
        return True

    def reset(self) -> bool:
        """Drop every stroke and leave that change on the undo stack."""
        if not self.strokes:
            return False
        self._past.append(self.strokes)
        self._future.clear()
        self.strokes = ()
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    @property
    def can_reset(self) -> bool:
        return bool(self.strokes)


def build_disease_mask_correction_record(
    *,
    whole_leaf_mask: np.ndarray,
    automatic_disease_mask: np.ndarray,
    strokes: Sequence[MaskStroke],
    pixels_per_mm: float,
    source_image_hash: str | None,
    leaf_mask_method: str,
    leaf_mask_settings_version: int,
    disease_mask_method: str,
    disease_mask_settings_version: int,
    pixel_grid: str = CURRENT_PIXEL_GRID,
    corrected_at: str | None = None,
) -> DiseaseMaskCorrectionRecord:
    """Measure the corrected candidate-disease mask without altering inputs."""
    corrected = apply_corrections(automatic_disease_mask, whole_leaf_mask, strokes)
    measurement = calculate_candidate_disease_measurement(
        whole_leaf_mask,
        corrected,
        pixels_per_mm,
    )
    return DiseaseMaskCorrectionRecord(
        status=DISEASE_MASK_CORRECTION_STATUS_VALID,
        result_label=RESEARCHER_CORRECTED_RESULT_LABEL,
        strokes=tuple(strokes),
        source_image_hash=source_image_hash,
        pixel_grid=pixel_grid,
        leaf_mask_method=leaf_mask_method,
        leaf_mask_settings_version=leaf_mask_settings_version,
        disease_mask_method=disease_mask_method,
        disease_mask_settings_version=disease_mask_settings_version,
        total_leaf_pixels=measurement.total_leaf_pixels,
        total_leaf_area_mm2=measurement.total_leaf_area_mm2,
        total_leaf_area_in2=measurement.total_leaf_area_in2,
        corrected_candidate_disease_pixels=measurement.candidate_disease_pixels,
        corrected_candidate_disease_area_mm2=measurement.candidate_disease_area_mm2,
        corrected_candidate_disease_area_in2=measurement.candidate_disease_area_in2,
        corrected_candidate_disease_coverage_percent=(
            measurement.candidate_disease_coverage_percent
        ),
        corrected_non_candidate_leaf_area_mm2=measurement.non_candidate_leaf_area_mm2,
        corrected_non_candidate_leaf_area_in2=measurement.non_candidate_leaf_area_in2,
        pixels_per_mm_used=measurement.pixels_per_mm_used,
        corrected_at=corrected_at or _utc_now_iso(),
        review_reason=None,
    )


def provenance_review_reason(
    record: DiseaseMaskCorrectionRecord,
    sample: SampleRecord,
) -> str | None:
    """Return why stored strokes must not be painted onto the current masks."""
    if not record.strokes:
        return None
    if sample.image_hash is None or record.source_image_hash is None:
        return "source_image_changed"
    if sample.image_hash != record.source_image_hash:
        return "source_image_changed"
    if record.pixel_grid != CURRENT_PIXEL_GRID:
        return "pixel_grid_changed"
    calibration = sample.calibration
    if calibration is None or calibration.pixel_grid != record.pixel_grid:
        return "pixel_grid_changed"
    if (
        record.leaf_mask_method != CURRENT_LEAF_MASK_METHOD
        or record.leaf_mask_settings_version != CURRENT_LEAF_MASK_SETTINGS_VERSION
    ):
        return "leaf_segmentation_changed"
    if (
        record.disease_mask_method != CURRENT_DISEASE_MASK_METHOD
        or record.disease_mask_settings_version != CURRENT_DISEASE_MASK_SETTINGS_VERSION
    ):
        return "disease_segmentation_changed"
    measurement = sample.candidate_disease_measurement
    if measurement is None:
        return "automatic_result_missing"
    if (
        measurement.leaf_mask_method != record.leaf_mask_method
        or measurement.leaf_mask_settings_version != record.leaf_mask_settings_version
    ):
        return "leaf_segmentation_changed"
    if (
        measurement.disease_mask_method != record.disease_mask_method
        or measurement.disease_mask_settings_version
        != record.disease_mask_settings_version
    ):
        return "disease_segmentation_changed"
    if measurement.source_image_hash != record.source_image_hash:
        return "source_image_changed"
    return None


def correction_scale_matches(
    record: DiseaseMaskCorrectionRecord,
    sample: SampleRecord,
) -> bool:
    calibration = sample.calibration
    if calibration is None or record.pixels_per_mm_used is None:
        return False
    return math.isclose(
        record.pixels_per_mm_used,
        calibration.pixels_per_mm,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def correction_display_state(
    sample: SampleRecord,
    *,
    calibration_grid_matches: bool = True,
) -> str:
    """Return ``none``, ``applied``, or ``review`` for inspector display."""
    record = sample.disease_mask_correction
    if record is None or not record.strokes:
        return "none"
    if not calibration_grid_matches:
        return "review"
    if record.status == DISEASE_MASK_CORRECTION_STATUS_REVIEW:
        return "review"
    if provenance_review_reason(record, sample):
        return "review"
    if not correction_scale_matches(record, sample):
        return "review"
    return "applied"


def correction_review_message(
    sample: SampleRecord,
    *,
    calibration_grid_matches: bool = True,
) -> str | None:
    if correction_display_state(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    ) != "review":
        return None
    return CORRECTION_REVIEW_MESSAGE


def result_status_for_sample(
    sample: SampleRecord,
    *,
    calculation_failed: bool = False,
    calibration_grid_matches: bool = True,
) -> str:
    """Inspector status, including researcher-corrected and review states."""
    state = correction_display_state(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    )
    if state == "applied":
        return RESEARCHER_CORRECTED_LABEL
    automatic_is_current = phase5_result_is_current(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    )
    if state == "review" and automatic_is_current:
        return REVIEW_REQUIRED_LABEL
    return phase5_result_status_display(
        sample,
        calculation_failed=calculation_failed,
        calibration_grid_matches=calibration_grid_matches,
    )


def format_automatic_candidate_summary(sample: SampleRecord) -> str | None:
    """Short automatic-result line shown while a correction is applied."""
    if sample.disease_mask_correction is None or not sample.disease_mask_correction.strokes:
        return None
    if correction_display_state(sample) != "applied":
        return None
    measurement = sample.candidate_disease_measurement
    if measurement is None or measurement.candidate_disease_area_in2 is None:
        return "Automatic result retained separately"
    numbers = (
        f"{format_area_in2_display(measurement.candidate_disease_area_in2)}"
        f" · {format_coverage_percent_display(measurement.candidate_disease_coverage_percent)}"
    )
    if measurement.status != "valid" or not candidate_disease_measurement_matches_provenance(
        measurement,
        calibration=sample.calibration,
        image_hash=sample.image_hash,
    ):
        return f"Automatic result retained separately: {numbers}"
    return f"Automatic: {numbers}"


def can_start_disease_mask_correction(
    sample: SampleRecord,
    *,
    image_available: bool,
    calibration_grid_matches: bool,
) -> bool:
    """True after an automatic result exists on the current image grid."""
    if not image_available or not calibration_grid_matches or sample.calibration is None:
        return False
    if phase5_result_is_current(
        sample,
        calibration_grid_matches=calibration_grid_matches,
    ):
        return True
    return (
        correction_display_state(
            sample,
            calibration_grid_matches=calibration_grid_matches,
        )
        == "applied"
    )


def refresh_correction_for_calibration(sample: SampleRecord) -> None:
    """Rescale corrected areas when only the calibration scale changed.

    Strokes stay stored. A grid or mask-provenance change marks the correction
    for review instead of painting it onto a different mask. The automatic
    measurement is left for the caller to mark stale.
    """
    record = sample.disease_mask_correction
    if record is None or not record.strokes:
        return
    reason = provenance_review_reason(record, sample)
    if reason:
        sample.disease_mask_correction = record.mark_review_required(reason)
        return
    calibration = sample.calibration
    if calibration is None or record.total_leaf_pixels is None:
        sample.disease_mask_correction = record.mark_review_required(
            "area_recalculation_failed"
        )
        return
    if record.corrected_candidate_disease_pixels is None:
        sample.disease_mask_correction = record.mark_review_required(
            "area_recalculation_failed"
        )
        return
    try:
        measurement = calculate_areas_from_pixel_counts(
            int(record.total_leaf_pixels),
            int(record.corrected_candidate_disease_pixels),
            calibration.pixels_per_mm,
        )
    except MeasurementError:
        sample.disease_mask_correction = record.mark_review_required(
            "area_recalculation_failed"
        )
        return
    sample.disease_mask_correction = record.with_rescaled_areas(
        total_leaf_pixels=measurement.total_leaf_pixels,
        total_leaf_area_mm2=measurement.total_leaf_area_mm2,
        total_leaf_area_in2=measurement.total_leaf_area_in2,
        corrected_candidate_disease_pixels=measurement.candidate_disease_pixels,
        corrected_candidate_disease_area_mm2=measurement.candidate_disease_area_mm2,
        corrected_candidate_disease_area_in2=measurement.candidate_disease_area_in2,
        corrected_candidate_disease_coverage_percent=(
            measurement.candidate_disease_coverage_percent
        ),
        corrected_non_candidate_leaf_area_mm2=measurement.non_candidate_leaf_area_mm2,
        corrected_non_candidate_leaf_area_in2=measurement.non_candidate_leaf_area_in2,
        pixels_per_mm_used=measurement.pixels_per_mm_used,
    )


def _correction_measurements_match(
    existing: DiseaseMaskCorrectionRecord,
    rebuilt: DiseaseMaskCorrectionRecord,
) -> bool:
    if existing.strokes != rebuilt.strokes:
        return False
    if existing.total_leaf_pixels != rebuilt.total_leaf_pixels:
        return False
    if (
        existing.corrected_candidate_disease_pixels
        != rebuilt.corrected_candidate_disease_pixels
    ):
        return False
    if existing.pixels_per_mm_used is None or rebuilt.pixels_per_mm_used is None:
        return False
    return math.isclose(
        existing.pixels_per_mm_used,
        rebuilt.pixels_per_mm_used,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def reconcile_correction_after_calculation(
    sample: SampleRecord,
    whole_leaf_mask: np.ndarray,
    automatic_disease_mask: np.ndarray,
    calibration: CalibrationRecord,
) -> CorrectionPersistenceUpdate:
    """Reapply stored strokes when the new automatic result has the same provenance.

    A segmentation or image change keeps the strokes and marks them for review.
    This does not change the automatic disease-mask version.
    """
    existing = sample.disease_mask_correction
    if existing is None or not existing.strokes:
        return CorrectionPersistenceUpdate(existing, False)
    reason = provenance_review_reason(existing, sample)
    if reason:
        reviewed = existing.mark_review_required(reason)
        changed = (
            reviewed.status != existing.status or reviewed.review_reason != existing.review_reason
        )
        return CorrectionPersistenceUpdate(reviewed if changed else existing, changed)
    rebuilt = build_disease_mask_correction_record(
        whole_leaf_mask=whole_leaf_mask,
        automatic_disease_mask=automatic_disease_mask,
        strokes=existing.strokes,
        pixels_per_mm=calibration.pixels_per_mm,
        source_image_hash=sample.image_hash,
        leaf_mask_method=existing.leaf_mask_method or CURRENT_LEAF_MASK_METHOD,
        leaf_mask_settings_version=int(
            existing.leaf_mask_settings_version
            if existing.leaf_mask_settings_version is not None
            else CURRENT_LEAF_MASK_SETTINGS_VERSION
        ),
        disease_mask_method=existing.disease_mask_method or CURRENT_DISEASE_MASK_METHOD,
        disease_mask_settings_version=int(
            existing.disease_mask_settings_version
            if existing.disease_mask_settings_version is not None
            else CURRENT_DISEASE_MASK_SETTINGS_VERSION
        ),
        pixel_grid=existing.pixel_grid or CURRENT_PIXEL_GRID,
    )
    if _correction_measurements_match(existing, rebuilt):
        return CorrectionPersistenceUpdate(existing, False)
    return CorrectionPersistenceUpdate(rebuilt, True)
