"""Workspace and sample metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from neopest.results import AnalysisStatus, CalibrationStatus

SCHEMA_VERSION = 1
WORKSPACE_JSON_FILENAME = "neopest_workspace.json"
DEFAULT_CONDITION_GROUP_NAME = "Ungrouped"
CALIBRATION_METHOD_MANUAL = "manual"
DISTANCE_UNIT_MM = "mm"
DISTANCE_UNIT_CM = "cm"
DISTANCE_UNIT_IN = "in"
SUPPORTED_DISTANCE_UNITS = (DISTANCE_UNIT_MM, DISTANCE_UNIT_CM, DISTANCE_UNIT_IN)
MM_PER_INCH = 25.4
MM2_PER_IN2 = 645.16
CALCULATION_METHOD_EXPLICIT_V1 = "explicit_leaf_and_candidate_disease_v1"
MASK_STROKE_ADD = "add"
MASK_STROKE_REMOVE = "remove"
DISEASE_MASK_CORRECTION_STATUS_VALID = "valid"
DISEASE_MASK_CORRECTION_STATUS_REVIEW = "review_required"
RESEARCHER_CORRECTED_RESULT_LABEL = "Researcher-corrected"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_workspace_id() -> str:
    """Return a stable, opaque workspace identifier."""
    return f"workspace_{uuid4().hex[:12]}"


def new_condition_group_id() -> str:
    """Return a stable, opaque condition group identifier."""
    return f"condition_group_{uuid4().hex[:12]}"


def new_sample_id() -> str:
    """Return a stable, opaque sample identifier."""
    return f"sample_{uuid4().hex[:12]}"


@dataclass
class ConditionGroupRecord:
    """A genetic or treatment grouping that owns one or more samples."""

    condition_group_id: str
    display_name: str
    created_at: str = field(default_factory=_utc_now_iso)
    notes: str = ""

    @classmethod
    def create(cls, *, display_name: str, notes: str = "") -> ConditionGroupRecord:
        return cls(
            condition_group_id=new_condition_group_id(),
            display_name=display_name,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_group_id": self.condition_group_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionGroupRecord:
        return cls(
            condition_group_id=str(data["condition_group_id"]),
            display_name=str(data["display_name"]),
            created_at=str(data.get("created_at", _utc_now_iso())),
            notes=str(data.get("notes", "")),
        )


@dataclass
class CalibrationRecord:
    """Manual scale calibration derived from a known distance between two image points."""

    pixels_per_mm: float
    known_distance_mm: float
    point_1_x: float
    point_1_y: float
    point_2_x: float
    point_2_y: float
    known_distance_value: float | None = None
    known_distance_unit: str | None = None
    calibration_method: str = CALIBRATION_METHOD_MANUAL
    calibrated_at: str = field(default_factory=_utc_now_iso)
    notes: str = ""
    pixel_grid: str | None = None

    def entered_distance_label(self) -> str:
        if self.known_distance_value is not None and self.known_distance_unit:
            return f"{self.known_distance_value:g} {self.known_distance_unit}"
        return f"{self.known_distance_mm:g} mm"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "calibration_method": self.calibration_method,
            "pixels_per_mm": self.pixels_per_mm,
            "known_distance_mm": self.known_distance_mm,
            "point_1_x": self.point_1_x,
            "point_1_y": self.point_1_y,
            "point_2_x": self.point_2_x,
            "point_2_y": self.point_2_y,
            "calibrated_at": self.calibrated_at,
            "notes": self.notes,
        }
        if self.known_distance_value is not None:
            payload["known_distance_value"] = self.known_distance_value
        if self.known_distance_unit is not None:
            payload["known_distance_unit"] = self.known_distance_unit
        if self.pixel_grid is not None:
            payload["pixel_grid"] = self.pixel_grid
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        known_distance_mm = float(data["known_distance_mm"])
        if "known_distance_value" in data and "known_distance_unit" in data:
            known_distance_value = float(data["known_distance_value"])
            known_distance_unit = str(data["known_distance_unit"])
        else:
            known_distance_value = known_distance_mm
            known_distance_unit = DISTANCE_UNIT_MM

        return cls(
            calibration_method=str(
                data.get("calibration_method", CALIBRATION_METHOD_MANUAL)
            ),
            pixels_per_mm=float(data["pixels_per_mm"]),
            known_distance_mm=known_distance_mm,
            known_distance_value=known_distance_value,
            known_distance_unit=known_distance_unit,
            point_1_x=float(data["point_1_x"]),
            point_1_y=float(data["point_1_y"]),
            point_2_x=float(data["point_2_x"]),
            point_2_y=float(data["point_2_y"]),
            calibrated_at=str(data.get("calibrated_at", _utc_now_iso())),
            notes=str(data.get("notes", "")),
            pixel_grid=(
                str(data["pixel_grid"]) if data.get("pixel_grid") is not None else None
            ),
        )


@dataclass
class LeafMeasurementRecord:
    """Persisted whole-leaf area measurement and reproducibility provenance."""

    status: str
    leaf_area_pixels: int | None = None
    leaf_area_in2: float | None = None
    leaf_area_mm2: float | None = None
    pixels_per_mm_used: float | None = None
    pixels_per_inch_used: float | None = None
    leaf_mask_method: str | None = None
    leaf_mask_settings_version: int | None = None
    calculated_at: str | None = None
    source_image_hash: str | None = None
    stale_reason: str | None = None

    def mark_stale(self, reason: str) -> LeafMeasurementRecord:
        """Return a copy marked stale while preserving prior numeric values."""
        return LeafMeasurementRecord(
            status="stale",
            leaf_area_pixels=self.leaf_area_pixels,
            leaf_area_in2=self.leaf_area_in2,
            leaf_area_mm2=self.leaf_area_mm2,
            pixels_per_mm_used=self.pixels_per_mm_used,
            pixels_per_inch_used=self.pixels_per_inch_used,
            leaf_mask_method=self.leaf_mask_method,
            leaf_mask_settings_version=self.leaf_mask_settings_version,
            calculated_at=self.calculated_at,
            source_image_hash=self.source_image_hash,
            stale_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.leaf_area_pixels is not None:
            payload["leaf_area_pixels"] = self.leaf_area_pixels
        if self.leaf_area_in2 is not None:
            payload["leaf_area_in2"] = self.leaf_area_in2
        if self.leaf_area_mm2 is not None:
            payload["leaf_area_mm2"] = self.leaf_area_mm2
        if self.pixels_per_mm_used is not None:
            payload["pixels_per_mm_used"] = self.pixels_per_mm_used
        if self.pixels_per_inch_used is not None:
            payload["pixels_per_inch_used"] = self.pixels_per_inch_used
        if self.leaf_mask_method is not None:
            payload["leaf_mask_method"] = self.leaf_mask_method
        if self.leaf_mask_settings_version is not None:
            payload["leaf_mask_settings_version"] = self.leaf_mask_settings_version
        if self.calculated_at is not None:
            payload["calculated_at"] = self.calculated_at
        if self.source_image_hash is not None:
            payload["source_image_hash"] = self.source_image_hash
        if self.stale_reason is not None:
            payload["stale_reason"] = self.stale_reason
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeafMeasurementRecord:
        return cls(
            status=str(data["status"]),
            leaf_area_pixels=data.get("leaf_area_pixels"),
            leaf_area_in2=data.get("leaf_area_in2"),
            leaf_area_mm2=data.get("leaf_area_mm2"),
            pixels_per_mm_used=data.get("pixels_per_mm_used"),
            pixels_per_inch_used=data.get("pixels_per_inch_used"),
            leaf_mask_method=data.get("leaf_mask_method"),
            leaf_mask_settings_version=data.get("leaf_mask_settings_version"),
            calculated_at=data.get("calculated_at"),
            source_image_hash=data.get("source_image_hash"),
            stale_reason=data.get("stale_reason"),
        )


@dataclass
class CandidateDiseaseMeasurementRecord:
    """Persisted explicit leaf and candidate-disease measurement with provenance."""

    status: str
    total_leaf_pixels: int | None = None
    candidate_disease_pixels: int | None = None
    total_leaf_area_mm2: float | None = None
    total_leaf_area_in2: float | None = None
    candidate_disease_area_mm2: float | None = None
    candidate_disease_area_in2: float | None = None
    candidate_disease_coverage_percent: float | None = None
    non_candidate_leaf_area_mm2: float | None = None
    non_candidate_leaf_area_in2: float | None = None
    pixels_per_mm_used: float | None = None
    source_image_hash: str | None = None
    leaf_mask_method: str | None = None
    leaf_mask_settings_version: int | None = None
    disease_mask_method: str | None = None
    disease_mask_settings_version: int | None = None
    calculation_method: str | None = None
    calculated_at: str | None = None
    stale_reason: str | None = None

    def mark_stale(self, reason: str) -> CandidateDiseaseMeasurementRecord:
        """Return a copy marked stale while preserving prior numeric values."""
        return CandidateDiseaseMeasurementRecord(
            status="stale",
            total_leaf_pixels=self.total_leaf_pixels,
            candidate_disease_pixels=self.candidate_disease_pixels,
            total_leaf_area_mm2=self.total_leaf_area_mm2,
            total_leaf_area_in2=self.total_leaf_area_in2,
            candidate_disease_area_mm2=self.candidate_disease_area_mm2,
            candidate_disease_area_in2=self.candidate_disease_area_in2,
            candidate_disease_coverage_percent=self.candidate_disease_coverage_percent,
            non_candidate_leaf_area_mm2=self.non_candidate_leaf_area_mm2,
            non_candidate_leaf_area_in2=self.non_candidate_leaf_area_in2,
            pixels_per_mm_used=self.pixels_per_mm_used,
            source_image_hash=self.source_image_hash,
            leaf_mask_method=self.leaf_mask_method,
            leaf_mask_settings_version=self.leaf_mask_settings_version,
            disease_mask_method=self.disease_mask_method,
            disease_mask_settings_version=self.disease_mask_settings_version,
            calculation_method=self.calculation_method,
            calculated_at=self.calculated_at,
            stale_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.total_leaf_pixels is not None:
            payload["total_leaf_pixels"] = self.total_leaf_pixels
        if self.candidate_disease_pixels is not None:
            payload["candidate_disease_pixels"] = self.candidate_disease_pixels
        if self.total_leaf_area_mm2 is not None:
            payload["total_leaf_area_mm2"] = self.total_leaf_area_mm2
        if self.total_leaf_area_in2 is not None:
            payload["total_leaf_area_in2"] = self.total_leaf_area_in2
        if self.candidate_disease_area_mm2 is not None:
            payload["candidate_disease_area_mm2"] = self.candidate_disease_area_mm2
        if self.candidate_disease_area_in2 is not None:
            payload["candidate_disease_area_in2"] = self.candidate_disease_area_in2
        if self.candidate_disease_coverage_percent is not None:
            payload["candidate_disease_coverage_percent"] = (
                self.candidate_disease_coverage_percent
            )
        if self.non_candidate_leaf_area_mm2 is not None:
            payload["non_candidate_leaf_area_mm2"] = self.non_candidate_leaf_area_mm2
        if self.non_candidate_leaf_area_in2 is not None:
            payload["non_candidate_leaf_area_in2"] = self.non_candidate_leaf_area_in2
        if self.pixels_per_mm_used is not None:
            payload["pixels_per_mm_used"] = self.pixels_per_mm_used
        if self.source_image_hash is not None:
            payload["source_image_hash"] = self.source_image_hash
        if self.leaf_mask_method is not None:
            payload["leaf_mask_method"] = self.leaf_mask_method
        if self.leaf_mask_settings_version is not None:
            payload["leaf_mask_settings_version"] = self.leaf_mask_settings_version
        if self.disease_mask_method is not None:
            payload["disease_mask_method"] = self.disease_mask_method
        if self.disease_mask_settings_version is not None:
            payload["disease_mask_settings_version"] = self.disease_mask_settings_version
        if self.calculation_method is not None:
            payload["calculation_method"] = self.calculation_method
        if self.calculated_at is not None:
            payload["calculated_at"] = self.calculated_at
        if self.stale_reason is not None:
            payload["stale_reason"] = self.stale_reason
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateDiseaseMeasurementRecord:
        return cls(
            status=str(data["status"]),
            total_leaf_pixels=data.get("total_leaf_pixels"),
            candidate_disease_pixels=data.get("candidate_disease_pixels"),
            total_leaf_area_mm2=data.get("total_leaf_area_mm2"),
            total_leaf_area_in2=data.get("total_leaf_area_in2"),
            candidate_disease_area_mm2=data.get("candidate_disease_area_mm2"),
            candidate_disease_area_in2=data.get("candidate_disease_area_in2"),
            candidate_disease_coverage_percent=data.get("candidate_disease_coverage_percent"),
            non_candidate_leaf_area_mm2=data.get("non_candidate_leaf_area_mm2"),
            non_candidate_leaf_area_in2=data.get("non_candidate_leaf_area_in2"),
            pixels_per_mm_used=data.get("pixels_per_mm_used"),
            source_image_hash=data.get("source_image_hash"),
            leaf_mask_method=data.get("leaf_mask_method"),
            leaf_mask_settings_version=data.get("leaf_mask_settings_version"),
            disease_mask_method=data.get("disease_mask_method"),
            disease_mask_settings_version=data.get("disease_mask_settings_version"),
            calculation_method=data.get("calculation_method"),
            calculated_at=data.get("calculated_at"),
            stale_reason=data.get("stale_reason"),
        )


@dataclass(frozen=True)
class MaskStroke:
    """One binary brush stroke in oriented image coordinates.

    ``add`` includes pixels in candidate disease. ``remove`` excludes them.
    Removal does not record a confirmed-healthy label.
    """

    operation: str
    radius_px: float
    points: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "radius_px": self.radius_px,
            "points": [[x, y] for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskStroke:
        operation = str(data["operation"])
        if operation not in {MASK_STROKE_ADD, MASK_STROKE_REMOVE}:
            raise ValueError(f"Unknown mask stroke operation: {operation}")
        points = tuple(
            (float(point[0]), float(point[1])) for point in data.get("points", [])
        )
        return cls(
            operation=operation,
            radius_px=float(data["radius_px"]),
            points=points,
        )


@dataclass
class DiseaseMaskCorrectionRecord:
    """Researcher brush corrections stored separately from the automatic result.

    Corrected areas describe candidate disease after these strokes. The automatic
    measurement remains on the sample and is not replaced by these values.
    """

    status: str
    strokes: tuple[MaskStroke, ...] = ()
    result_label: str = RESEARCHER_CORRECTED_RESULT_LABEL
    source_image_hash: str | None = None
    pixel_grid: str | None = None
    leaf_mask_method: str | None = None
    leaf_mask_settings_version: int | None = None
    disease_mask_method: str | None = None
    disease_mask_settings_version: int | None = None
    total_leaf_pixels: int | None = None
    total_leaf_area_mm2: float | None = None
    total_leaf_area_in2: float | None = None
    corrected_candidate_disease_pixels: int | None = None
    corrected_candidate_disease_area_mm2: float | None = None
    corrected_candidate_disease_area_in2: float | None = None
    corrected_candidate_disease_coverage_percent: float | None = None
    corrected_non_candidate_leaf_area_mm2: float | None = None
    corrected_non_candidate_leaf_area_in2: float | None = None
    pixels_per_mm_used: float | None = None
    corrected_at: str | None = None
    review_reason: str | None = None

    def mark_review_required(self, reason: str) -> DiseaseMaskCorrectionRecord:
        """Keep the strokes and mark them as not safe to apply."""
        return replace(
            self,
            status=DISEASE_MASK_CORRECTION_STATUS_REVIEW,
            review_reason=reason,
        )

    def with_rescaled_areas(
        self,
        *,
        total_leaf_pixels: int,
        total_leaf_area_mm2: float,
        total_leaf_area_in2: float,
        corrected_candidate_disease_pixels: int,
        corrected_candidate_disease_area_mm2: float,
        corrected_candidate_disease_area_in2: float,
        corrected_candidate_disease_coverage_percent: float,
        corrected_non_candidate_leaf_area_mm2: float,
        corrected_non_candidate_leaf_area_in2: float,
        pixels_per_mm_used: float,
    ) -> DiseaseMaskCorrectionRecord:
        """Replace corrected areas after a scale change. Strokes stay put."""
        return replace(
            self,
            status=DISEASE_MASK_CORRECTION_STATUS_VALID,
            review_reason=None,
            result_label=RESEARCHER_CORRECTED_RESULT_LABEL,
            total_leaf_pixels=total_leaf_pixels,
            total_leaf_area_mm2=total_leaf_area_mm2,
            total_leaf_area_in2=total_leaf_area_in2,
            corrected_candidate_disease_pixels=corrected_candidate_disease_pixels,
            corrected_candidate_disease_area_mm2=corrected_candidate_disease_area_mm2,
            corrected_candidate_disease_area_in2=corrected_candidate_disease_area_in2,
            corrected_candidate_disease_coverage_percent=(
                corrected_candidate_disease_coverage_percent
            ),
            corrected_non_candidate_leaf_area_mm2=corrected_non_candidate_leaf_area_mm2,
            corrected_non_candidate_leaf_area_in2=corrected_non_candidate_leaf_area_in2,
            pixels_per_mm_used=pixels_per_mm_used,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "result_label": self.result_label,
            "strokes": [stroke.to_dict() for stroke in self.strokes],
        }
        optional = {
            "source_image_hash": self.source_image_hash,
            "pixel_grid": self.pixel_grid,
            "leaf_mask_method": self.leaf_mask_method,
            "leaf_mask_settings_version": self.leaf_mask_settings_version,
            "disease_mask_method": self.disease_mask_method,
            "disease_mask_settings_version": self.disease_mask_settings_version,
            "total_leaf_pixels": self.total_leaf_pixels,
            "total_leaf_area_mm2": self.total_leaf_area_mm2,
            "total_leaf_area_in2": self.total_leaf_area_in2,
            "corrected_candidate_disease_pixels": self.corrected_candidate_disease_pixels,
            "corrected_candidate_disease_area_mm2": self.corrected_candidate_disease_area_mm2,
            "corrected_candidate_disease_area_in2": self.corrected_candidate_disease_area_in2,
            "corrected_candidate_disease_coverage_percent": (
                self.corrected_candidate_disease_coverage_percent
            ),
            "corrected_non_candidate_leaf_area_mm2": self.corrected_non_candidate_leaf_area_mm2,
            "corrected_non_candidate_leaf_area_in2": self.corrected_non_candidate_leaf_area_in2,
            "pixels_per_mm_used": self.pixels_per_mm_used,
            "corrected_at": self.corrected_at,
            "review_reason": self.review_reason,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiseaseMaskCorrectionRecord:
        return cls(
            status=str(data.get("status", DISEASE_MASK_CORRECTION_STATUS_VALID)),
            result_label=str(
                data.get("result_label", RESEARCHER_CORRECTED_RESULT_LABEL)
            ),
            strokes=tuple(
                MaskStroke.from_dict(item) for item in data.get("strokes", [])
            ),
            source_image_hash=data.get("source_image_hash"),
            pixel_grid=data.get("pixel_grid"),
            leaf_mask_method=data.get("leaf_mask_method"),
            leaf_mask_settings_version=data.get("leaf_mask_settings_version"),
            disease_mask_method=data.get("disease_mask_method"),
            disease_mask_settings_version=data.get("disease_mask_settings_version"),
            total_leaf_pixels=data.get("total_leaf_pixels"),
            total_leaf_area_mm2=data.get("total_leaf_area_mm2"),
            total_leaf_area_in2=data.get("total_leaf_area_in2"),
            corrected_candidate_disease_pixels=data.get(
                "corrected_candidate_disease_pixels"
            ),
            corrected_candidate_disease_area_mm2=data.get(
                "corrected_candidate_disease_area_mm2"
            ),
            corrected_candidate_disease_area_in2=data.get(
                "corrected_candidate_disease_area_in2"
            ),
            corrected_candidate_disease_coverage_percent=data.get(
                "corrected_candidate_disease_coverage_percent"
            ),
            corrected_non_candidate_leaf_area_mm2=data.get(
                "corrected_non_candidate_leaf_area_mm2"
            ),
            corrected_non_candidate_leaf_area_in2=data.get(
                "corrected_non_candidate_leaf_area_in2"
            ),
            pixels_per_mm_used=data.get("pixels_per_mm_used"),
            corrected_at=data.get("corrected_at"),
            review_reason=data.get("review_reason"),
        )


@dataclass
class SampleRecord:
    """One imported still image representing a single leaf observation.

    Each sample is one image capture. Automatic measurements and optional
    researcher mask corrections are stored with the sample. The mask images
    themselves are not saved as separate files.
    """

    sample_id: str
    condition_group_id: str
    display_name: str
    raw_image_path: str
    original_source_path: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    image_hash: str | None = None
    imported_at: str = field(default_factory=_utc_now_iso)
    notes: str = ""
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_RUN
    calibration: CalibrationRecord | None = None
    leaf_measurement: LeafMeasurementRecord | None = None
    candidate_disease_measurement: CandidateDiseaseMeasurementRecord | None = None
    disease_mask_correction: DiseaseMaskCorrectionRecord | None = None

    @classmethod
    def create(
        cls,
        *,
        condition_group_id: str,
        display_name: str,
        raw_image_path: str,
        original_source_path: str | None = None,
        width_px: int | None = None,
        height_px: int | None = None,
        image_hash: str | None = None,
    ) -> SampleRecord:
        return cls(
            sample_id=new_sample_id(),
            condition_group_id=condition_group_id,
            display_name=display_name,
            raw_image_path=raw_image_path,
            original_source_path=original_source_path,
            width_px=width_px,
            height_px=height_px,
            image_hash=image_hash,
        )

    def raw_image_absolute(self, workspace_root: Path) -> Path:
        """Resolve the workspace-relative raw image path."""
        return (workspace_root / self.raw_image_path).resolve()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "condition_group_id": self.condition_group_id,
            "display_name": self.display_name,
            "raw_image_path": self.raw_image_path,
            "original_source_path": self.original_source_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "image_hash": self.image_hash,
            "imported_at": self.imported_at,
            "notes": self.notes,
            "calibration_status": self.calibration_status.value,
            "analysis_status": self.analysis_status.value,
            "calibration": (
                self.calibration.to_dict() if self.calibration is not None else None
            ),
            "leaf_measurement": (
                self.leaf_measurement.to_dict()
                if self.leaf_measurement is not None
                else None
            ),
            "candidate_disease_measurement": (
                self.candidate_disease_measurement.to_dict()
                if self.candidate_disease_measurement is not None
                else None
            ),
        }
        if self.disease_mask_correction is not None:
            payload["disease_mask_correction"] = self.disease_mask_correction.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleRecord:
        return cls(
            sample_id=str(data["sample_id"]),
            condition_group_id=str(data["condition_group_id"]),
            display_name=str(data["display_name"]),
            raw_image_path=str(data["raw_image_path"]),
            original_source_path=(
                str(data["original_source_path"])
                if data.get("original_source_path")
                else None
            ),
            width_px=data.get("width_px"),
            height_px=data.get("height_px"),
            image_hash=data.get("image_hash"),
            imported_at=str(data.get("imported_at", _utc_now_iso())),
            notes=str(data.get("notes", "")),
            calibration_status=CalibrationStatus(
                str(data.get("calibration_status", CalibrationStatus.UNCALIBRATED.value))
            ),
            analysis_status=AnalysisStatus(
                str(data.get("analysis_status", AnalysisStatus.NOT_RUN.value))
            ),
            calibration=(
                CalibrationRecord.from_dict(data["calibration"])
                if data.get("calibration")
                else None
            ),
            leaf_measurement=(
                LeafMeasurementRecord.from_dict(data["leaf_measurement"])
                if data.get("leaf_measurement")
                else None
            ),
            candidate_disease_measurement=(
                CandidateDiseaseMeasurementRecord.from_dict(
                    data["candidate_disease_measurement"]
                )
                if data.get("candidate_disease_measurement")
                else None
            ),
            disease_mask_correction=(
                DiseaseMaskCorrectionRecord.from_dict(data["disease_mask_correction"])
                if data.get("disease_mask_correction")
                else None
            ),
        )


@dataclass
class WorkspaceMetadata:
    """Top-level workspace descriptor persisted on disk."""

    workspace_id: str
    root_path: Path
    display_name: str
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    condition_groups: list[ConditionGroupRecord] = field(default_factory=list)
    samples: list[SampleRecord] = field(default_factory=list)
    legacy_inherited_calibration_repairs: int = field(default=0, repr=False)

    @classmethod
    def create(cls, root_path: Path, *, display_name: str | None = None) -> WorkspaceMetadata:
        resolved = root_path.expanduser().resolve()
        label = display_name or resolved.name
        now = _utc_now_iso()
        return cls(
            workspace_id=new_workspace_id(),
            root_path=resolved,
            display_name=label,
            created_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        """Update the workspace modification timestamp."""
        self.updated_at = _utc_now_iso()

    def get_condition_group(self, condition_group_id: str) -> ConditionGroupRecord | None:
        for group in self.condition_groups:
            if group.condition_group_id == condition_group_id:
                return group
        return None

    def get_sample(self, sample_id: str) -> SampleRecord | None:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None

    def samples_for_group(self, condition_group_id: str) -> list[SampleRecord]:
        return [
            sample
            for sample in self.samples
            if sample.condition_group_id == condition_group_id
        ]

    def workspace_json_path(self) -> Path:
        return self.root_path / WORKSPACE_JSON_FILENAME

    def to_dict(self, *, app_version: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "app_version": app_version,
            "workspace_id": self.workspace_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "condition_groups": [group.to_dict() for group in self.condition_groups],
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        root_path: Path,
        legacy_inherited_calibration_repairs: int = 0,
    ) -> WorkspaceMetadata:
        resolved = root_path.expanduser().resolve()
        return cls(
            workspace_id=str(data["workspace_id"]),
            root_path=resolved,
            display_name=str(data.get("display_name", resolved.name)),
            created_at=str(data.get("created_at", _utc_now_iso())),
            updated_at=str(data.get("updated_at", _utc_now_iso())),
            condition_groups=[
                ConditionGroupRecord.from_dict(item)
                for item in data.get("condition_groups", [])
            ],
            samples=[SampleRecord.from_dict(item) for item in data.get("samples", [])],
            legacy_inherited_calibration_repairs=legacy_inherited_calibration_repairs,
        )
