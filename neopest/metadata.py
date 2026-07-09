"""Workspace and sample metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from neopest.results import AnalysisStatus, CalibrationStatus

SCHEMA_VERSION = 1
WORKSPACE_JSON_FILENAME = "neopest_workspace.json"
DEFAULT_CONDITION_GROUP_NAME = "Ungrouped"


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
class SampleRecord:
    """One imported still image representing a single strawberry leaf."""

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
        return {
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
        }

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
    def from_dict(cls, data: dict[str, Any], *, root_path: Path) -> WorkspaceMetadata:
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
        )
