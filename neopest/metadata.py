"""Workspace and sample metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


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


@dataclass
class SampleRecord:
    """One imported still image representing a single strawberry leaf."""

    sample_id: str
    condition_group_id: str
    source_path: Path
    display_name: str
    imported_at: str = field(default_factory=_utc_now_iso)
    notes: str = ""

    @classmethod
    def create(
        cls,
        source_path: Path,
        *,
        condition_group_id: str,
        display_name: str | None = None,
    ) -> SampleRecord:
        resolved = source_path.expanduser().resolve()
        return cls(
            sample_id=new_sample_id(),
            condition_group_id=condition_group_id,
            source_path=resolved,
            display_name=display_name or resolved.name,
        )


@dataclass
class WorkspaceMetadata:
    """Top-level workspace descriptor persisted on disk."""

    workspace_id: str
    root_path: Path
    created_at: str = field(default_factory=_utc_now_iso)
    condition_groups: list[ConditionGroupRecord] = field(default_factory=list)
    samples: list[SampleRecord] = field(default_factory=list)

    @classmethod
    def create(cls, root_path: Path) -> WorkspaceMetadata:
        resolved = root_path.expanduser().resolve()
        return cls(
            workspace_id=new_workspace_id(),
            root_path=resolved,
        )
