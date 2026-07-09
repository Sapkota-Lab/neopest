"""Workspace lifecycle, persistence, and image import."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from neopest import __version__
from neopest.image_io import compute_image_hash, is_supported_image, load_image_info
from neopest.metadata import (
    DEFAULT_CONDITION_GROUP_NAME,
    WORKSPACE_JSON_FILENAME,
    ConditionGroupRecord,
    SampleRecord,
    WorkspaceMetadata,
)

RAW_IMAGES_DIR = "raw_images"
ANALYSIS_DIR = "analysis"
EXPORTS_DIR = "exports"
WORKSPACE_DIRS = (RAW_IMAGES_DIR, ANALYSIS_DIR, EXPORTS_DIR)


class WorkspaceError(Exception):
    """Raised when workspace operations fail."""


def get_default_workspace_root() -> Path:
    """Return the default NeoPest workspace folder under the user's Documents."""
    return Path.home() / "Documents" / "NeoPest"


def default_workspace_metadata_path() -> Path:
    """Return the expected metadata file path for the default workspace."""
    return get_default_workspace_root() / WORKSPACE_JSON_FILENAME


def ensure_workspace_layout(root: Path) -> None:
    """Create the on-disk workspace folder tree if missing."""
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_DIRS:
        (resolved / name).mkdir(exist_ok=True)


def save_workspace(workspace: WorkspaceMetadata) -> Path:
    """Persist workspace metadata to JSON and return the file path."""
    ensure_workspace_layout(workspace.root_path)
    workspace.touch()
    payload = workspace.to_dict(app_version=__version__)
    target = workspace.workspace_json_path()
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_workspace(root: Path) -> WorkspaceMetadata:
    """Load workspace metadata from an existing workspace folder."""
    resolved = root.expanduser().resolve()
    json_path = resolved / WORKSPACE_JSON_FILENAME
    if not json_path.is_file():
        raise WorkspaceError(
            f"No workspace metadata found at {json_path}. "
            "Use initialize_workspace() to create a new workspace."
        )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return WorkspaceMetadata.from_dict(data, root_path=resolved)


def initialize_workspace(
    root: Path,
    *,
    display_name: str | None = None,
) -> WorkspaceMetadata:
    """Create a new workspace with folders, default condition group, and JSON."""
    resolved = root.expanduser().resolve()
    json_path = resolved / WORKSPACE_JSON_FILENAME
    if json_path.is_file():
        raise WorkspaceError(
            f"A NeoPest workspace already exists at {json_path}. "
            "Use load_workspace() or File → Open Workspace to open it."
        )
    ensure_workspace_layout(resolved)
    workspace = WorkspaceMetadata.create(
        resolved,
        display_name=display_name or resolved.name,
    )
    ensure_default_condition_group(workspace)
    save_workspace(workspace)
    return workspace


def create_condition_group(
    workspace: WorkspaceMetadata,
    *,
    display_name: str,
    notes: str = "",
    save: bool = True,
) -> ConditionGroupRecord:
    """Add a new condition group to the workspace."""
    group = ConditionGroupRecord.create(display_name=display_name, notes=notes)
    workspace.condition_groups.append(group)
    (workspace.root_path / RAW_IMAGES_DIR / group.condition_group_id).mkdir(
        parents=True,
        exist_ok=True,
    )
    if save:
        save_workspace(workspace)
    return group


def ensure_default_condition_group(
    workspace: WorkspaceMetadata,
    *,
    save: bool = False,
) -> str:
    """Ensure the default 'Ungrouped' condition group exists; return its ID."""
    for group in workspace.condition_groups:
        if group.display_name == DEFAULT_CONDITION_GROUP_NAME:
            return group.condition_group_id

    group = create_condition_group(
        workspace,
        display_name=DEFAULT_CONDITION_GROUP_NAME,
        save=save,
    )
    return group.condition_group_id


def _workspace_relative_path(workspace_root: Path, absolute_path: Path) -> str:
    return absolute_path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _raw_image_destination(
    workspace: WorkspaceMetadata,
    *,
    condition_group_id: str,
    sample_id: str,
    source_name: str,
) -> Path:
    group_dir = workspace.root_path / RAW_IMAGES_DIR / condition_group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir / f"{sample_id}_{source_name}"


def import_images(
    workspace: WorkspaceMetadata,
    source_paths: list[Path | str],
    *,
    condition_group_id: str | None = None,
    save: bool = True,
) -> list[SampleRecord]:
    """Import one or more still images as samples under a condition group."""
    if not source_paths:
        return []

    target_group_id = condition_group_id or ensure_default_condition_group(workspace)
    if workspace.get_condition_group(target_group_id) is None:
        raise WorkspaceError(f"Unknown condition group: {target_group_id}")

    imported: list[SampleRecord] = []
    for raw_path in source_paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file():
            raise WorkspaceError(f"Image file not found: {source}")
        if not is_supported_image(source):
            raise WorkspaceError(f"Unsupported image format: {source.suffix}")

        info = load_image_info(source)
        image_hash = compute_image_hash(source)
        sample = SampleRecord.create(
            condition_group_id=target_group_id,
            display_name=source.name,
            raw_image_path="",
            original_source_path=str(source),
            width_px=info.width_px,
            height_px=info.height_px,
            image_hash=image_hash,
        )
        destination = _raw_image_destination(
            workspace,
            condition_group_id=target_group_id,
            sample_id=sample.sample_id,
            source_name=source.name,
        )
        shutil.copy2(source, destination)
        sample.raw_image_path = _workspace_relative_path(workspace.root_path, destination)
        workspace.samples.append(sample)
        imported.append(sample)

    if save:
        save_workspace(workspace)
    return imported
