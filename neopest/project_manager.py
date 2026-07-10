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
    CalibrationRecord,
    ConditionGroupRecord,
    LeafMeasurementRecord,
    SampleRecord,
    WorkspaceMetadata,
)
from neopest.results import CalibrationStatus

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


def ensure_default_workspace(*, display_name: str = "NeoPest") -> WorkspaceMetadata:
    """Create or load the app-owned default workspace at ~/Documents/NeoPest."""
    root = get_default_workspace_root()
    metadata_path = root / WORKSPACE_JSON_FILENAME
    if metadata_path.is_file():
        return load_workspace(root)
    return initialize_workspace(root, display_name=display_name)


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


def mark_sample_leaf_measurement_stale(sample: SampleRecord, reason: str) -> None:
    """Mark a persisted leaf measurement stale when scientific inputs change."""
    if sample.leaf_measurement is None:
        return
    if sample.leaf_measurement.status == "valid":
        sample.leaf_measurement = sample.leaf_measurement.mark_stale(reason)


def clear_sample_leaf_measurement(sample: SampleRecord) -> None:
    """Remove a persisted leaf measurement from a sample."""
    sample.leaf_measurement = None


def save_sample_leaf_measurement(
    workspace: WorkspaceMetadata,
    sample_id: str,
    measurement: LeafMeasurementRecord,
    *,
    save: bool = True,
) -> SampleRecord:
    """Persist a whole-leaf area measurement for a sample."""
    sample = workspace.get_sample(sample_id)
    if sample is None:
        raise WorkspaceError(f"Unknown sample: {sample_id}")

    sample.leaf_measurement = measurement
    if save:
        save_workspace(workspace)
    return sample


def rename_condition_group(
    workspace: WorkspaceMetadata,
    condition_group_id: str,
    *,
    display_name: str,
    save: bool = True,
) -> ConditionGroupRecord:
    """Rename a condition group's display name."""
    group = workspace.get_condition_group(condition_group_id)
    if group is None:
        raise WorkspaceError(f"Unknown condition group: {condition_group_id}")

    cleaned = display_name.strip()
    if not cleaned:
        raise WorkspaceError("Condition group name cannot be empty.")

    group.display_name = cleaned
    if save:
        save_workspace(workspace)
    return group


def rename_sample(
    workspace: WorkspaceMetadata,
    sample_id: str,
    *,
    display_name: str,
    save: bool = True,
) -> SampleRecord:
    """Rename a sample's display name."""
    sample = workspace.get_sample(sample_id)
    if sample is None:
        raise WorkspaceError(f"Unknown sample: {sample_id}")

    cleaned = display_name.strip()
    if not cleaned:
        raise WorkspaceError("Sample name cannot be empty.")

    sample.display_name = cleaned
    if save:
        save_workspace(workspace)
    return sample


def _managed_workspace_file(path: Path, workspace_root: Path) -> bool:
    """Return True when ``path`` is inside the workspace root."""
    try:
        path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return False
    return True


def remove_sample(
    workspace: WorkspaceMetadata,
    sample_id: str,
    *,
    save: bool = True,
) -> SampleRecord:
    """Remove a sample and delete its managed workspace raw image copy only."""
    sample = workspace.get_sample(sample_id)
    if sample is None:
        raise WorkspaceError(f"Unknown sample: {sample_id}")

    raw_path = sample.raw_image_absolute(workspace.root_path)
    if raw_path.is_file() and _managed_workspace_file(raw_path, workspace.root_path):
        raw_path.unlink()

    workspace.samples = [
        item for item in workspace.samples if item.sample_id != sample_id
    ]
    if save:
        save_workspace(workspace)
    return sample


def remove_condition_group(
    workspace: WorkspaceMetadata,
    condition_group_id: str,
    *,
    save: bool = True,
) -> None:
    """Remove a condition group and its managed samples/files within the workspace."""
    group = workspace.get_condition_group(condition_group_id)
    if group is None:
        raise WorkspaceError(f"Unknown condition group: {condition_group_id}")

    for sample in list(workspace.samples_for_group(condition_group_id)):
        remove_sample(workspace, sample.sample_id, save=False)

    workspace.condition_groups = [
        item
        for item in workspace.condition_groups
        if item.condition_group_id != condition_group_id
    ]

    group_dir = workspace.root_path / RAW_IMAGES_DIR / condition_group_id
    if group_dir.is_dir() and _managed_workspace_file(group_dir, workspace.root_path):
        shutil.rmtree(group_dir)

    if save:
        save_workspace(workspace)


def _resolve_unique_managed_filename(directory: Path, filename: str) -> str:
    """Return a managed filename that does not collide with an existing file."""
    candidate = Path(filename)
    if not (directory / candidate.name).is_file():
        return candidate.name

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        unique_name = f"{stem}_{counter}{suffix}"
        if not (directory / unique_name).is_file():
            return unique_name
        counter += 1


def move_sample_to_condition_group(
    workspace: WorkspaceMetadata,
    sample_id: str,
    destination_condition_group_id: str,
    *,
    save: bool = True,
) -> SampleRecord:
    """Move a sample and its managed raw image to another condition group."""
    sample = workspace.get_sample(sample_id)
    if sample is None:
        raise WorkspaceError(f"Unknown sample: {sample_id}")

    destination_group = workspace.get_condition_group(destination_condition_group_id)
    if destination_group is None:
        raise WorkspaceError(f"Unknown condition group: {destination_condition_group_id}")

    if sample.condition_group_id == destination_condition_group_id:
        return sample

    source_path = sample.raw_image_absolute(workspace.root_path)
    if not source_path.is_file():
        raise WorkspaceError(f"Managed workspace image not found: {source_path}")
    if not _managed_workspace_file(source_path, workspace.root_path):
        raise WorkspaceError(
            "Sample image path is outside the active workspace and cannot be moved."
        )

    destination_dir = workspace.root_path / RAW_IMAGES_DIR / destination_condition_group_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not _managed_workspace_file(destination_dir, workspace.root_path):
        raise WorkspaceError(
            "Destination condition group path is outside the active workspace."
        )

    managed_filename = _resolve_unique_managed_filename(
        destination_dir,
        source_path.name,
    )
    destination_path = destination_dir / managed_filename

    previous_group_id = sample.condition_group_id
    previous_relative_path = sample.raw_image_path

    try:
        shutil.move(str(source_path), str(destination_path))
    except OSError as exc:
        raise WorkspaceError(
            f"Could not move managed workspace image for sample '{sample.display_name}'."
        ) from exc

    sample.condition_group_id = destination_condition_group_id
    sample.raw_image_path = _workspace_relative_path(workspace.root_path, destination_path)

    if not save:
        return sample

    try:
        save_workspace(workspace)
    except OSError as exc:
        try:
            shutil.move(str(destination_path), str(source_path))
        except OSError as rollback_exc:
            raise WorkspaceError(
                "Workspace metadata could not be saved after moving the sample image, "
                "and NeoPest could not restore the original managed file location. "
                "The workspace may be inconsistent; inspect the workspace folder before retrying."
            ) from rollback_exc

        sample.condition_group_id = previous_group_id
        sample.raw_image_path = previous_relative_path
        raise WorkspaceError(
            f"Could not save workspace metadata after moving sample '{sample.display_name}'."
        ) from exc

    return sample


def save_sample_calibration(
    workspace: WorkspaceMetadata,
    sample_id: str,
    calibration: CalibrationRecord,
    *,
    save: bool = True,
) -> SampleRecord:
    """Attach manual calibration metadata to a sample and persist the workspace."""
    sample = workspace.get_sample(sample_id)
    if sample is None:
        raise WorkspaceError(f"Unknown sample: {sample_id}")

    mark_sample_leaf_measurement_stale(sample, "calibration_changed")
    sample.calibration = calibration
    sample.calibration_status = CalibrationStatus.MANUAL
    # Phase 3C: calibration save does not run final analysis. Preserve analysis_status
    # and invalidate whole-leaf area measurements tied to prior calibration.

    if save:
        save_workspace(workspace)
    return sample
