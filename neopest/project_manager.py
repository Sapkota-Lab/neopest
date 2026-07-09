"""Workspace lifecycle helpers (scaffold)."""

from __future__ import annotations

from pathlib import Path

from neopest.metadata import WorkspaceMetadata

WORKSPACE_DIRS = ("raw", "processed", "metadata", "exports")


def ensure_workspace_layout(root: Path) -> WorkspaceMetadata:
    """Create the on-disk workspace folder tree if missing."""
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_DIRS:
        (resolved / name).mkdir(exist_ok=True)
    return WorkspaceMetadata.create(resolved)
