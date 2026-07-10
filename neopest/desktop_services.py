"""Cross-platform desktop integration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def reveal_image_menu_label() -> str:
    """Return the platform-appropriate label for revealing a managed image file."""
    if sys.platform == "darwin":
        return "Reveal Image in Finder"
    if sys.platform == "win32":
        return "Show Image in File Explorer"
    return "Show Image in Folder"


def open_path_in_file_manager(path: Path) -> bool:
    """Open a folder or reveal a file using the platform file manager."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved)))
