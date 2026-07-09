"""Still-image loading, probing, and hashing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QImageReader

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ImageLoadInfo:
    """Minimal metadata returned when probing an image file."""

    path: Path
    width_px: int | None
    height_px: int | None


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def load_image_info(path: Path) -> ImageLoadInfo:
    """Return basic image dimensions using Qt image probing."""
    resolved = path.expanduser().resolve()
    if not is_supported_image(resolved):
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    reader = QImageReader(str(resolved))
    size = reader.size()
    if size.isValid():
        return ImageLoadInfo(
            path=resolved,
            width_px=size.width(),
            height_px=size.height(),
        )
    return ImageLoadInfo(path=resolved, width_px=None, height_px=None)


def compute_image_hash(path: Path, *, algorithm: str = "sha256") -> str:
    """Return a content hash prefixed with the algorithm name."""
    resolved = path.expanduser().resolve()
    digest = hashlib.new(algorithm)
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"
