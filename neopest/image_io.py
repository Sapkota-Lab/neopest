"""Still-image loading helpers (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageLoadInfo:
    """Minimal metadata returned when probing an image file."""

    path: Path
    width_px: int | None
    height_px: int | None


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def load_image_info(path: Path) -> ImageLoadInfo:
    """Return basic image dimensions without decoding full pixel data yet."""
    resolved = path.expanduser().resolve()
    if not is_supported_image(resolved):
        raise ValueError(f"Unsupported image format: {resolved.suffix}")

    # Real decoding will use OpenCV or Pillow in a later phase.
    return ImageLoadInfo(path=resolved, width_px=None, height_px=None)
