"""Still-image loading, probing, and hashing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QImageReader

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
HASH_CHUNK_SIZE = 1024 * 1024
LEAF_MASK_OVERLAY_RGBA = (50, 220, 80, 110)


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


def load_image_rgb_array(path: Path) -> np.ndarray:
    """Load a still image as an RGB ``uint8`` array with shape (H, W, 3)."""
    resolved = path.expanduser().resolve()
    if not is_supported_image(resolved):
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    bgr = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {resolved}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def leaf_mask_to_qimage(mask: np.ndarray) -> QImage:
    """Convert a boolean leaf mask into a semi-transparent RGBA ``QImage``."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask; got shape {mask.shape}")

    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    red, green, blue, alpha = LEAF_MASK_OVERLAY_RGBA
    rgba[mask, 0] = red
    rgba[mask, 1] = green
    rgba[mask, 2] = blue
    rgba[mask, 3] = alpha
    bytes_per_line = width * 4
    return QImage(
        rgba.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGBA8888,
    ).copy()


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
