"""Still-image loading, probing, and hashing helpers."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QImageReader, QPixmap

HEIF_IMAGE_SUFFIXES = frozenset({".heic", ".heif"})
SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    *HEIF_IMAGE_SUFFIXES,
}
HASH_CHUNK_SIZE = 1024 * 1024
LEAF_MASK_OVERLAY_RGBA = (50, 220, 80, 110)
DISEASE_MASK_OVERLAY_RGBA = (255, 95, 60, 115)
HEALTHY_TISSUE_MASK_OVERLAY_RGBA = (55, 210, 75, 105)


@dataclass(frozen=True)
class ImageLoadInfo:
    """Minimal metadata returned when probing an image file."""

    path: Path
    width_px: int | None
    height_px: int | None


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _is_heif_image(path: Path) -> bool:
    return path.suffix.lower() in HEIF_IMAGE_SUFFIXES


def _heif_plugin_available() -> bool:
    """Return True when this Qt build can decode HEIC or HEIF."""
    available = {
        bytes(fmt).decode("ascii", errors="ignore").lower()
        for fmt in QImageReader.supportedImageFormats()
    }
    return "heic" in available or "heif" in available


def _open_heif_reader(path: Path) -> QImageReader:
    """Open a HEIC/HEIF reader and apply its orientation metadata once.

    ``autoTransform`` covers HEIF rotation, mirrors, and embedded EXIF.
    Callers must not rotate the result again.
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    return reader


def _read_heif_qimage(path: Path) -> QImage:
    """Decode one HEIC/HEIF file without requiring a QApplication or widgets."""
    if not _heif_plugin_available():
        raise ValueError(
            "HEIC/HEIF decoding is not available on this system "
            "because the Qt HEIC image plugin is missing."
        )

    reader = _open_heif_reader(path)
    image = reader.read()
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise ValueError(f"Unable to decode HEIC/HEIF image: {path.name}")
    return image


def qimage_to_rgb_uint8(image: QImage) -> np.ndarray:
    """Copy ``image`` into an RGB ``uint8`` array of shape (H, W, 3).

    Qt rows may be padded. Only the live RGB bytes of each row are kept.
    """
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    width = converted.width()
    height = converted.height()
    bytes_per_line = converted.bytesPerLine()
    row_bytes = width * 3
    if converted.isNull() or width <= 0 or height <= 0 or bytes_per_line < row_bytes:
        raise ValueError("Unable to convert image to RGB.")

    raw = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=bytes_per_line * height,
    )
    rows = raw.reshape(height, bytes_per_line)
    packed = rows[:, :row_bytes].copy()
    return packed.reshape(height, width, 3)


def rgb_uint8_to_qimage(array: np.ndarray) -> QImage:
    """Copy an RGB ``uint8`` array into a ``QImage`` without a second decode."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    contiguous = np.ascontiguousarray(array)
    height, width, _channels = contiguous.shape
    image = QImage(
        contiguous.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply one EXIF orientation value, including mirrored orientations."""
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 0)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _orientation_in_tiff(tiff: bytes) -> int | None:
    """Return the first IFD0 orientation tag, if the TIFF header is readable."""
    if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
        return None
    endian = "<" if tiff[:2] == b"II" else ">"

    def u16(offset: int) -> int:
        return struct.unpack_from(endian + "H", tiff, offset)[0]

    def u32(offset: int) -> int:
        return struct.unpack_from(endian + "I", tiff, offset)[0]

    try:
        if u16(2) != 42:
            return None
        ifd = u32(4)
        if ifd <= 0 or ifd + 2 > len(tiff):
            return None
        count = u16(ifd)
        for index in range(count):
            entry = ifd + 2 + (12 * index)
            if entry + 12 > len(tiff):
                return None
            tag = u16(entry)
            value_type = u16(entry + 2)
            value_count = u32(entry + 4)
            if tag == 0x0112 and value_type == 3 and value_count == 1:
                orientation = u16(entry + 8)
                if 1 <= orientation <= 8:
                    return orientation
                return None
    except struct.error:
        return None
    return None


def _exif_orientation(path: Path) -> int | None:
    """Return EXIF orientation 1–8, or None when the file has no usable tag."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _orientation_in_tiff(data)

    search_from = 0
    while True:
        found = data.find(b"Exif\x00\x00", search_from)
        if found < 0:
            return None
        orientation = _orientation_in_tiff(data[found + 6 :])
        if orientation is not None:
            return orientation
        search_from = found + 6


def _heif_box_byte(data: bytes, box_type: bytes) -> int | None:
    """Return the first payload byte of a small HEIF box, if one is present."""
    search_from = 0
    while True:
        found = data.find(box_type, search_from)
        if found < 4:
            return None
        size = int.from_bytes(data[found - 4 : found], "big")
        payload_start = found + 4
        if 9 <= size <= 16 and payload_start < len(data):
            return data[payload_start]
        search_from = found + 4


def display_orientation_is_identity(path: Path) -> bool:
    """Return True when display pixels match the file's stored pixel grid.

    Legacy calibration points were clicked on that stored grid. A non-identity
    orientation means those points belong to a different coordinate system.
    """
    resolved = path.expanduser()
    if not resolved.is_file():
        return True
    if _is_heif_image(resolved):
        try:
            data = resolved.read_bytes()
        except OSError:
            return True
        rotation = _heif_box_byte(data, b"irot")
        mirror = _heif_box_byte(data, b"imir")
        if mirror is not None or (rotation is not None and (rotation & 0x03) != 0):
            return False
    orientation = _exif_orientation(resolved)
    return orientation in {None, 1}


def load_image_info(path: Path) -> ImageLoadInfo:
    """Return basic image dimensions using Qt image probing."""
    resolved = path.expanduser().resolve()
    if not is_supported_image(resolved):
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    if _is_heif_image(resolved):
        image = _read_heif_qimage(resolved)
        return ImageLoadInfo(
            path=resolved,
            width_px=image.width(),
            height_px=image.height(),
        )

    oriented = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if oriented is not None:
        height_px, width_px = oriented.shape[:2]
        return ImageLoadInfo(path=resolved, width_px=width_px, height_px=height_px)

    reader = QImageReader(str(resolved))
    size = reader.size()
    if size.isValid():
        return ImageLoadInfo(
            path=resolved,
            width_px=size.width(),
            height_px=size.height(),
        )
    return ImageLoadInfo(path=resolved, width_px=None, height_px=None)


# One decoded frame, keyed by path, size, and mtime. Callers must not write into it.
_IMAGE_ARRAY_CACHE: tuple[tuple[str, int, int], np.ndarray] | None = None


def load_image_rgb_array(path: Path) -> np.ndarray:
    """Load a still image as an RGB ``uint8`` array with shape (H, W, 3)."""
    resolved = path.expanduser().resolve()
    if not is_supported_image(resolved):
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    global _IMAGE_ARRAY_CACHE
    stat = resolved.stat()
    token = (str(resolved), stat.st_mtime_ns, stat.st_size)
    if _IMAGE_ARRAY_CACHE is not None and _IMAGE_ARRAY_CACHE[0] == token:
        return _IMAGE_ARRAY_CACHE[1]

    array = _decode_image_rgb_array(resolved)
    _IMAGE_ARRAY_CACHE = (token, array)
    return array


def _decode_image_rgb_array(resolved: Path) -> np.ndarray:
    if _is_heif_image(resolved):
        return qimage_to_rgb_uint8(_read_heif_qimage(resolved))

    bgr = cv2.imread(str(resolved), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {resolved}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _apply_exif_orientation(rgb, _exif_orientation(resolved) or 1)


def load_preview_pixmap(path: Path) -> QPixmap:
    """Return the preview pixmap for the same pixels ``load_image_rgb_array`` uses."""
    return QPixmap.fromImage(rgb_uint8_to_qimage(load_image_rgb_array(path)))


def mask_to_qimage(mask: np.ndarray, *, overlay_rgba: tuple[int, int, int, int]) -> QImage:
    """Convert a boolean mask into a semi-transparent RGBA ``QImage``."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask; got shape {mask.shape}")

    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    red, green, blue, alpha = overlay_rgba
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


def write_overlay_region(
    image: QImage,
    membership: np.ndarray,
    origin_x: int,
    origin_y: int,
    rgba: tuple[int, int, int, int],
) -> None:
    """Write a boolean patch into an existing RGBA8888 overlay image.

    ``membership`` is the patch at ``(origin_x, origin_y)``. True pixels take
    ``rgba``; false pixels become transparent. The image buffer is updated
    in place.
    """
    if image.format() != QImage.Format.Format_RGBA8888:
        raise ValueError("Overlay image must be Format_RGBA8888.")
    if membership.ndim != 2:
        raise ValueError(f"Expected a 2D membership patch; got shape {membership.shape}")

    patch_height, patch_width = membership.shape
    x0 = max(0, int(origin_x))
    y0 = max(0, int(origin_y))
    x1 = min(image.width(), int(origin_x) + patch_width)
    y1 = min(image.height(), int(origin_y) + patch_height)
    if x0 >= x1 or y0 >= y1:
        return

    local_x = x0 - int(origin_x)
    local_y = y0 - int(origin_y)
    region = np.asarray(
        membership[local_y : local_y + (y1 - y0), local_x : local_x + (x1 - x0)],
        dtype=bool,
    )
    red, green, blue, alpha = rgba
    pixels = np.zeros((region.shape[0], region.shape[1], 4), dtype=np.uint8)
    pixels[..., 0] = np.where(region, red, 0)
    pixels[..., 1] = np.where(region, green, 0)
    pixels[..., 2] = np.where(region, blue, 0)
    pixels[..., 3] = np.where(region, alpha, 0)

    bytes_per_line = image.bytesPerLine()
    buffer = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), bytes_per_line)
    row_bytes = region.shape[1] * 4
    buffer[y0:y1, x0 * 4 : x0 * 4 + row_bytes] = pixels.reshape(region.shape[0], row_bytes)


def leaf_mask_to_qimage(mask: np.ndarray) -> QImage:
    """Convert a boolean whole-leaf mask into a semi-transparent green overlay."""
    return mask_to_qimage(mask, overlay_rgba=LEAF_MASK_OVERLAY_RGBA)


def disease_mask_to_qimage(mask: np.ndarray) -> QImage:
    """Convert a boolean disease mask into a semi-transparent warm overlay."""
    return mask_to_qimage(mask, overlay_rgba=DISEASE_MASK_OVERLAY_RGBA)


def healthy_tissue_mask_to_qimage(mask: np.ndarray) -> QImage:
    """Convert a boolean healthy-tissue mask into a semi-transparent green overlay."""
    return mask_to_qimage(mask, overlay_rgba=HEALTHY_TISSUE_MASK_OVERLAY_RGBA)


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
