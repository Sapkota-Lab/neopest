"""Deterministic leaf mask preview segmentation.

Scientific note (Phase 3 preview only):
Experiments apply inoculum to the central leaf region and image the same unit
over multiple days. Central brown/dark tissue is biologically meaningful
inoculation/necrosis and must remain inside the leaf blade mask — it is not
an artifact and is not disease segmentation. Future phases may quantify disease
area and spread per timepoint; this module returns a preview leaf mask only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

PREVIEW_METHOD = "green_seeded_color_thresholding"
SETTINGS_VERSION = 2


class LeafMaskPreviewStatus(str, Enum):
    """Outcome of a non-persistent leaf mask preview."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class LeafMaskPreviewSettings:
    """Versioned defaults for Phase 3B green-seeded leaf preview."""

    settings_version: int = SETTINGS_VERSION
    # Blue petri-dish background (spatial constraint cue).
    blue_hue_min: int = 85
    blue_hue_max: int = 135
    blue_sat_min: int = 40
    blue_val_min: int = 30
    # Green leaf tissue seed (HSV + ExG).
    green_hue_min: int = 32
    green_hue_max: int = 92
    green_sat_min: int = 35
    green_val_min: int = 25
    exg_threshold: float = 12.0
    min_green_seed_area_px: int = 200
    # Brown inoculation/necrosis: included in leaf blade when connected to green
    # seed (midrib/central disease regions are part of the leaf, not excluded).
    brown_hue_min: int = 8
    brown_hue_max: int = 28
    brown_sat_min: int = 30
    brown_val_min: int = 25
    brown_val_max: int = 170
    brown_bridge_radius: int = 6
    # White wrap / glare / tape highlights.
    white_sat_max: int = 40
    white_val_min: int = 185
    glare_sat_max: int = 55
    glare_val_min: int = 170
    # General cleanup.
    min_component_area_px: int = 500
    morphology_radius: int = 3
    dish_dilate_radius: int = 8
    dish_shrink_radius: int = 4
    max_leaf_dish_fraction: float = 0.82


@dataclass
class LeafMaskPreview:
    """In-memory leaf mask preview result (not final analysis output)."""

    mask: np.ndarray | None
    status: LeafMaskPreviewStatus
    warnings: list[str] = field(default_factory=list)
    method: str = PREVIEW_METHOD
    settings_snapshot: dict[str, Any] = field(default_factory=dict)


class SegmentationError(ValueError):
    """Raised when segmentation input is invalid."""


def _validate_image_array(image_array: np.ndarray) -> tuple[int, int]:
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise SegmentationError(
            f"Expected RGB image array with shape (H, W, 3); got {image_array.shape}"
        )
    height, width = image_array.shape[:2]
    if height == 0 or width == 0:
        raise SegmentationError("Image array has zero width or height")
    return height, width


def _kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _image_hsv(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def detect_blue_background(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return a boolean mask of blue petri-dish background pixels."""
    hsv = _image_hsv(image_rgb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return (
        (hue >= settings.blue_hue_min)
        & (hue <= settings.blue_hue_max)
        & (sat >= settings.blue_sat_min)
        & (val >= settings.blue_val_min)
    )


def detect_green_leaf_tissue(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return confident green vegetation pixels using HSV and ExG."""
    hsv = _image_hsv(image_rgb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_hsv = (
        (hue >= settings.green_hue_min)
        & (hue <= settings.green_hue_max)
        & (sat >= settings.green_sat_min)
        & (val >= settings.green_val_min)
    )

    red = image_rgb[:, :, 0].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    blue = image_rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * green - red - blue
    green_exg = exg >= settings.exg_threshold
    return green_hsv & green_exg


def detect_brown_necrosis_tissue(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return brown/darker central or necrotic leaf tissue pixels.

    Used only to expand the leaf blade mask from a green seed. This is not
    disease segmentation — brown inoculation zones remain part of the leaf.
    """
    hsv = _image_hsv(image_rgb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return (
        (hue >= settings.brown_hue_min)
        & (hue <= settings.brown_hue_max)
        & (sat >= settings.brown_sat_min)
        & (val >= settings.brown_val_min)
        & (val <= settings.brown_val_max)
    )


def detect_glare_and_wrapping(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return bright low-saturation pixels such as white wrap or tape glare."""
    hsv = _image_hsv(image_rgb)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_wrap = (sat <= settings.white_sat_max) & (val >= settings.white_val_min)
    glare = (sat <= settings.glare_sat_max) & (val >= settings.glare_val_min)
    return white_wrap | glare


def exclude_white_wrapping(
    image_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Remove low-saturation bright pixels such as white stem wrapping."""
    return candidate_mask & ~detect_glare_and_wrapping(image_rgb, settings)


def _largest_component_mask(
    binary_mask: np.ndarray,
    *,
    min_area_px: int,
) -> tuple[np.ndarray, int]:
    """Keep the largest connected component above ``min_area_px``."""
    component_count, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    if component_count <= 1:
        return np.zeros_like(binary_mask, dtype=bool), 0

    best_label = 0
    best_area = 0
    for label in range(1, component_count):
        area = int(np.count_nonzero(labels == label))
        if area >= min_area_px and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return np.zeros_like(binary_mask, dtype=bool), 0
    return labels == best_label, best_area


def _largest_component_overlapping_seed(
    binary_mask: np.ndarray,
    seed_mask: np.ndarray,
    *,
    min_area_px: int,
) -> tuple[np.ndarray, int]:
    """Keep the largest component that overlaps the green seed mask."""
    component_count, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    if component_count <= 1:
        return np.zeros_like(binary_mask, dtype=bool), 0

    best_label = 0
    best_area = 0
    for label in range(1, component_count):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < min_area_px:
            continue
        if not np.any(component & seed_mask):
            continue
        if area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return np.zeros_like(binary_mask, dtype=bool), 0
    return labels == best_label, best_area


def _petri_dish_interior_mask(
    blue_mask: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Approximate the petri dish interior as a spatial analysis constraint."""
    largest_blue, _ = _largest_component_mask(
        blue_mask,
        min_area_px=max(100, settings.min_component_area_px // 5),
    )
    if not np.any(largest_blue):
        return np.zeros_like(blue_mask, dtype=bool)

    points_yx = np.column_stack(np.nonzero(largest_blue))
    if points_yx.shape[0] < 3:
        kernel = _kernel(settings.dish_dilate_radius)
        return cv2.dilate(largest_blue.astype(np.uint8), kernel).astype(bool) > 0

    points_xy = points_yx[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(points_xy)
    interior = np.zeros(blue_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(interior, hull, 1)
    kernel = _kernel(settings.dish_dilate_radius)
    interior = cv2.dilate(interior, kernel)
    if settings.dish_shrink_radius > 0:
        shrink = _kernel(settings.dish_shrink_radius)
        interior = cv2.erode(interior, shrink)
    return interior.astype(bool)


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill fully enclosed holes, preserving leaf-enclosed brown tissue."""
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    filled = binary_mask.astype(np.uint8).copy()
    height, width = filled.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 1)
    holes = (filled == 0) & binary_mask
    return binary_mask | holes


def _brown_connected_to_green(
    green_mask: np.ndarray,
    brown_mask: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Grow from green leaf margin into connected brown inoculation/necrosis tissue."""
    if not np.any(green_mask):
        return np.zeros_like(green_mask, dtype=bool)

    combined = green_mask.copy()
    growth_kernel = _kernel(max(1, settings.brown_bridge_radius // 2))
    while True:
        grown = cv2.dilate(combined.astype(np.uint8), growth_kernel).astype(bool)
        newly_reachable_brown = brown_mask & grown & ~combined
        if not np.any(newly_reachable_brown):
            break
        combined = combined | newly_reachable_brown

    return _fill_holes(combined)


def clean_leaf_mask(
    binary_mask: np.ndarray,
    settings: LeafMaskPreviewSettings,
    *,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply morphology and keep the largest plausible green-seeded component."""
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    kernel = _kernel(settings.morphology_radius)
    cleaned = cv2.morphologyEx(
        binary_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned_bool = cleaned.astype(bool)

    if seed_mask is not None and np.any(seed_mask):
        largest, _ = _largest_component_overlapping_seed(
            cleaned_bool,
            seed_mask,
            min_area_px=settings.min_component_area_px,
        )
    else:
        largest, _ = _largest_component_mask(
            cleaned_bool,
            min_area_px=settings.min_component_area_px,
        )
    return largest


def segment_leaf_preview(
    image_array: np.ndarray,
    settings: LeafMaskPreviewSettings | None = None,
) -> LeafMaskPreview:
    """Generate a preview-only leaf blade mask using green-seeded color rules."""
    active_settings = settings or LeafMaskPreviewSettings()
    snapshot = asdict(active_settings)
    warnings: list[str] = []

    try:
        height, width = _validate_image_array(image_array)
    except SegmentationError as exc:
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=[str(exc)],
            settings_snapshot=snapshot,
        )

    image_rgb = np.ascontiguousarray(image_array, dtype=np.uint8)
    blue_mask = detect_blue_background(image_rgb, active_settings)
    blue_area = int(np.count_nonzero(blue_mask))
    if blue_area == 0:
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["No blue petri-dish background detected."],
            settings_snapshot=snapshot,
        )

    image_area = height * width
    if blue_area / image_area < 0.02:
        warnings.append("Blue dish detection is weak; verify dish boundary visually.")

    dish_region = _petri_dish_interior_mask(blue_mask, active_settings)
    if not np.any(dish_region):
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["Could not estimate petri dish interior region."],
            settings_snapshot=snapshot,
        )

    exclusion_mask = (
        blue_mask
        | detect_glare_and_wrapping(image_rgb, active_settings)
    )

    green_seed = (
        detect_green_leaf_tissue(image_rgb, active_settings)
        & dish_region
        & ~exclusion_mask
    )
    green_seed_area = int(np.count_nonzero(green_seed))
    if green_seed_area < active_settings.min_green_seed_area_px:
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=[
                "No confident green leaf tissue seed detected inside the petri dish."
            ],
            settings_snapshot=snapshot,
        )

    if green_seed_area < active_settings.min_green_seed_area_px * 2:
        warnings.append("Green leaf seed area is small; preview confidence is low.")

    green_core, _ = _largest_component_overlapping_seed(
        green_seed,
        green_seed,
        min_area_px=active_settings.min_green_seed_area_px,
    )
    if not np.any(green_core):
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["Green leaf seed could not be isolated inside the dish."],
            settings_snapshot=snapshot,
        )

    brown_candidate = (
        detect_brown_necrosis_tissue(image_rgb, active_settings)
        & dish_region
        & ~exclusion_mask
    )
    leaf_expanded = _brown_connected_to_green(
        green_core,
        brown_candidate,
        active_settings,
    )
    leaf_expanded = leaf_expanded & ~exclusion_mask
    leaf_mask = clean_leaf_mask(
        leaf_expanded,
        active_settings,
        seed_mask=green_core,
    )
    leaf_mask = leaf_mask & ~exclusion_mask

    if not np.any(leaf_mask):
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["No leaf tissue detected after green-seeded segmentation."],
            settings_snapshot=snapshot,
        )

    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area < active_settings.min_component_area_px:
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["Detected leaf region is too small to be plausible."],
            settings_snapshot=snapshot,
        )

    dish_area = int(np.count_nonzero(dish_region))
    if dish_area > 0:
        leaf_fraction = leaf_area / dish_area
        if leaf_fraction > active_settings.max_leaf_dish_fraction:
            warnings.append(
                "Leaf mask covers a large fraction of the dish interior; verify visually."
            )

    green_overlap = int(np.count_nonzero(leaf_mask & green_core))
    if green_overlap < active_settings.min_green_seed_area_px // 2:
        warnings.append("Final mask has weak overlap with green leaf seed.")

    return LeafMaskPreview(
        mask=leaf_mask,
        status=LeafMaskPreviewStatus.SUCCESS,
        warnings=warnings,
        settings_snapshot=snapshot,
    )


def segment_leaf_and_disease(*_args, **_kwargs):
    """Separate leaf tissue from background and identify necrotic regions."""
    raise NotImplementedError(
        "Final leaf/disease segmentation is not implemented in Phase 3."
    )
