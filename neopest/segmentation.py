"""Deterministic whole-leaf segmentation for white moisture-pad assays.

Central brown or dark tissue stays inside the leaf blade. It is not treated as
background. Candidate disease is identified by a separate mask and is not a
diagnosis. This module returns the whole-leaf mask only.

The dish interior is white moisture-retaining paper. Leaf area is green-seeded:
connected brown, yellow, pale, and dark damaged tissue stay in the blade.
Paper texture, wrapping, the rim, the exterior, and a blue ruler do not
become leaf.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

PREVIEW_METHOD = "green_seeded_color_thresholding"
# Version 3 drops blue-pad dish detection. Saved version 2 results stay on
# disk and no longer match current provenance until a researcher recalculates.
SETTINGS_VERSION = 3


class LeafMaskPreviewStatus(str, Enum):
    """Outcome of a non-persistent leaf mask preview."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class LeafMaskPreviewSettings:
    """Versioned defaults for Phase 3B green-seeded leaf preview."""

    settings_version: int = SETTINGS_VERSION
    # Blue ruler and blue-cast cloth. Used only to keep those pixels out of the leaf.
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
    # White moisture-retaining paper is the only dish interior.
    paper_sat_max: int = 55
    paper_val_min: int = 85
    paper_exg_abs_max: float = 14.0
    paper_min_area_fraction: float = 0.06
    paper_max_area_fraction: float = 0.82
    paper_max_aspect: float = 2.2
    paper_max_border_fraction: float = 0.50
    yellow_hue_min: int = 8
    yellow_hue_max: int = 45
    yellow_sat_min: int = 40
    yellow_val_min: int = 40
    yellow_val_max: int = 250
    pale_hue_min: int = 8
    pale_hue_max: int = 95
    pale_val_min: int = 80
    pale_exg_min: float = 4.0
    dark_damage_val_max: int = 62
    dark_damage_sat_min: int = 25


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
    """Return blue ruler or blue-cast cloth pixels to exclude from the leaf."""
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


def _component_aspect(component: np.ndarray) -> float:
    ys, xs = np.nonzero(component)
    if ys.size == 0:
        return 0.0
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return max(height, width) / float(max(1, min(height, width)))


def _border_fraction(component: np.ndarray) -> float:
    border = np.zeros(component.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_count = int(np.count_nonzero(border))
    if border_count == 0:
        return 1.0
    return float(np.count_nonzero(component & border) / border_count)


def _dish_component_min_area(settings: LeafMaskPreviewSettings) -> int:
    return max(100, settings.min_component_area_px // 5)


def _largest_compact_paper_dish(
    paper_mask: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> tuple[np.ndarray, int]:
    """Return the largest paper component that can be a dish interior.

    A wide ruler, label, or table can be brighter than the pad. Those regions
    fail the aspect or border check and are not used as the dish.
    """
    if paper_mask.size == 0:
        return np.zeros_like(paper_mask, dtype=bool), 0
    component_count, labels = cv2.connectedComponents(paper_mask.astype(np.uint8))
    min_area = _dish_component_min_area(settings)
    image_area = float(paper_mask.size)
    best = np.zeros_like(paper_mask, dtype=bool)
    best_area = 0
    for label in range(1, component_count):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < min_area or area <= best_area:
            continue
        fraction = area / image_area
        if fraction < settings.paper_min_area_fraction or fraction > settings.paper_max_area_fraction:
            continue
        if _component_aspect(component) > settings.paper_max_aspect:
            continue
        if _border_fraction(component) > settings.paper_max_border_fraction:
            continue
        best = component
        best_area = area
    return best, best_area


def _exg(image_rgb: np.ndarray) -> np.ndarray:
    red = image_rgb[:, :, 0].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    blue = image_rgb[:, :, 2].astype(np.float32)
    return 2.0 * green - red - blue


def detect_extended_leaf_tissue(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return yellow, pale, reddish, and dark damaged leaf pixels.

    Used only to grow a green seed on white paper. Achromatic paper, glare,
    and white wrapping are not included. This is not disease classification.
    """
    hsv = _image_hsv(image_rgb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = image_rgb[:, :, 0].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    exg = _exg(image_rgb)

    yellow = (
        (hue >= settings.yellow_hue_min)
        & (hue <= settings.yellow_hue_max)
        & (sat >= settings.yellow_sat_min)
        & (val >= settings.yellow_val_min)
        & (val <= settings.yellow_val_max)
    )
    pale = (
        (hue >= settings.pale_hue_min)
        & (hue <= settings.pale_hue_max)
        & (sat > settings.glare_sat_max)
        & (val >= settings.pale_val_min)
        & (exg >= settings.pale_exg_min)
    )
    dark_damage = (
        (val <= settings.dark_damage_val_max)
        & (val >= settings.brown_val_min)
        & (sat >= settings.dark_damage_sat_min)
        & (red >= green)
    )
    reddish = (
        ((hue <= settings.brown_hue_min) | (hue >= 170))
        & (sat >= settings.brown_sat_min)
        & (val >= settings.brown_val_min)
        & (val <= settings.yellow_val_max)
        & (red >= green)
    )
    blue_pixels = (
        (hue >= settings.blue_hue_min)
        & (hue <= settings.blue_hue_max)
        & (sat >= settings.blue_sat_min)
    )
    return (yellow | pale | dark_damage | reddish) & ~blue_pixels & ~detect_glare_and_wrapping(
        image_rgb,
        settings,
    )


def detect_white_paper_background(
    image_rgb: np.ndarray,
    settings: LeafMaskPreviewSettings,
) -> np.ndarray:
    """Return white moisture-pad pixels, including folds and soft shadows.

    A 3x3 median and a small close bridge paper texture. Pixels that already
    match leaf-tissue rules are removed so the pad does not consume the blade.
    """
    smoothed = cv2.medianBlur(np.ascontiguousarray(image_rgb, dtype=np.uint8), 3)
    hsv = _image_hsv(smoothed)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    paper = (
        (sat <= settings.paper_sat_max)
        & (val >= settings.paper_val_min)
        & (np.abs(_exg(smoothed)) <= settings.paper_exg_abs_max)
    )
    closed = cv2.morphologyEx(
        paper.astype(np.uint8),
        cv2.MORPH_CLOSE,
        _kernel(max(1, settings.morphology_radius // 2)),
    ).astype(bool)
    leafish = (
        detect_green_leaf_tissue(image_rgb, settings)
        | detect_brown_necrosis_tissue(image_rgb, settings)
        | detect_extended_leaf_tissue(image_rgb, settings)
    )
    return closed & ~leafish


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
    image_area = height * width
    paper_mask = detect_white_paper_background(image_rgb, active_settings)
    dish_source, paper_area = _largest_compact_paper_dish(paper_mask, active_settings)
    if paper_area == 0:
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["No white moisture-pad background detected."],
            settings_snapshot=snapshot,
        )
    if image_area > 0 and paper_area / image_area < 0.08:
        warnings.append(
            "White paper dish detection is weak; verify dish boundary visually."
        )
    extended_tissue = detect_extended_leaf_tissue(image_rgb, active_settings)
    blue_mask = detect_blue_background(image_rgb, active_settings)

    dish_region = _petri_dish_interior_mask(dish_source, active_settings)
    if not np.any(dish_region):
        return LeafMaskPreview(
            mask=None,
            status=LeafMaskPreviewStatus.FAILED,
            warnings=["Could not estimate petri dish interior region."],
            settings_snapshot=snapshot,
        )

    # Paper, cotton wrapping, and blue ruler or blue-cast cloth stay outside the leaf.
    exclusion_mask = (
        paper_mask
        | blue_mask
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
        (
            detect_brown_necrosis_tissue(image_rgb, active_settings)
            | extended_tissue
        )
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


def render_leaf_mask_diagnostic(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return an RGB inspection image with the leaf mask tinted and outlined.

    The tint matches the GUI whole-leaf overlay. The mask and measurements
    are not modified.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SegmentationError(
            f"Expected RGB image array with shape (H, W, 3); got {image_rgb.shape}"
        )
    if mask.shape != image_rgb.shape[:2]:
        raise SegmentationError(
            "Diagnostic mask shape must match the image "
            f"{image_rgb.shape[:2]}; got {mask.shape}"
        )

    diagnostic = np.array(image_rgb, copy=True, dtype=np.uint8)
    tint = mask.astype(bool)
    if np.any(tint):
        # LEAF_MASK_OVERLAY_RGBA = (50, 220, 80, 110)
        alpha = 110.0 / 255.0
        color = np.array([50.0, 220.0, 80.0], dtype=np.float32)
        pixels = diagnostic[tint].astype(np.float32)
        diagnostic[tint] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(
            np.uint8
        )
        contours, _hierarchy = cv2.findContours(
            tint.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(diagnostic, contours, -1, (255, 255, 0), 2, lineType=cv2.LINE_8)
    return diagnostic


def segment_leaf_and_disease(*_args, **_kwargs):
    """Combined entry point kept unavailable.

    Whole-leaf and candidate-disease masks are produced separately.
    """
    raise NotImplementedError(
        "Combined leaf and disease segmentation is not available. "
        "Use the separate whole-leaf and candidate-disease functions."
    )
