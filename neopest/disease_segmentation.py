"""Deterministic brown/necrotic candidate-mask preview within a whole-leaf mask.

Phase 4 preview only: highlights candidate brown/necrotic tissue inside an
existing whole-leaf mask. Does not compute or persist disease area metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from neopest.segmentation import (
    PREVIEW_METHOD,
    SETTINGS_VERSION as LEAF_SETTINGS_VERSION,
    detect_glare_and_wrapping,
    detect_green_leaf_tissue,
)

DISEASE_PREVIEW_METHOD = "brown_necrotic_candidate"
DISEASE_SETTINGS_VERSION = 2


class DiseaseMaskPreviewStatus(str, Enum):
    """Outcome of a non-persistent disease mask preview."""

    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class DiseaseMaskPreviewSettings:
    """Versioned defaults for Phase 4 brown/necrotic candidate preview."""

    settings_version: int = DISEASE_SETTINGS_VERSION
    # Confident healthy-green reference (HSV + ExG, aligned with leaf preview).
    green_hue_min: int = 32
    green_hue_max: int = 92
    green_sat_min: int = 35
    green_val_min: int = 25
    exg_threshold: float = 12.0
    min_healthy_reference_px: int = 150
    # Brown / tan necrotic hues (HSV).
    brown_hue_min: int = 5
    brown_hue_max: int = 35
    brown_sat_min: int = 22
    brown_val_min: int = 18
    brown_val_max: int = 185
    # Dark necrosis and near-black dead tissue.
    necrosis_val_max: int = 62
    necrosis_lab_l_max: int = 48
    near_black_val_max: int = 38
    near_black_sat_max: int = 85
    # Deviation from sample healthy reference (disabled when reference is weak).
    exg_deficit_threshold: float = 10.0
    lab_ab_distance_threshold: float = 18.0
    # Optional morphology for future experiments only; 0 disables close/open.
    morphology_radius: int = 0
    healthy_reference_erode_radius: int = 3
    vein_hue_min: int = 32
    vein_hue_max: int = 92
    vein_sat_min: int = 30
    vein_val_min: int = 25
    vein_val_max: int = 98
    vein_exg_min: float = 8.0


@dataclass
class DiseaseMaskPreview:
    """In-memory candidate disease and healthy-tissue preview masks."""

    disease_mask: np.ndarray | None
    healthy_tissue_mask: np.ndarray | None
    status: DiseaseMaskPreviewStatus
    warnings: list[str] = field(default_factory=list)
    method: str = DISEASE_PREVIEW_METHOD
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    whole_leaf_method: str = PREVIEW_METHOD
    whole_leaf_settings_version: int = LEAF_SETTINGS_VERSION


class DiseaseSegmentationError(ValueError):
    """Raised when disease segmentation input is invalid."""


def _kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _validate_inputs(
    image_array: np.ndarray,
    whole_leaf_mask: np.ndarray,
) -> tuple[int, int]:
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise DiseaseSegmentationError(
            f"Expected RGB image array with shape (H, W, 3); got {image_array.shape}"
        )
    if whole_leaf_mask.ndim != 2:
        raise DiseaseSegmentationError(
            f"Expected 2D whole-leaf mask; got shape {whole_leaf_mask.shape}"
        )
    height, width = image_array.shape[:2]
    if whole_leaf_mask.shape != (height, width):
        raise DiseaseSegmentationError(
            "Whole-leaf mask shape must match image shape "
            f"{(height, width)}; got {whole_leaf_mask.shape}"
        )
    if height == 0 or width == 0:
        raise DiseaseSegmentationError("Image array has zero width or height")
    return height, width


def _erode_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(binary_mask):
        return binary_mask.copy()
    kernel = _kernel(radius)
    return cv2.erode(binary_mask.astype(np.uint8), kernel).astype(bool)


def _compute_exg(image_rgb: np.ndarray) -> np.ndarray:
    red = image_rgb[:, :, 0].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    blue = image_rgb[:, :, 2].astype(np.float32)
    return 2.0 * green - red - blue


def _healthy_reference_mask(
    image_rgb: np.ndarray,
    whole_leaf_mask: np.ndarray,
    settings: DiseaseMaskPreviewSettings,
) -> np.ndarray:
    """Confident healthy-green pixels inside the whole-leaf mask."""
    green_leaf = detect_green_leaf_tissue(
        image_rgb,
        _green_settings_from_disease(settings),
    )
    confident = green_leaf & whole_leaf_mask
    return _erode_mask(confident, settings.healthy_reference_erode_radius)


def _green_settings_from_disease(settings: DiseaseMaskPreviewSettings):
    """Build minimal green-detection kwargs compatible with leaf preview."""
    from neopest.segmentation import LeafMaskPreviewSettings

    return LeafMaskPreviewSettings(
        green_hue_min=settings.green_hue_min,
        green_hue_max=settings.green_hue_max,
        green_sat_min=settings.green_sat_min,
        green_val_min=settings.green_val_min,
        exg_threshold=settings.exg_threshold,
    )


def _glare_settings_from_disease(settings: DiseaseMaskPreviewSettings):
    from neopest.segmentation import LeafMaskPreviewSettings

    return LeafMaskPreviewSettings()


def _median_lab_reference(
    image_rgb: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[float, float, float]:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if not np.any(reference_mask):
        return 0.0, 0.0, 0.0
    pixels = lab[reference_mask]
    return (
        float(np.median(pixels[:, 1])),
        float(np.median(pixels[:, 2])),
        float(np.median(pixels[:, 0])),
    )


def _detect_vein_like(
    hsv: np.ndarray,
    exg: np.ndarray,
    settings: DiseaseMaskPreviewSettings,
) -> np.ndarray:
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return (
        (hue >= settings.vein_hue_min)
        & (hue <= settings.vein_hue_max)
        & (sat >= settings.vein_sat_min)
        & (val >= settings.vein_val_min)
        & (val <= settings.vein_val_max)
        & (exg >= settings.vein_exg_min)
    )


def _apply_optional_morphology(
    candidate_mask: np.ndarray,
    settings: DiseaseMaskPreviewSettings,
) -> np.ndarray:
    """Apply close/open only when ``morphology_radius`` is positive."""
    if settings.morphology_radius <= 0 or not np.any(candidate_mask):
        return candidate_mask.astype(bool).copy()

    kernel = _kernel(settings.morphology_radius)
    cleaned = cv2.morphologyEx(
        candidate_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned.astype(bool)


def enforce_mask_invariants(
    whole_leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce disease ⊆ leaf and healthy = leaf \\ disease."""
    disease = disease_mask.astype(bool) & whole_leaf_mask.astype(bool)
    healthy = whole_leaf_mask.astype(bool) & ~disease
    return disease, healthy


def segment_disease_preview(
    image_array: np.ndarray,
    whole_leaf_mask: np.ndarray,
    *,
    whole_leaf_method: str = PREVIEW_METHOD,
    whole_leaf_settings_version: int = LEAF_SETTINGS_VERSION,
    settings: DiseaseMaskPreviewSettings | None = None,
) -> DiseaseMaskPreview:
    """Classify candidate brown/necrotic tissue inside ``whole_leaf_mask`` for preview."""
    active_settings = settings or DiseaseMaskPreviewSettings()
    snapshot = asdict(active_settings)
    warnings: list[str] = []

    try:
        _validate_inputs(image_array, whole_leaf_mask)
    except DiseaseSegmentationError as exc:
        return DiseaseMaskPreview(
            disease_mask=None,
            healthy_tissue_mask=None,
            status=DiseaseMaskPreviewStatus.FAILED,
            warnings=[str(exc)],
            settings_snapshot=snapshot,
            whole_leaf_method=whole_leaf_method,
            whole_leaf_settings_version=whole_leaf_settings_version,
        )

    leaf_mask = whole_leaf_mask.astype(bool)
    if not np.any(leaf_mask):
        return DiseaseMaskPreview(
            disease_mask=None,
            healthy_tissue_mask=None,
            status=DiseaseMaskPreviewStatus.FAILED,
            warnings=["Whole-leaf mask is empty; disease preview requires a valid leaf mask."],
            settings_snapshot=snapshot,
            whole_leaf_method=whole_leaf_method,
            whole_leaf_settings_version=whole_leaf_settings_version,
        )

    image_rgb = np.ascontiguousarray(image_array, dtype=np.uint8)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    exg = _compute_exg(image_rgb)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_l = lab[:, :, 0]
    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]

    healthy_ref = _healthy_reference_mask(image_rgb, leaf_mask, active_settings)
    healthy_ref_px = int(np.count_nonzero(healthy_ref))
    reference_ready = healthy_ref_px >= active_settings.min_healthy_reference_px
    if not reference_ready:
        warnings.append(
            "Insufficient healthy-reference pixels; reference-dependent contrast cues disabled."
        )

    green_confident = detect_green_leaf_tissue(
        image_rgb,
        _green_settings_from_disease(active_settings),
    ) & leaf_mask

    brown_tan = (
        (hue >= active_settings.brown_hue_min)
        & (hue <= active_settings.brown_hue_max)
        & (sat >= active_settings.brown_sat_min)
        & (val >= active_settings.brown_val_min)
        & (val <= active_settings.brown_val_max)
    )
    dark_necrosis = (val <= active_settings.necrosis_val_max) | (
        lab_l <= active_settings.necrosis_lab_l_max
    )
    near_black = (val <= active_settings.near_black_val_max) | (
        (lab_l <= active_settings.necrosis_lab_l_max) & (sat <= active_settings.near_black_sat_max)
    )

    reference_independent = leaf_mask & (
        (brown_tan & ~green_confident)
        | (dark_necrosis & ~green_confident)
        | near_black
    )

    if reference_ready:
        healthy_exg_median = float(np.median(exg[healthy_ref]))
        ref_a, ref_b, _ref_l = _median_lab_reference(image_rgb, healthy_ref)
        ab_distance = np.sqrt((lab_a - ref_a) ** 2 + (lab_b - ref_b) ** 2)
        reduced_green = exg < (healthy_exg_median - active_settings.exg_deficit_threshold)
        chromatic_shift = ab_distance > active_settings.lab_ab_distance_threshold
        reference_dependent = leaf_mask & (
            (brown_tan & reduced_green)
            | (chromatic_shift & reduced_green & ~green_confident)
        )
        disease_evidence = reference_independent | reference_dependent
    else:
        disease_evidence = reference_independent

    vein_like = _detect_vein_like(hsv, exg, active_settings)
    glare = detect_glare_and_wrapping(image_rgb, _glare_settings_from_disease(active_settings))
    disease_evidence = disease_evidence & ~vein_like & ~glare

    disease_mask = _apply_optional_morphology(disease_evidence, active_settings)
    disease_mask, healthy_tissue_mask = enforce_mask_invariants(leaf_mask, disease_mask)

    if int(np.count_nonzero(disease_mask)) == 0:
        warnings.append("No candidate disease pixels found.")

    return DiseaseMaskPreview(
        disease_mask=disease_mask,
        healthy_tissue_mask=healthy_tissue_mask,
        status=DiseaseMaskPreviewStatus.READY,
        warnings=warnings,
        settings_snapshot=snapshot,
        whole_leaf_method=whole_leaf_method,
        whole_leaf_settings_version=whole_leaf_settings_version,
    )
