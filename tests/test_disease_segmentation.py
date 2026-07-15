"""Tests for deterministic disease mask preview segmentation."""

from __future__ import annotations

import inspect
import unittest

import cv2
import numpy as np

from neopest.disease_segmentation import (
    DiseaseMaskPreviewSettings,
    DiseaseMaskPreviewStatus,
    enforce_mask_invariants,
    segment_disease_preview,
)
from neopest.metadata import SampleRecord
from neopest.results import AnalysisStatus
from neopest.segmentation import LeafMaskPreviewStatus, segment_leaf_preview
from tests.test_segmentation import (
    _make_central_inoculation_leaf_image,
    _make_realistic_lab_image,
    _make_synthetic_leaf_image,
)


def _make_tan_lesion_leaf_image(
    *,
    width: int = 400,
    height: int = 400,
) -> tuple[np.ndarray, tuple[int, int]]:
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (width // 2, height // 2 + 30)
    cv2.ellipse(image, center, (35, 25), 0, 0, 360, (140, 120, 70), thickness=-1)
    return image, center


def _make_dark_necrosis_leaf_image() -> tuple[np.ndarray, tuple[int, int]]:
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 25)
    cv2.ellipse(image, center, (30, 35), 0, 0, 360, (55, 40, 25), thickness=-1)
    return image, center


def _make_near_black_necrosis_leaf_image() -> tuple[np.ndarray, tuple[int, int]]:
    image = _make_synthetic_leaf_image(include_brown=True)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 20)
    cv2.ellipse(image, center, (24, 30), 0, 0, 360, (28, 22, 18), thickness=-1)
    return image, center


def _make_multi_lesion_leaf_image() -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = _make_synthetic_leaf_image(include_brown=False)
    centers = [
        (image.shape[1] // 2, image.shape[0] // 2 + 35),
        (image.shape[1] // 2 - 45, image.shape[0] // 2 + 10),
        (image.shape[1] // 2 + 40, image.shape[0] // 2 + 55),
    ]
    for center in centers:
        cv2.circle(image, center, 18, (100, 80, 40), thickness=-1)
    return image, centers


def _make_veined_leaf_image() -> tuple[np.ndarray, tuple[int, int]]:
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 10)
    vein_y = center[1]
    cv2.line(
        image,
        (center[0] - 60, vein_y),
        (center[0] + 60, vein_y),
        (25, 95, 35),
        3,
    )
    cv2.line(
        image,
        (center[0] - 20, vein_y - 40),
        (center[0] - 20, vein_y + 40),
        (25, 95, 35),
        2,
    )
    return image, (center[0], vein_y)


def _make_peripheral_brown_lesion_image() -> tuple[np.ndarray, tuple[int, int]]:
    """Brown candidate placed near the leaf edge (not centrally)."""
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 10)
    peripheral = (center[0] + 72, center[1] + 48)
    cv2.circle(image, peripheral, 20, (100, 80, 40), thickness=-1)
    return image, peripheral


def _make_matched_lesion_pair_image() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Identical brown patches at center and periphery for symmetry testing."""
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 30)
    peripheral = (center[0] + 70, center[1] + 40)
    color = (100, 80, 40)
    cv2.circle(image, center, 18, color, thickness=-1)
    cv2.circle(image, peripheral, 18, color, thickness=-1)
    return image, center, peripheral


def _make_chlorotic_only_patch_image() -> tuple[np.ndarray, tuple[int, int]]:
    """Yellow chlorotic patch without brown/necrotic hues."""
    image = _make_synthetic_leaf_image(include_brown=False)
    center = (image.shape[1] // 2, image.shape[0] // 2 + 25)
    cv2.circle(image, center, 22, (60, 210, 210), thickness=-1)
    return image, center


def _make_tiny_brown_patch_image() -> tuple[np.ndarray, np.ndarray]:
    """2x2 brown candidate pixels inside the leaf."""
    image = _make_synthetic_leaf_image(include_brown=False)
    leaf = segment_leaf_preview(image)
    assert leaf.mask is not None
    ys, xs = np.where(leaf.mask)
    anchor = (int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    patch_mask = np.zeros(leaf.mask.shape, dtype=bool)
    x0, y0 = anchor
    for dy in range(2):
        for dx in range(2):
            image[y0 + dy, x0 + dx] = (95, 75, 35)
            patch_mask[y0 + dy, x0 + dx] = True
    return image, patch_mask


def _segment_pair(image: np.ndarray):
    leaf = segment_leaf_preview(image)
    assert leaf.status is LeafMaskPreviewStatus.SUCCESS
    assert leaf.mask is not None
    disease = segment_disease_preview(image, leaf.mask)
    return leaf, disease


class TestDiseaseMaskPreview(unittest.TestCase):
    def test_segment_disease_preview_is_deterministic(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        first = segment_disease_preview(image, leaf.mask)
        second = segment_disease_preview(image, leaf.mask)
        assert first.disease_mask is not None
        assert second.disease_mask is not None
        assert first.healthy_tissue_mask is not None
        assert second.healthy_tissue_mask is not None
        self.assertTrue(np.array_equal(first.disease_mask, second.disease_mask))
        self.assertTrue(np.array_equal(first.healthy_tissue_mask, second.healthy_tissue_mask))

    def test_inputs_are_not_mutated(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        image_copy = image.copy()
        mask_copy = leaf.mask.copy()
        segment_disease_preview(image, leaf.mask)
        self.assertTrue(np.array_equal(image, image_copy))
        self.assertTrue(np.array_equal(leaf.mask, mask_copy))

    def test_peripheral_brown_candidate_retained(self) -> None:
        image, peripheral = _make_peripheral_brown_lesion_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(disease.disease_mask[peripheral[1], peripheral[0]])

    def test_equivalent_center_and_peripheral_candidates_treated_symmetrically(self) -> None:
        image, center, peripheral = _make_matched_lesion_pair_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        center_hit = bool(disease.disease_mask[center[1], center[0]])
        peripheral_hit = bool(disease.disease_mask[peripheral[1], peripheral[0]])
        self.assertEqual(center_hit, peripheral_hit)
        self.assertTrue(center_hit)

    def test_no_candidate_pixels_warns_without_crashing(self) -> None:
        image = _make_synthetic_leaf_image(include_brown=False)
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        conservative = DiseaseMaskPreviewSettings(
            brown_hue_min=200,
            brown_hue_max=210,
            necrosis_val_max=0,
            near_black_val_max=0,
            exg_deficit_threshold=1000.0,
            lab_ab_distance_threshold=1000.0,
        )
        preview = segment_disease_preview(image, leaf.mask, settings=conservative)
        self.assertEqual(preview.status, DiseaseMaskPreviewStatus.READY)
        assert preview.disease_mask is not None
        self.assertEqual(int(np.count_nonzero(preview.disease_mask)), 0)
        self.assertTrue(
            any("no candidate disease pixels" in warning.lower() for warning in preview.warnings)
        )

    def test_tiny_candidate_regions_are_not_removed_by_size(self) -> None:
        image, patch_mask = _make_tiny_brown_patch_image()
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        preview = segment_disease_preview(image, leaf.mask)
        assert preview.disease_mask is not None
        retained = int(np.count_nonzero(preview.disease_mask & patch_mask))
        self.assertEqual(retained, int(np.count_nonzero(patch_mask)))

    def test_default_settings_disable_morphology(self) -> None:
        settings = DiseaseMaskPreviewSettings()
        self.assertEqual(settings.morphology_radius, 0)

    def test_morphology_is_not_applied_by_default(self) -> None:
        from neopest.disease_segmentation import _apply_optional_morphology

        candidate = np.zeros((20, 20), dtype=bool)
        candidate[5:15, 5:15] = True
        candidate[5, 5] = False
        candidate[14, 14] = False
        default_settings = DiseaseMaskPreviewSettings()
        morphed_settings = DiseaseMaskPreviewSettings(morphology_radius=2)
        unchanged = _apply_optional_morphology(candidate, default_settings)
        changed = _apply_optional_morphology(candidate, morphed_settings)
        self.assertTrue(np.array_equal(unchanged, candidate))
        self.assertFalse(np.array_equal(changed, candidate))

    def test_chlorotic_only_tissue_not_in_candidate_mask(self) -> None:
        image, center = _make_chlorotic_only_patch_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertFalse(disease.disease_mask[center[1], center[0]])

    def test_insufficient_healthy_reference_disables_reference_dependent_cues(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        with_reference = segment_disease_preview(image, leaf.mask)
        without_reference = segment_disease_preview(
            image,
            leaf.mask,
            settings=DiseaseMaskPreviewSettings(min_healthy_reference_px=10**9),
        )
        self.assertTrue(
            any(
                "insufficient healthy-reference" in warning.lower()
                for warning in without_reference.warnings
            )
        )
        assert with_reference.disease_mask is not None
        assert without_reference.disease_mask is not None
        extra_from_reference = with_reference.disease_mask & ~without_reference.disease_mask
        self.assertEqual(int(np.count_nonzero(extra_from_reference)), 0)

    def test_insufficient_reference_still_detects_brown_necrosis(self) -> None:
        image, center = _make_dark_necrosis_leaf_image()
        leaf = segment_leaf_preview(image)
        assert leaf.mask is not None
        preview = segment_disease_preview(
            image,
            leaf.mask,
            settings=DiseaseMaskPreviewSettings(min_healthy_reference_px=10**9),
        )
        assert preview.disease_mask is not None
        self.assertTrue(preview.disease_mask[center[1], center[0]])

    def test_healthy_mask_equals_leaf_minus_disease(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf, disease = _segment_pair(image)
        assert leaf.mask is not None
        assert disease.disease_mask is not None
        assert disease.healthy_tissue_mask is not None
        expected = leaf.mask & ~disease.disease_mask
        self.assertTrue(np.array_equal(disease.healthy_tissue_mask, expected))

    def test_segment_disease_preview_returns_masks_for_synthetic_image(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf, disease = _segment_pair(image)
        self.assertEqual(disease.status, DiseaseMaskPreviewStatus.READY)
        self.assertIsNotNone(disease.disease_mask)
        self.assertIsNotNone(disease.healthy_tissue_mask)
        assert disease.disease_mask is not None
        assert disease.healthy_tissue_mask is not None
        self.assertEqual(disease.disease_mask.shape, leaf.mask.shape)
        self.assertEqual(disease.healthy_tissue_mask.shape, leaf.mask.shape)
        self.assertEqual(disease.disease_mask.dtype, bool)
        self.assertEqual(disease.healthy_tissue_mask.dtype, bool)

    def test_disease_mask_never_outside_whole_leaf(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf, disease = _segment_pair(image)
        assert leaf.mask is not None
        assert disease.disease_mask is not None
        outside = disease.disease_mask & ~leaf.mask
        self.assertEqual(int(np.count_nonzero(outside)), 0)

    def test_healthy_and_disease_do_not_overlap(self) -> None:
        image = _make_synthetic_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        assert disease.healthy_tissue_mask is not None
        overlap = int(np.count_nonzero(disease.disease_mask & disease.healthy_tissue_mask))
        self.assertEqual(overlap, 0)

    def test_healthy_plus_disease_reconstructs_whole_leaf(self) -> None:
        image = _make_synthetic_leaf_image()
        leaf, disease = _segment_pair(image)
        assert leaf.mask is not None
        assert disease.disease_mask is not None
        assert disease.healthy_tissue_mask is not None
        union = disease.disease_mask | disease.healthy_tissue_mask
        self.assertTrue(np.array_equal(union, leaf.mask))

    def test_central_brown_inoculation_detected(self) -> None:
        image, center = _make_central_inoculation_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(disease.disease_mask[center[1], center[0]])

    def test_tan_lesion_detected(self) -> None:
        image, center = _make_tan_lesion_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(disease.disease_mask[center[1], center[0]])

    def test_dark_brown_necrosis_detected(self) -> None:
        image, center = _make_dark_necrosis_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(disease.disease_mask[center[1], center[0]])

    def test_near_black_necrosis_detected(self) -> None:
        image, center = _make_near_black_necrosis_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(disease.disease_mask[center[1], center[0]])

    def test_multiple_lesion_regions_may_be_detected(self) -> None:
        image, centers = _make_multi_lesion_leaf_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        detected = sum(
            1 for center in centers if disease.disease_mask[center[1], center[0]]
        )
        self.assertGreaterEqual(detected, 2)

    def test_healthy_green_blade_mostly_healthy(self) -> None:
        image = _make_synthetic_leaf_image(include_brown=False)
        leaf, disease = _segment_pair(image)
        assert leaf.mask is not None
        assert disease.healthy_tissue_mask is not None
        healthy_fraction = np.count_nonzero(disease.healthy_tissue_mask) / np.count_nonzero(
            leaf.mask
        )
        self.assertGreater(healthy_fraction, 0.7)

    def test_realistic_scene_excludes_outside_brown(self) -> None:
        image, markers = _make_realistic_lab_image(include_disconnected_brown=True)
        leaf, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertTrue(
            disease.disease_mask[
                markers["brown_inside_leaf"][1],
                markers["brown_inside_leaf"][0],
            ]
        )
        self.assertFalse(
            disease.disease_mask[
                markers["brown_disconnected"][1],
                markers["brown_disconnected"][0],
            ]
        )

    def test_realistic_scene_excludes_tape_glare_and_wrapping(self) -> None:
        image, markers = _make_realistic_lab_image()
        _, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        self.assertFalse(
            disease.disease_mask[markers["tape_glare"][1], markers["tape_glare"][0]]
        )
        self.assertFalse(
            disease.disease_mask[markers["white_wrap"][1], markers["white_wrap"][0]]
        )

    def test_veins_not_broadly_classified_as_disease(self) -> None:
        image, vein_center = _make_veined_leaf_image()
        leaf, disease = _segment_pair(image)
        assert disease.disease_mask is not None
        vein_region = np.zeros(leaf.mask.shape, dtype=bool)  # type: ignore[union-attr]
        y, x = vein_center[1], vein_center[0]
        vein_region[y - 2 : y + 3, x - 60 : x + 60] = True
        vein_in_disease = int(np.count_nonzero(disease.disease_mask & vein_region & leaf.mask))  # type: ignore[union-attr]
        vein_pixels = int(np.count_nonzero(vein_region & leaf.mask))  # type: ignore[union-attr]
        if vein_pixels > 0:
            self.assertLess(vein_in_disease / vein_pixels, 0.6)

    def test_empty_whole_leaf_mask_fails_controlled(self) -> None:
        image = _make_synthetic_leaf_image()
        empty_mask = np.zeros(image.shape[:2], dtype=bool)
        preview = segment_disease_preview(image, empty_mask)
        self.assertEqual(preview.status, DiseaseMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.disease_mask)
        self.assertIsNone(preview.healthy_tissue_mask)

    def test_shape_mismatch_fails_controlled(self) -> None:
        image = _make_synthetic_leaf_image()
        wrong_mask = np.zeros((10, 10), dtype=bool)
        preview = segment_disease_preview(image, wrong_mask)
        self.assertEqual(preview.status, DiseaseMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.disease_mask)

    def test_failed_whole_leaf_cannot_produce_disease_preview_via_gui_pattern(self) -> None:
        image = np.full((120, 120, 3), 245, dtype=np.uint8)
        leaf = segment_leaf_preview(image)
        self.assertEqual(leaf.status, LeafMaskPreviewStatus.FAILED)
        self.assertIsNone(leaf.mask)

    def test_disease_preview_does_not_change_analysis_status(self) -> None:
        sample = SampleRecord.create(
            condition_group_id="condition_group_test",
            display_name="Synthetic",
            raw_image_path="raw_images/group/sample.png",
        )
        image = _make_synthetic_leaf_image()
        leaf, disease = _segment_pair(image)
        self.assertEqual(disease.status, DiseaseMaskPreviewStatus.READY)
        self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)

    def test_enforce_mask_invariants(self) -> None:
        whole_leaf = np.zeros((20, 20), dtype=bool)
        whole_leaf[5:15, 5:15] = True
        disease = np.zeros_like(whole_leaf)
        disease[8:12, 8:12] = True
        disease[0, 0] = True
        disease, healthy = enforce_mask_invariants(whole_leaf, disease)
        self.assertFalse(disease[0, 0])
        self.assertTrue(np.array_equal(whole_leaf, disease | healthy))
        self.assertEqual(int(np.count_nonzero(disease & healthy)), 0)


class TestDiseaseSegmentationArchitecture(unittest.TestCase):
    def test_disease_segmentation_has_no_gui_imports(self) -> None:
        import neopest.disease_segmentation as disease_segmentation

        source = inspect.getsource(disease_segmentation)
        self.assertNotIn("PySide6", source)
        self.assertNotIn("gui", source)

    def test_gui_canvas_does_not_import_disease_segmentation(self) -> None:
        import neopest.gui_canvas as gui_canvas

        source = inspect.getsource(gui_canvas)
        self.assertNotIn("disease_segmentation", source)

    def test_gui_layout_builders_do_not_import_disease_segmentation(self) -> None:
        import neopest.gui_layout_builders as layout_builders

        source = inspect.getsource(layout_builders)
        self.assertNotIn("disease_segmentation", source)


if __name__ == "__main__":
    unittest.main()
