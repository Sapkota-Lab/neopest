"""Tests for deterministic leaf mask preview segmentation."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from neopest.image_io import load_image_rgb_array
from neopest.metadata import SampleRecord
from neopest.results import AnalysisStatus, CalibrationStatus
from neopest.segmentation import (
    SETTINGS_VERSION,
    LeafMaskPreviewSettings,
    LeafMaskPreviewStatus,
    clean_leaf_mask,
    detect_green_leaf_tissue,
    detect_white_paper_background,
    exclude_white_wrapping,
    render_leaf_mask_diagnostic,
    segment_leaf_preview,
)


def _make_synthetic_leaf_image(
    *,
    width: int = 400,
    height: int = 400,
    include_brown: bool = True,
    include_white_wrapping: bool = True,
) -> np.ndarray:
    """Build a white-pad / green-leaf / optional brown / white-wrap test image."""
    image = np.full((height, width, 3), (16, 16, 18), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(image, center, 140, (236, 234, 230), thickness=-1)
    cv2.ellipse(image, (center[0], center[1] + 10), (85, 115), 0, 0, 360, (45, 170, 55), -1)
    if include_brown:
        cv2.ellipse(image, (center[0], center[1] + 20), (28, 38), 0, 0, 360, (95, 75, 35), -1)
    if include_white_wrapping:
        cv2.rectangle(
            image,
            (center[0] - 18, center[1] - 95),
            (center[0] + 18, center[1] - 55),
            (250, 250, 250),
            thickness=-1,
        )
    return image


def _make_realistic_lab_image(
    *,
    width: int = 800,
    height: int = 600,
    include_disconnected_brown: bool = True,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Build a lab-like scene with dish, tape, wrap, leaf, and outside clutter."""
    image = np.full((height, width, 3), 25, dtype=np.uint8)

    # White table and ruler outside the dish.
    cv2.rectangle(image, (0, 430), (width, height), (238, 238, 238), thickness=-1)
    cv2.rectangle(image, (40, 445), (width - 40, 505), (205, 205, 205), thickness=-1)
    for tick_x in range(80, width - 80, 50):
        cv2.line(image, (tick_x, 445), (tick_x, 505), (40, 40, 40), 2)

    # Label-like block outside dish.
    cv2.rectangle(image, (20, 20), (150, 90), (255, 248, 200), thickness=-1)

    dish_center = (width // 2, 270)
    cv2.circle(image, dish_center, 165, (236, 234, 230), thickness=-1)

    # Large tape/glare rectangle inside dish, away from the leaf body.
    cv2.rectangle(image, (dish_center[0] + 35, dish_center[1] - 95), (dish_center[0] + 150, dish_center[1] - 15), (225, 232, 245), thickness=-1)

    # Green leaf blade in lower dish area.
    cv2.ellipse(image, (dish_center[0] - 10, dish_center[1] + 55), (90, 105), 0, 0, 360, (45, 170, 55), thickness=-1)

    # Brown necrosis inside leaf.
    cv2.ellipse(image, (dish_center[0] - 10, dish_center[1] + 65), (30, 40), 0, 0, 360, (95, 75, 35), thickness=-1)

    # White stem wrap above leaf petiole (drawn after leaf).
    cv2.rectangle(image, (dish_center[0] - 30, dish_center[1] - 25), (dish_center[0] + 10, dish_center[1] + 15), (252, 252, 252), thickness=-1)

    markers = {
        "brown_inside_leaf": (dish_center[0] - 10, dish_center[1] + 65),
        "white_wrap": (dish_center[0] - 10, dish_center[1] - 5),
        "tape_glare": (dish_center[0] + 95, dish_center[1] - 55),
        "ruler": (120, 475),
        "label": (80, 55),
        "outside_dark": (30, height - 30),
    }

    if include_disconnected_brown:
        disconnected_center = (dish_center[0] - 130, dish_center[1] - 60)
        cv2.circle(image, disconnected_center, 24, (90, 70, 30), thickness=-1)
        markers["brown_disconnected"] = disconnected_center

    return image, markers


def _make_central_inoculation_leaf_image(
    *,
    width: int = 400,
    height: int = 400,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Leaf with a large central brown inoculation zone (green margin ring)."""
    image = np.full((height, width, 3), (16, 16, 18), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(image, center, 140, (236, 234, 230), thickness=-1)
    cv2.ellipse(image, (center[0], center[1] + 10), (85, 115), 0, 0, 360, (45, 170, 55), -1)
    # Large central inoculation/necrosis region replacing much of the blade interior.
    cv2.ellipse(image, (center[0], center[1] + 15), (58, 78), 0, 0, 360, (95, 75, 35), -1)
    inoculation_center = (center[0], center[1] + 15)
    return image, inoculation_center


def _make_white_paper_assay_image() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """White moisture pad inside a dish, with clutter that must stay outside the leaf.

    Texture, a fold, and a shadow are part of the pad. The blue ruler, label,
    rim, exterior, wrapping, and a disconnected yellow spot are not leaf.
    """
    height, width = 640, 480
    image = np.full((height, width, 3), (12, 14, 18), dtype=np.uint8)
    center = (240, 280)
    cv2.circle(image, center, 158, (48, 50, 52), thickness=-1)
    cv2.circle(image, center, 142, (236, 234, 230), thickness=-1)

    yy, xx = np.ogrid[:height, :width]
    paper_region = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= 142**2
    noise = np.random.default_rng(7).integers(-14, 15, size=image.shape, dtype=np.int16)
    textured = image.astype(np.int16)
    textured[paper_region] = np.clip(
        textured[paper_region] + noise[paper_region],
        0,
        255,
    )
    image = textured.astype(np.uint8)

    cv2.line(image, (120, 170), (132, 390), (168, 166, 162), 4)
    cv2.ellipse(image, (150, 210), (26, 16), 15, 0, 360, (148, 146, 142), thickness=-1)

    leaf_center = (center[0] - 8, center[1] + 18)
    cv2.ellipse(image, leaf_center, (78, 96), 0, 0, 360, (42, 158, 52), thickness=-1)
    cv2.ellipse(
        image,
        (leaf_center[0], leaf_center[1] + 8),
        (24, 30),
        0,
        0,
        360,
        (96, 74, 36),
        thickness=-1,
    )
    yellow_at = (leaf_center[0] - 30, leaf_center[1] - 20)
    cv2.circle(image, yellow_at, 14, (220, 190, 50), thickness=-1)
    pale_at = (leaf_center[0] + 24, leaf_center[1] + 36)
    cv2.circle(image, pale_at, 12, (196, 188, 120), thickness=-1)
    dark_at = (leaf_center[0] - 10, leaf_center[1] + 28)
    cv2.circle(image, dark_at, 8, (42, 24, 16), thickness=-1)
    hole_at = (leaf_center[0] + 8, leaf_center[1] - 36)
    cv2.circle(image, hole_at, 7, (236, 234, 230), thickness=-1)
    clutter_yellow = (center[0] + 95, center[1] - 90)
    cv2.circle(image, clutter_yellow, 10, (220, 190, 50), thickness=-1)
    disconnected_brown = (180, 160)
    cv2.circle(image, disconnected_brown, 12, (90, 70, 30), thickness=-1)

    wrap_at = (leaf_center[0] + 70, leaf_center[1] - 10)
    cv2.rectangle(
        image,
        (wrap_at[0] - 16, wrap_at[1] - 14),
        (wrap_at[0] + 22, wrap_at[1] + 16),
        (250, 250, 250),
        thickness=-1,
    )
    petiole_at = (wrap_at[0] - 28, wrap_at[1])

    cv2.rectangle(image, (20, 560), (460, 620), (36, 78, 188), thickness=-1)
    for tick_x in range(40, 450, 30):
        cv2.line(image, (tick_x, 560), (tick_x, 588), (20, 20, 20), 2)
    cv2.rectangle(image, (30, 24), (150, 70), (248, 248, 246), thickness=-1)

    markers = {
        "green_blade": (leaf_center[0], leaf_center[1] - 50),
        "brown": (leaf_center[0], leaf_center[1] + 8),
        "yellow": yellow_at,
        "pale": pale_at,
        "dark_damage": dark_at,
        "paper_hole": hole_at,
        "petiole": petiole_at,
        "white_wrap": wrap_at,
        "paper_fold": (126, 280),
        "paper_shadow": (150, 210),
        "disconnected_yellow": clutter_yellow,
        "disconnected_brown": disconnected_brown,
        "rim": (center[0], center[1] - 150),
        "exterior": (12, 12),
        "blue_ruler": (240, 590),
        "label": (80, 46),
    }
    return image, markers


class TestLeafMaskPreview(unittest.TestCase):
    def test_segment_leaf_preview_returns_mask_for_synthetic_image(self) -> None:
        image = _make_synthetic_leaf_image()
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        self.assertIsNotNone(preview.mask)
        assert preview.mask is not None
        self.assertTrue(np.any(preview.mask))
        self.assertEqual(preview.mask.shape[:2], image.shape[:2])

    def test_brown_pixels_remain_in_final_leaf_mask(self) -> None:
        image = _make_synthetic_leaf_image(include_brown=True)
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        assert preview.mask is not None
        final_mask = preview.mask

        brown_pixels = (
            (image[:, :, 0] < 120)
            & (image[:, :, 1] < 90)
            & (image[:, :, 2] < 50)
        )
        brown_in_final_mask = int(np.count_nonzero(final_mask & brown_pixels))
        self.assertGreater(
            brown_in_final_mask,
            0,
            "Expected brown necrosis pixels in the final returned mask after cleanup",
        )

        brown_center = (image.shape[1] // 2, image.shape[0] // 2 + 20)
        self.assertTrue(
            final_mask[brown_center[1], brown_center[0]],
            "Brown patch center must remain in final LeafMaskPreview.mask",
        )

    def test_central_inoculation_brown_remains_in_leaf_mask(self) -> None:
        """Central inoculation/necrosis must stay in the leaf blade, not be treated as artifact."""
        image, inoculation_center = _make_central_inoculation_leaf_image()
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        assert preview.mask is not None
        ix, iy = inoculation_center
        self.assertTrue(
            preview.mask[iy, ix],
            "Central inoculation brown must remain inside the leaf blade mask",
        )

    def test_white_wrapping_excluded_when_present(self) -> None:
        image = _make_synthetic_leaf_image(include_white_wrapping=True)
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        assert preview.mask is not None

        wrap_y = image.shape[0] // 2 - 75
        wrap_x = image.shape[1] // 2
        self.assertFalse(preview.mask[wrap_y, wrap_x])

    def test_empty_image_returns_failed_preview(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        preview = segment_leaf_preview(empty)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.mask)
        self.assertTrue(preview.warnings)

    def test_no_white_pad_fails_controlled(self) -> None:
        image = np.full((120, 120, 3), 245, dtype=np.uint8)
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.mask)
        self.assertIn("No white moisture-pad background detected.", preview.warnings)

    def test_saved_leaf_settings_version_is_current_white_pad_version(self) -> None:
        preview = segment_leaf_preview(_make_synthetic_leaf_image())
        self.assertEqual(preview.settings_snapshot["settings_version"], SETTINGS_VERSION)
        self.assertEqual(SETTINGS_VERSION, 3)

    def test_invalid_shape_returns_failed_preview(self) -> None:
        preview = segment_leaf_preview(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(preview.status, LeafMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.mask)

    def test_preview_does_not_require_gui(self) -> None:
        image = _make_synthetic_leaf_image()
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)

    def test_preview_failure_does_not_change_analysis_status(self) -> None:
        sample = SampleRecord.create(
            condition_group_id="condition_group_test",
            display_name="Synthetic",
            raw_image_path="raw_images/group/sample.png",
        )
        preview = segment_leaf_preview(np.full((120, 120, 3), 245, dtype=np.uint8))
        self.assertEqual(preview.status, LeafMaskPreviewStatus.FAILED)
        self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)

    def test_previewing_does_not_change_analysis_status(self) -> None:
        sample = SampleRecord.create(
            condition_group_id="condition_group_test",
            display_name="Synthetic",
            raw_image_path="raw_images/group/sample.png",
        )
        self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)
        preview = segment_leaf_preview(_make_synthetic_leaf_image())
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        self.assertEqual(sample.analysis_status, AnalysisStatus.NOT_RUN)

    def test_calibration_metadata_independent_of_preview(self) -> None:
        sample = SampleRecord.create(
            condition_group_id="condition_group_test",
            display_name="Synthetic",
            raw_image_path="raw_images/group/sample.png",
        )
        sample.calibration_status = CalibrationStatus.MANUAL
        segment_leaf_preview(_make_synthetic_leaf_image())
        self.assertEqual(sample.calibration_status, CalibrationStatus.MANUAL)

    def test_load_image_rgb_array_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.png"
            image = _make_synthetic_leaf_image()
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            loaded = load_image_rgb_array(path)
            self.assertEqual(loaded.shape, image.shape)

    def test_exclude_white_wrapping_removes_bright_pixels(self) -> None:
        image = _make_synthetic_leaf_image(include_white_wrapping=True)
        settings = LeafMaskPreviewSettings()
        candidate = np.ones(image.shape[:2], dtype=bool)
        filtered = exclude_white_wrapping(image, candidate, settings)
        wrap_y = image.shape[0] // 2 - 75
        wrap_x = image.shape[1] // 2
        self.assertFalse(filtered[wrap_y, wrap_x])

    def test_clean_leaf_mask_keeps_largest_component(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        mask[2:8, 2:8] = True
        cleaned = clean_leaf_mask(
            mask,
            LeafMaskPreviewSettings(min_component_area_px=50, morphology_radius=1),
        )
        self.assertGreaterEqual(int(np.count_nonzero(cleaned)), 3500)


class TestRealisticLabSegmentation(unittest.TestCase):
    def test_realistic_scene_segments_green_seeded_leaf(self) -> None:
        image, markers = _make_realistic_lab_image()
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        assert preview.mask is not None
        final_mask = preview.mask

        self.assertTrue(final_mask[markers["brown_inside_leaf"][1], markers["brown_inside_leaf"][0]])
        self.assertFalse(final_mask[markers["white_wrap"][1], markers["white_wrap"][0]])
        self.assertFalse(final_mask[markers["tape_glare"][1], markers["tape_glare"][0]])
        self.assertFalse(final_mask[markers["ruler"][1], markers["ruler"][0]])
        self.assertFalse(final_mask[markers["label"][1], markers["label"][0]])
        self.assertFalse(final_mask[markers["outside_dark"][1], markers["outside_dark"][0]])

    def test_tape_glare_not_selected_over_leaf(self) -> None:
        image, markers = _make_realistic_lab_image()
        preview = segment_leaf_preview(image)
        assert preview.mask is not None
        final_mask = preview.mask

        tape_region = np.zeros(image.shape[:2], dtype=bool)
        tape_region[
            markers["tape_glare"][1] - 40 : markers["tape_glare"][1] + 40,
            markers["tape_glare"][0] - 90 : markers["tape_glare"][0] + 90,
        ] = True
        leaf_region = np.zeros(image.shape[:2], dtype=bool)
        leaf_region[
            markers["brown_inside_leaf"][1] - 120 : markers["brown_inside_leaf"][1] + 120,
            markers["brown_inside_leaf"][0] - 100 : markers["brown_inside_leaf"][0] + 100,
        ] = True

        tape_in_mask = int(np.count_nonzero(final_mask & tape_region))
        leaf_in_mask = int(np.count_nonzero(final_mask & leaf_region))
        self.assertGreater(leaf_in_mask, tape_in_mask)
        self.assertGreater(leaf_in_mask, 0)

        green_seed = detect_green_leaf_tissue(image, LeafMaskPreviewSettings())
        self.assertGreater(int(np.count_nonzero(final_mask & green_seed)), 0)

    def test_disconnected_brown_blob_is_excluded(self) -> None:
        image, markers = _make_realistic_lab_image(include_disconnected_brown=True)
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS)
        assert preview.mask is not None

        disconnected = markers["brown_disconnected"]
        self.assertFalse(
            preview.mask[disconnected[1], disconnected[0]],
            "Brown tissue disconnected from green leaf should not be in final mask",
        )


class TestWhitePaperSegmentation(unittest.TestCase):
    def test_white_paper_keeps_leaf_tissue_and_excludes_background(self) -> None:
        image, markers = _make_white_paper_assay_image()
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.SUCCESS, preview.warnings)
        assert preview.mask is not None
        mask = preview.mask

        for name in ("green_blade", "brown", "yellow", "pale", "dark_damage", "petiole"):
            x, y = markers[name]
            self.assertTrue(mask[y, x], f"{name} should remain inside the leaf mask")

        for name in (
            "paper_hole",
            "white_wrap",
            "paper_fold",
            "paper_shadow",
            "disconnected_yellow",
            "disconnected_brown",
            "rim",
            "exterior",
            "blue_ruler",
            "label",
        ):
            x, y = markers[name]
            self.assertFalse(mask[y, x], f"{name} should stay outside the leaf mask")

        paper = detect_white_paper_background(image, LeafMaskPreviewSettings())
        self.assertFalse(np.any(mask & paper))
        self.assertGreater(int(np.count_nonzero(mask)), 0)

    def test_blue_ruler_alone_is_not_a_dish(self) -> None:
        image = np.full((320, 480, 3), (12, 14, 18), dtype=np.uint8)
        cv2.rectangle(image, (20, 250), (460, 300), (36, 78, 188), thickness=-1)
        cv2.ellipse(image, (240, 140), (70, 80), 0, 0, 360, (42, 158, 52), thickness=-1)
        preview = segment_leaf_preview(image)
        self.assertEqual(preview.status, LeafMaskPreviewStatus.FAILED)
        self.assertIsNone(preview.mask)
        self.assertTrue(preview.warnings)

    def test_white_paper_preview_is_deterministic(self) -> None:
        image, _markers = _make_white_paper_assay_image()
        first = segment_leaf_preview(image)
        second = segment_leaf_preview(image)
        self.assertEqual(first.status, second.status)
        assert first.mask is not None and second.mask is not None
        self.assertTrue(np.array_equal(first.mask, second.mask))

    def test_render_leaf_mask_diagnostic_outlines_without_mutating_inputs(self) -> None:
        image, _markers = _make_white_paper_assay_image()
        preview = segment_leaf_preview(image)
        assert preview.mask is not None
        image_before = image.copy()
        mask_before = preview.mask.copy()
        diagnostic = render_leaf_mask_diagnostic(image, preview.mask)
        self.assertTrue(np.array_equal(image, image_before))
        self.assertTrue(np.array_equal(preview.mask, mask_before))
        self.assertEqual(diagnostic.shape, image.shape)
        self.assertTrue(np.any(np.any(diagnostic != image, axis=2) & preview.mask))
        contours, _hierarchy = cv2.findContours(
            preview.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        self.assertGreater(len(contours), 0)
        boundary = tuple(int(v) for v in contours[0][0, 0])
        self.assertFalse(np.array_equal(diagnostic[boundary[1], boundary[0]], image[boundary[1], boundary[0]]))


class TestGuiSegmentationSeparation(unittest.TestCase):
    def test_gui_canvas_does_not_import_segmentation(self) -> None:
        import neopest.gui_canvas as gui_canvas

        source = inspect.getsource(gui_canvas)
        self.assertNotIn("segmentation", source)
        self.assertNotIn("segment_leaf", source)

    def test_gui_layout_builders_do_not_import_segmentation(self) -> None:
        import neopest.gui_layout_builders as layout_builders

        source = inspect.getsource(layout_builders)
        self.assertNotIn("segmentation", source)
        self.assertNotIn("segment_leaf", source)

    def test_run_analysis_remains_disabled_in_layout(self) -> None:
        import neopest.gui_layout_builders as layout_builders

        source = inspect.getsource(layout_builders.SampleInspectorPanel.__init__)
        self.assertIn("setEnabled(False)", source)
        self.assertIn("setVisible(False)", source)


if __name__ == "__main__":
    unittest.main()
