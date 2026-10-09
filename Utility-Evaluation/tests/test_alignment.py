"""Label alignment and misalignment diagnosis."""

from __future__ import annotations

import numpy as np
from PIL import Image

from utility_evaluation.alignment import Box, coordinates_valid, diagnose, load_mask, spatial_status


def mask_with_patch(value, patch=(slice(20, 60), slice(20, 60)), background=3):
    mask = np.full((100, 100), background, dtype=np.uint8)
    mask[patch] = value
    return mask


BOX = Box(class_id=1, x_center=0.4, y_center=0.4, width=0.4, height=0.4)


class TestCoordinates:
    def test_box_inside_frame_is_valid(self):
        assert coordinates_valid(BOX)

    def test_box_past_the_edge_is_invalid(self):
        assert not coordinates_valid(Box(1, 0.95, 0.5, 0.3, 0.2))

    def test_zero_width_is_invalid(self):
        assert not coordinates_valid(Box(1, 0.5, 0.5, 0.0, 0.2))


class TestSpatial:
    def test_box_on_its_object_is_aligned(self):
        assert spatial_status(BOX, mask_with_patch(1)) == "ALIGNED"

    def test_box_on_background_is_misaligned(self):
        assert spatial_status(BOX, mask_with_patch(3)).startswith("MISALIGNED")

    def test_tiny_box_is_skipped(self):
        tiny = Box(1, 0.5, 0.5, 0.02, 0.02)
        assert spatial_status(tiny, mask_with_patch(1)).startswith("SKIPPED")


class TestDiagnose:
    def test_background_region_is_occlusion(self):
        assert diagnose(BOX, mask_with_patch(3)).kind == "occlusion_explained"

    def test_other_class_region_is_true_misalignment(self):
        result = diagnose(BOX, mask_with_patch(0))
        assert result.kind == "true_misalignment"
        assert "car" in result.category

    def test_aligned_box_has_no_diagnosis(self):
        assert diagnose(BOX, mask_with_patch(1)) is None

    def test_unexpected_mask_value_does_not_crash(self):
        result = diagnose(BOX, mask_with_patch(9, background=9))
        assert result.kind == "other"
        assert result.dominant_class == 9


class TestLoadMask:
    def test_rgb_mask_is_reduced_to_class_ids(self, tmp_path):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[..., 0] = 2
        Image.fromarray(rgb).save(tmp_path / "m.png")
        mask = load_mask(tmp_path / "m.png")
        assert mask.shape == (10, 10)
        assert set(np.unique(mask)) == {2}
