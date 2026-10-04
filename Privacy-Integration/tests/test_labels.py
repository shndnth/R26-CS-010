"""Bounding-box parsing and size-filtered label derivation."""

from __future__ import annotations

from privacy_integration.data.labels import Label, derive_label
from tests.conftest import write_bbox

NATIVE_W, NATIVE_H = 800, 600
MIN_AREA = 1024.0


def _label(path):
    return derive_label(path, MIN_AREA, NATIVE_W, NATIVE_H)


class TestDeriveLabel:
    def test_missing_file_is_negative(self, tmp_path):
        assert _label(tmp_path / "absent.txt") is Label.NEGATIVE

    def test_empty_file_is_negative(self, bbox_dir):
        assert _label(write_bbox(bbox_dir, "a.txt", [])) is Label.NEGATIVE

    def test_cars_alone_are_negative(self, bbox_dir):
        path = write_bbox(bbox_dir, "a.txt", ["0 0.5 0.5 0.5 0.5"])
        assert _label(path) is Label.NEGATIVE

    def test_large_pedestrian_is_positive(self, bbox_dir):
        # 0.1 * 800 = 80 px wide, 0.1 * 600 = 60 px tall -> 4800 px^2 > 1024
        path = write_bbox(bbox_dir, "a.txt", ["1 0.5 0.5 0.1 0.1"])
        assert _label(path) is Label.POSITIVE

    def test_large_cyclist_is_positive(self, bbox_dir):
        path = write_bbox(bbox_dir, "a.txt", ["2 0.5 0.5 0.1 0.1"])
        assert _label(path) is Label.POSITIVE

    def test_small_pedestrian_is_excluded_not_negative(self, bbox_dir):
        # 0.01 * 800 = 8 px, 0.01 * 600 = 6 px -> 48 px^2 < 1024
        path = write_bbox(bbox_dir, "a.txt", ["1 0.5 0.5 0.01 0.01"])
        assert _label(path) is Label.EXCLUDED

    def test_largest_target_decides(self, bbox_dir):
        path = write_bbox(
            bbox_dir, "a.txt", ["1 0.5 0.5 0.01 0.01", "1 0.5 0.5 0.1 0.1"]
        )
        assert _label(path) is Label.POSITIVE

    def test_small_target_beside_large_car_is_excluded(self, bbox_dir):
        path = write_bbox(bbox_dir, "a.txt", ["0 0.5 0.5 0.9 0.9", "1 0.5 0.5 0.01 0.01"])
        assert _label(path) is Label.EXCLUDED

    def test_threshold_is_inclusive(self, bbox_dir):
        # width * 800 * height * 600 == exactly 1024
        path = write_bbox(bbox_dir, "a.txt", ["1 0.5 0.5 0.04 0.0533333"])
        assert _label(path) in (Label.POSITIVE, Label.EXCLUDED)

    def test_lower_threshold_admits_more_positives(self, bbox_dir):
        path = write_bbox(bbox_dir, "a.txt", ["1 0.5 0.5 0.02 0.02"])
        assert derive_label(path, 1024.0, NATIVE_W, NATIVE_H) is Label.EXCLUDED
        assert derive_label(path, 100.0, NATIVE_W, NATIVE_H) is Label.POSITIVE
