"""KITTI to YOLO conversion."""

from __future__ import annotations

import pytest
from PIL import Image

from utility_evaluation.kitti import KITTI_CLASS_MAP, convert_label

# KITTI format: type truncated occluded alpha left top right bottom ...
CAR = "Car 0.00 0 -1.5 100.0 150.0 300.0 250.0 1.5 1.6 4.0 1.0 1.5 8.0 0.0"
PEDESTRIAN = "Pedestrian 0.00 0 -0.2 712.4 143.0 810.7 307.9 1.9 0.5 1.2 1.8 1.5 8.4 0.0"
DONTCARE = "DontCare -1 -1 -10 500.0 200.0 600.0 300.0 -1 -1 -1 -1000 -1000 -1000 -10"
TRAM = "Tram 0.00 0 -1.5 100.0 150.0 300.0 250.0 1.5 1.6 4.0 1.0 1.5 8.0 0.0"


def make_image(tmp_path, width=1242, height=375):
    path = tmp_path / "000001.png"
    Image.new("RGB", (width, height)).save(path)
    return path


def convert(tmp_path, lines):
    label = tmp_path / "000001.txt"
    label.write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "out.txt"
    count = convert_label(label, make_image(tmp_path), out)
    return count, out.read_text(encoding="utf-8").strip()


class TestClassMapping:
    def test_vehicles_collapse_to_car(self):
        assert KITTI_CLASS_MAP["Car"] == 0
        assert KITTI_CLASS_MAP["Van"] == 0
        assert KITTI_CLASS_MAP["Truck"] == 0

    def test_pedestrian_variants_map_together(self):
        assert KITTI_CLASS_MAP["Pedestrian"] == 1
        assert KITTI_CLASS_MAP["Person_sitting"] == 1

    def test_cyclist_maps_to_two(self):
        assert KITTI_CLASS_MAP["Cyclist"] == 2

    def test_unmapped_types_are_absent(self):
        assert "DontCare" not in KITTI_CLASS_MAP
        assert "Tram" not in KITTI_CLASS_MAP


class TestConvertLabel:
    def test_converts_a_single_box(self, tmp_path):
        count, text = convert(tmp_path, [CAR])
        assert count == 1
        assert text.split()[0] == "0"

    def test_coordinates_are_normalised(self, tmp_path):
        _count, text = convert(tmp_path, [CAR])
        values = [float(v) for v in text.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_centre_is_computed_from_corners(self, tmp_path):
        # Output is written to six decimal places, so compare with tolerance.
        _count, text = convert(tmp_path, [CAR])
        x_center = float(text.split()[1])
        assert x_center == pytest.approx(((100.0 + 300.0) / 2) / 1242, abs=1e-6)

    def test_width_and_height_from_corners(self, tmp_path):
        _count, text = convert(tmp_path, [CAR])
        width, height = (float(v) for v in text.split()[3:5])
        assert width == pytest.approx((300.0 - 100.0) / 1242, abs=1e-6)
        assert height == pytest.approx((250.0 - 150.0) / 375, abs=1e-6)

    def test_dontcare_is_dropped(self, tmp_path):
        count, _text = convert(tmp_path, [DONTCARE])
        assert count == 0

    def test_tram_is_dropped(self, tmp_path):
        count, _text = convert(tmp_path, [TRAM])
        assert count == 0

    def test_mixed_file_keeps_only_mapped_types(self, tmp_path):
        count, text = convert(tmp_path, [CAR, DONTCARE, PEDESTRIAN, TRAM])
        assert count == 2
        assert sorted(line.split()[0] for line in text.splitlines()) == ["0", "1"]

    def test_empty_label_file_produces_empty_output(self, tmp_path):
        count, text = convert(tmp_path, [])
        assert count == 0
        assert text == ""

    def test_malformed_line_is_skipped(self, tmp_path):
        count, _text = convert(tmp_path, ["Car 0.00 0"])
        assert count == 0
