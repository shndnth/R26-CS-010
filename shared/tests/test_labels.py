"""YOLO label parsing and the frame layout."""

from __future__ import annotations

import pytest
from r26_common.labels import BoundingBox, ObjectClass, format_bbox_line, iter_frame_pairs, parse_bbox_file


def write(tmp_path, lines, name="bbox_f000001.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestParse:
    def test_missing_file_yields_no_boxes(self, tmp_path):
        assert parse_bbox_file(tmp_path / "absent.txt") == []

    def test_empty_and_blank_files_are_valid(self, tmp_path):
        assert parse_bbox_file(write(tmp_path, [])) == []
        assert parse_bbox_file(write(tmp_path, ["", "   "])) == []

    def test_parses_all_five_fields(self, tmp_path):
        [box] = parse_bbox_file(write(tmp_path, ["1 0.5 0.4 0.1 0.2"]))
        assert box == BoundingBox(ObjectClass.PEDESTRIAN, 0.5, 0.4, 0.1, 0.2)

    def test_skips_truncated_and_non_numeric_lines(self, tmp_path):
        boxes = parse_bbox_file(write(tmp_path, ["1 0.5 0.5", "x 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1"]))
        assert [b.object_class for b in boxes] == [2]


class TestBox:
    def test_area_scales_with_native_resolution(self):
        assert BoundingBox(0, 0.5, 0.5, 0.1, 0.2).area_native(800, 600) == pytest.approx(80 * 120)

    def test_within_and_outside_frame(self):
        assert BoundingBox(0, 0.5, 0.5, 0.2, 0.2).is_within_frame
        assert not BoundingBox(0, 0.95, 0.5, 0.3, 0.2).is_within_frame

    def test_format_round_trips(self, tmp_path):
        box = BoundingBox(2, 0.25, 0.75, 0.125, 0.5)
        assert parse_bbox_file(write(tmp_path, [format_bbox_line(box)])) == [box]


def test_frame_pairs_follow_frame_order_and_alternate_label_folder(tmp_path):
    for i in (2, 0, 1):
        (tmp_path / f"rgb_f{i:06d}.png").touch()
    pairs = list(iter_frame_pairs(tmp_path))
    assert [fid for _, _, fid in pairs] == ["000000", "000001", "000002"]
    assert pairs[0][1] == tmp_path / "labels" / "bbox_f000000.txt"
    other = list(iter_frame_pairs(tmp_path, tmp_path / "clean"))
    assert other[0][1].parent == tmp_path / "clean"
