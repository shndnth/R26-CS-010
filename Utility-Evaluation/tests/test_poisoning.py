"""Data poisoning detection."""

from __future__ import annotations

from tests.conftest import write_label
from utility_evaluation.poisoning import scan_frame

MAX_OBJECTS = 50
MIN_AREA = 1e-6


def scan(path):
    return scan_frame(path, MAX_OBJECTS, MIN_AREA)


class TestCleanInput:
    def test_valid_file_produces_no_anomalies(self, label_dir):
        path = write_label(label_dir, "a.txt", ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"])
        classes, anomalies = scan(path)
        assert anomalies == []
        assert classes == [0, 1]

    def test_empty_file_produces_no_anomalies(self, label_dir):
        classes, anomalies = scan(write_label(label_dir, "a.txt", []))
        assert anomalies == []
        assert classes == []

    def test_missing_file_produces_no_anomalies(self, tmp_path):
        classes, anomalies = scan(tmp_path / "absent.txt")
        assert anomalies == []
        assert classes == []


class TestDetectsPoisoning:
    def test_coordinate_outside_range(self, label_dir):
        path = write_label(label_dir, "a.txt", ["1 1.5 0.5 0.2 0.2"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "out_of_range" for a in anomalies)

    def test_unknown_class_id(self, label_dir):
        path = write_label(label_dir, "a.txt", ["7 0.5 0.5 0.1 0.1"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "unknown_class" for a in anomalies)

    def test_degenerate_box(self, label_dir):
        path = write_label(label_dir, "a.txt", ["1 0.5 0.5 0.0000001 0.0000001"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "degenerate_box" for a in anomalies)

    def test_box_extending_beyond_frame(self, label_dir):
        path = write_label(label_dir, "a.txt", ["1 0.95 0.5 0.3 0.2"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "outside_frame" for a in anomalies)

    def test_implausible_object_count(self, label_dir):
        lines = ["0 0.5 0.5 0.01 0.01"] * (MAX_OBJECTS + 5)
        _classes, anomalies = scan(write_label(label_dir, "a.txt", lines))
        assert any(a.kind == "implausible_count" for a in anomalies)

    def test_malformed_line(self, label_dir):
        path = write_label(label_dir, "a.txt", ["1 0.5 0.5"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "malformed" for a in anomalies)

    def test_non_numeric_values(self, label_dir):
        path = write_label(label_dir, "a.txt", ["1 abc 0.5 0.1 0.1"])
        _classes, anomalies = scan(path)
        assert any(a.kind == "non_numeric" for a in anomalies)

    def test_multiple_problems_all_reported(self, label_dir):
        path = write_label(label_dir, "a.txt", ["9 1.5 0.5 0.2 0.2"])
        _classes, anomalies = scan(path)
        kinds = {a.kind for a in anomalies}
        assert "unknown_class" in kinds
        assert "out_of_range" in kinds
