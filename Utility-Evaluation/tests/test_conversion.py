"""Dataset conversion into the Ultralytics layout."""

from __future__ import annotations

import yaml

from utility_evaluation.conversion import build_yolo_dataset


def make_pairs(tmp_path, count):
    src = tmp_path / "src"
    src.mkdir()
    pairs = []
    for i in range(count):
        rgb = src / f"rgb_{i:04d}.png"
        bbox = src / f"bbox_{i:04d}.txt"
        rgb.write_bytes(b"\x89PNG placeholder")
        bbox.write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
        pairs.append((rgb, bbox, f"town01_{i:04d}"))
    return pairs


class TestBuildYoloDataset:
    def test_split_sizes_match_fractions(self, tmp_path):
        counts = build_yolo_dataset(
            make_pairs(tmp_path, 100), tmp_path / "out",
            {0: "car"}, 0.70, 0.15, seed=42,
        )
        assert counts.train == 70
        assert counts.val == 15
        assert counts.test == 15
        assert counts.total == 100

    def test_creates_expected_directories(self, tmp_path):
        out = tmp_path / "out"
        build_yolo_dataset(make_pairs(tmp_path, 20), out, {0: "car"}, 0.70, 0.15, 42)
        for split in ("train", "val", "test"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

    def test_writes_data_yaml_with_class_names(self, tmp_path):
        out = tmp_path / "out"
        names = {0: "car", 1: "pedestrian", 2: "cyclist"}
        build_yolo_dataset(make_pairs(tmp_path, 20), out, names, 0.70, 0.15, 42)
        config = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert config["names"] == names
        assert config["train"] == "images/train"

    def test_image_and_label_counts_match(self, tmp_path):
        out = tmp_path / "out"
        build_yolo_dataset(make_pairs(tmp_path, 40), out, {0: "car"}, 0.70, 0.15, 42)
        for split in ("train", "val", "test"):
            images = list((out / "images" / split).glob("*.png"))
            labels = list((out / "labels" / split).glob("*.txt"))
            assert len(images) == len(labels)

    def test_split_is_reproducible_for_a_seed(self, tmp_path):
        pairs = make_pairs(tmp_path, 50)
        a = tmp_path / "a"
        b = tmp_path / "b"
        build_yolo_dataset(pairs, a, {0: "car"}, 0.70, 0.15, 42)
        build_yolo_dataset(pairs, b, {0: "car"}, 0.70, 0.15, 42)
        names_a = sorted(p.name for p in (a / "images" / "train").glob("*.png"))
        names_b = sorted(p.name for p in (b / "images" / "train").glob("*.png"))
        assert names_a == names_b

    def test_filenames_are_prefixed_so_towns_do_not_collide(self, tmp_path):
        out = tmp_path / "out"
        build_yolo_dataset(make_pairs(tmp_path, 20), out, {0: "car"}, 0.70, 0.15, 42)
        every = [p.name for split in ("train", "val", "test")
                 for p in (out / "images" / split).glob("*.png")]
        assert all(name.startswith("town01_") for name in every)
        assert len(set(every)) == len(every)
