"""Clean-label sourcing, overwrite protection, block splits and equivalence with the original split."""

from __future__ import annotations

import random
import shutil

import pytest

from utility_evaluation.conversion import build_yolo_dataset, collect_pairs, split_pairs


def make_decrypted(root, towns=("town01", "town02"), frames=40):
    for town in towns:
        labels = root / town / "labels"
        labels.mkdir(parents=True)
        for i in range(frames):
            (root / town / f"rgb_f{i:06d}.png").write_bytes(b"png")
            (labels / f"bbox_f{i:06d}.txt").write_text("0 0.5 0.5 0.1 0.1")
    return root


def original_split(source, seed=42):
    """The split as convert_to_yolo.py originally computed it."""
    pairs = []
    for town in sorted(source.glob("town*")):
        for rgb in sorted(town.glob("rgb_f*.png")):
            frame_id = rgb.stem.replace("rgb_f", "")
            label = town / "labels" / f"bbox_f{frame_id}.txt"
            if label.exists():
                pairs.append(f"{town.name}_{frame_id}")
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


class TestCleanLabels:
    def test_frames_without_a_clean_label_are_dropped(self, tmp_path):
        data = make_decrypted(tmp_path / "data")
        clean = tmp_path / "clean"
        shutil.copytree(data / "town01" / "labels", clean / "town01")
        shutil.copytree(data / "town02" / "labels", clean / "town02")
        (clean / "town02" / "bbox_f000005.txt").unlink()

        pairs = collect_pairs(data, ("town01", "town02"), clean)
        ids = {uid for _, _, uid in pairs}
        assert len(pairs) == 79
        assert "town02_000005" not in ids
        assert all(label.is_relative_to(clean) for _, label, _ in pairs)

    def test_raw_mode_uses_the_decrypted_labels(self, tmp_path):
        data = make_decrypted(tmp_path / "data")
        pairs = collect_pairs(data, ("town01", "town02"))
        assert len(pairs) == 80
        assert all(label.parent.name == "labels" for _, label, _ in pairs)


class TestOverwrite:
    def test_existing_dataset_is_not_written_into(self, tmp_path):
        data = make_decrypted(tmp_path / "data")
        pairs = collect_pairs(data, ("town01",))
        out = tmp_path / "yolo"
        build_yolo_dataset(pairs, out, {0: "car"}, 0.7, 0.15, 42)
        with pytest.raises(FileExistsError, match="--overwrite"):
            build_yolo_dataset(pairs, out, {0: "car"}, 0.7, 0.15, 43)

    def test_overwrite_removes_files_from_the_previous_build(self, tmp_path):
        data = make_decrypted(tmp_path / "data")
        pairs = collect_pairs(data, ("town01", "town02"))
        out = tmp_path / "yolo"
        build_yolo_dataset(pairs, out, {0: "car"}, 0.7, 0.15, 42)
        build_yolo_dataset(pairs, out, {0: "car"}, 0.7, 0.15, 7, overwrite=True)
        every = [p.name for s in ("train", "val", "test") for p in (out / "images" / s).iterdir()]
        assert len(every) == len(set(every)) == 80


class TestSplits:
    def test_random_split_matches_the_original_script(self, tmp_path):
        data = make_decrypted(tmp_path / "data", towns=("town01", "town02", "town03"), frames=57)
        pairs = collect_pairs(data, ("town01", "town02", "town03"))
        splits = split_pairs(pairs, 0.70, 0.15, 42)
        ours = tuple([uid for _, _, uid in splits[s]] for s in ("train", "val", "test"))
        assert ours == original_split(data)

    def test_block_split_keeps_neighbouring_frames_together(self, tmp_path):
        data = make_decrypted(tmp_path / "data", frames=200)
        pairs = collect_pairs(data, ("town01", "town02"))
        splits = split_pairs(pairs, 0.70, 0.15, 42, mode="block", block_size=50)
        where = {uid: split for split, members in splits.items() for _, _, uid in members}
        for town in ("town01", "town02"):
            for start in range(0, 200, 50):
                assert len({where[f"{town}_{i:06d}"] for i in range(start, start + 50)}) == 1
        assert sum(len(m) for m in splits.values()) == 400

    def test_unknown_mode_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="split mode"):
            split_pairs([], 0.7, 0.15, 42, mode="town")
