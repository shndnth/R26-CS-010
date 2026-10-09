"""Batch chi-square, cross-town outliers and clean-label output."""

from __future__ import annotations

import random

import numpy as np

from utility_evaluation.poisoning import (
    Thresholds,
    audit,
    chi_square_batch,
    town_outliers,
    write_clean_labels,
)

TOWNS = ("town01", "town02", "town03", "town04", "town05")


def make_town(root, town, frames=500, seed=0, weights=(0.70, 0.22, 0.08)):
    rng = random.Random(seed)
    labels = root / town / "labels"
    labels.mkdir(parents=True)
    for i in range(frames):
        lines = [
            f"{rng.choices((0, 1, 2), weights=weights)[0]} 0.5 0.5 0.05 0.08"
            for _ in range(rng.randint(1, 4))
        ]
        (labels / f"bbox_f{i:06d}.txt").write_text("\n".join(lines), encoding="utf-8")
    return labels


class TestChiSquareBatch:
    def test_matching_distribution_is_accepted(self):
        _, _, status = chi_square_batch([70, 22, 8], [700, 220, 80], Thresholds())
        assert status == "ACCEPTED"

    def test_relabelled_batch_is_rejected(self):
        p, v, status = chi_square_batch([0, 0, 100], [700, 220, 80], Thresholds())
        assert status == "REJECTED"
        assert p < 1e-4 and v > 0.40

    def test_small_batch_is_skipped_not_rejected(self):
        assert chi_square_batch([0, 0, 5], [700, 220, 80], Thresholds())[2] == "SKIPPED"

    def test_significant_but_small_effect_is_accepted(self):
        _, v, status = chi_square_batch([600, 300, 100], [7000, 2200, 800], Thresholds())
        assert v < 0.40 and status == "ACCEPTED"


class TestTownOutliers:
    def test_plain_z_score_cannot_reach_four_with_five_towns(self):
        values = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
        plain = np.abs((values - values.mean()) / values.std())
        assert plain.max() < 4.0

    def test_leave_one_out_flags_the_skewed_town(self):
        rates = {t: {0: 0.70, 1: 0.22, 2: 0.08} for t in TOWNS}
        rates["town01"] = {0: 0.70, 1: 0.2205, 2: 0.0795}
        rates["town02"] = {0: 0.71, 1: 0.215, 2: 0.075}
        rates["town03"] = {0: 0.69, 1: 0.225, 2: 0.085}
        rates["town05"] = {0: 0.10, 1: 0.10, 2: 0.80}
        warnings = town_outliers(rates, z_limit=4.0)
        assert {w.file for w in warnings} == {"town05"}

    def test_similar_towns_raise_nothing(self):
        rates = {t: {0: 0.70 + i * 0.01, 1: 0.22 - i * 0.005, 2: 0.08 - i * 0.005}
                 for i, t in enumerate(TOWNS)}
        assert town_outliers(rates, z_limit=4.0) == []

    def test_fewer_than_three_towns_is_not_tested(self):
        assert town_outliers({"a": {0: 1.0, 1: 0.0, 2: 0.0}, "b": {0: 0.0, 1: 1.0, 2: 0.0}}, 4.0) == []


class TestAuditAndCleanLabels:
    def test_clean_dataset_passes_and_keeps_everything(self, tmp_path):
        make_town(tmp_path, "town01")
        result = audit(tmp_path, ("town01",))
        assert result.verdict == "PASS"
        assert len(result.towns["town01"].accepted) == 500

    def test_structural_and_batch_rejections_are_excluded(self, tmp_path):
        labels = make_town(tmp_path, "town01", frames=1000)
        (labels / "bbox_f000003.txt").write_text("7 0.5 0.5 0.1 0.1", encoding="utf-8")
        for i in range(300, 400):
            path = labels / f"bbox_f{i:06d}.txt"
            path.write_text("\n".join("2" + ln[1:] for ln in path.read_text().splitlines()))

        result = audit(tmp_path, ("town01",), Thresholds(batch_size=100))
        accepted = set(result.towns["town01"].accepted)
        assert result.verdict == "FILTERED"
        assert "bbox_f000003.txt" not in accepted
        assert not any(f"bbox_f{i:06d}.txt" in accepted for i in range(300, 400))
        assert result.frames_excluded == 101

    def test_malformed_line_does_not_crash_the_batch_test(self, tmp_path):
        labels = make_town(tmp_path, "town01")
        (labels / "bbox_f000010.txt").write_text("car 0.5 0.5 0.1 0.1", encoding="utf-8")
        result = audit(tmp_path, ("town01",))
        assert "bbox_f000010.txt" in result.towns["town01"].structurally_rejected

    def test_clean_labels_are_rebuilt_not_accumulated(self, tmp_path):
        make_town(tmp_path, "town01")
        clean = tmp_path / "clean"
        stale = clean / "town01" / "bbox_f999999.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("0 0.5 0.5 0.1 0.1")

        written = write_clean_labels(audit(tmp_path, ("town01",)), tmp_path, clean)
        assert written == 500
        assert not stale.exists()

    def test_report_dict_is_serialisable(self, tmp_path):
        import json

        make_town(tmp_path, "town01")
        payload = json.loads(json.dumps(audit(tmp_path, ("town01",)).as_dict()))
        assert payload["verdict"] == "PASS"
        assert payload["towns"]["town01"]["frames"] == 500
