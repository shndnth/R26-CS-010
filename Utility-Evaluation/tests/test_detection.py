"""Per-class metric mapping and size-stratified AP."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utility_evaluation.detection import (
    SizeStratifiedAP,
    average_precision,
    bucket_of,
    iou,
    summarise_validation,
)


def fake_metrics(class_index, ap50, ap):
    box = SimpleNamespace(map50=0.5, map=0.3, ap50=ap50, ap=ap, ap_class_index=class_index)
    return SimpleNamespace(box=box)


class TestSummariseValidation:
    def test_all_classes_present(self):
        record = summarise_validation(fake_metrics([0, 1, 2], [0.9, 0.5, 0.2], [0.6, 0.3, 0.1]), "x")
        assert record["per_class"]["cyclist"] == {"ap50": 0.2, "ap50_95": 0.1}

    def test_missing_class_is_none_and_others_are_not_shifted(self):
        record = summarise_validation(fake_metrics([0, 2], [0.9, 0.2], [0.6, 0.1]), "x")
        assert record["per_class"]["pedestrian"] is None
        assert record["per_class"]["cyclist"] == {"ap50": 0.2, "ap50_95": 0.1}

    def test_extra_fields_are_kept(self):
        record = summarise_validation(fake_metrics([0], [0.9], [0.6]), "x", split="test")
        assert record["split"] == "test" and record["label"] == "x"


class TestPrimitives:
    def test_iou_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_buckets(self):
        assert bucket_of(31 * 31) == "small"
        assert bucket_of(50 * 50) == "medium"
        assert bucket_of(100 * 100) == "large"

    def test_perfect_detection_scores_one(self):
        assert average_precision([1, 1], 2) == pytest.approx(1.0)

    def test_no_ground_truth_is_none(self):
        assert average_precision([0], 0) is None

    def test_no_detections_scores_zero(self):
        assert average_precision([], 3) == 0.0


class TestSizeStratifiedAP:
    def test_perfect_predictions_score_one_in_their_bucket(self):
        scorer = SizeStratifiedAP()
        gt = [(0, 0, 0, 20, 20), (0, 100, 100, 300, 300)]
        preds = [(0, 0.9, 0, 0, 20, 20), (0, 0.8, 100, 100, 300, 300)]
        scorer.add_image(gt, preds)
        results = scorer.results()
        assert results["small"]["per_class"]["car"]["ap50"] == pytest.approx(1.0)
        assert results["large"]["per_class"]["car"]["ap50"] == pytest.approx(1.0)
        assert results["medium"]["per_class"]["car"]["ap50"] is None

    def test_match_across_a_bucket_boundary_is_ignored_not_scored(self):
        # 31x31 ground truth is small and the 33x33 prediction medium; they still match.
        scorer = SizeStratifiedAP()
        scorer.add_image([(0, 0, 0, 31, 31)], [(0, 0.9, 0, 0, 33, 33)])
        results = scorer.results()
        assert results["small"]["per_class"]["car"]["ap50"] == pytest.approx(1.0)
        assert results["medium"]["per_class"]["car"]["detections"] == 0

    def test_unmatched_detection_is_a_false_positive_only_in_its_own_bucket(self):
        scorer = SizeStratifiedAP()
        scorer.add_image([(0, 0, 0, 20, 20)], [(0, 0.95, 400, 400, 600, 600), (0, 0.5, 0, 0, 20, 20)])
        results = scorer.results()
        assert results["small"]["per_class"]["car"]["ap50"] == pytest.approx(1.0)
        assert results["large"]["per_class"]["car"]["detections"] == 1

    def test_wrong_class_does_not_match(self):
        scorer = SizeStratifiedAP()
        scorer.add_image([(1, 0, 0, 20, 20)], [(0, 0.9, 0, 0, 20, 20)])
        results = scorer.results()
        assert results["small"]["per_class"]["pedestrian"]["ap50"] == 0.0


def test_evaluate_accepts_record_fields_named_like_its_parameters(monkeypatch, tmp_path):
    import sys
    import types

    from utility_evaluation.detection import evaluate

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def val(self, **kwargs):
            return fake_metrics([0, 1, 2], [0.9, 0.5, 0.2], [0.6, 0.3, 0.1])

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    record = evaluate(tmp_path / "best.pt", tmp_path / "data.yaml", "x", "cpu", tmp_path, "run",
                      weights="runs/a/best.pt", data="yolo_dataset/data.yaml", split="test")
    assert record["weights"] == "runs/a/best.pt" and record["data"] == "yolo_dataset/data.yaml"
