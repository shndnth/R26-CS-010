"""k-anonymity measurement."""

from __future__ import annotations

import pandas as pd
import pytest

from risk_compliance.anonymity import measure

QUASI = ("Weather_Category", "Time_Of_Day")


def frame(rows):
    return pd.DataFrame(rows, columns=["Weather_Category", "Time_Of_Day"])


class TestMeasure:
    def test_k_is_the_smallest_class_size(self):
        data = frame([["clear", "day"]] * 5 + [["rain", "night"]] * 3)
        result = measure(data, QUASI, threshold=2)
        assert result.k == 3

    def test_unique_combination_gives_k_of_one(self):
        data = frame([["clear", "day"]] * 5 + [["fog", "dawn"]])
        result = measure(data, QUASI, threshold=5)
        assert result.k == 1
        assert result.singletons == 1

    def test_passes_when_k_meets_threshold(self):
        data = frame([["clear", "day"]] * 10 + [["rain", "night"]] * 10)
        assert measure(data, QUASI, threshold=5).passed

    def test_fails_when_k_below_threshold(self):
        data = frame([["clear", "day"]] * 10 + [["rain", "night"]] * 2)
        result = measure(data, QUASI, threshold=5)
        assert not result.passed
        assert result.verdict == "FAIL"

    def test_counts_distinct_combinations(self):
        data = frame([["clear", "day"], ["rain", "night"], ["fog", "dawn"]] * 4)
        assert measure(data, QUASI, threshold=2).distinct_combinations == 3

    def test_counts_classes_below_threshold(self):
        data = frame([["clear", "day"]] * 10 + [["rain", "night"]] * 2 + [["fog", "dawn"]] * 1)
        assert measure(data, QUASI, threshold=5).below_threshold == 2

    def test_singleton_interpretation_mentions_unique_identification(self):
        data = frame([["clear", "day"]] * 5 + [["fog", "dawn"]])
        assert "uniquely identifiable" in measure(data, QUASI, threshold=5).interpretation()

    def test_missing_column_raises(self):
        data = frame([["clear", "day"]] * 3)
        with pytest.raises(KeyError, match="not present"):
            measure(data, ("Weather_Category", "Absent_Column"), threshold=2)
