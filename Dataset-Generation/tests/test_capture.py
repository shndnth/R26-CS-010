"""Pure parts of the capture script: settings, schedules and the class mask."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from conftest import import_capture
from PIL import Image

TOWNS = ("town01", "town02", "town03", "town04", "town05")


def test_defaults(monkeypatch):
    capture = import_capture(monkeypatch)
    assert capture.TOWN_NAME == "town01"
    assert capture.NUM_SETS == 2000
    assert capture.CAPTURE_INTERVAL == 16


def test_short_test_run(monkeypatch):
    capture = import_capture(monkeypatch, "--town", "town04", "--sets", "20")
    assert capture.TOWN_NAME == "town04" and capture.CARLA_TOWN == "Town04"
    assert capture.NUM_SETS == 20


def test_every_town_has_a_profile(monkeypatch):
    capture = import_capture(monkeypatch)
    assert set(capture.TOWN_PROFILES) == set(TOWNS)


def test_exact_sequence_matches_the_distribution(monkeypatch):
    capture = import_capture(monkeypatch)
    random.seed(0)
    sequence = capture.build_exact_sequence({"clear": 0.62, "rain": 0.24, "cloudy": 0.14}, 100)
    assert Counter(sequence) == {"clear": 62, "rain": 24, "cloudy": 14}


def test_rounding_still_fills_every_set(monkeypatch):
    capture = import_capture(monkeypatch)
    assert len(capture.build_exact_sequence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100)) == 100


def test_schedule_follows_the_calibration_profile(monkeypatch):
    capture = import_capture(monkeypatch)
    schedule = capture.build_scenario_schedule(200)
    assert len(schedule) == 200
    weather = Counter(s["weather_category"] for s in schedule)
    expected = capture.CALIBRATION_PROFILE["weather_distribution"]
    for category, share in expected.items():
        assert abs(weather.get(category, 0) - share * 200) <= 1


def test_night_scenes_have_the_sun_below_the_horizon(monkeypatch):
    capture = import_capture(monkeypatch)
    assert capture.weather_params_from_categories("rain", "night")["sun_altitude_angle"] < 0
    assert capture.weather_params_from_categories("unknown", "afternoon")["cloudiness"] == 5


def test_class_mask_maps_palette_colours(monkeypatch, tmp_path):
    capture = import_capture(monkeypatch)
    target = capture.CLASS_IDS["pedestrian"]
    pedestrian = next(colour for colour, cid in capture.CLASS_COLOR_MAP.items() if cid == target)
    semantic = np.zeros((4, 4, 3), dtype=np.uint8)
    semantic[0, 0] = pedestrian
    Image.fromarray(semantic).save(tmp_path / "semantic.png")
    capture.remap_semantic_to_class_mask(tmp_path / "semantic.png", tmp_path / "mask.png")
    mask = np.array(Image.open(tmp_path / "mask.png"))
    assert mask[0, 0] == capture.CLASS_IDS["pedestrian"]
    assert mask[3, 3] == capture.BACKGROUND_CLASS_ID
