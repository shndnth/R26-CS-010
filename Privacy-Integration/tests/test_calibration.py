import unittest

from privacy_integration.calibration.laplace import LaplaceCalibrationModule
from privacy_integration.legacy_constants import EPSILON_CALIBRATION_MAX

VALID_STATS = {
    "vehicle_density_per_km":    12.4,
    "pedestrian_density_per_km":  3.8,
    "avg_vehicle_speed_kmh":     47.2,
    "avg_pedestrian_speed_kmh":   4.1,
    "intersection_count_per_km":  2.3,
    "weather_distribution":      {"clear": 0.6, "cloudy": 0.2, "rain": 0.15, "fog": 0.05},
    "time_of_day_distribution":  {"morning": 0.3, "afternoon": 0.4, "evening": 0.2, "night": 0.1},
}


class TestLaplaceCalibrationModule(unittest.TestCase):

    def setUp(self):
        self.module = LaplaceCalibrationModule(epsilon_calibration=1.8, random_seed=42)

    def test_rejects_epsilon_at_ceiling(self):
        with self.assertRaises(ValueError):
            LaplaceCalibrationModule(epsilon_calibration=EPSILON_CALIBRATION_MAX)

    def test_rejects_epsilon_above_ceiling(self):
        with self.assertRaises(ValueError):
            LaplaceCalibrationModule(epsilon_calibration=3.0)

    def test_rejects_missing_fields(self):
        incomplete = {"vehicle_density_per_km": 10.0}
        with self.assertRaises(ValueError):
            self.module.process(incomplete)

    def test_result_has_all_required_keys(self):
        result = self.module.process(VALID_STATS)
        for key in LaplaceCalibrationModule.REQUIRED_FIELDS:
            self.assertIn(key, result.stats)

    def test_epsilon_spent_is_the_sequential_total(self):
        result = self.module.process(VALID_STATS)
        self.assertEqual(result.stats["calibration_epsilon_per_query"], 1.8)
        self.assertEqual(result.stats["num_queries"], 7)
        self.assertAlmostEqual(result.stats["calibration_epsilon_spent"], 12.6)
        self.assertAlmostEqual(result.epsilon_spent, 12.6)

    def test_weather_distribution_sums_to_one(self):
        result = self.module.process(VALID_STATS)
        total = sum(result.stats["weather_distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_time_of_day_sums_to_one(self):
        result = self.module.process(VALID_STATS)
        total = sum(result.stats["time_of_day_distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_all_histogram_bins_non_negative(self):
        result = self.module.process(VALID_STATS)
        for v in result.stats["weather_distribution"].values():
            self.assertGreaterEqual(v, 0.0)
        for v in result.stats["time_of_day_distribution"].values():
            self.assertGreaterEqual(v, 0.0)

    def test_query_log_length(self):
        result = self.module.process(VALID_STATS)
        self.assertEqual(len(result.query_log), len(LaplaceCalibrationModule.REQUIRED_FIELDS))

    def test_reproducibility_with_same_seed(self):
        m1 = LaplaceCalibrationModule(epsilon_calibration=1.8, random_seed=42)
        m2 = LaplaceCalibrationModule(epsilon_calibration=1.8, random_seed=42)
        r1 = m1.process(VALID_STATS)
        r2 = m2.process(VALID_STATS)
        self.assertEqual(r1.stats["vehicle_density_per_km"],
                         r2.stats["vehicle_density_per_km"])

    def test_different_seeds_produce_different_noise(self):
        m1 = LaplaceCalibrationModule(epsilon_calibration=1.8, random_seed=1)
        m2 = LaplaceCalibrationModule(epsilon_calibration=1.8, random_seed=2)
        r1 = m1.process(VALID_STATS)
        r2 = m2.process(VALID_STATS)
        self.assertNotEqual(r1.stats["vehicle_density_per_km"],
                            r2.stats["vehicle_density_per_km"])

    def test_generated_at_present(self):
        result = self.module.process(VALID_STATS)
        self.assertIn("generated_at", result.stats)

    def test_note_present(self):
        result = self.module.process(VALID_STATS)
        self.assertIn("note", result.stats)


if __name__ == "__main__":
    unittest.main()
