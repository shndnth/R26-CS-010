from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from privacy_integration.legacy_constants import EPSILON_CALIBRATION_MAX
from privacy_integration.logging_config import get_logger as setup_logging
from privacy_integration.serialization import utc_timestamp as timestamp
from privacy_integration.serialization import write_json as save_json
from privacy_integration.settings import settings  # noqa: F401

logger = setup_logging(__name__)


@dataclass
class QueryRecord:
    name: str
    sensitivity: float
    noise: float
    epsilon_used: float


@dataclass
class CalibrationResult:
    stats: dict[str, Any]
    query_log: list[QueryRecord] = field(default_factory=list)
    epsilon_spent: float = 0.0
    generated_at: str = field(default_factory=timestamp)


class LaplaceCalibrationModule:
    """Laplace mechanism on aggregate calibration statistics, before they reach the simulator."""

    REQUIRED_FIELDS: frozenset[str] = frozenset({
        "vehicle_density_per_km",
        "pedestrian_density_per_km",
        "avg_vehicle_speed_kmh",
        "avg_pedestrian_speed_kmh",
        "intersection_count_per_km",
        "weather_distribution",
        "time_of_day_distribution",
    })

    SCALAR_SENSITIVITIES: dict[str, float] = {
        "vehicle_density_per_km":    1.0,
        "pedestrian_density_per_km": 1.0,
        "avg_vehicle_speed_kmh":     2.0,
        "avg_pedestrian_speed_kmh":  1.0,
        "intersection_count_per_km": 1.0,
    }

    def __init__(
        self,
        epsilon_calibration: float,
        random_seed: int = 42,
    ) -> None:
        if epsilon_calibration >= EPSILON_CALIBRATION_MAX:
            raise ValueError(
                f"epsilon_calibration={epsilon_calibration} must be below "
                f"{EPSILON_CALIBRATION_MAX}, the per-query cap."
            )
        self._epsilon = epsilon_calibration
        self._rng = np.random.default_rng(random_seed)
        logger.info("Initialised | epsilon=%.4f", epsilon_calibration)

    def _laplace(self, sensitivity: float) -> float:
        return float(self._rng.laplace(loc=0.0, scale=sensitivity / self._epsilon))

    def _privatise_scalar(
        self,
        name: str,
        value: float,
        sensitivity: float,
    ) -> tuple[float, QueryRecord]:
        noise = self._laplace(sensitivity)
        record = QueryRecord(
            name=name,
            sensitivity=sensitivity,
            noise=round(noise, 6),
            epsilon_used=round(sensitivity / self._epsilon, 6),
        )
        logger.debug(
            "%s | true=%.4f noise=%.4f noised=%.4f",
            name, value, noise, value + noise,
        )
        return round(value + noise, 4), record

    def _privatise_histogram(
        self,
        name: str,
        counts: dict[str, float],
        sensitivity: float,
    ) -> tuple[dict[str, float], QueryRecord]:
        noised = {k: max(0.0, v + self._laplace(sensitivity)) for k, v in counts.items()}
        total = sum(noised.values())
        if total > 0:
            noised = {k: round(v / total, 4) for k, v in noised.items()}
        else:
            n = len(counts)
            noised = {k: round(1.0 / n, 4) for k in counts}
        record = QueryRecord(
            name=name,
            sensitivity=sensitivity,
            noise=float("nan"),
            epsilon_used=round(sensitivity / self._epsilon, 6),
        )
        logger.debug("%s | noised=%s", name, noised)
        return noised, record

    def _validate(self, raw_stats: dict[str, Any]) -> None:
        missing = self.REQUIRED_FIELDS - raw_stats.keys()
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    def process(self, raw_stats: dict[str, Any]) -> CalibrationResult:
        """Privatise aggregate statistics."""
        self._validate(raw_stats)
        logger.info("Processing %d stat fields", len(self.REQUIRED_FIELDS))

        privatised: dict[str, Any] = {}
        query_log: list[QueryRecord] = []

        for field_name, sensitivity in self.SCALAR_SENSITIVITIES.items():
            value, record = self._privatise_scalar(
                field_name, raw_stats[field_name], sensitivity
            )
            privatised[field_name] = value
            query_log.append(record)

        for hist_field in ("weather_distribution", "time_of_day_distribution"):
            noised, record = self._privatise_histogram(
                hist_field, raw_stats[hist_field], sensitivity=1.0
            )
            privatised[hist_field] = noised
            query_log.append(record)

        # Sequential composition: every query reads the same records.
        total = self._epsilon * len(query_log)
        privatised["calibration_epsilon_spent"] = round(total, 4)
        privatised["calibration_epsilon_per_query"] = round(self._epsilon, 4)
        privatised["num_queries"] = len(query_log)
        privatised["generated_at"] = timestamp()
        privatised["note"] = (
            "All values Laplace-noised. No raw sensor data included. "
            "Safe to share with simulator operator (IT22541284)."
        )

        return CalibrationResult(
            stats=privatised,
            query_log=query_log,
            epsilon_spent=total,
        )

    def save(self, result: CalibrationResult, path: Path | str) -> None:
        save_json(result.stats, path)
        logger.info("Calibration stats saved -> %s", path)

    def save_audit_log(self, result: CalibrationResult, path: Path | str) -> None:
        log = {
            "epsilon_calibration": self._epsilon,
            "epsilon_per_query": self._epsilon,
            "epsilon_total": round(self._epsilon * len(result.query_log), 6),
            "composition": "sequential: every query reads the same records",
            "generated_at": result.generated_at,
            "queries": [
                {
                    "query": r.name,
                    "sensitivity": r.sensitivity,
                    "noise_added": r.noise,
                    "epsilon_used": r.epsilon_used,
                    "laplace_scale": r.epsilon_used,
                    "epsilon_spent": self._epsilon,
                }
                for r in result.query_log
            ],
        }
        save_json(log, path)
        logger.info("Audit log saved -> %s", path)
