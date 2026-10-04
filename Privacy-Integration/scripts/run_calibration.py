from __future__ import annotations

import argparse
import json
from pathlib import Path

from privacy_integration.calibration.laplace import LaplaceCalibrationModule
from privacy_integration.legacy_constants import EPSILON_CALIBRATION_MAX
from privacy_integration.logging_config import get_logger as setup_logging
from privacy_integration.serialization import read_json as load_json
from privacy_integration.settings import settings

logger = setup_logging("run_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-Aware Calibration Module (R26-CS-010)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON file containing raw aggregate calibration statistics.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=f"Epsilon per calibration query (default: from config.yaml, below {EPSILON_CALIBRATION_MAX}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/calibration/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = settings()

    epsilon = args.epsilon or cfg.privacy.epsilon_calibration_max - 0.2
    seed = args.seed or cfg.privacy.random_seed
    output_dir = args.output_dir or cfg.outputs_dir / "calibration"

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_stats = load_json(args.input)
    logger.info("Loaded input from %s", args.input)

    module = LaplaceCalibrationModule(
        epsilon_calibration=epsilon,
        random_seed=seed,
    )

    result = module.process(raw_stats)

    stats_path = output_dir / "calibration_stats.json"
    audit_path = output_dir / "calibration_audit_log.json"

    module.save(result, stats_path)
    module.save_audit_log(result, audit_path)

    logger.info("Calibration complete")
    logger.info("  epsilon per query : %.4f (cap %.1f)", epsilon, EPSILON_CALIBRATION_MAX)
    logger.info("  epsilon total     : %.4f over %d queries", result.epsilon_spent, len(result.query_log))
    logger.info("  stats         : %s", stats_path)
    logger.info("  audit log     : %s", audit_path)

    print(json.dumps(result.stats, indent=2))


if __name__ == "__main__":
    main()
