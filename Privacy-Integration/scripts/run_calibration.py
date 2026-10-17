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
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--epsilon-total",
        type=float,
        default=None,
        help="Total calibration budget, split evenly over the 7 queries "
             "(default: epsilon_calibration_total in config.yaml).",
    )
    budget.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=f"Legacy: epsilon PER QUERY (below {EPSILON_CALIBRATION_MAX}); the total is 7x this.",
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

    seed = args.seed or cfg.privacy.random_seed
    output_dir = args.output_dir or cfg.outputs_dir / "calibration"

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_stats = load_json(args.input)
    logger.info("Loaded input from %s", args.input)

    if args.epsilon is not None:
        logger.warning("--epsilon is per query; the 7 queries compose to %.4f", 7 * args.epsilon)
        module = LaplaceCalibrationModule(epsilon_calibration=args.epsilon, random_seed=seed)
    else:
        total = args.epsilon_total or cfg.privacy.epsilon_calibration_total
        module = LaplaceCalibrationModule.from_total(total, random_seed=seed)
    epsilon = module.epsilon_per_query

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
