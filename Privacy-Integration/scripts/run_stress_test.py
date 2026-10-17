"""DP stress test against a deliberately overfitted baseline (PP2 panel response).

Example:
    pi-stress --data-dir data/decrypted --labels-dir data/clean_labels --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv

from privacy_integration.attack.stress import (
    NON_PRIVATE,
    OVERFIT_LR,
    POSITIVE_CONTROL_MIN_AUC,
    StressRegime,
    StressSummary,
    pick_members,
    run_one,
)
from privacy_integration.data.splits import stratified_split
from privacy_integration.logging_config import get_logger
from privacy_integration.paths import outputs_root
from privacy_integration.seeding import seed_everything
from privacy_integration.serialization import utc_timestamp, write_json
from privacy_integration.training.experiment import DP_DEFAULT_LR
from scripts._common import add_dataset_args, load_dataset_and_split, resolve_device

logger = get_logger("stress_test")


DEFAULTS = StressRegime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP stress test against an overfitted baseline")
    add_dataset_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--n-members",
        type=int,
        default=DEFAULTS.n_members,
        help="Members per target (small on purpose, so the baseline memorises them).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULTS.epsilons))
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--bootstrap", type=int, default=DEFAULTS.bootstrap)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, _split, cfg, _ = load_dataset_and_split(args)
    device = resolve_device(args.device)
    regime = StressRegime(args.n_members, args.epochs, args.batch_size, tuple(args.epsilons), args.bootstrap)
    out_dir = outputs_root() / "stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = dataset.labels

    summary = StressSummary()
    for seed in args.seeds:
        split = stratified_split(labels, cfg.dataset.train_fraction, cfg.dataset.val_fraction, seed)
        members, nonmembers = pick_members(split.train, split.test, regime.n_members, seed)
        held_out = set(nonmembers)
        test_idx = [i for i in split.test if i not in held_out] or split.test
        logger.info(
            "seed %d | %d members, %d non-members, %d epochs",
            seed,
            len(members),
            len(nonmembers),
            regime.epochs,
        )

        for epsilon in (None, *regime.epsilons):
            seed_everything(seed)
            run = run_one(dataset, labels, members, nonmembers, test_idx, epsilon, seed, regime, device, cfg)
            summary.runs.append(run)
            write_json(
                {**run.as_dict(), "generated_at": utc_timestamp()}, out_dir / f"seed_{seed}_{run.config}.json"
            )
            logger.info(
                "seed %d %-8s | attack AUC %.4f [%.4f, %.4f] | TPR@1%%FPR %.3f "
                "| loss gap %.4f | test AUC %.4f",
                seed,
                run.config,
                run.attack_auc,
                *run.attack_auc_ci,
                run.tpr_at_1pct_fpr,
                run.train_test_loss_gap,
                run.target_test_auc,
            )

    comparisons = summary.compare()
    verdict = summary.verdict()
    report = {
        "description": (
            "Stress test of DP-SGD against membership inference. Every target trains on a small "
            "member set for many epochs, so the non-private model overfits and leaks strongly "
            "(positive control). DP models train in the same regime, so the drop in attack AUC "
            "is attributable to DP-SGD."
        ),
        "panel_comment": "PP2: even with differential privacy, the reduction of attacks is not significant",
        "generated_at": utc_timestamp(),
        "regime": {
            "n_members": regime.n_members,
            "n_nonmembers": regime.n_members,
            "epochs": regime.epochs,
            "batch_size": regime.batch_size,
            "baseline_optimizer": f"Adam lr={OVERFIT_LR}",
            "dp_optimizer": f"SGD momentum lr={DP_DEFAULT_LR}",
            "epsilons": list(regime.epsilons),
            "delta": cfg.privacy.delta,
            "clipping_norm": cfg.privacy.clipping_norm,
            "bootstrap_resamples": regime.bootstrap,
            "seeds": args.seeds,
            "positive_control_min_auc": POSITIVE_CONTROL_MIN_AUC,
        },
        "runs": [r.as_dict() for r in summary.runs],
        "comparisons": comparisons,
        "verdict": verdict,
    }
    write_json(report, out_dir / "stress_test_summary.json")

    with (out_dir / "stress_test_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [r.as_dict() for r in summary.runs]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{'config':<10}{'no_dp AUC':>11}{'DP AUC':>9}{'drop':>8}{'removed':>9}"
        f"{'p (1-sided)':>13}  CIs disjoint"
    )
    for c in comparisons:
        p = "n/a" if c["p_value_one_sided"] is None else f"{c['p_value_one_sided']:.4g}"
        print(
            f"{c['config']:<10}{c['no_dp_attack_auc_mean']:>11.4f}{c['dp_attack_auc_mean']:>9.4f}"
            f"{c['attack_auc_drop_mean']:>8.4f}{c['advantage_removed_pct']:>8.1f}%{p:>13}  "
            f"{c['ci95_disjoint_every_seed']}"
        )
    print(f"\npositive control ({NON_PRIVATE} leaks): {verdict['positive_control']}")
    print(f"DP reduction: {verdict['dp_reduction']}")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
