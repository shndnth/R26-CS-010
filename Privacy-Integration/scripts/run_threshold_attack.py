"""Loss-threshold attack with TPR at low FPR against every saved checkpoint. No retraining."""

from __future__ import annotations

import argparse
import csv

import numpy as np
from torch.utils.data import DataLoader, Subset

from privacy_integration.artifacts import BASELINE_TAG, ArtifactStore
from privacy_integration.attack import features as attack_features
from privacy_integration.attack.threshold import LOW_FPRS, loss_threshold_attack
from privacy_integration.data.splits import stratified_split, subsample_to
from privacy_integration.evaluation.metrics import evaluate
from privacy_integration.logging_config import get_logger
from privacy_integration.serialization import utc_timestamp, write_json
from privacy_integration.training.experiment import load_checkpoint_model
from scripts._common import add_dataset_args, add_run_args, load_dataset_and_split, resolve_device

logger = get_logger("threshold_attack")

AUC_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loss-threshold MIA on saved checkpoints")
    add_dataset_args(parser)
    add_run_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: every completed seed.")
    parser.add_argument("--config", dest="configs", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, _split, cfg, _seed = load_dataset_and_split(args)
    device = resolve_device(args.device)
    batch_size = args.batch_size or cfg.dataset.batch_size
    store = ArtifactStore()
    seeds = args.seeds or store.completed_seeds()
    tags = args.configs or [*cfg.ablation_names, BASELINE_TAG]

    rows = []
    for seed in seeds:
        split = stratified_split(dataset.labels, cfg.dataset.train_fraction, cfg.dataset.val_fraction, seed)
        n_eval = min(len(split.train), len(split.test))
        member_idx = subsample_to(split.train, n_eval, seed)
        nonmember_idx = subsample_to(split.test, n_eval, seed)
        members = DataLoader(Subset(dataset, member_idx), batch_size=batch_size)
        nonmembers = DataLoader(Subset(dataset, nonmember_idx), batch_size=batch_size)
        test_loader = DataLoader(Subset(dataset, split.test), batch_size=batch_size)

        for tag in tags:
            paths = store.run(seed, tag)
            if not paths.checkpoint.is_file():
                logger.warning("skipping seed %d %s | no checkpoint", seed, tag)
                continue

            model, checkpoint = load_checkpoint_model(paths.checkpoint, device)
            stored_auc = checkpoint.get("test_metrics", {}).get("auc")
            recomputed_auc = evaluate(model, test_loader, device).auc
            split_matches = stored_auc is None or abs(recomputed_auc - stored_auc) <= AUC_TOLERANCE
            if not split_matches:
                logger.error(
                    "seed %d %s | test AUC %.4f differs from training (%.4f); "
                    "use the same --data-dir, --labels-dir and --towns as pi-train",
                    seed, tag, recomputed_auc, stored_auc,
                )

            outcome = loss_threshold_attack(
                attack_features.extract(model, members, device)[:, attack_features.LOSS],
                attack_features.extract(model, nonmembers, device)[:, attack_features.LOSS],
            )
            epsilon = None if tag == BASELINE_TAG else cfg.ablation_by_name(tag).target_epsilon
            record = {
                **outcome.as_dict(),
                "config": tag,
                "epsilon": epsilon,
                "seed": seed,
                "n_members": n_eval,
                "n_nonmembers": n_eval,
                "target_test_auc_recomputed": round(recomputed_auc, 4),
                "split_matches_training": split_matches,
                "generated_at": utc_timestamp(),
            }
            write_json(record, paths.run_dir / "threshold_attack.json")
            rows.append(record)
            logger.info(
                "seed %d %-9s | AUC %.4f | TPR@1%%FPR %.4f | TPR@5%%FPR %.4f",
                seed, tag, outcome.auc, outcome.tpr_at_fpr[0.01], outcome.tpr_at_fpr[0.05],
            )

    if not rows:
        raise SystemExit("no checkpoints found")

    summary = store.root / "threshold_attack_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'config':<10}{'seeds':>6}{'AUC':>16}{'TPR@1%FPR':>18}{'TPR@5%FPR':>18}")
    for tag in tags:
        picked = [r for r in rows if r["config"] == tag]
        if not picked:
            continue
        keys = ["attack_auc", *(f"tpr_at_{int(f * 100)}pct_fpr" for f in LOW_FPRS)]
        cols = [np.array([r[k] for r in picked]) for k in keys]
        cells = (f"{c.mean():>10.4f} ± {c.std(ddof=1) if len(c) > 1 else 0:.4f}" for c in cols)
        print(f"{tag:<10}{len(picked):>6}" + "".join(cells))
    print("\nrandom guessing: AUC 0.5, TPR@1%FPR 0.01, TPR@5%FPR 0.05")
    print(f"per-run JSON in each run folder, summary at {summary}")
    if not all(r["split_matches_training"] for r in rows):
        print("\nWARNING: some runs did not reproduce their training split; see the errors above")


if __name__ == "__main__":
    main()
