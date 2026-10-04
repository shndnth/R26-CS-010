"""Train the presence classifier under DP-SGD, or the non-private baseline."""

from __future__ import annotations

import argparse

from privacy_integration.artifacts import BASELINE_TAG, ArtifactStore
from privacy_integration.evaluation.metrics import evaluate
from privacy_integration.logging_config import get_logger
from privacy_integration.seeding import seed_everything
from privacy_integration.serialization import utc_timestamp
from privacy_integration.training.experiment import (
    BASELINE_DEFAULT_LR,
    DP_DEFAULT_LR,
    ExperimentResult,
    build_loaders,
    persist,
    train_baseline,
    train_private,
)
from scripts._common import add_dataset_args, add_run_args, load_dataset_and_split, resolve_device

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP training for R26-CS-010 Privacy Integration")
    add_dataset_args(parser)
    add_run_args(parser)
    parser.add_argument(
        "--config",
        dest="configs",
        nargs="+",
        default=None,
        help="Ablation configs to run. Omit for all. Use 'no_dp' for the baseline.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, split, cfg, seed = load_dataset_and_split(args)

    seed_everything(seed)
    device = resolve_device(args.device)
    batch_size = args.batch_size or cfg.dataset.batch_size
    store = ArtifactStore()

    train_loader, val_loader, test_loader = build_loaders(dataset, split, batch_size)

    requested = args.configs or [*cfg.ablation_names, BASELINE_TAG]
    run_meta = {
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "resolution": f"{cfg.dataset.train_width}x{cfg.dataset.train_height}",
        "min_box_area_native": cfg.dataset.min_box_area_native,
        "split_sizes": {"train": split.sizes[0], "val": split.sizes[1], "test": split.sizes[2]},
        "dataset_stats": dataset.stats.as_dict(),
        "generated_at": utc_timestamp(),
    }

    for tag in requested:
        paths = store.run(seed, tag)
        logger.info("=" * 70)

        if tag == BASELINE_TAG:
            epochs = args.epochs or 15
            lr = args.lr or BASELINE_DEFAULT_LR
            model = train_baseline(train_loader, val_loader, epochs, lr, device)
            wrapper = training_result = None
            epsilon = final_epsilon = noise_multiplier = None
        else:
            ablation = cfg.ablation_by_name(tag)
            epochs = args.epochs or ablation.epochs
            lr = args.lr or DP_DEFAULT_LR
            epsilon = ablation.target_epsilon
            logger.info("%s | target_epsilon=%.1f epochs=%d lr=%s", tag, epsilon, epochs, lr)
            model, training_result, noise_multiplier = train_private(
                epsilon, train_loader, epochs, lr, device, cfg
            )
            final_epsilon = training_result.final_epsilon
            from privacy_integration.training.dp_trainer import DPTrainingWrapper  # noqa: F401

            wrapper = _wrapper_from(model, noise_multiplier)

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        logger.info("%s | val  %s", tag, val_metrics.summary())
        logger.info("%s | test %s", tag, test_metrics.summary())

        result = ExperimentResult(
            tag=tag,
            seed=seed,
            epsilon=epsilon,
            final_epsilon=final_epsilon,
            noise_multiplier=noise_multiplier,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        persist(
            paths, model, result, {**run_meta, "epochs": epochs, "lr": lr},
            training_result, wrapper, cfg,
        )

    logger.info("=" * 70)
    logger.info("All runs complete | artifacts under %s", store.root / f"seed_{seed}")


def _wrapper_from(model, noise_multiplier):
    """Adapter exposing the wrapper interface persist() needs."""

    class _Adapter:
        def __init__(self) -> None:
            self.model = model
            self.noise_multiplier = noise_multiplier

        @staticmethod
        def save_budget_log(result, path) -> None:
            import pandas as pd

            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "step": r.step,
                        "epoch": r.epoch,
                        "cumulative_eps": r.cumulative_eps,
                        "loss": r.loss,
                    }
                    for r in result.budget_log
                ]
            ).to_csv(path, index=False)

    return _Adapter()


if __name__ == "__main__":
    main()
