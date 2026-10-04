"""Shadow-model membership inference attack against trained checkpoints."""

from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader, Subset

from privacy_integration.artifacts import BASELINE_TAG, ArtifactStore
from privacy_integration.attack import features as attack_features
from privacy_integration.attack.classifier import (
    RANDOM_BASELINE_PCT,
    evaluate_attack,
    train_classifier,
)
from privacy_integration.data.splits import balance_by_oversampling, subsample_to
from privacy_integration.logging_config import get_logger
from privacy_integration.seeding import seed_everything
from privacy_integration.serialization import utc_timestamp, write_json
from privacy_integration.training.experiment import (
    BASELINE_DEFAULT_LR,
    DP_DEFAULT_LR,
    load_checkpoint_model,
    train_baseline,
    train_private,
)
from scripts._common import add_dataset_args, add_run_args, load_dataset_and_split, resolve_device

logger = get_logger("mia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIA calibration study for R26-CS-010")
    add_dataset_args(parser)
    add_run_args(parser)
    parser.add_argument("--config", dest="configs", nargs="+", default=None)
    parser.add_argument("--n-shadow", type=int, default=None)
    parser.add_argument("--shadow-epochs", type=int, default=None)
    parser.add_argument("--attack-epochs", type=int, default=None)
    return parser.parse_args()


def _train_shadow(dataset, member_idx, labels, epsilon, epochs, lr, batch_size, device, cfg):
    balanced = balance_by_oversampling(member_idx, labels)
    loader = DataLoader(Subset(dataset, balanced), batch_size=batch_size, shuffle=True)
    if epsilon is None:
        return train_baseline(loader, None, epochs, BASELINE_DEFAULT_LR, device)
    model, _result, _sigma = train_private(epsilon, loader, epochs, lr, device, cfg)
    return model


def build_attack_set(dataset, labels, epsilon, n_shadow, epochs, lr, batch_size, device, cfg, seed):
    """Train shadow models under the target's own DP configuration."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    all_features: list[list[float]] = []
    all_labels: list[int] = []
    gaps: list[float] = []

    for i in range(n_shadow):
        indices = rng.permutation(n)
        half = n // 2
        members, nonmembers = indices[:half].tolist(), indices[half:].tolist()

        logger.info("  shadow %d/%d | %d member samples", i + 1, n_shadow, len(members))
        shadow = _train_shadow(dataset, members, labels, epsilon, epochs, lr, batch_size, device, cfg)

        member_feat = attack_features.extract(
            shadow, DataLoader(Subset(dataset, members), batch_size=batch_size), device
        )
        nonmember_feat = attack_features.extract(
            shadow, DataLoader(Subset(dataset, nonmembers), batch_size=batch_size), device
        )

        all_features.extend(member_feat.tolist() + nonmember_feat.tolist())
        all_labels.extend([1] * len(member_feat) + [0] * len(nonmember_feat))
        gap = attack_features.loss_gap(member_feat, nonmember_feat)
        gaps.append(gap)
        logger.info("  shadow %d/%d | loss gap %.4f", i + 1, n_shadow, gap)

    return np.asarray(all_features), np.asarray(all_labels), float(np.mean(gaps))


def main() -> None:
    args = parse_args()
    dataset, split, cfg, seed = load_dataset_and_split(args)

    seed_everything(seed)
    device = resolve_device(args.device)
    batch_size = args.batch_size or cfg.dataset.batch_size
    n_shadow = args.n_shadow or cfg.attack.n_shadow_models
    shadow_epochs = args.shadow_epochs or cfg.attack.shadow_epochs
    attack_epochs = args.attack_epochs or cfg.attack.attack_epochs

    store = ArtifactStore()
    labels = dataset.labels

    # Size-match members and non-members so accuracy reflects the attack, not the class prior.
    n_eval = min(len(split.train), len(split.test))
    member_idx = subsample_to(split.train, n_eval, seed)
    nonmember_idx = subsample_to(split.test, n_eval, seed)
    logger.info(
        "Membership boundary | train=%d test=%d -> balanced eval %d vs %d (baseline %.0f%%)",
        len(split.train), len(split.test), len(member_idx), len(nonmember_idx), RANDOM_BASELINE_PCT,
    )

    member_loader = DataLoader(Subset(dataset, member_idx), batch_size=batch_size)
    nonmember_loader = DataLoader(Subset(dataset, nonmember_idx), batch_size=batch_size)

    requested = args.configs or [*cfg.ablation_names, BASELINE_TAG]
    results = []

    for tag in requested:
        paths = store.run(seed, tag)
        if not paths.checkpoint.is_file():
            logger.warning("skipping %s | no checkpoint at %s", tag, paths.checkpoint)
            continue

        epsilon = None if tag == BASELINE_TAG else cfg.ablation_by_name(tag).target_epsilon
        logger.info("=" * 70)
        logger.info("attacking %s | epsilon=%s", tag, epsilon)

        target, checkpoint = load_checkpoint_model(paths.checkpoint, device)

        shadow_features, shadow_labels, shadow_gap = build_attack_set(
            dataset, labels, epsilon, n_shadow, shadow_epochs,
            DP_DEFAULT_LR, batch_size, device, cfg, seed,
        )
        classifier = train_classifier(shadow_features, shadow_labels, attack_epochs, device)

        outcome = evaluate_attack(
            classifier,
            attack_features.extract(target, member_loader, device),
            attack_features.extract(target, nonmember_loader, device),
            device,
            cfg.attack.success_rate_ceiling,
        )

        record = {
            **outcome.as_dict(),
            "config": tag,
            "epsilon": epsilon,
            "seed": seed,
            "n_shadow_models": n_shadow,
            "shadow_loss_gap": round(shadow_gap, 4),
            "target_test_auc": checkpoint.get("test_metrics", {}).get("auc"),
        }
        write_json(record, paths.mia_results)
        results.append(record)

        logger.info(
            "%s | attack_rate=%.2f%% attack_auc=%.4f target_loss_gap=%.4f | %s",
            tag, outcome.success_rate_pct, outcome.auc, outcome.loss_gap, outcome.interpretation,
        )

    if not results:
        raise SystemExit("no checkpoints found; run training first")

    write_json(
        {
            "description": (
                "Empirical epsilon-to-attack-resistance calibration curve on real CARLA "
                "synthetic data. Shadow-model membership inference with a two-feature "
                "signal (confidence + per-sample loss), extending Shokri et al. (2017)."
            ),
            "generated_at": utc_timestamp(),
            "seed": seed,
            "random_baseline_pct": RANDOM_BASELINE_PCT,
            "success_ceiling_pct": cfg.attack.success_rate_ceiling,
            "evaluation_balanced": True,
            "eval_members": n_eval,
            "eval_nonmembers": n_eval,
            "data_points": results,
        },
        store.curve_path(seed),
    )
    logger.info("=" * 70)
    logger.info("MIA study complete | %d configs | curve at %s", len(results), store.curve_path(seed))


if __name__ == "__main__":
    main()
