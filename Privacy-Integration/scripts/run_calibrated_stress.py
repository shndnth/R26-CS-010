"""Stress test with a stronger, reference-calibrated attack.

Retrains the same stress-test targets (same seed, members and regime, so each
target matches the one pi-stress trained) and attacks them twice: with the
plain loss-threshold attack and with a reference-calibrated attack that uses a
few non-private reference models per seed. Results go to
outputs/stress_test/calibrated/. Safe to stop and rerun: finished seeds and
configurations are skipped, and reference results are cached per seed.

Example:
    pi-stress-calibrated --data-dir <decrypted> --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset

from privacy_integration.attack import features as attack_features
from privacy_integration.attack.calibrated import (
    calibrated_scores,
    evaluate_scores,
    reference_members,
    reference_pool,
    true_label_logits,
)
from privacy_integration.attack.stress import (
    NON_PRIVATE,
    OVERFIT_LR,
    SIGNIFICANCE_LEVEL,
    StressRegime,
    StressRun,
    bootstrap_auc_ci,
    eval_indices,
    from_run_files,
    pick_members,
    train_target,
)
from privacy_integration.attack.threshold import loss_threshold_attack
from privacy_integration.data.splits import stratified_split
from privacy_integration.evaluation.metrics import evaluate
from privacy_integration.logging_config import get_logger
from privacy_integration.paths import outputs_root
from privacy_integration.seeding import seed_everything
from privacy_integration.serialization import utc_timestamp, write_json
from privacy_integration.training.experiment import DP_DEFAULT_LR, train_baseline
from scripts._common import add_dataset_args, load_dataset_and_split, resolve_device
from scripts.run_stress_test import print_results

logger = get_logger("calibrated_stress")
DEFAULTS = StressRegime()
REPRODUCE_TOLERANCE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test with a reference-calibrated attack")
    add_dataset_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-ref", type=int, default=4, help="Reference models per seed (default 4).")
    parser.add_argument("--n-members", type=int, default=DEFAULTS.n_members)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULTS.epsilons))
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--bootstrap", type=int, default=DEFAULTS.bootstrap)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reference_logits(dataset, labels, pool, attacked, seed, n_ref, regime, device, cache):
    """Train the reference models for one seed (or load them) and score the attacked frames."""
    if cache.is_file():
        data = np.load(cache)
        if data["attacked"].tolist() == list(attacked) and data["phi"].shape[0] >= n_ref:
            logger.info("seed %d | reusing %d cached reference models", seed, data["phi"].shape[0])
            return data["phi"][:n_ref]
    rows = []
    for k in range(n_ref):
        ref_seed = seed * 1000 + k
        seed_everything(ref_seed)
        members = reference_members(pool, labels, regime.n_members, ref_seed)
        loader = DataLoader(Subset(dataset, members), batch_size=regime.batch_size, shuffle=True)
        logger.info("seed %d | reference %d/%d | %d frames", seed, k + 1, n_ref, len(members))
        model = train_baseline(loader, None, regime.epochs, OVERFIT_LR, device)
        rows.append(true_label_logits(model, dataset, attacked, regime.batch_size, device))
        _free(model)
    phi = np.vstack(rows)
    np.savez(cache, phi=phi, attacked=np.asarray(attacked))
    return phi


def low_fpr_comparison(summary) -> list[dict]:
    """No-DP versus each budget on TPR at 1% FPR, paired over seeds."""
    grouped = summary.by_config()
    base = {r.seed: r.tpr_at_1pct_fpr for r in grouped.get(NON_PRIVATE, [])}
    rows = []
    for tag, runs in grouped.items():
        if tag == NON_PRIVATE:
            continue
        pairs = [(base[r.seed], r.tpr_at_1pct_fpr) for r in runs if r.seed in base]
        if not pairs:
            continue
        a, b = np.array(pairs).T
        p = None
        if len(a) > 1 and np.std(a - b) > 0:
            p = float(stats.ttest_rel(a, b, alternative="greater").pvalue)
        rows.append(
            {
                "config": tag,
                "seeds": len(a),
                "no_dp_tpr_at_1pct_fpr_mean": round(float(a.mean()), 4),
                "dp_tpr_at_1pct_fpr_mean": round(float(b.mean()), 4),
                "random_guess": 0.01,
                "p_value_one_sided": None if p is None else round(p, 5),
                "significant": p is not None and p < SIGNIFICANCE_LEVEL,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset, _split, cfg, _ = load_dataset_and_split(args)
    device = resolve_device(args.device)
    regime = StressRegime(args.n_members, args.epochs, args.batch_size, tuple(args.epsilons), args.bootstrap)
    stress_dir = outputs_root() / "stress_test"
    out_dir = stress_dir / "calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = dataset.labels
    configs = [None, *regime.epsilons]

    for seed in args.seeds:
        tags = [NON_PRIVATE if e is None else f"eps_{e:g}" for e in configs]
        if all((out_dir / f"seed_{seed}_{t}.json").is_file() for t in tags):
            logger.info("seed %d | already done, skipping", seed)
            continue

        split = stratified_split(labels, cfg.dataset.train_fraction, cfg.dataset.val_fraction, seed)
        members, nonmembers = pick_members(split.train, split.test, regime.n_members, seed)
        test_idx = eval_indices(split, nonmembers)
        attacked = [*members, *nonmembers]
        pool = reference_pool(split.train, split.val, attacked)
        n = len(members)
        logger.info("seed %d | %d members, %d non-members, %d reference frames", seed, n, n, len(pool))

        ref_phi = reference_logits(
            dataset,
            labels,
            pool,
            attacked,
            seed,
            args.n_ref,
            regime,
            device,
            out_dir / f"seed_{seed}_refs.npz",
        )

        for epsilon, tag in zip(configs, tags, strict=True):
            path = out_dir / f"seed_{seed}_{tag}.json"
            if path.is_file():
                continue
            seed_everything(seed)
            model, final_epsilon = train_target(dataset, labels, members, epsilon, regime, device, cfg)

            phi = true_label_logits(model, dataset, attacked, regime.batch_size, device)
            scores = calibrated_scores(phi, ref_phi)
            outcome = evaluate_scores(scores[:n], scores[n:])
            ci = bootstrap_auc_ci(scores[:n], scores[n:], regime.bootstrap, seed)

            def losses(idx, model=model):
                loader = DataLoader(Subset(dataset, list(idx)), batch_size=regime.batch_size)
                return attack_features.extract(model, loader, device)[:, attack_features.LOSS]

            loss_outcome = loss_threshold_attack(losses(members), losses(nonmembers))
            test_auc = evaluate(
                model, DataLoader(Subset(dataset, test_idx), batch_size=regime.batch_size), device
            ).auc
            _free(model)

            earlier = stress_dir / f"seed_{seed}_{tag}.json"
            earlier_auc = json.loads(earlier.read_text())["attack_auc"] if earlier.is_file() else None
            run = StressRun(
                config=tag,
                seed=seed,
                epsilon=epsilon,
                final_epsilon=final_epsilon,
                attack_auc=outcome.auc,
                attack_auc_ci=ci,
                tpr_at_1pct_fpr=outcome.tpr_at_fpr[0.01],
                tpr_at_5pct_fpr=outcome.tpr_at_fpr[0.05],
                member_loss_mean=loss_outcome.member_loss_mean,
                nonmember_loss_mean=loss_outcome.nonmember_loss_mean,
                train_test_loss_gap=loss_outcome.nonmember_loss_mean - loss_outcome.member_loss_mean,
                target_test_auc=test_auc,
            )
            record = {
                **run.as_dict(),
                "attack": "reference_calibrated",
                "n_reference_models": int(ref_phi.shape[0]),
                "loss_attack_auc": round(loss_outcome.auc, 4),
                "loss_attack_tpr_at_1pct_fpr": round(loss_outcome.tpr_at_fpr[0.01], 4),
                "stress_run_loss_attack_auc": earlier_auc,
                "reproduces_stress_run": None
                if earlier_auc is None
                else abs(loss_outcome.auc - earlier_auc) <= REPRODUCE_TOLERANCE,
                "generated_at": utc_timestamp(),
            }
            write_json(record, path)
            logger.info(
                "seed %d %-6s | calibrated AUC %.4f [%.4f, %.4f] TPR@1%%FPR %.3f "
                "| loss AUC %.4f (stress run %s)",
                seed,
                tag,
                outcome.auc,
                *ci,
                outcome.tpr_at_fpr[0.01],
                loss_outcome.auc,
                earlier_auc,
            )

    summary = from_run_files(out_dir)
    comparisons, verdict, ordering = summary.compare(), summary.verdict(), summary.ordering()
    low_fpr = low_fpr_comparison(summary)
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("seed_*_*.json"))]
    report = {
        "description": (
            "Stress test targets attacked with a reference-calibrated membership inference attack "
            "(offline, fixed variance; after Carlini et al. 2022 and Watson et al. 2022). Reference "
            "models are non-private and never saw the attacked frames."
        ),
        "generated_at": utc_timestamp(),
        "regime": {
            "n_members": regime.n_members,
            "epochs": regime.epochs,
            "batch_size": regime.batch_size,
            "baseline_optimizer": f"Adam lr={OVERFIT_LR}",
            "dp_optimizer": f"SGD momentum lr={DP_DEFAULT_LR}",
            "epsilons": list(regime.epsilons),
            "n_reference_models": args.n_ref,
            "seeds": summary.seeds(),
            "significance_level": SIGNIFICANCE_LEVEL,
        },
        "runs": records,
        "comparisons": comparisons,
        "low_fpr_comparisons": low_fpr,
        "epsilon_ordering": ordering,
        "verdict": verdict,
        "targets_match_stress_runs": all(r.get("reproduces_stress_run") is not False for r in records),
    }
    write_json(report, out_dir / "calibrated_summary.json")

    print_results(comparisons, verdict, ordering, summary.seeds())
    print("\nTPR at 1% FPR (random guessing 0.01), paired one-sided:")
    for row in low_fpr:
        p = "n/a" if row["p_value_one_sided"] is None else f"{row['p_value_one_sided']:.4g}"
        print(
            f"  no_dp {row['no_dp_tpr_at_1pct_fpr_mean']:.3f} vs {row['config']} "
            f"{row['dp_tpr_at_1pct_fpr_mean']:.3f}  p {p}"
        )
    print(f"\ntargets match the earlier stress runs: {report['targets_match_stress_runs']}")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
