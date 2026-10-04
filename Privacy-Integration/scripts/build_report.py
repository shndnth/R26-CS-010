"""Aggregate runs across seeds and render the privacy-utility figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from privacy_integration.artifacts import BASELINE_TAG, ArtifactStore  # noqa: E402
from privacy_integration.logging_config import get_logger  # noqa: E402
from privacy_integration.serialization import read_json, utc_timestamp, write_json  # noqa: E402

logger = get_logger("report")

FIGURE_DPI = 200


def collect(store: ArtifactStore) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    for seed in store.completed_seeds():
        for tag in store.completed_runs(seed):
            paths = store.run(seed, tag)
            entry = read_json(paths.metrics)
            if paths.mia_results.is_file():
                entry["mia"] = read_json(paths.mia_results)
            records[tag].append(entry)
    return records


def summarise(records: dict[str, list[dict]]) -> dict:
    summary = {}
    for tag, entries in records.items():
        aucs = [e["test_metrics"]["auc"] for e in entries]
        attack = [e["mia"]["attack_auc"] for e in entries if "mia" in e]
        gaps = [e["mia"]["target_loss_gap"] for e in entries if "mia" in e]
        summary[tag] = {
            "n_seeds": len(entries),
            "seeds": sorted(e["seed"] for e in entries),
            "target_epsilon": entries[0].get("target_epsilon"),
            "utility_auc_mean": round(mean(aucs), 4),
            "utility_auc_sd": round(stdev(aucs), 4) if len(aucs) > 1 else None,
            "attack_auc_mean": round(mean(attack), 4) if attack else None,
            "attack_auc_sd": round(stdev(attack), 4) if len(attack) > 1 else None,
            "loss_gap_mean": round(mean(gaps), 4) if gaps else None,
        }
    return summary


def _ordered(summary: dict) -> list[tuple[str, dict]]:
    private = [(t, s) for t, s in summary.items() if s["target_epsilon"] is not None]
    private.sort(key=lambda item: item[1]["target_epsilon"])
    baseline = [(t, s) for t, s in summary.items() if s["target_epsilon"] is None]
    return private + baseline


def plot_privacy_utility(summary: dict, path) -> None:
    ordered = _ordered(summary)
    private = [(t, s) for t, s in ordered if s["target_epsilon"] is not None]
    if not private:
        return

    epsilons = [s["target_epsilon"] for _, s in private]
    means = [s["utility_auc_mean"] for _, s in private]
    errors = [s["utility_auc_sd"] or 0.0 for _, s in private]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(epsilons, means, yerr=errors, marker="o", capsize=4, linewidth=2, label="DP-SGD")

    baseline = summary.get(BASELINE_TAG)
    if baseline:
        ax.axhline(
            baseline["utility_auc_mean"], linestyle="--", color="tab:green",
            label=f"Non-private baseline ({baseline['utility_auc_mean']:.3f})",
        )
    ax.axhline(0.5, linestyle=":", color="grey", label="Random (0.50)")

    ax.set_xscale("log")
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Privacy budget ε (log scale)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Privacy-utility trade-off")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_attack_resistance(summary: dict, path) -> None:
    ordered = _ordered(summary)
    labels = [t for t, s in ordered if s["attack_auc_mean"] is not None]
    if not labels:
        return

    values = [summary[t]["attack_auc_mean"] for t in labels]
    errors = [summary[t]["attack_auc_sd"] or 0.0 for t in labels]
    colours = ["tab:red" if t == BASELINE_TAG else "tab:blue" for t in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, yerr=errors, capsize=4, color=colours)
    ax.axhline(0.5, linestyle=":", color="grey", label="Random (0.50)")
    ax.set_ylabel("Attack AUC")
    ax.set_ylim(0.45, max(0.70, max(values) + 0.05))
    ax.set_title("Membership inference resistance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_loss_gap(summary: dict, path) -> None:
    ordered = _ordered(summary)
    labels = [t for t, s in ordered if s["loss_gap_mean"] is not None]
    if not labels:
        return

    values = [summary[t]["loss_gap_mean"] for t in labels]
    colours = ["tab:red" if t == BASELINE_TAG else "tab:blue" for t in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=colours)
    ax.set_ylabel("Member / non-member loss gap")
    ax.set_title("Memorisation suppressed by DP-SGD")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and render figures")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    store = ArtifactStore()
    records = collect(store)
    if not records:
        raise SystemExit(f"no runs found under {store.root}")

    summary = summarise(records)
    write_json({"generated_at": utc_timestamp(), "configs": summary}, store.summary_path())

    print(f"\n{'config':<10}{'eps':>6}{'seeds':>7}{'utility AUC':>16}{'attack AUC':>14}{'loss gap':>11}")
    print("-" * 66)
    for tag, stats in _ordered(summary):
        eps = "n/a" if stats["target_epsilon"] is None else f"{stats['target_epsilon']:.1f}"
        sd = f" ±{stats['utility_auc_sd']:.3f}" if stats["utility_auc_sd"] else ""
        attack = f"{stats['attack_auc_mean']:.4f}" if stats["attack_auc_mean"] else "n/a"
        gap = f"{stats['loss_gap_mean']:.4f}" if stats["loss_gap_mean"] else "n/a"
        print(
            f"{tag:<10}{eps:>6}{stats['n_seeds']:>7}"
            f"{stats['utility_auc_mean']:>11.4f}{sd:<5}{attack:>14}{gap:>11}"
        )
    print()

    if not args.no_figures:
        figures = store.figures_dir()
        plot_privacy_utility(summary, figures / "privacy_utility_tradeoff.png")
        plot_attack_resistance(summary, figures / "attack_resistance.png")
        plot_loss_gap(summary, figures / "memorisation_loss_gap.png")

    logger.info("summary written to %s", store.summary_path())


if __name__ == "__main__":
    main()
