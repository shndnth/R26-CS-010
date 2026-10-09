"""Steps 8 and 9: aggregate the results into the utility findings, with figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utility_evaluation.config import add_root_argument, workspace_from
from utility_evaluation.reporting import EXPERIMENT_ORDER, load

FIGURE_DPI = 200


def plot_experiments(report, path: Path) -> None:
    labels = [name for name in EXPERIMENT_ORDER if name in report.experiments]
    if not labels:
        return

    values = [report.experiments[name].map50_95 for name in labels]
    # Real-data baseline in a distinct colour; it is the reference point.
    colours = ["tab:green" if name == "real -> real" else "tab:blue" for name in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([name.replace(" -> ", "\n-> ") for name in labels], values, color=colours)
    ax.set_ylabel("mAP50-95")
    ax.set_title("Detection performance across train and test domains")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"wrote {path}")


def plot_per_class(report, path: Path) -> None:
    experiment = report.experiments.get("synthetic -> synthetic")
    if experiment is None:
        return

    scored = [
        (name, values["ap50_95"])
        for name, values in experiment.per_class.items()
        if isinstance(values, dict) and "ap50_95" in values
    ]
    if not scored:
        return

    names = [n for n, _ in scored]
    values = [v for _, v in scored]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, values, color="tab:blue")
    ax.set_ylabel("AP50-95")
    ax.set_title("Per-class detection, synthetic data")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the utility results")
    add_root_argument(parser)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results = workspace_from(args).results
    output = results / "utility_results.json"
    report = load(results)
    if not report.experiments:
        raise SystemExit(f"no evaluations found under {results}. Run cross_evaluation.py first.")

    print(f"{'experiment':<26}{'mAP50':>10}{'mAP50-95':>12}")
    print("-" * 48)
    for label in EXPERIMENT_ORDER:
        experiment = report.experiments.get(label)
        if experiment is None:
            print(f"{label:<26}{'not run':>10}{'':>12}")
        else:
            print(f"{label:<26}{experiment.map50:>10.4f}{experiment.map50_95:>12.4f}")

    print()
    if report.retention_pct is not None:
        print(f"utility retention      {report.retention_pct:.1f}%")
    if report.sim_to_real_gap is not None:
        print(f"sim-to-real gap        {report.sim_to_real_gap:.4f} mAP50-95")
    if report.fid is not None:
        print(f"FID                    {report.fid:.2f}  (n={report.fid_sample_size})")
    if report.poisoning_verdict:
        print(f"poisoning audit        {report.poisoning_verdict}")

    weakest = report.weakest_class()
    if weakest:
        print(f"weakest class          {weakest[0]} at {weakest[1]:.4f} AP50-95")

    print(f"\n{report.headline()}")

    missing = report.as_dict()["experiments_missing"]
    if missing:
        print(f"\nstill to run: {', '.join(missing)}")

    output.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {output}")

    if not args.no_figures:
        figures = results / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plot_experiments(report, figures / "domain_transfer.png")
        plot_per_class(report, figures / "per_class_detection.png")


if __name__ == "__main__":
    main()
