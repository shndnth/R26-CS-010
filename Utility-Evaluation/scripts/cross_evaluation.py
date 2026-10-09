"""The four train/test domain experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utility_evaluation.config import add_root_argument, pick_device, workspace_from
from utility_evaluation.detection import evaluate
from utility_evaluation.reporting import REAL_TO_REAL, REAL_TO_SYNTH, SYNTH_TO_REAL, SYNTH_TO_SYNTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four cross-domain evaluations")
    add_root_argument(parser)
    parser.add_argument("--synthetic-name", default="synthetic_baseline")
    parser.add_argument("--real-name", default="kitti_baseline")
    parser.add_argument("--synthetic-weights", type=Path)
    parser.add_argument("--real-weights", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    ws = workspace_from(args)

    def locate(explicit: Path | None, name: str) -> Path | None:
        if explicit is not None:
            return explicit if explicit.is_file() else None
        try:
            return ws.weights(name)
        except FileNotFoundError:
            return None

    synthetic_weights = locate(args.synthetic_weights, args.synthetic_name)
    real_weights = locate(args.real_weights, args.real_name)
    synthetic_data = ws.yolo_dataset / "data.yaml"
    real_data = ws.kitti_dataset / "data.yaml"
    device = pick_device(args.device)

    experiments = [
        (SYNTH_TO_SYNTH, synthetic_weights, synthetic_data),
        (SYNTH_TO_REAL, synthetic_weights, real_data),
        (REAL_TO_REAL, real_weights, real_data),
        (REAL_TO_SYNTH, real_weights, synthetic_data),
    ]

    runnable = [(name, w, d) for name, w, d in experiments if w is not None and d.is_file()]
    for label, weights, data in experiments:
        if (label, weights, data) not in runnable:
            missing = "model" if weights is None else str(ws.display(data))
            print(f"skipping {label}: {missing} not available")
    if not runnable:
        raise SystemExit("no experiment can run: train a model and build its dataset first")

    records = []
    for label, weights, data in runnable:
        print(f"\n{'=' * 60}\nRunning: {label}\n{'=' * 60}")
        record = evaluate(weights, data, label, device, ws.runs / "val", label.replace(" -> ", "_to_"),
                          weights=ws.display(weights), data=ws.display(data), split="test")
        records.append(record)
        print(f"{label:<24} mAP50={record['map50']:.4f} mAP50-95={record['map50_95']:.4f}")

    def ap(record: dict, name: str) -> str:
        values = record["per_class"][name]
        return f"{values['ap50']:>8.4f}" if values else f"{'n/a':>8}"

    print(f"\n{'=' * 60}\nSUMMARY TABLE (per-class columns are AP50)\n{'=' * 60}")
    print(f"{'Experiment':<24} {'mAP50':>8} {'mAP50-95':>10} {'car':>8} {'ped':>8} {'cyc':>8}")
    for r in records:
        print(f"{r['label']:<24} {r['map50']:>8.4f} {r['map50_95']:>10.4f} "
              f"{ap(r, 'car')} {ap(r, 'pedestrian')} {ap(r, 'cyclist')}")

    ws.results.mkdir(parents=True, exist_ok=True)
    out = ws.results / "evaluations.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
