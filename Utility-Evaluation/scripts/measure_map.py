"""Final mAP of one trained model on a held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utility_evaluation.config import add_root_argument, workspace_from
from utility_evaluation.detection import evaluate


def print_record(record: dict) -> None:
    print(f"mAP50 {record['map50']:.4f}")
    print(f"mAP50-95 {record['map50_95']:.4f}")
    for name, values in record["per_class"].items():
        if values is None:
            print(f"{name:<12} not present in this split")
        else:
            print(f"{name:<12} AP50={values['ap50']:.4f} AP50-95={values['ap50_95']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure test-split mAP for one model")
    add_root_argument(parser)
    parser.add_argument("--name", default="synthetic_baseline", help="training run name")
    parser.add_argument("--weights", type=Path, help="explicit best.pt, overrides --name")
    parser.add_argument("--dataset", choices=["synthetic", "kitti"], default="synthetic")
    parser.add_argument("--device")
    args = parser.parse_args()

    ws = workspace_from(args)
    weights = args.weights or ws.weights(args.name)
    data = (ws.yolo_dataset if args.dataset == "synthetic" else ws.kitti_dataset) / "data.yaml"

    record = evaluate(weights, data, f"{args.name} on {args.dataset}", args.device,
                      ws.runs / "val", f"{args.name}_on_{args.dataset}",
                      weights=ws.display(weights), data=ws.display(data), split="test")
    print_record(record)

    ws.results.mkdir(parents=True, exist_ok=True)
    out = ws.results / f"map_{args.name}_on_{args.dataset}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
