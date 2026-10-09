"""Train YOLOv8 on the synthetic set or on KITTI with identical settings."""

from __future__ import annotations

import argparse

from utility_evaluation.config import SEED, add_root_argument, pick_device, workspace_from
from utility_evaluation.training import TrainSettings, train

DEFAULT_NAMES = {"synthetic": "synthetic_baseline", "kitti": "kitti_baseline"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 baseline")
    add_root_argument(parser)
    parser.add_argument("--dataset", choices=list(DEFAULT_NAMES), default="synthetic")
    parser.add_argument("--name", help="run name (default: synthetic_baseline or kitti_baseline)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="lower to 4 if memory runs out")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", help="'cpu', '0' for the first GPU (default: GPU if available)")
    parser.add_argument("--workers", type=int, default=8, help="data loader workers; try 0 if Windows hangs")
    args = parser.parse_args()

    ws = workspace_from(args)
    data = (ws.yolo_dataset if args.dataset == "synthetic" else ws.kitti_dataset) / "data.yaml"
    if not data.is_file():
        raise SystemExit(f"{data} not found. Build the dataset first.")

    name = args.name or DEFAULT_NAMES[args.dataset]
    settings = TrainSettings(model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                             patience=args.patience, seed=args.seed, workers=args.workers,
                             device=args.device)
    print(f"training {name} on {data} | device {pick_device(args.device)}")
    best = train(data, ws.runs, name, settings)
    print(f"best weights: {ws.display(best)}")

if __name__ == "__main__":
    main()
