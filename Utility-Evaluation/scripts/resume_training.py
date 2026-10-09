"""Resume an interrupted training run from its last.pt checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from utility_evaluation.config import add_root_argument, workspace_from


def is_resumable(checkpoint: Path) -> bool:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return state.get("optimizer") is not None and state.get("epoch", -1) >= 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume an interrupted training run")
    add_root_argument(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="run name, e.g. kitti_baseline")
    group.add_argument("--checkpoint", type=Path, help="explicit path to last.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    checkpoint = args.checkpoint or workspace_from(args).weights(args.name, "last")
    if not is_resumable(checkpoint):
        raise SystemExit(f"{checkpoint} is from a finished run; there is nothing to resume.")
    print(f"resuming from {checkpoint}")
    YOLO(str(checkpoint)).train(resume=True)


if __name__ == "__main__":
    main()
