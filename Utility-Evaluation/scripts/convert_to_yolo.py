"""Build yolo_dataset/ (Ultralytics layout) from the decrypted CARLA frames."""

from __future__ import annotations

import argparse

from utility_evaluation.config import (
    CLASS_NAMES,
    SEED,
    SPLITS,
    TOWNS,
    add_root_argument,
    workspace_from,
)
from utility_evaluation.conversion import build_yolo_dataset, collect_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the CARLA dataset to the YOLO layout")
    add_root_argument(parser)
    parser.add_argument("--labels", choices=["clean", "raw"], default="clean")
    parser.add_argument("--split-mode", choices=["random", "block"], default="random",
                        help="block keeps runs of consecutive frames in the same split")
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild yolo_dataset/")
    args = parser.parse_args()

    ws = workspace_from(args)
    clean = ws.clean_labels if args.labels == "clean" else None
    if clean is not None and not clean.is_dir():
        raise SystemExit(f"'{clean}' not found. Run Step 3 (detect_poisoning.py) first, "
                         "or pass --labels raw to skip the filter.")

    pairs = collect_pairs(ws.decrypted, TOWNS, clean)
    if not pairs:
        raise SystemExit(f"no image and label pairs found under {ws.decrypted}")

    try:
        counts = build_yolo_dataset(
            pairs, ws.yolo_dataset, CLASS_NAMES, SPLITS["train"], SPLITS["val"], args.seed,
            mode=args.split_mode, block_size=args.block_size, overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    print(f"labels: {args.labels} | split: {args.split_mode} | seed: {args.seed}")
    print(f"total {counts.total} | train={counts.train} val={counts.val} test={counts.test}")
    print(f"written to {ws.yolo_dataset}")


if __name__ == "__main__":
    main()
