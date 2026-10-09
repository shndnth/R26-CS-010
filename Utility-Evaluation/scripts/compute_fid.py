"""Frechet Inception Distance between real (KITTI) and synthetic (CARLA) test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utility_evaluation.config import SEED, add_root_argument, workspace_from
from utility_evaluation.fid import center_square, list_images, sample_paths

BATCH = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID between real and synthetic images")
    add_root_argument(parser)
    parser.add_argument("--real-dir", type=Path, help="default: <root>/kitti_dataset/images/test")
    parser.add_argument("--synthetic-dir", type=Path, help="default: <root>/yolo_dataset/images/test")
    parser.add_argument("--limit", type=int, default=1000,
                        help="images per set; FID is unreliable below ~1000")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--crop", choices=["resize", "center-square"], default="resize")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms

    ws = workspace_from(args)
    real_dir = args.real_dir or ws.kitti_dataset / "images" / "test"
    synthetic_dir = args.synthetic_dir or ws.yolo_dataset / "images" / "test"
    real = sample_paths(list_images(real_dir), args.limit, args.seed)
    synthetic = sample_paths(list_images(synthetic_dir), args.limit, args.seed)
    if not real or not synthetic:
        raise SystemExit("no images found in one of the two folders")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Real images: {len(real)} | Synthetic images: {len(synthetic)} | preprocessing: {args.crop}")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    to_tensor = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    def add(paths: list[Path], real_set: bool) -> None:
        batch = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if args.crop == "center-square":
                    image = center_square(image)
                batch.append(to_tensor(image))
            if len(batch) == BATCH:
                fid.update(torch.stack(batch).to(device), real=real_set)
                batch = []
        if batch:
            fid.update(torch.stack(batch).to(device), real=real_set)

    add(real, True)
    add(synthetic, False)
    score = float(fid.compute().item())
    print(f"\nFID: {score:.2f}")

    towns: dict[str, int] = {}
    for path in synthetic:
        town = path.stem.split("_", 1)[0]
        towns[town] = towns.get(town, 0) + 1
    print(f"synthetic sample by town: {towns}")

    ws.results.mkdir(parents=True, exist_ok=True)
    out = ws.results / "fid.json"
    out.write_text(json.dumps({
        "fid": round(score, 4),
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "seed": args.seed,
        "preprocessing": args.crop,
        "synthetic_by_town": towns,
        "real_dir": ws.display(real_dir),
        "synthetic_dir": ws.display(synthetic_dir),
    }, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
