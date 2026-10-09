"""Size-stratified AP50: small, medium and large objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utility_evaluation.config import add_root_argument, pick_device, workspace_from
from utility_evaluation.detection import SIZE_BUCKETS, SizeStratifiedAP, load_gt_boxes
from utility_evaluation.fid import list_images

CONF_THRESHOLD = 0.001


def main() -> None:
    parser = argparse.ArgumentParser(description="Size-stratified AP50")
    add_root_argument(parser)
    parser.add_argument("--name", default="synthetic_baseline")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", choices=["synthetic", "kitti"], default="synthetic")
    parser.add_argument("--device")
    args = parser.parse_args()

    from ultralytics import YOLO

    ws = workspace_from(args)
    weights = args.weights or ws.weights(args.name)
    dataset = ws.yolo_dataset if args.dataset == "synthetic" else ws.kitti_dataset
    images = list_images(dataset / "images" / "test")
    labels_dir = dataset / "labels" / "test"
    device = pick_device(args.device)

    print("Size-Stratified AP50 (small / medium / large)")
    print(f"model {weights} | {len(images)} test images | device {device}")

    model = YOLO(str(weights))
    scorer = SizeStratifiedAP(iou_threshold=0.5)
    for index, image_path in enumerate(images, start=1):
        result = model.predict(source=str(image_path), conf=CONF_THRESHOLD, verbose=False, device=device)[0]
        height, width = result.orig_shape
        gt = load_gt_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
        predictions = []
        if result.boxes is not None:
            for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                       result.boxes.xyxy.tolist(), strict=True):
                predictions.append((int(cls), float(conf), *xyxy))
        scorer.add_image(gt, predictions)
        if index % 300 == 0:
            print(f"  ...{index}/{len(images)} images processed")

    results = scorer.results()
    print("\n" + "=" * 60)
    print("RESULTS - Size-Stratified AP50 (single IoU=0.5 threshold)")
    print("=" * 60)
    for bucket in SIZE_BUCKETS:
        print(f"\n--- {bucket.upper()} objects ---")
        for name, values in results[bucket]["per_class"].items():
            if values["ap50"] is None:
                print(f"  {name:<12} AP50=N/A     (no ground truth in this bucket)")
            else:
                print(f"  {name:<12} AP50={values['ap50']:.4f}  "
                      f"(GT instances: {values['gt_instances']}, detections: {values['detections']})")
        if results[bucket]["map50"] is not None:
            print(f"  {'mAP50':<12} {results[bucket]['map50']:.4f}")

    ws.results.mkdir(parents=True, exist_ok=True)
    out = ws.results / f"size_breakdown_{args.name}_on_{args.dataset}.json"
    out.write_text(json.dumps({"weights": ws.display(weights), "dataset": args.dataset,
                               "iou_threshold": 0.5, "buckets": results}, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
