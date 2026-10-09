"""Show that the detector fires, using injected defects in a scratch folder."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from utility_evaluation.poisoning import Thresholds, audit

FRAMES = 1000
BATCH = 100


def build_town(labels: Path, seed: int) -> None:
    rng = random.Random(seed)
    labels.mkdir(parents=True)
    for i in range(FRAMES):
        lines = []
        for _ in range(rng.randint(1, 4)):
            cid = rng.choices((0, 1, 2), weights=(0.70, 0.22, 0.08))[0]
            lines.append(f"{cid} {rng.uniform(0.2, 0.8):.6f} {rng.uniform(0.2, 0.8):.6f} 0.050000 0.080000")
        (labels / f"bbox_f{i:06d}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove the poisoning detector catches injected defects")
    parser.add_argument("--workdir", type=Path, default=Path("test_poison"))
    args = parser.parse_args()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    labels = args.workdir / "town01" / "labels"
    build_town(labels, seed=7)

    clean = audit(args.workdir, ("town01",), Thresholds(batch_size=BATCH))
    print(f"Before injection: {len(clean.anomalies)} structural anomalies, "
          f"{clean.batches_rejected} batches rejected, verdict {clean.verdict}\n")

    (labels / "bbox_f000224.txt").write_text("1 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "bbox_f000225.txt").write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    poisoned_batch = 5
    for i in range(poisoned_batch * BATCH, (poisoned_batch + 1) * BATCH):
        path = labels / f"bbox_f{i:06d}.txt"
        relabelled = ["2" + line[1:] for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("\n".join(relabelled), encoding="utf-8")

    print("Injected:")
    print("  bbox_f000224.txt   x_center = 1.5, outside [0, 1]")
    print("  bbox_f000225.txt   class id 7, valid ids are 0, 1, 2")
    print(f"  batch {poisoned_batch}            every box relabelled as cyclist\n")

    result = audit(args.workdir, ("town01",), Thresholds(batch_size=BATCH))
    town = result.towns["town01"]
    for anomaly in result.anomalies:
        print(f"  {anomaly}")
    for batch, _ in town.batches:
        if batch.rejected:
            print(f"  batch {batch.index} REJECTED  p={batch.p_value:.2e}  V={batch.cramers_v:.2f}")

    caught = {
        "out-of-range box": "bbox_f000224.txt" in town.structurally_rejected,
        "unknown class id": "bbox_f000225.txt" in town.structurally_rejected,
        "relabelled batch": any(b.rejected and b.index == poisoned_batch for b, _ in town.batches),
    }
    print()
    for name, ok in caught.items():
        print(f"  {name:<18} {'CAUGHT' if ok else 'MISSED'}")

    if all(caught.values()):
        print("\nPROOF SUCCESSFUL: every injected defect was caught.")
    else:
        print("\nPROOF FAILED: at least one injected defect was missed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
