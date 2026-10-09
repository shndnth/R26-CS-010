"""Confirm every class mask holds only valid ids (0 car, 1 pedestrian, 2 cyclist, 3 background)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utility_evaluation.alignment import MASK_NAMES, load_mask
from utility_evaluation.config import TOWNS, add_root_argument, workspace_from

VALID_VALUES = frozenset(MASK_NAMES)


def inspect_one(path: Path) -> None:
    mask = load_mask(path)
    values, counts = np.unique(mask, return_counts=True)
    print("Unique pixel values:", values.tolist())
    print("Value counts:", {int(v): int(c) for v, c in zip(values, counts, strict=True)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate class mask pixel values")
    add_root_argument(parser)
    parser.add_argument("--file", type=Path, help="inspect one mask instead of all of them")
    args = parser.parse_args()

    if args.file:
        inspect_one(args.file)
        return

    data = workspace_from(args).decrypted
    total = 0
    frequency = {v: 0 for v in sorted(VALID_VALUES)}
    invalid: list[tuple[Path, set[int]]] = []

    for town in TOWNS:
        mask_dir = data / town / "class_masks"
        if not mask_dir.is_dir():
            print(f"{town}: no class_masks/ folder found, skipping.")
            continue
        files = sorted(mask_dir.glob("*.png"))
        print(f"{town}: checking {len(files)} class_mask files...")
        for path in files:
            total += 1
            present = {int(v) for v in np.unique(load_mask(path))}
            for v in present & VALID_VALUES:
                frequency[v] += 1
            if present - VALID_VALUES:
                invalid.append((path, present - VALID_VALUES))

    if not total:
        raise SystemExit(f"no class masks found under {data}")

    print("\n" + "=" * 55)
    print(f"DONE. Total class_mask files checked: {total}")
    print(f"\nFiles containing each class (out of {total}):")
    for value, count in frequency.items():
        print(f"  {value} ({MASK_NAMES[value]}): {count} files ({100 * count / total:.1f}%)")

    print(f"\nFiles with INVALID pixel values (outside 0-3): {len(invalid)}")
    for path, bad in invalid[:20]:
        print(f"  {path}: unexpected values {sorted(bad)}")
    if len(invalid) > 20:
        print(f"  ... and {len(invalid) - 20} more")

    if invalid:
        print("\nVERDICT: FAIL - some files contain unexpected pixel values, investigate.")
        raise SystemExit(1)
    print("\nVERDICT: PASS - every class mask contains only valid class ids (0-3).")


if __name__ == "__main__":
    main()
