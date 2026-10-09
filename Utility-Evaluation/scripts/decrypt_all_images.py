"""Step 2: decrypt the CARLA dataset and verify every file against its manifest."""

from __future__ import annotations

import argparse

from r26_common.env import dataset_aes_key

from utility_evaluation.config import TOWNS, add_root_argument, workspace_from
from utility_evaluation.decryption import decrypt_town


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: decrypt and verify the CARLA dataset")
    add_root_argument(parser)
    parser.add_argument("--towns", nargs="+", default=list(TOWNS))
    parser.add_argument("--strict", action="store_true",
                        help="fail if any file could not be checked against the manifest")
    args = parser.parse_args()

    ws = workspace_from(args)
    print("Step 2 - Receive and Decrypt the CARLA Dataset")
    print(f"Source:      {ws.encrypted}")
    print(f"Destination: {ws.decrypted}\n")
    if not ws.encrypted.is_dir():
        raise SystemExit(f"'{ws.encrypted}' not found. Extract the dataset's output/ folder there.")

    key = dataset_aes_key()
    totals = {"verified": 0, "unverified": 0, "failed": 0}
    for town in args.towns:
        town_dir = ws.encrypted / town
        if not town_dir.is_dir():
            print(f"{town} not found under output/, skipping.")
            continue
        for k, v in decrypt_town(town_dir, ws.decrypted / town, key).items():
            totals[k] += v

    print("=" * 55)
    print(f"   DONE. Decrypted and hash-verified: {totals['verified']}")
    print(f"   Decrypted, no manifest entry:      {totals['unverified']}")
    print(f"   Failed:                            {totals['failed']}")
    print(f"   Output location: {ws.decrypted}")

    if totals["failed"] or (args.strict and totals["unverified"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
