"""Constants and the working folder layout shared by every script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TOWNS: tuple[str, ...] = ("town01", "town02", "town03", "town04", "town05")
CLASS_NAMES: dict[int, str] = {0: "car", 1: "pedestrian", 2: "cyclist"}
BACKGROUND_CLASS = 3
NATIVE_WIDTH = 800
NATIVE_HEIGHT = 600
SEED = 42
SPLITS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True, slots=True)
class Workspace:
    """The working folder layout used since the first runs."""

    root: Path
    encrypted_dir: Path | None = None
    decrypted_dir: Path | None = None
    kitti_dir: Path | None = None

    @property
    def encrypted(self) -> Path:
        return self.encrypted_dir or self.root / "output"

    @property
    def decrypted(self) -> Path:
        return self.decrypted_dir or self.root / "decrypted_output"

    @property
    def poisoning_report(self) -> Path:
        return self.root / "poisoning_detection_report"

    @property
    def clean_labels(self) -> Path:
        return self.poisoning_report / "clean_labels"

    @property
    def ground_truth(self) -> Path:
        return self.root / "ground_truth_extraction"

    @property
    def alignment_report(self) -> Path:
        return self.root / "label_alignment_report"

    @property
    def yolo_dataset(self) -> Path:
        return self.root / "yolo_dataset"

    @property
    def kitti_raw(self) -> Path:
        return self.kitti_dir or self.root / "kitti_raw" / "training"

    @property
    def kitti_dataset(self) -> Path:
        return self.root / "kitti_dataset"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "outputs"

    def display(self, path: Path) -> str:
        """A path as it should appear in result files: relative to the root."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.name

    def weights(self, run_name: str, which: str = "best") -> Path:
        """Locate a run's weights, including the nested layout older runs used."""
        candidates = (
            self.runs / run_name / "weights" / f"{which}.pt",
            self.runs / "detect" / "runs" / run_name / "weights" / f"{which}.pt",
            self.runs / "detect" / run_name / "weights" / f"{which}.pt",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"no {which}.pt for run '{run_name}'. Searched:\n  {searched}")


def pick_device(requested: str | None = None) -> str:
    """'0' for the first GPU when one is visible, otherwise 'cpu'."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def add_root_argument(parser) -> None:
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="working folder holding output/, decrypted_output/ and the rest (default: current folder)",
    )
    parser.add_argument("--encrypted-dir", type=Path, help="encrypted dataset (default: <root>/output)")
    parser.add_argument("--decrypted-dir", type=Path,
                        help="decrypted dataset (default: <root>/decrypted_output)")
    parser.add_argument("--kitti-dir", type=Path,
                        help="KITTI image_2 and label_2 folder (default: <root>/kitti_raw/training)")


def workspace_from(args) -> Workspace:
    return Workspace(
        root=args.root,
        encrypted_dir=getattr(args, "encrypted_dir", None),
        decrypted_dir=getattr(args, "decrypted_dir", None),
        kitti_dir=getattr(args, "kitti_dir", None),
    )
