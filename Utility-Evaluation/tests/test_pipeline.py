"""End to end: encrypted fixture dataset through every non-training script, in order."""

from __future__ import annotations

import hashlib
import io
import json
import sys

import numpy as np
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from PIL import Image

from scripts import (
    analyze_misalignment,
    build_report,
    check_class_masks,
    convert_to_yolo,
    decrypt_all_images,
    detect_poisoning,
    explore_data,
    extract_ground_truth,
    kitti_to_yolo,
    verify_label_alignment,
)

KEY = b"0123456789abcdef0123456789abcdef"
TOWNS = ("town01", "town02")
FRAMES = 120


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def encrypt(data: bytes) -> bytes:
    cipher = AES.new(KEY, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(data, AES.block_size))


def build_encrypted_dataset(root):
    for town in TOWNS:
        town_dir = root / "output" / town
        (town_dir / "labels").mkdir(parents=True)
        (town_dir / "class_masks").mkdir()
        manifest = {}
        for i in range(FRAMES):
            fid = f"{i:06d}"
            mask = np.full((60, 80), 3, dtype=np.uint8)
            mask[20:40, 30:50] = 0
            files = {
                f"rgb_f{fid}.png": png_bytes(np.full((60, 80, 3), i % 255, dtype=np.uint8)),
                f"depth_f{fid}.png": png_bytes(np.zeros((60, 80), dtype=np.uint8)),
                f"class_masks/class_mask_f{fid}.png": png_bytes(mask),
            }
            for relative, data in files.items():
                manifest[relative.split("/")[-1]] = hashlib.sha256(data).hexdigest()
                (town_dir / f"{relative}.enc").write_bytes(encrypt(data))
            (town_dir / "labels" / f"bbox_f{fid}.txt").write_text("0 0.5 0.5 0.25 0.333333\n")
        (town_dir / f"{town}_manifest.json").write_text(json.dumps(manifest))
        (town_dir / f"{town}_metadata.csv").write_text("frame,town\n")


def run(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("R26_DATASET_AES_KEY", KEY.decode())
    build_encrypted_dataset(tmp_path)
    return tmp_path


def test_full_pipeline(workspace, monkeypatch, capsys):
    root = str(workspace)
    encrypted_before = sorted(p.name for p in (workspace / "output").rglob("*.enc"))

    run(decrypt_all_images, monkeypatch, "--root", root, "--strict")
    decrypted = workspace / "decrypted_output" / "town01"
    assert (decrypted / "rgb_f000000.png").is_file()
    assert (decrypted / "class_masks" / "class_mask_f000000.png").is_file()
    assert (decrypted / "labels" / "bbox_f000000.txt").is_file()
    assert sorted(p.name for p in (workspace / "output").rglob("*.enc")) == encrypted_before

    run(check_class_masks, monkeypatch, "--root", root)
    run(explore_data, monkeypatch, "--root", root)

    poisoned = decrypted / "labels" / "bbox_f000007.txt"
    poisoned.write_text("7 0.5 0.5 0.1 0.1\n")
    run(detect_poisoning, monkeypatch, "--root", root)
    audit = json.loads((workspace / "outputs" / "poisoning_audit.json").read_text())
    assert audit["verdict"] == "FILTERED"
    assert audit["frames_excluded"] == 1
    clean = workspace / "poisoning_detection_report" / "clean_labels" / "town01"
    assert not (clean / "bbox_f000007.txt").exists()
    assert (workspace / "poisoning_detection_report" / "town01_batch_report.csv").is_file()

    run(extract_ground_truth, monkeypatch, "--root", root)
    records = json.loads((workspace / "ground_truth_extraction" / "town01" / "ground_truth.json").read_text())
    assert len(records) == FRAMES - 1
    assert records[0]["rgb_path"] and records[0]["semantic_mask_path"]

    run(verify_label_alignment, monkeypatch, "--root", root)
    out = capsys.readouterr().out
    assert f"Aligned: {2 * FRAMES - 1} | Misaligned: 0" in out

    run(analyze_misalignment, monkeypatch, "--root", root)

    run(convert_to_yolo, monkeypatch, "--root", root)
    data_yaml = workspace / "yolo_dataset" / "data.yaml"
    assert data_yaml.is_file()
    images = [p.name for p in (workspace / "yolo_dataset" / "images").rglob("*.png")]
    assert len(images) == 2 * FRAMES - 1
    assert "town01_000007.png" not in images

    with pytest.raises(SystemExit):
        run(convert_to_yolo, monkeypatch, "--root", root)
    run(convert_to_yolo, monkeypatch, "--root", root, "--overwrite", "--split-mode", "block")


def test_tampered_file_fails_decryption(workspace, monkeypatch):
    target = workspace / "output" / "town02" / "rgb_f000003.png.enc"
    target.write_bytes(encrypt(b"not the original frame"))
    with pytest.raises(SystemExit):
        run(decrypt_all_images, monkeypatch, "--root", str(workspace))
    assert not (workspace / "decrypted_output" / "town02" / "rgb_f000003.png").exists()


def test_kitti_conversion(tmp_path, monkeypatch):
    raw = tmp_path / "kitti_raw" / "training"
    (raw / "image_2").mkdir(parents=True)
    (raw / "label_2").mkdir()
    for i in range(20):
        Image.new("RGB", (1242, 375)).save(raw / "image_2" / f"{i:06d}.png")
        (raw / "label_2" / f"{i:06d}.txt").write_text(
            "Car 0.00 0 -1.58 587.0 173.0 614.0 200.0 1.6 1.7 3.9 1.8 1.5 46.7 -1.6\n"
            "DontCare -1 -1 -10 100.0 100.0 120.0 120.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
    run(kitti_to_yolo, monkeypatch, "--root", str(tmp_path))
    labels = list((tmp_path / "kitti_dataset" / "labels").rglob("*.txt"))
    assert len(labels) == 20
    assert all(p.read_text().count("\n") == 0 and p.read_text().startswith("0 ") for p in labels)


def test_report_from_results(tmp_path, monkeypatch, capsys):
    results = tmp_path / "outputs"
    results.mkdir()
    per_class = {n: {"ap50": 0.5, "ap50_95": 0.3} for n in ("car", "pedestrian", "cyclist")}
    records = [
        {"label": label, "map50": m50, "map50_95": m, "per_class": per_class}
        for label, m50, m in (
            ("synthetic -> synthetic", 0.80, 0.55),
            ("synthetic -> real", 0.30, 0.15),
            ("real -> real", 0.70, 0.45),
            ("real -> synthetic", 0.20, 0.10),
        )
    ]
    (results / "evaluations.json").write_text(json.dumps(records))
    (results / "fid.json").write_text(json.dumps({"fid": 120.5, "n_real": 1000, "n_synthetic": 1000}))
    run(build_report, monkeypatch, "--root", str(tmp_path), "--no-figures")
    summary = json.loads((results / "utility_results.json").read_text())
    assert summary["utility_retention_pct"] == pytest.approx(33.3, abs=0.1)
    assert "33%" in capsys.readouterr().out


def test_shared_decrypted_folder(workspace, monkeypatch):
    shared = workspace / "elsewhere" / "decrypted"
    root = workspace / "ue_root"
    run(decrypt_all_images, monkeypatch, "--root", str(root),
        "--encrypted-dir", str(workspace / "output"), "--decrypted-dir", str(shared))
    assert (shared / "town01" / "rgb_f000000.png").is_file()
    run(detect_poisoning, monkeypatch, "--root", str(root), "--decrypted-dir", str(shared))
    assert (root / "poisoning_detection_report" / "clean_labels" / "town01").is_dir()
    assert not (root / "decrypted_output").exists()
