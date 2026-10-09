"""Independent manifest verification."""

from __future__ import annotations

import hashlib
import json

from risk_compliance.manifest import verify_dataset


def build(tmp_path, files=None):
    files = files or {"rgb_f000001.png": b"rgb", "class_mask_f000001.png": b"mask"}
    enc = tmp_path / "enc" / "town01"
    dec = tmp_path / "dec" / "town01"
    enc.mkdir(parents=True)
    (dec / "class_masks").mkdir(parents=True)
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    (enc / "town01_manifest.json").write_text(json.dumps(manifest))
    for name, data in files.items():
        target = dec / "class_masks" / name if name.startswith("class_mask") else dec / name
        target.write_bytes(data)
    return tmp_path / "enc", tmp_path / "dec"


def test_matching_dataset_passes(tmp_path):
    enc, dec = build(tmp_path)
    [result] = verify_dataset(enc, dec)
    assert result.passed and result.verified == 2


def test_altered_file_fails(tmp_path):
    enc, dec = build(tmp_path)
    (dec / "town01" / "rgb_f000001.png").write_bytes(b"altered")
    [result] = verify_dataset(enc, dec)
    assert not result.passed and result.mismatched == ["rgb_f000001.png"]


def test_missing_file_fails(tmp_path):
    enc, dec = build(tmp_path)
    (dec / "town01" / "class_masks" / "class_mask_f000001.png").unlink()
    [result] = verify_dataset(enc, dec)
    assert not result.passed and result.missing == ["class_mask_f000001.png"]


def test_absent_manifest_fails(tmp_path):
    enc, dec = build(tmp_path)
    (enc / "town01" / "town01_manifest.json").unlink()
    [result] = verify_dataset(enc, dec)
    assert not result.manifest_found and not result.passed
