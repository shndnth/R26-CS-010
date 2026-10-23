"""Security audit of the Dataset Generation encryption engine.

Six tests, all on the real SecurityEngine:

  1. round_trip           encrypt then decrypt returns the exact original bytes
  2. nonce_uniqueness     the same file encrypted 30 times gives 30 different IVs or nonces
  3. byte_distribution    ciphertext bytes are uniform (chi-square, 255 degrees of freedom)
  4. ciphertext_change    a 1-bit change in the file changes about half the ciphertext bits
  5. tamper_detection     a changed ciphertext byte is rejected (GCM) or noted as a CBC limitation
  6. real_folder          a captured folder holds no readable PNG; optionally every file
                          decrypts in memory and matches its SHA-256 manifest entry

Tests 1 to 5 use synthetic bytes and a random key that exists only in memory for
this run. Test 6 reads the dataset key from the project .env, as the capture does.
No plaintext is written to disk.

Run from the Dataset-Generation folder:

    python scripts/security_audit.py --mode gcm
    python scripts/security_audit.py --mode gcm --data-dir output/town01 --verify

Results go to outputs/security_audit/ (never committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security_engine import CIPHER_MODES, SecurityEngine, dataset_key  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Critical value of chi-square with 255 degrees of freedom at alpha 0.05.
CHI_SQUARE_CRITICAL_255 = 293.25
CHANGE_RANGE_PCT = (45.0, 55.0)


def synthetic_bytes(size, seed):
    """Repeatable pseudo-random test bytes (works on Python 3.8)."""
    return random.Random(seed).getrandbits(size * 8).to_bytes(size, "little")


def _bits_different(a, b):
    n = min(len(a), len(b))
    x = np.bitwise_xor(np.frombuffer(a[:n], dtype=np.uint8), np.frombuffer(b[:n], dtype=np.uint8))
    return int(np.unpackbits(x).sum()), n * 8


def test_round_trip(engine, size=4096):
    data = synthetic_bytes(size, 1)
    t0 = time.perf_counter()
    blob = engine.encrypt_bytes(data)
    t1 = time.perf_counter()
    recovered = engine.decrypt_bytes(blob)
    t2 = time.perf_counter()
    return {
        "passed": recovered == data,
        "bytes": size,
        "overhead_bytes": len(blob) - size,
        "encrypt_ms": round((t1 - t0) * 1000, 3),
        "decrypt_ms": round((t2 - t1) * 1000, 3),
    }


def test_nonce_uniqueness(engine, runs=30):
    data = synthetic_bytes(4096, 1)
    blobs = [engine.encrypt_bytes(data) for _ in range(runs)]
    head = 16
    unique_nonces = len({b[:head] for b in blobs})
    unique_ciphertexts = len(set(blobs))
    return {
        "passed": unique_nonces == runs and unique_ciphertexts == runs,
        "runs": runs,
        "unique_nonces": unique_nonces,
        "unique_ciphertexts": unique_ciphertexts,
    }


def test_byte_distribution(engine, size=200_000):
    blob = engine.encrypt_bytes(synthetic_bytes(size, 7))
    body = np.frombuffer(blob[engine.header_bytes():], dtype=np.uint8)
    counts = np.bincount(body, minlength=256)
    expected = len(body) / 256
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    return {
        "passed": chi_square < CHI_SQUARE_CRITICAL_255,
        "sample_bytes": int(len(body)),
        "expected_count": round(expected, 2),
        "std_dev": round(float(np.std(counts)), 2),
        "chi_square": round(chi_square, 2),
        "critical_value_alpha_0_05": CHI_SQUARE_CRITICAL_255,
        "counts": counts.tolist(),
    }


def test_ciphertext_change(engine, size=4096):
    """1-bit change in the file, each encryption with its own fresh IV or nonce.

    About half the bits differ because every encryption uses a fresh IV or nonce.
    In GCM, a stream mode, reusing a nonce would change only the flipped bit, which
    is why test 2 matters.
    """
    a = bytearray(synthetic_bytes(size, 3))
    b = bytearray(a)
    b[0] ^= 0x01
    head = engine.header_bytes()
    ca = engine.encrypt_bytes(bytes(a))[head:]
    cb = engine.encrypt_bytes(bytes(b))[head:]
    diff, total = _bits_different(ca, cb)
    pct = round(100 * diff / total, 2)
    return {
        "passed": CHANGE_RANGE_PCT[0] <= pct <= CHANGE_RANGE_PCT[1],
        "bits_compared": total,
        "bits_different": diff,
        "percent_different": pct,
        "expected": "about 50%",
    }


def test_tamper_detection(engine, size=4096):
    blob = bytearray(engine.encrypt_bytes(synthetic_bytes(size, 9)))
    blob[engine.header_bytes() + 8] ^= 0xFF
    try:
        engine.decrypt_bytes(bytes(blob))
        rejected = False
    except ValueError:
        rejected = True
    result = {"rejected": rejected}
    if engine.mode == "gcm":
        result["passed"] = rejected
        result["note"] = "GCM checks an authentication tag, so a changed byte is rejected."
    else:
        result["passed"] = None
        result["note"] = (
            "CBC has no authentication tag; changed files are caught by the SHA-256 manifest "
            "after decryption instead."
        )
    return result


def _manifest(folder):
    for name in os.listdir(folder):
        if name.endswith("_manifest.json"):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                return json.load(handle)
    return None


def test_real_folder(folder, engine=None):
    """Scan captured .png.enc files. With an engine, also decrypt each in memory."""
    files = []
    for root, _dirs, names in os.walk(folder):
        files.extend(os.path.join(root, n) for n in names if n.lower().endswith(".png.enc"))
    readable = 0
    for path in files:
        with open(path, "rb") as handle:
            if handle.read(8) == PNG_MAGIC:
                readable += 1
    result = {
        "folder": os.path.basename(os.path.normpath(folder)),
        "encrypted_files": len(files),
        "readable_without_key": readable,
        "passed": bool(files) and readable == 0,
    }
    if engine is None:
        return result

    manifest = _manifest(folder) or {}
    verified = failed = missing = 0
    for path in files:
        name = os.path.basename(path)[:-4]
        with open(path, "rb") as handle:
            try:
                digest = hashlib.sha256(engine.decrypt_bytes(handle.read())).hexdigest()
            except ValueError:
                failed += 1
                continue
        if name not in manifest:
            missing += 1
        elif manifest[name] == digest:
            verified += 1
        else:
            failed += 1
    result.update({
        "cipher": engine.label,
        "manifest_entries": len(manifest),
        "decrypted_and_matched": verified,
        "failed": failed,
        "not_in_manifest": missing,
    })
    result["passed"] = result["passed"] and failed == 0 and missing == 0 and verified == len(files)
    return result


def run_audit(mode="gcm", data_dir=None, verify=False, key=None):
    engine = SecurityEngine(key=key or os.urandom(32), mode=mode)
    tests = {
        "round_trip": test_round_trip(engine),
        "nonce_uniqueness": test_nonce_uniqueness(engine),
        "byte_distribution": test_byte_distribution(engine),
        "ciphertext_change": test_ciphertext_change(engine),
        "tamper_detection": test_tamper_detection(engine),
    }
    if data_dir:
        real_engine = SecurityEngine(key=dataset_key(), mode=mode) if verify else None
        tests["real_folder"] = test_real_folder(data_dir, real_engine)
    decided = [t["passed"] for t in tests.values() if t["passed"] is not None]
    return {
        "component": "Dataset Generation (IT22541284)",
        "cipher": engine.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "passed": sum(decided),
        "decided": len(decided),
        "verdict": "PASS" if all(decided) else "FAIL",
    }


def _bar(ax, labels, values, colors, title, ylabel, ymax=None):
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)


def save_charts(report, folder):
    """Write one PNG per test and a dashboard. Needs matplotlib (pip install -e ".[audit]")."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = report["tests"]
    blue, green, red, grey = "#2563eb", "#16a34a", "#dc2626", "#9ca3af"
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = os.path.join(folder, name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    n = t["nonce_uniqueness"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _bar(ax, ["Unique IVs / nonces", "Unique ciphertexts"], [n["unique_nonces"], n["unique_ciphertexts"]],
         [blue, green], "Nonce uniqueness", f"Count of {n['runs']} encryptions", n["runs"] + 3)
    save(fig, "nonce_uniqueness.png")

    b = t["byte_distribution"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(256), b["counts"], width=1, color=blue, alpha=0.8)
    ax.axhline(b["expected_count"], color=red, linestyle="--", label="Expected (uniform)")
    ax.set_title(f"Ciphertext byte distribution (chi-square {b['chi_square']} < {CHI_SQUARE_CRITICAL_255})",
                 fontweight="bold")
    ax.set_xlabel("Byte value")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, "byte_distribution.png")

    c = t["ciphertext_change"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _bar(ax, ["Observed", "Expected"], [c["percent_different"], 50], [blue, green],
         "Ciphertext change (1-bit file change)", "% of bits different", 100)
    save(fig, "ciphertext_change.png")

    if "real_folder" in t:
        r = t["real_folder"]
        labels = ["Encrypted files", "Readable without key"]
        values = [r["encrypted_files"], r["readable_without_key"]]
        if "decrypted_and_matched" in r:
            labels.append("Matched manifest")
            values.append(r["decrypted_and_matched"])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        _bar(ax, labels, values, [blue, red, green][: len(values)], f"Real folder: {r['folder']}", "Files")
        save(fig, "real_folder.png")

    names = list(t)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(names) + 1))
    colors = [grey if t[k]["passed"] is None else (green if t[k]["passed"] else red) for k in names]
    ax.barh([k.replace("_", " ") for k in names], [1] * len(names), color=colors)
    for i, k in enumerate(names):
        text = "NOT APPLICABLE (CBC)" if t[k]["passed"] is None else ("PASS" if t[k]["passed"] else "FAIL")
        ax.text(0.5, i, text, ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title(f"{report['cipher']} security audit: {report['passed']} / {report['decided']} passed",
                 fontweight="bold")
    save(fig, "dashboard.png")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Security audit of the dataset encryption engine")
    parser.add_argument("--mode", choices=CIPHER_MODES, default="gcm", help="cipher to audit (default: gcm)")
    parser.add_argument("--data-dir", help="captured town folder to scan, for example output/town01")
    parser.add_argument("--verify", action="store_true",
                        help="also decrypt every file in memory with the project key and check the manifest")
    parser.add_argument("--out", default=os.path.join("outputs", "security_audit"), help="output folder")
    parser.add_argument("--no-charts", action="store_true", help="write the JSON only")
    args = parser.parse_args(argv)

    if args.verify and not args.data_dir:
        parser.error("--verify needs --data-dir")
    if args.data_dir and not os.path.isdir(args.data_dir):
        parser.error(f"folder not found: {args.data_dir}")

    report = run_audit(args.mode, args.data_dir, args.verify)
    os.makedirs(args.out, exist_ok=True)
    stored = json.loads(json.dumps(report))
    stored["tests"]["byte_distribution"].pop("counts")
    out_json = os.path.join(args.out, "security_audit.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2)

    for name, result in report["tests"].items():
        status = "N/A " if result["passed"] is None else ("PASS" if result["passed"] else "FAIL")
        print(f"  [{status}] {name}")
    summary = f"{report['passed']} / {report['decided']} passed, verdict {report['verdict']}"
    print(f"\n{report['cipher']}: {summary}")
    print(f"Results: {out_json}")
    if not args.no_charts:
        try:
            for path in save_charts(report, args.out):
                print(f"Chart:   {path}")
        except ImportError:
            print('Charts skipped: install matplotlib with pip install -e ".[audit]"')
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
