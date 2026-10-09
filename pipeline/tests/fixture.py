"""A small but complete stand-in for the real inputs: encrypted towns, KITTI, calibration source."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from PIL import Image, ImageDraw

KEY = b"0123456789abcdef0123456789abcdef"
HMAC_KEY = "fixture-signing-key-not-a-secret"
COLOURS = {0: (220, 30, 30), 1: (30, 220, 30), 2: (30, 30, 220)}
KITTI_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
WIDTH, HEIGHT = 320, 240


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _encrypt(data: bytes) -> bytes:
    cipher = AES.new(KEY, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(data, AES.block_size))


def build_dataset(root: Path, towns=("town01", "town02", "town03", "town04", "town05"),
                  frames: int = 40, seed: int = 0) -> Path:
    rng = random.Random(seed)
    output = root / "output"
    for town in towns:
        town_dir = output / town
        (town_dir / "labels").mkdir(parents=True)
        (town_dir / "class_masks").mkdir()
        manifest, rows = {}, ["RGB_File,Weather_Category,Time_Of_Day,Event_Type,Town"]
        for i in range(frames):
            fid = f"{i:06d}"
            image = Image.new("RGB", (WIDTH, HEIGHT), (90, 90, 90))
            mask = np.full((HEIGHT, WIDTH), 3, np.uint8)
            draw, lines = ImageDraw.Draw(image), []
            for _ in range(rng.randint(1, 3)):
                cid = rng.choice((0, 0, 1, 2))
                bw, bh = rng.randint(20, 90), rng.randint(20, 90)
                x, y = rng.randint(0, WIDTH - bw), rng.randint(0, HEIGHT - bh)
                draw.rectangle([x, y, x + bw, y + bh], fill=COLOURS[cid])
                mask[y:y + bh, x:x + bw] = cid
                lines.append(f"{cid} {(x + bw / 2) / WIDTH:.6f} {(y + bh / 2) / HEIGHT:.6f} "
                             f"{bw / WIDTH:.6f} {bh / HEIGHT:.6f}")
            files = {
                f"rgb_f{fid}.png": _png(image),
                f"depth_f{fid}.png": _png(Image.new("L", (WIDTH, HEIGHT))),
                f"class_masks/class_mask_f{fid}.png": _png(Image.fromarray(mask)),
            }
            for relative, data in files.items():
                manifest[relative.split("/")[-1]] = hashlib.sha256(data).hexdigest()
                (town_dir / f"{relative}.enc").write_bytes(_encrypt(data))
            (town_dir / "labels" / f"bbox_f{fid}.txt").write_text("\n".join(lines))
            rows.append(f"rgb_f{fid}.png,{('Clear', 'Rain')[i % 2]},{('Day', 'Night')[i % 4 // 2]},"
                        f"Normal,{town}")
        (town_dir / f"{town}_manifest.json").write_text(json.dumps(manifest))
        (town_dir / f"{town}_metadata.csv").write_text("\n".join(rows) + "\n")
    return output


def build_kitti(root: Path, images: int = 40, seed: int = 1) -> Path:
    rng = random.Random(seed)
    base = root / "kitti_raw" / "training"
    (base / "image_2").mkdir(parents=True)
    (base / "label_2").mkdir()
    for i in range(images):
        image = Image.new("RGB", (620, 188), (120, 110, 100))
        draw, lines = ImageDraw.Draw(image), []
        for _ in range(rng.randint(1, 3)):
            cid = rng.choice((0, 0, 1, 2))
            bw, bh = rng.randint(20, 120), rng.randint(20, 100)
            x, y = rng.randint(0, 620 - bw), rng.randint(0, 188 - bh)
            draw.rectangle([x, y, x + bw, y + bh], fill=COLOURS[cid])
            lines.append(f"{KITTI_NAMES[cid]} 0.00 0 0 {x} {y} {x + bw} {y + bh} 1 1 1 1 1 1 0")
        image.save(base / "image_2" / f"{i:06d}.png")
        (base / "label_2" / f"{i:06d}.txt").write_text("\n".join(lines))
    return base


def build_calibration_source(root: Path) -> Path:
    path = root / "calibration_source" / "calibration_stats_source_TRUE_VALUES.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "vehicle_density_per_km": 2.4,
        "pedestrian_density_per_km": 0.9,
        "avg_vehicle_speed_kmh": 23.0,
        "avg_pedestrian_speed_kmh": 1.1,
        "intersection_count_per_km": 0.3,
        "weather_distribution": {"clear": 600, "cloudy": 150, "rain": 230, "fog": 20},
        "time_of_day_distribution": {"morning": 150, "afternoon": 130, "evening": 550, "night": 170},
    }))
    return path
