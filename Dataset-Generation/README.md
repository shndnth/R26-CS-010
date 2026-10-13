# Dataset Generation

Component of R26-CS-010: Privacy-Enhanced Synthetic Data Generation for
Autonomous Vehicle Training.

**Owner:** IT22541284, Yubitha S.

Generates synthetic multi-sensor driving data in CARLA, configured from the
privatized calibration statistics produced by Privacy Integration, and secures
the output with AES-256 encryption, SHA-256 integrity hashing and role-based
access control.

**Consumes:** `calibration_stats.json` from Privacy Integration
**Produces:** the encrypted dataset, manifests, YOLO labels and configuration metadata

Five towns, about 2000 frame sets each. Every set yields RGB, depth, semantic
segmentation, a class mask and a YOLO bounding-box label file.

## Setup

```bash
conda create -n carla python=3.11      # or the version your CARLA client requires
conda activate carla
pip install -e ".[dev]"
pip install <carla>/PythonAPI/carla/dist/carla-<version>-<platform>.whl
```

The CARLA client comes from the CARLA installation and must match the server
version. It also decides the Python version for this component, which is why
this folder does not depend on `shared/`.

The dataset key comes from the project `.env` at the repository root (copy
`.env.example` there), from `R26_DATASET_AES_KEY` in the environment, or from a
file named by `R26_ENV_FILE`.

## Running

Start the CARLA server, then capture one town at a time:

```bash
python scripts/capture_town.py --town town01
python scripts/capture_town.py --town town04 --sets 20     # short test run
```

Output goes to `output/<town>/` under the current folder. Each run captures
`--sets` frame sets (default 2000), one every 16 simulator frames (0.8 s at
20 FPS) so consecutive sets are not near-duplicates. Both values are recorded in
`<town>_config_metadata.json`.

## Per-town settings

One script serves every town; the differences are in `TOWN_PROFILES`.

| | town01 | town02 | town03 | town04 | town05 |
|---|---|---|---|---|---|
| `ego_vehicle` | tesla.model3 | mini.cooper_s | tesla.model3 | tesla.model3 | tesla.model3 |
| `brake_threshold` | 0.60 | 0.80 | 0.80 | 0.80 | 0.80 |
| `steer_threshold` | 0.15 | 0.15 | 0.15 | 0.12 | 0.15 |
| `spectator_z` | 40 | 40 | 45 | 50 | 50 |
| `stuck_distance` | 1.0 | 0.5 | 0.5 | 0.2 | 0.5 |
| `stuck_check_ticks` | 50 | 50 | 50 | 100 | 50 |
| `stuck_strikes_limit` | 4 | 4 | 5 | 2 | 6 |
| `high_speed_threshold` | none | none | none | 80 | 90 |
| `high_speed_label` | | | | High_Speed_Cruise | High_Speed_Flow |

`capture_town.py` replaces five per-town scripts that differed only in these
values. Its logic was checked line by line against all five, and every setting
above matches the original. The five originals are not part of this repository.

**Pending confirmation:** the consolidated script has not yet produced a live
capture. Run `--sets 20` for one town and compare the output layout and metadata
with an earlier capture before the next full run.

## Output layout, per town

```
town0X/
├── rgb_f000224.png.enc              AES-256-CBC encrypted
├── depth_f000224.png.enc
├── semantic_f000224.png.enc
├── class_masks/
│   └── class_mask_f000224.png.enc
├── labels/
│   └── bbox_f000224.txt             not encrypted
├── town0X_manifest.json             SHA-256 of every file, before encryption
├── town0X_metadata.csv              per-frame weather, time of day, event
└── town0X_config_metadata.json      scene configuration and calibration profile used
```

Encryption format is `[16-byte IV][AES-256-CBC ciphertext, PKCS7 padded]`.

## Privacy requirement

CARLA is configured from the **privatized** `calibration_stats.json`, never from
raw source statistics: the simulator configuration is the channel through which
real-world information would otherwise leak. The same values apply to every
town and sensor type, and `town0X_config_metadata.json` records the profile used
so the claim can be audited.

## Security modules

`security_engine.py` encrypts each frame and removes the plaintext only after
the encrypted copy is written. A missing or wrong-length key is an error rather
than a silent random key, and decryption keeps the encrypted file unless asked
to remove it. `rbac_gateway.py` restricts actions by role.

## Tests

```bash
pytest
```

The tests replace the CARLA client with a stub, so they run without a CARLA
installation. File formats are defined in `shared/INTERFACES.md`. Never commit
`.env`, `output/` or any generated imagery.
