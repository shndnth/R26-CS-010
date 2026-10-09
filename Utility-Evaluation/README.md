# Utility Evaluation

Component of R26-CS-010: Privacy-Enhanced Synthetic Data Generation for
Autonomous Vehicle Training.

**Owner:** IT22110220, Apeksha M.K.S.R

This component answers one question: is the privacy-preserving synthetic data
still good enough to train a working perception model? It measures that with
**mAP** (how well a detector trained on the data finds objects) and **FID** (how
close the synthetic images are to real ones), and it guards the training data
with **data poisoning detection**, the component's security contribution.

These are the scripts used for the component's runs, reorganized so the shared
logic is tested and nothing depends on one machine's folder layout.

**In the pipeline**, `pipeline/` runs these steps automatically. It decrypts the
dataset once into a shared folder (`--decrypted-dir`), and Privacy Integration
trains on this component's `clean_labels/`, so the poisoning gate protects both
components' models.

---

## Setup

```bash
conda create -n utility python=3.11
conda activate utility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -e ../shared -e ".[dev]"
pytest -q
```

The dataset key comes from the project `.env` at the repository root (copy
`.env.example` there). A `.env` in the working folder, or any folder above it,
also works. **`.env` is never committed.**

---

## Working folder

Every command works on one folder, the current directory by default, or the one
given with `--root`. The layout is the one used since the first runs, so an
existing working folder needs no changes:

```
<working folder>/
├── output/townXX/                 encrypted dataset from IT22541284
├── decrypted_output/townXX/        Step 2
├── poisoning_detection_report/    Step 3, including clean_labels/
├── ground_truth_extraction/       Step 4
├── label_alignment_report/        Step 5
├── yolo_dataset/                  CARLA in YOLO layout
├── kitti_raw/training/            KITTI download (image_2, label_2)
├── kitti_dataset/                 KITTI in YOLO layout
├── runs/                          trained models
└── outputs/                       JSON results and figures
```

---

## Pipeline

| Step | Command | Script |
|---|---|---|
| 2 Decrypt and verify | `ue-decrypt` | `decrypt_all_images.py` |
| Inspect the data | `ue-explore`, `ue-check-masks` | `explore_data.py`, `check_class_masks.py` |
| 3 Poisoning detection | `ue-detect` | `detect_poisoning.py` |
| 3 Evidence it works | `ue-prove-detection` | `prove_poison_detection.py` |
| 4 Ground truth | `ue-extract-gt` | `extract_ground_truth.py` |
| 5 Label alignment | `ue-verify-alignment`, `ue-analyze-misalignment` | `verify_label_alignment.py`, `analyze_misalignment.py` |
| Build datasets | `ue-build`, `ue-convert-kitti` | `convert_to_yolo.py`, `kitti_to_yolo.py` |
| 1, 6 Train | `ue-train`, `ue-train --dataset kitti` | `train_yolo.py` |
| Resume after a crash | `ue-resume --name <run>` | `resume_training.py` |
| 7 mAP | `ue-measure-map`, `ue-cross-eval`, `ue-size-map` | `measure_map.py`, `cross_evaluation.py`, `size_breakdown_map.py` |
| 7 FID | `ue-fid` | `compute_fid.py` |
| 8, 9 Findings | `ue-report` | `build_report.py` |
| 6 Per-epsilon study | `ue-epsilon-study` | `epsilon_study.py` |
| 9 Joint figure | `ue-joint-figure` | `joint_figure.py` |

In order:

```bash
ue-decrypt
ue-detect
ue-extract-gt
ue-verify-alignment
ue-analyze-misalignment
ue-build
ue-convert-kitti
ue-train
ue-train --dataset kitti
ue-cross-eval
ue-size-map
ue-fid
ue-report
```

Every command takes `--help`. The scripts also run directly, for example
`python scripts/detect_poisoning.py --root <working folder>`.

**Order matters at Step 3.** `ue-build` reads `poisoning_detection_report/clean_labels/`,
so anything the detector rejects never reaches training. `ue-build --labels raw`
skips the filter and reproduces the original split exactly.

---

## The four experiments

| Train on | Test on | Measures |
|---|---|---|
| synthetic | synthetic | whether a model can learn from this data at all |
| synthetic | **real** | **sim-to-real transfer, the headline number** |
| real | real | the real-data upper bound |
| real | synthetic | the domain gap in reverse |

`ue-report` turns rows two and three into **utility retention**: synthetic-to-real
mAP50-95 as a percentage of real-to-real. That is the data-level cost of privacy,
and it pairs with the Privacy Integration component's model-level finding:

| Level | Component | Finding |
|---|---|---|
| Data | Utility Evaluation | synthetic data retains ___% of real-data utility |
| Model | Privacy Integration | DP-SGD costs 0.16 to 0.21 utility AUC and brings membership inference down to chance (attack AUC about 0.51) |

---

## Data poisoning detection

The threat: an attacker who cannot touch the images can still edit the label
files. A shifted box teaches the model to look in the wrong place, and a
relabeled batch teaches it the wrong classes. Neither shows up in the imagery.

Three layers, in `utility_evaluation/poisoning.py`:

1. **Structural, per frame.** Malformed lines, unknown class ids, coordinates
   outside the frame, degenerate boxes, implausible object counts. Failing
   frames are excluded.
2. **Statistical, per batch.** Each batch of 100 frames is tested against the rest
   of its own town with a chi-square goodness-of-fit test. A batch is rejected
   only when the deviation is both significant (p < 0.0001) and large (effect
   size above 0.40). Rejected batches are excluded.
3. **Cross-town.** Each town's class shares are compared with the other four.
   This is a warning, not a rejection, because CARLA towns differ by design.

`ue-prove-detection` builds a scratch town, injects an out-of-range box, an
unknown class id and a fully relabeled batch, and shows all three are caught.
Together with a PASS on the real data, this shows the detector both accepts
clean labels and rejects tampered ones.

---

## Interpreting results

- **Per-class results** matter alongside the average, since cars outnumber the
  other classes.
- **Small objects.** `ue-size-map` gives AP by object size. It is a simplified
  single-threshold (IoU 0.5) metric and should be labeled that way.
- **Resolution and aspect ratio.** KITTI is about 1242x375, CARLA 800x600. Part
  of any sim-to-real gap comes from that, not from the privacy mechanism.
  `ue-fid --crop center-square` removes the aspect-ratio part from FID.
- **Near-duplicate frames.** Consecutive CARLA frames are 0.8 s apart, so a random
  split puts near-copies of test frames into training and flatters
  synthetic-to-synthetic scores. `ue-build --split-mode block` keeps runs of
  frames together. Results should state which split was used.
- **Reproducibility.** Seed, YOLO version, image size, batch size and epochs are
  recorded with each run under `runs/<name>/args.yaml`.

---

## Per-epsilon study and the joint figure

Step 6 trains one detector per privacy configuration; Step 9 sets the results
beside Privacy Integration's. A **variant** is the synthetic dataset produced
under one configuration. Each variant is listed in a plan file (copy
`epsilon_study.example.yaml`) with its epsilon and either its encrypted delivery
or an already decrypted folder.

```bash
ue-epsilon-study --plan epsilon_study.yaml
ue-joint-figure  --study <root>/epsilon_study.json \
                 --privacy-summary <Privacy Integration outputs>/runs/summary.json
```

Every variant goes through the same poisoning gate, dataset build, training
settings and evaluation, and is tested on its own synthetic test split and on
KITTI. With the KITTI baseline in the plan, each variant's synthetic-to-real
score is also reported as a percentage of the real-data score. Trained variants
are reused on later runs; `--only <name>` reruns one and `--force` retrains.

The figure has one panel per measure, against the same epsilon axis: attack AUC
and classifier AUC from Privacy Integration, and detection utility from this
component. The non-private configuration is drawn apart from the epsilon series
because it is a reference, not a point on that scale. The same numbers are
written to `joint_privacy_utility.json`.

**To agree before the real runs:** how each variant is produced. Calibration
epsilon is capped at 2.0 in `Privacy-Integration/config.yaml`, so variants at
epsilon 3 and 8 cannot come from calibration alone.

`utility_results.json` is the file Risk and Compliance reads; its fields are
fixed in `shared/INTERFACES.md`.

---

## Layout

```
utility_evaluation/
├── config.py       towns, classes, working folder layout
├── poisoning.py    the three detection layers
├── alignment.py    label alignment and misalignment diagnosis
├── conversion.py   YOLO dataset building and splits
├── kitti.py        KITTI conversion
├── detection.py    per-class metrics, size-stratified AP
├── fid.py          FID sampling and preprocessing
├── decryption.py   decrypt and verify one town
├── training.py     YOLOv8 training, one set of settings for every model
├── study.py        per-epsilon study
├── joint.py        joint privacy-utility rows and figure
└── reporting.py    utility retention and the findings

scripts/            one file per step, names as originally used
tests/              unit tests plus an end-to-end run on an encrypted fixture
```

Decryption primitives, label parsing and key loading come from `r26_common` in
`shared/`.

## Never commit

`.env`, dataset imagery, model weights, or anything in the working folder. Run
`git status` and read the list before every push.
