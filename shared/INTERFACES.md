# Component Interfaces

Every file passed between components, what it contains, who produces it and who
reads it. The pipeline in `pipeline/` wires the components together through
exactly these files, and checks each one against its schema in
`shared/r26_contracts/schemas/` as soon as it is produced. A renamed or missing
field therefore stops the run at the stage that caused it.

---

## Handoff map

```
Real calibration statistics (outside the repository)
      |
      v
Privacy Integration: calibration -- calibration_stats.json --> Dataset Generation
      |                                                              |
      | calibration_audit_log.json                                   v
      |                                                encrypted dataset + manifests
      |                                                              |
      |                             +--------------------------------+
      |                             v
      |                 decrypt once (Utility Evaluation)
      |                             |
      |      manifest re-hash  <----+----> poisoning gate (Utility Evaluation)
      |      (Risk and Compliance)  |              |
      |                             |      clean_labels/
      |                             |              |
      |              +--------------+--------------+
      |              v                             v
      |   Privacy Integration                Utility Evaluation
      |   DP training, MIA                   YOLO, mAP, FID
      |              |                             |
      |   audit reports, summary          utility_results.json
      |              |                             |
      |              +----> privacy_certificate.json (joint)
      |                             |
      +-------------> Risk and Compliance <--------+
                                    |
                         compliance_report.json
                                    |
                                    v
                release/ with pipeline_report.json
```

## Machine-checked contracts

| File | Contract | Producer | Read by |
|---|---|---|---|
| `calibration_stats.json` | `calibration_stats` | Privacy Integration | Dataset Generation, Risk and Compliance |
| `calibration_audit_log.json` | `calibration_audit_log` | Privacy Integration | Risk and Compliance only, never released |
| `runs/seed_N/<config>/audit_report.json` | `audit_report` | Privacy Integration | Risk and Compliance |
| `runs/seed_N/<config>/audit_certificate.json` | `audit_certificate` | Privacy Integration | Risk and Compliance |
| `runs/seed_N/<config>/metrics.json` | `run_metrics` | Privacy Integration | Privacy Integration report |
| `runs/seed_N/<config>/mia_results.json` | `mia_results` | Privacy Integration | Privacy Integration report |
| `runs/summary.json` | `privacy_summary` | Privacy Integration | Pipeline report |
| `privacy_certificate.json` | `privacy_certificate` | Privacy Integration, with Utility Evaluation's results | Everyone |
| `poisoning_audit.json` | `poisoning_audit` | Utility Evaluation | Risk and Compliance |
| `utility_results.json` | `utility_results` | Utility Evaluation | Privacy Integration certificate, Risk and Compliance |
| `manifest_verification.json` | `manifest_verification` | Risk and Compliance | Pipeline gate |
| `calibration_composition.json` | `calibration_composition` | Risk and Compliance | Compliance report |
| `epsilon_study.json` | `epsilon_study` | Utility Evaluation | Joint figure |
| `joint_privacy_utility.json` | `joint_privacy_utility` | Utility Evaluation, from both components | Everyone |
| `compliance_report.json` | `compliance_report` | Risk and Compliance | Everyone |
| `pipeline_report.json` | `pipeline_report` | Pipeline | Everyone |

Validate any of them by hand:

```python
from pathlib import Path
from r26_contracts import validate_file
validate_file(Path("utility_results.json"), "utility_results")
```

---

## 1. Privacy Integration → Dataset Generation

### `calibration_stats.json`

Privatised aggregate statistics. Laplace noise already applied. Safe to share.

```json
{
  "vehicle_density_per_km":      2.5139,
  "pedestrian_density_per_km":   0.9153,
  "avg_vehicle_speed_kmh":      22.0135,
  "avg_pedestrian_speed_kmh":    0.2789,
  "intersection_count_per_km":   0.2796,
  "weather_distribution":     { "clear": 0.62, "cloudy": 0.14,
                                "rain": 0.24,  "fog": 0.00 },
  "time_of_day_distribution": { "morning": 0.14, "afternoon": 0.12,
                                "evening": 0.57, "night": 0.17 },
  "calibration_epsilon_spent": 12.6,
  "calibration_epsilon_per_query": 1.8,
  "num_queries": 7,
  "generated_at": "ISO-8601 UTC"
}
```

`calibration_epsilon_spent` is the total over all queries: each query reads
the same records, so their costs add under sequential composition (7 x 1.8 =
12.6). Files delivered before the correction record the per-query value (1.8)
in this field; their statistics are identical, so data generated from them is
unaffected.

**Rules**

- Use these values, not raw source statistics. Configuring the simulator from
  unprotected values defeats the entire privacy mechanism.
- Apply the same values to every town and every sensor type. The noise is
  applied once, at population level.
- Record the values used in `town_config_metadata.json` so provenance can be
  verified afterwards.
- Distributions may contain zeros. A zeroed weather bin is a legitimate outcome
  of noise addition, not an error.

### `calibration_audit_log.json`

The audit trail for the above. Records sensitivity and noise applied per query.
Consumed by Risk and Compliance, not by Dataset Generation.

---

## 2. Dataset Generation → everyone

### Folder layout, per town

```
town0X/
├── rgb_f000224.png.enc          AES-256-CBC encrypted
├── depth_f000224.png.enc
├── semantic_f000224.png.enc
├── class_masks/
│   └── class_mask_f000224.png.enc
├── labels/
│   └── bbox_f000224.txt         NOT encrypted
├── town0X_manifest.json
├── town0X_metadata.csv
└── town0X_config_metadata.json
```

### Encryption format

```
[16-byte IV][AES-256-CBC ciphertext, PKCS7 padded]
```

The key is shared out of band and must never be committed.

### `bbox_f*.txt`: YOLO format

```
class_id  x_center  y_center  width  height
```

All coordinates normalised to `[0, 1]`.

```
0 = car          includes vans, trucks, buses
1 = pedestrian
2 = cyclist      includes motorcycles
```

**An empty file is valid** and means no objects were detected in that frame.
Consumers must handle this without failing.

### `town0X_manifest.json`

SHA-256 digest of every file, keyed by the **unencrypted** filename.

```json
{ "rgb_f000224.png": "99a3445f..." }
```

Consumers verify after decrypting, before use.

### `town0X_config_metadata.json`

Scene configuration, including a `calibration_profile` block recording the
privatised values used. This is what makes the privacy claim auditable.
`epsilon_reference` is the per-query calibration epsilon; captures made after
the correction also record `epsilon_total`.

### `town0X_metadata.csv`

One row per frame set.

```
RGB_File, Depth_File, Semantic_File, Weather_Category,
Time_Of_Day, Weather_Label, Event_Type, Town, Security
```

---

## 3. Privacy Integration → Risk and Compliance

Delivered per seed, per configuration, under `runs/seed_NN/config_X/`.

| File | Contents |
|---|---|
| `audit_report.json` | Final epsilon, clipping norm, noise multiplier, budget status |
| `audit_certificate.json` | HMAC-SHA256 signature over the report |
| `budget_log.csv` | `step, epoch, cumulative_eps, loss`, one row per training step |
| `metrics.json` | Utility metrics on validation and test splits |
| `mia_results.json` | Attack success rate, attack AUC, member/non-member loss gap |
| `model.pt` | Trained weights |

Plus `privacy_certificate.json` aggregating all runs, and a `MANIFEST.json`
giving a SHA-256 digest of the handoff itself.

**The non-private baseline has no budget log**, and its audit report is an
explicit declaration that no privacy mechanism was applied. Absence of an
epsilon value there is intentional and must not be read as epsilon = 0.

**Loading a checkpoint:** Opacus wraps models in `GradSampleModule`, which
prefixes `state_dict` keys with `_module.`. Strip it before loading.

```python
state = {k.removeprefix("_module."): v for k, v in ckpt["state_dict"].items()}
```

---

## 4. Utility Evaluation → Risk and Compliance

Written to `outputs/` in the Utility Evaluation working folder.

### `utility_results.json`

Produced by `ue-report`. The file Risk and Compliance reads.

```json
{
  "experiments": {
    "synthetic -> synthetic": { "map50": 0.0, "map50_95": 0.0,
      "per_class": { "car": { "ap50": 0.0, "ap50_95": 0.0 },
                     "pedestrian": { "ap50": 0.0, "ap50_95": 0.0 },
                     "cyclist": null } }
  },
  "utility_retention_pct": 0.0,
  "sim_to_real_gap_map50_95": 0.0,
  "weakest_class": ["pedestrian", 0.0],
  "fid": 0.0,
  "fid_sample_size": 1000,
  "poisoning_verdict": "PASS",
  "experiments_missing": []
}
```

**Rules**

- Experiment keys are exactly `synthetic -> synthetic`, `synthetic -> real`,
  `real -> real`, `real -> synthetic`.
- A per-class value of `null` means the class did not appear in that test split.
  It is not zero.
- `poisoning_verdict` is `PASS` (nothing removed) or `FILTERED` (rejected frames
  were excluded before training). `FILTERED` is not a failure.
- Values that cannot be computed yet are `null`.

### Supporting files

| File | Producer | Content |
|---|---|---|
| `poisoning_audit.json` | `ue-detect` | Verdict, frames and boxes scanned, frames excluded, structural anomalies, rejected batches, per-town class shares, cross-town warnings |
| `evaluations.json` | `ue-cross-eval` | One record per experiment, same fields as above plus weights and data paths |
| `fid.json` | `ue-fid` | FID, sample sizes, seed, preprocessing, synthetic sample by town |
| `size_breakdown_*.json` | `ue-size-map` | AP50 per class and object size |

---

## 5. Risk and Compliance → all

### `compliance_report.json`

Independent verification results and the regulatory mapping. Combines evidence
from all three upstream components into a single verdict: `PASS` needs every
check run and passed, `FAIL` means a check failed, and `INCOMPLETE` means one is
outstanding, which is not a failure.

### `manifest_verification.json`

Every decrypted file re-hashed against the generator's manifest. The pipeline
stops here if anything does not match, before any model is trained.

---

## 6. Utility Evaluation → Privacy Integration

### `poisoning_detection_report/clean_labels/townXX/`

The labels that passed the poisoning gate. Privacy Integration trains on these
through `--labels-dir`. A frame whose label file is absent here was rejected,
and is skipped rather than treated as an empty frame.

### `utility_results.json`

Read by `build_certificate.py --utility-results` to fill the utility section of
the joint certificate.

---

## 7. Pipeline → everyone

### `release/`

What may be shared outside the team: the joint certificate and its signature,
the privacy summary, `calibration_stats.json`, the utility and poisoning
results, the compliance report, figures, `pipeline_report.json`, and a
`MANIFEST.json` of SHA-256 digests. The leakage scan runs over this folder.

`calibration_audit_log.json` is never released. It records the noise added to
each query, and the published value minus that noise is the real statistic.

### `pipeline_report.json`

The integrated verdict and the lineage of the run: every stage, its status, the
digest of each output, and the digest of the component code that produced it.

---

## Rules that apply to everyone

**Never commit**

- `.env` files or any key material
- Unprivatised source statistics, anything matching `*TRUE_VALUES*`
- Dataset imagery, encrypted or decrypted
- Model weights
- Generated outputs

The repository `.gitignore` covers all of these. Run `git status` and read the
list before your first commit of any component.

**Credentials**

There is one `.env` for the whole project, at the repository root, made from
`.env.example`. Every component reads its keys through `r26_common.env`, and
variables already set in the environment take priority. Hardcoded keys in a
repository are permanent once pushed, because they remain in git history even
after deletion.

**Changing an interface**

Open a pull request rather than sending a message. A renamed field breaks the
consumer silently, and silent breakage is expensive to find later.
