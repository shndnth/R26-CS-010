<div align="center">

  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/PyTorch-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/ScikitLearn-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/UnrealEngine.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Anaconda-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/GithubActions-Dark.svg" width="30" />

  <h1>R26-CS-010</h1>
  <p><b>Privacy-Enhanced Synthetic Data Generation for Autonomous Vehicle Training</b></p>
  <p>A framework that guards the point where real data enters a synthetic driving dataset: differential privacy on the statistics that calibrate the simulator and on the models trained from it, attacks that test those guarantees, a measure of what usefulness survives, and an independent audit against the GDPR.</p>
  <p><a href="https://shndnth.github.io/R26-CS-010/"><b>Visit the project website</b></a></p>

</div>

---

## Overview

Autonomous vehicle perception models need large volumes of labeled road
footage, and that footage records people. Simulated data avoids recording
anyone, but the simulator is tuned from real driving statistics, so the privacy
risk comes back through that channel. **The framework** places formal guarantees
where real information enters, then checks that they hold, measures what they
cost, and audits the result.

```
 OWNER'S MACHINE ONLY                    DATASET GENERATION (CARLA)
┌----------------------------------┐     ┌--------------------------------------------┐
│ Real driving statistics          │     │ Five towns: RGB, depth, semantic, labels   │
│ Laplace: 7 queries, ε 1.8 total  │ ──► │ AES-256 at source, SHA-256 manifests       │
└----------------------------------┘     └----------------------┬---------------------┘
                                                                │ encrypted delivery
 PIPELINE (r26, one environment per component)                  ▼
┌----------------------------------------------------------------------------------------------┐
│ Decrypt ──► Manifest gate ──► Poisoning gate ──┬──► DP-SGD (ε 1, 3, 8) ──► Membership attack │
│ (UE)        (RC, stops run)   (UE, filters)    └──► YOLOv8 ──► mAP, FID against KITTI        │
│                                                                                              │
│ Joint certificate (PI) ◄── privacy summary + utility results                                 │
│ Compliance checks (RC) ──► Release ──► Leakage scan ──► Integrated verdict                   │
└----------------------------------------------------------------------------------------------┘
```

---

## Key Features

### Privacy Integration
* **Private Calibration:** The Laplace mechanism privatizes seven aggregate driving statistics with a total budget of ε = 1.8 (about 0.257 per query under sequential composition), before they configure CARLA. The real values never leave the owner's machine.
* **DP-SGD Training:** Opacus 1.4.1 with an RDP accountant, per-sample clipping (C = 1.0), δ = 1e-5 and target budgets of ε 1, 3 and 8, with an automatic halt at ε = 10. A non-private baseline sets the utility ceiling.
* **Guarantees Tested, Not Assumed:** A shadow-model membership inference attack on confidence and per-sample loss, plus a loss-threshold attack reporting the true-positive rate at 1% and 5% false positives. Under DP both attacks fall to chance level (AUC about 0.50 to 0.51).
* **Stress Test:** Models trained to memorize a small member set make the no-DP control leak on purpose (attack AUC 0.628 over five seeds); DP-SGD brings it to 0.501 at ε = 1, significant at every budget (p ≤ 0.001), with stronger privacy giving a weaker attack.
* **Calibrated Attack:** A reference-calibrated attack after LiRA catches 7.9% of members at a 1% false-positive rate without DP and 1.1% (chance) at ε = 1.
* **Signed Evidence:** Every training run writes an audit report signed with HMAC-SHA256. The privacy certificate combines the privacy evidence with the utility results and is also printed as a one-page PDF carrying its SHA-256 fingerprint, signature and QR code.

### Utility Evaluation
* **Poisoning Gate:** Structural checks per frame, a chi-square test per batch of 100 frames and a cross-town comparison. Rejected frames never reach any model, including the privacy classifiers.
* **Four Domain Experiments:** YOLOv8 trained on synthetic data and on KITTI with identical settings, each tested on both. Synthetic-to-real as a percentage of real-to-real is the utility retention.
* **FID and Object Size:** Frechet Inception Distance between synthetic and real test images, and AP by object size following the COCO area ranges.
* **Per-Epsilon Study:** One detector per privacy configuration, drawn beside the attack results in a joint privacy and utility figure.

### Risk and Compliance
* **Independent Verification:** Signatures, budgets and manifests are re-checked with this component's own code, so a defect in the checked code cannot pass unnoticed.
* **Attacks on the Calibration Channel:** A reconstruction attack on the published statistics and a composition check on the calibration budget.
* **Leakage and k-Anonymity:** Released files are scanned for paths, credentials and coordinates, and frame metadata must reach k = 5.
* **Regulatory Verdict:** Each provision of GDPR Articles 5, 17, 25 and 32 is marked verified only when the check behind it passed, with declared weaknesses carried into every report.
* **Compliance Certificate:** The compliance report is printed as a one-page PDF with its SHA-256 fingerprint, HMAC signature and QR code; `rc-certificate --verify` shows whether the report or the PDF changed.

### Dataset Generation
* **Calibrated Capture:** Five CARLA towns, about 2000 frame sets each, with weather, time of day, traffic density and speed drawn from the privatized statistics.
* **Five Outputs per Frame:** RGB, depth, semantic segmentation, a class mask and a YOLO bounding-box label.
* **Encrypted at Source:** AES-256-CBC with a SHA-256 manifest per town, role-based access control, and the plaintext removed after encryption.

### One Framework
* **One Command:** `r26 run` runs every stage in its component's own environment and repeats only what changed: inputs, settings or code.
* **Checked Handoffs:** 16 JSON Schema contracts. Each file that crosses a component boundary is validated as it is produced, so a missing field stops the run at the stage that caused it.
* **Gates:** A manifest mismatch or a stray seed stops the run before anything downstream uses the data.
* **Lineage:** Every stage records what it read, what it wrote and a digest of the code that ran, so each result traces back to its inputs.

---

## Installation and Usage

### Quick Start

Requirements: Anaconda or Miniconda, Python 3.11, and an NVIDIA GPU for training (CPU works, slowly). CARLA is needed only for capture.

**1. Create the environments** (one per component, because their dependencies conflict)
```bash
conda create -n research python=3.11 && conda activate research
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e shared -e Privacy-Integration

conda create -n utility python=3.11 && conda activate utility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e shared -e Utility-Evaluation

conda create -n compliance python=3.11 && conda activate compliance
pip install -e shared -e Risk-Compliance

conda create -n r26 python=3.11 && conda activate r26
pip install -e shared -e pipeline
```

**2. Add the keys** (one `.env` for the whole project, at the repository root)
```bash
cp .env.example .env
```
```
R26_DATASET_AES_KEY=<32 bytes, from IT22541284>
R26_AUDIT_HMAC_KEY=<from IT22309556>
```
Both keys are shared out of band. `.env` is never committed.

**3. Point the pipeline at the data**
```bash
cd pipeline
cp pipeline.example.yaml pipeline.yaml
```
```yaml
workspace: ../../R26_runs/main                # outside the repository
inputs:
  encrypted_dataset: ../../Dataset/output     # town01 to town05 from Dataset Generation
  calibration_source:                         # real statistics; empty skips calibration
  kitti:                                      # KITTI training folder; empty skips the real baseline
```

**4. Run it**
```bash
r26 check        # environments, inputs and keys
r26 plan         # every stage and whether it will run
r26 run          # run whatever is not up to date
r26 status       # what happened
```

| Command | What it does |
|---|---|
| `r26 run --only utility` | One component |
| `r26 run --from privacy.report` | Resume from a stage |
| `r26 run --only utility.fid --force` | Repeat one stage |
| `r26 run --dry-run` | Print the commands only |
| `r26 validate` | Recheck every handoff file against its contract |

The integrated verdict is in `release/pipeline_report.json`:

| Verdict | Meaning |
|---|---|
| **PASS** | Privacy certificate and compliance report both pass, all four utility experiments ran, no stage skipped |
| **FAIL** | The privacy or the compliance verdict failed |
| **INCOMPLETE** | Anything else, for example no KITTI set yet. Not a failure |

> **Real statistics:** keep the unprivatized calibration file outside the repository. The commit checks refuse it.

### Running a Component on Its Own

| Component | Environment | Commands |
|---|---|---|
| Privacy Integration | `research` | `pi-calibrate`, `pi-train`, `pi-attack`, `pi-report`, `pi-stress`, `pi-stress-rescore`, `pi-stress-calibrated`, `pi-certificate` |
| Utility Evaluation | `utility` | `ue-decrypt`, `ue-detect`, `ue-build`, `ue-train`, `ue-cross-eval`, `ue-fid`, `ue-report`, `ue-epsilon-study`, `ue-joint-figure` |
| Risk and Compliance | `compliance` | `rc-verify-manifest`, `rc-verify-certs`, `rc-audit-budgets`, `rc-reconstruct`, `rc-calibration-composition`, `rc-anonymity`, `rc-scan-leakage`, `rc-report`, `rc-certificate` |
| Dataset Generation | CARLA client | `python scripts/capture_town.py --town town01` |

Every command takes `--help`. Each component's README has the full sequence.

### Tests and Checks

```bash
cd shared && pytest                  # 50 tests
cd pipeline && pytest                # 37 tests
cd Privacy-Integration && pytest     # 135 tests
cd Utility-Evaluation && pytest      # 117 tests
cd Risk-Compliance && pytest         # 79 tests
cd Dataset-Generation && pytest      # 14 tests, CARLA stubbed
R26_E2E=1 pytest pipeline/tests/test_end_to_end.py   # the whole pipeline on generated data
ruff check .
```

The suites cover decryption and manifest checks, key loading, every handoff contract, the stage graph and gates, Laplace noise and its accounting, DP-SGD budgets and the halt, the attack features, the stress test and calibrated attack, certificate signing, the PDF certificates and the tamper tests, the poisoning detector against injected defects, size-stratified AP, FID sampling, budget audits, the reconstruction attack, composition, leakage patterns and the regulatory mapping. The end-to-end test runs every real stage on an encrypted fixture and expects an overall PASS.

Install the commit checks once per clone. They block `.env` files, keys, unprivatized statistics, datasets, model weights and em or en dashes, and run the linter:

```bash
pip install pre-commit
pre-commit install
```

Every push runs the same checks on GitHub Actions: the commit checks, each component's tests in its own environment, then the end-to-end run.

---

## Privacy Guarantees

| Stage | Mechanism | Budget | Protects | Evidence |
|---|---|---|---|---|
| Calibration | Laplace | ε = 1.8 in total, 0.257 per query over 7 queries | The real driving statistics | Audit log, reconstruction attack, composition check |
| Training | DP-SGD (Opacus, RDP) | ε 1, 3 or 8, δ = 1e-5, halt at 10 | The synthetic training frames | Signed audit report and budget log per run |
| Validation | Membership inference | | | Shadow-model and loss-threshold attacks per configuration and seed; a five-seed stress test against an overfitted control; a reference-calibrated attack at 1% FPR |

The two budgets protect different data, so they are reported separately and never added.

## Team

| Member | Student ID | Component | Folder |
|---|---|---|---|
| Priyadarshi S.A.S.D. | IT22309556 | Privacy Integration, framework integration | `Privacy-Integration/`, `pipeline/` |
| Apeksha M.K.S.R | IT22110220 | Utility Evaluation | `Utility-Evaluation/` |
| Wijesinghe R.D.P.T. | IT22066916 | Risk and Compliance | `Risk-Compliance/` |
| Yubitha S. | IT22541284 | Dataset Generation | `Dataset-Generation/` |

BSc (Hons) in Information Technology, specializing in Cyber Security. Sri Lanka Institute of Information Technology. Project R26-CS-010.

---

## Under the Hood (Architecture)

* `shared/r26_common`: decryption and manifest checks, YOLO label parsing, key loading and hashing, used in the same form by every component.
* `shared/r26_contracts`: the 16 handoff schemas and the validator.
* `pipeline/r26_pipeline`: the runner, run state and lineage, the integrated report, and one stage module per component.
* `Privacy-Integration/`: calibration, the DP-SGD trainer, the attacks, the stress tests and the certificate.
* `Utility-Evaluation/`: decryption, the poisoning detector, dataset building, YOLO training and evaluation.
* `Risk-Compliance/`: the independent checks, the regulatory mapping and the compliance certificate.
* `Dataset-Generation/`: the CARLA capture script and its security engine. Runs on its own; its output feeds the pipeline.

```
shared/
  r26_common/           crypto, labels, env, hashing
  r26_contracts/        schemas/ (16), validator
  INTERFACES.md         every handoff: format, producer, consumer
pipeline/
  r26_pipeline/         cli, config, runner, state, report
    stages/             base, privacy, utility, compliance
Privacy-Integration/
  privacy_integration/  calibration, training, attack, evaluation, audit, data, models
  scripts/              run_calibration, run_training, run_mia_study, run_stress_test,
                        run_calibrated_stress, build_report, build_certificate
Utility-Evaluation/
  utility_evaluation/   decryption, poisoning, alignment, conversion, kitti, detection, fid,
                        training, study, joint, reporting
  scripts/              one file per step
Risk-Compliance/
  risk_compliance/      verification, budget_audit, reconstruction, composition, manifest,
                        leakage, anonymity, mapping, certificate
  scripts/              one file per check
Dataset-Generation/
  scripts/              capture_town, security_engine, rbac_gateway
.github/workflows/      tests.yml
```

### Handoffs

| File | From | To |
|---|---|---|
| `calibration_stats.json` | Privacy Integration | Dataset Generation, Risk and Compliance |
| Encrypted towns and `*_manifest.json` | Dataset Generation | Utility Evaluation, Risk and Compliance |
| `clean_labels/`, `poisoning_audit.json` | Utility Evaluation | Privacy Integration, Risk and Compliance |
| `audit_report.json`, `audit_certificate.json` | Privacy Integration | Risk and Compliance |
| `utility_results.json` | Utility Evaluation | Privacy Integration, Risk and Compliance |
| `privacy_certificate.json` | Privacy Integration | Release |
| `compliance_report.json` | Risk and Compliance | Release |
| `privacy_certificate.pdf`, `compliance_certificate.pdf` | Privacy Integration, Risk and Compliance | Printed for review and signing (not copied into `release/`) |
| `pipeline_report.json` | Pipeline | Release |

`calibration_audit_log.json` is never released: published value minus logged noise is the real statistic.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `R26_DATASET_AES_KEY` | *(required)* | AES-256 key for the CARLA dataset, exactly 32 bytes |
| `R26_AUDIT_HMAC_KEY` | *(required)* | Signs and verifies the privacy audit reports |
| `R26_ENV_FILE` | *(empty)* | A `.env` somewhere other than the repository root |
| `R26_OUTPUTS_DIR` | `Privacy-Integration/outputs` | Where Privacy Integration writes; the pipeline sets it per run |
| `R26_E2E` | *(unset)* | `1` runs the end-to-end test |

Values already set in the environment win. Otherwise keys come from the file named by `R26_ENV_FILE`, then a `.env` in the current folder or any folder above it, then the one at the repository root.

---

## Documentation

| Document | For |
|---|---|
| [shared/INTERFACES.md](shared/INTERFACES.md) | Every handoff between components: format, producer and consumer |
| [Privacy-Integration/README.md](Privacy-Integration/README.md) | Calibration, DP-SGD, the attacks, the stress tests, the certificate and the design decisions |
| [Utility-Evaluation/README.md](Utility-Evaluation/README.md) | The working folder, the steps in order, poisoning detection and interpreting results |
| [Risk-Compliance/README.md](Risk-Compliance/README.md) | Each check, its method, the regulatory mapping, declared weaknesses and the certificate |
| [Dataset-Generation/README.md](Dataset-Generation/README.md) | Running a capture, per-town settings and the output layout |
