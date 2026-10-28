<div align="center">

  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/PyTorch-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/ScikitLearn-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/UnrealEngine.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Anaconda-Dark.svg" width="30" />
  <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/GithubActions-Dark.svg" width="30" />

  <h1>R26-CS-010</h1>
  <p><b>Privacy-Enhanced Synthetic Data Generation for Autonomous Vehicle Training</b></p>
  <p>A framework that guards the points where real data enters a synthetic driving dataset: differential privacy on the statistics that calibrate the simulator and on the models trained from it, attacks that test those guarantees, encryption and integrity checks on every generated file, a measure of what usefulness survives, and an independent audit against the GDPR.</p>
  <p>
    <a href="https://shndnth.github.io/R26-CS-010/"><b>Project website</b></a> ·
    <a href="Website/docs/documents/research-paper.pdf"><b>Research paper</b></a> ·
    <a href="Website/docs/documents/final-report-group.pdf"><b>Final report</b></a> ·
    <a href="Website/docs/slides/final.pdf"><b>Final slides</b></a>
  </p>

</div>

---

## Overview

Autonomous vehicle perception models need large volumes of labeled road
footage, and that footage records people. Simulated data avoids recording
anyone, but synthetic data is not private by default. The simulator is tuned
from real driving statistics, trained models can reveal which records they were
trained on, stored files can be read or changed, and privacy claims are rarely
checked by anyone other than their authors.

**The framework** answers this with four components that run as one pipeline.
It places formal guarantees where real information enters, protects every file
from the moment of capture, measures what the protection costs, and has every
claim checked by code written apart from the code that makes it.

```
 OWNER'S MACHINE ONLY                    DATASET GENERATION (CARLA)
┌----------------------------------┐     ┌--------------------------------------------┐
│ Real driving statistics          │     │ Five towns: RGB, depth, semantic, labels   │
│ Laplace: 7 queries, ε 1.8 total  │ ──► │ SHA-256 manifests, RBAC, AES-256 at source │
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

### Results at a Glance

Evaluated on 10,000 synthetic frames (five CARLA towns, 2,000 frame sets each) with KITTI as the independent real-data baseline.

| Question | Result | Answer |
|---|---|---|
| Does DP stop membership inference, and at what cost? | Attack AUC 0.639 without DP, 0.504 to 0.515 at ε = 1, 3 and 8, for 0.16 to 0.21 in test AUC. On an overfitted model, ε = 1 removed 99.5% of the attacker's advantage | Yes, at a measured cost |
| Are the generated files confidential and tamper-evident? | Every security audit test passed; a changed byte was rejected; 0 of 6,093 captured files readable; 40,000 of 40,000 decrypted files matched their manifests | Yes |
| Is the synthetic data still useful? | mAP50 of 0.3655 on synthetic images; 10.46% of the real-data mAP50 on KITTI; FID 115.99 | Useful in the synthetic domain, weak on real images |
| Do the claims hold when checked independently? | Every check passed on the final outputs; 8 of 8 GDPR provisions verified; one understated calibration budget found and fixed | Yes, within the scope of the checks |

---

## Contributions

1. **Differential privacy at both entry points.** Laplace noise on seven calibration statistics within a total budget of ε = 1.8 under sequential composition, before they configure CARLA; and DP-SGD with Rényi DP accounting and a hard ceiling of ε = 10 during training.
2. **Guarantees tested, not assumed.** Two membership inference attacks (shadow-model and loss-threshold) against private models and a non-private positive control, plus a five-seed stress test on a deliberately overfitted model, with bootstrap intervals and paired significance tests.
3. **Protection from the moment of capture.** Every generated file is hashed (SHA-256), access-checked (deny-by-default RBAC) and encrypted (AES-256) inside the capture step, with a six-test security audit that runs without the simulator.
4. **Two levels of privacy cost.** The cost at the model level (test AUC under DP-SGD) and at the data level (cross-domain YOLOv8 mAP and FID against KITTI), with a three-layer poisoning gate so that tampered labels never reach a model.
5. **Independent verification mapped to the law.** Seven checks written apart from the code they test, a compliance report in which each GDPR provision is verified only when the check behind it passed, and signed one-page PDF certificates.
6. **One modular pipeline.** 16 JSON Schema contracts, gates that stop a run at the stage that caused a fault, and recorded lineage for every stage. Four components built on four machines were joined without changing their internal code.

---

## Key Features

### Privacy Integration
* **Private Calibration:** The Laplace mechanism privatizes seven aggregate driving statistics (five scalars and two four-bin histograms) with a total budget of ε = 1.8, about 0.2571 per query under sequential composition, before they configure CARLA. The real values never leave the owner's machine.
* **DP-SGD Training:** Opacus 1.4.1 with an RDP accountant, per-sample clipping (C = 1.0), δ = 1e-5 and target budgets of ε 1, 3 and 8 (noise multipliers 0.977, 0.665 and 0.507), with a `BudgetExceededError` halt at ε = 10. A non-private baseline sets the utility ceiling and serves as the attack's positive control.
* **Two Attacks:** A shadow-model membership inference attack on confidence and per-sample loss, and a loss-threshold attack reporting the true-positive rate at 1% and 5% false positives.
* **Stress Test:** Targets trained for 60 epochs on 500 member frames make the no-DP control leak on purpose, so a failed attack under DP can be trusted.
* **Reference-Calibrated Attack:** An offline attack after LiRA (`pi-stress-calibrated`) that scores each frame against four non-private reference models, built in response to the second progress review.
* **Signed Evidence:** Every training run writes an audit report signed with HMAC-SHA256 and a per-step budget log. The privacy certificate combines the privacy evidence with the utility results and is also printed as a one-page PDF carrying its SHA-256 fingerprint, signature and QR code.

### Dataset Generation
* **Calibrated Capture:** Five CARLA towns, 2,000 frame sets each, with weather, time of day, traffic density and speed drawn from the privatized statistics.
* **Five Outputs per Frame:** RGB, depth, semantic segmentation, a class mask and a YOLO bounding-box label, plus per-frame metadata.
* **Protected at Source:** A SHA-256 digest of each plaintext file, a deny-by-default RBAC check, then AES-256 encryption with a fresh IV or nonce per file. The plaintext is removed only after the encrypted copy is written.
* **Two Cipher Modes:** The delivered dataset uses AES-256-CBC (16-byte IV, PKCS7 padding), so its integrity rests on the SHA-256 manifests. The engine also supports AES-256-GCM (`--cipher gcm`), whose tag rejects a changed byte during decryption at a cost of 32 bytes per file.
* **Security Audit:** Six tests (round trip, nonce uniqueness, byte distribution, bit change, tamper detection and a real captured folder) that run on the real `SecurityEngine` without CARLA.

### Utility Evaluation
* **Poisoning Gate:** Structural checks per label line, a chi-square test per batch of 100 frames (rejected only when p < 0.0001 and the effect size is above 0.40) and a cross-town comparison. Rejected frames never reach any model, including the privacy classifiers.
* **Four Cross-Domain Experiments:** YOLOv8n trained on synthetic data and on KITTI with identical settings, each tested on both. Synthetic-to-real as a percentage of real-to-real is the utility retention.
* **FID and Per-Class AP:** Fréchet Inception Distance between synthetic and real test images, per-class AP50 and AP by object size following the COCO area ranges.
* **Per-Epsilon Study:** One detector per privacy configuration, drawn beside the attack results in a joint privacy and utility figure.

### Risk and Compliance
* **Independent Verification:** Signatures, budgets, calibration logs and manifests are re-checked with this component's own code. No check reuses the signer, hash routine or noise function of the component it tests.
* **Attacks on the Calibration Channel:** A reconciliation of the noise log, a reconstruction attack on the published statistics and a composition check on the calibration budget.
* **Leakage and k-Anonymity:** Released files are scanned for paths, credentials, addresses and coordinates, and frame metadata must reach k = 5.
* **Regulatory Verdict:** Each provision of GDPR Articles 5, 17, 25 and 32 is marked verified only when the check behind it passed, with declared weaknesses carried into every report.
* **Compliance Certificate:** The compliance report is printed as a one-page PDF with its SHA-256 fingerprint, HMAC signature and QR code; `rc-certificate --verify` shows whether the report or the PDF changed.

### One Framework
* **One Command:** `r26 run` runs every stage in its component's own environment and repeats only what changed: inputs, settings or code.
* **Checked Handoffs:** 16 JSON Schema contracts. Each file that crosses a component boundary is validated as it is produced, so a missing field stops the run at the stage that caused it.
* **Gates:** A manifest mismatch or a stray seed stops the run before anything downstream uses the data.
* **Lineage:** Every stage records what it read, what it wrote and a digest of the code that ran, so each result traces back to its inputs.

---

## Results

All results below come from the final five-town capture, made from the corrected calibration release. The detailed experiments are in the [final report](Website/docs/documents/final-report-group.pdf), the [research paper](Website/docs/documents/research-paper.pdf) and the four individual reports.

### Privacy at Calibration

The Town01 simulator profile matched the privatized file field by field. Of Town01's 2,000 frames, 1,248 are clear, 270 cloudy and 482 rainy, the same shares as the privatized weather values (0.624, 0.135, 0.241).

| Query | Type | Sensitivity Δf | Noise scale b |
|---|---|---|---|
| Vehicle density per km | Scalar | 1.0 | 3.889 |
| Pedestrian density per km | Scalar | 1.0 | 3.889 |
| Average vehicle speed (km/h) | Scalar | 2.0 | 7.778 |
| Average pedestrian speed (km/h) | Scalar | 1.0 | 3.889 |
| Intersections per km | Scalar | 1.0 | 3.889 |
| Weather distribution (4 bins) | Histogram | 1.0 | 3.889 per bin |
| Time-of-day distribution (4 bins) | Histogram | 1.0 | 3.889 per bin |

Negative histogram bins are set to zero and the bins rescaled to sum to one; a value outside its natural range is mapped into range at simulator setup. Both are post-processing and cost no budget.

### Privacy at Training

Mean ± standard deviation over seeds 42, 43 and 44.

| Configuration | Target ε | Final logged ε | Test AUC | Attack AUC | Attack success | Loss gap |
|---|---|---|---|---|---|---|
| No DP | none | | 0.829 ± 0.009 | 0.639 ± 0.013 | 56.7% | 0.360 |
| Config A | 1.0 | 0.991 | 0.621 ± 0.045 | 0.504 ± 0.002 | 50.3% | 0.001 |
| Config B | 3.0 | 3.000 | 0.655 ± 0.035 | 0.515 ± 0.002 | 51.6% | 0.013 |
| Config C | 8.0 | 7.995 | 0.668 ± 0.020 | 0.506 ± 0.007 | 50.8% | 0.026 |

* The attack worked against the non-private baseline and fell to near chance on every private model. Single runs did not overlap: the nine private runs scored 0.499 to 0.516, the three baselines 0.625 to 0.650.
* Paired two-sided t-tests against same-seed baselines: the drops in test AUC (0.160 to 0.208, p ≤ 0.0099) and in attack AUC (0.124 to 0.134, p ≤ 0.0032) were significant for every configuration. With three seeds, the budgets did not separate from each other.
* The loss-threshold attack gave the same picture: AUC 0.610 without DP and 0.501 to 0.506 with it. At 1% and 5% FPR it was at chance for every configuration, including the baseline, so the baseline leaked only a small average shift in loss.

### Stress Test

To answer the second progress review (the drop looked small), targets were trained to memorize 500 member frames for 60 epochs, over five seeds, and attacked with the loss-threshold attack. One-sided paired t-tests against the non-private model.

| Configuration | Attack AUC | Drop | p-value | Advantage removed |
|---|---|---|---|---|
| No DP | 0.628 | | | |
| ε = 1 | 0.501 | 0.128 | 0.00004 | 99.5% |
| ε = 3 | 0.526 | 0.102 | 0.00006 | 79.4% |
| ε = 8 | 0.547 | 0.081 | 0.001 | 63.0% |

The 95% bootstrap interval of every no-DP seed lay above 0.5. The budgets separated in the order theory predicts: ε = 1 vs 3 (p = 0.013) and ε = 3 vs 8 (p = 0.010). The first version of the stress test used stricter, arbitrary verdict rules; they were replaced by standard tests after the first run, and the privacy certificate reports both verdicts.

The reference-calibrated attack recorded 7.9% of members caught at a 1% FPR without DP and 1.1% (chance) at ε = 1. The final report lists a full evaluation of this attack, with more seeds and separate code in Risk and Compliance, as future work.

### Protection of Generated Files

Security audit of the AES-256-GCM engine. The first five tests use synthetic bytes and an in-memory key; the sixth reads a real 6,093-image Town01 capture with the project key and writes no plaintext.

| Property | Evidence | Outcome |
|---|---|---|
| Round-trip correctness | Bytes identical after decryption | Pass |
| Nonce uniqueness | 30 of 30 unique nonces and ciphertexts | Pass |
| Byte distribution | Chi-square 272.59 (5% critical value 293.25) | Pass |
| Ciphertext change | 50.0% of bits differ | Pass |
| Tamper detection | One flipped byte: "MAC check failed", no output written | Pass |
| Real folder | 0 of 6,093 encrypted files readable as PNG | Pass |

Encrypting a 4 KB file took about 15 ms and decrypting it about 26 ms. For the CBC dataset the tamper test is reported as not applicable, and integrity rests on the manifest gate: all 40,000 decrypted image files of the five towns matched their SHA-256 entries.

### Utility of the Synthetic Data

The poisoning gate checked 10,000 frames and 15,084 boxes: no structural faults, 24 batches (2,400 frames) excluded by the batch layer, verdict FILTERED, and a cross-town warning for Town01 (cyclist share 0.030 against 0.116 elsewhere). In planted-defect tests it caught an out-of-range box, an unknown class and a fully relabeled batch.

| Experiment | mAP50 | mAP50-95 | Car AP50 | Pedestrian AP50 | Cyclist AP50 |
|---|---|---|---|---|---|
| Synthetic to Synthetic | 0.3655 | 0.2273 | 0.6076 | 0.1793 | 0.3096 |
| Synthetic to Real (KITTI) | 0.0889 | 0.0343 | 0.2614 | 0.0051 | 0.0001 |
| Real (KITTI) to Real (KITTI) | 0.8500 | 0.5531 | 0.9519 | 0.7769 | 0.8213 |
| Real (KITTI) to Synthetic | 0.0507 | 0.0199 | 0.1273 | 0.0079 | 0.0171 |

* **Utility retention:** 10.46% of the real-data mAP50 (6.2% on mAP50-95). Cars kept about 27.5% of their real-data AP50; pedestrians and cyclists almost disappeared.
* **FID:** 115.99 between 1,000 synthetic and 1,000 KITTI test images, a moderate difference on the 50 to 200 scale. Detection utility was much worse than moderate, so FID and mAP should be read together.
* **Label imbalance:** 11,008 cars, 2,424 pedestrians and 1,652 cyclists; 32.6% of frames have no objects; 86.4% of pedestrian and 67.3% of cyclist boxes are under 32 × 32 pixels.
* The detectors were trained on all 10,000 frames before the batch layer of the gate was added.

### Independent Verification

| Check | What was tested | Result | Verdict |
|---|---|---|---|
| Certificate verification | Every signed audit report verifies | 12 of 12 reports (4 configurations × 3 seeds) | PASS |
| Tamper test | A changed report is rejected | ε changed from 0.990572 to 0.01 was caught | PASS |
| Budget audit | Reported ε equals the last logged ε; never falls; at most 10 | 9 private runs agree; 3 baselines declared | PASS |
| Calibration log | Source value plus logged noise equals the published value | 5 of 5 scalar fields | PASS |
| Composition | Queries compose to the claimed total | 7 × 0.2571 = 1.8 | PASS |
| Reconstruction | Noise has the size the mechanism requires | Ratio 0.94 (4.37 observed vs 4.67 expected) | Consistent |
| Manifest verification | Every decrypted file matches the manifest | 40,000 of 40,000 | PASS |
| k-anonymity | Smallest group sharing all quasi-identifiers | k = 5 over 192 groups; no unique rows | PASS |
| Leakage scan | No paths, addresses or credentials in released files | 9 files, 0 findings | PASS |

The compliance report combined six of these checks with the poisoning verdict. All seven passed and all eight GDPR mappings were verified:

| GDPR provision | Requirement | Evidence check | State |
|---|---|---|---|
| Art. 5(1)(b) | Purpose limitation | Calibration audit log | Verified |
| Art. 5(1)(c) | Data minimization | Calibration audit log | Verified |
| Art. 5(2) | Accountability | Certificate verification | Verified |
| Art. 17 | Right to erasure | k-anonymity | Verified |
| Art. 25 | Data protection by design | Budget audit | Verified |
| Art. 25 | Calibration budget accounting | Calibration composition | Verified |
| Art. 32 | Security of processing | Manifest verification | Verified |
| Art. 32 | Integrity of processed data | Poisoning detection | Verified |

The overall verdict was **PASS** with two declared weaknesses: Opacus secure mode was off, and earlier calibration releases had understated their total ε. The report was fingerprinted with SHA-256, signed with HMAC-SHA256 and printed as certificate **R26-RC-F3335EAE8286**. It covers only the listed checks and is not a full GDPR assessment.

### Findings from Integration

* **Understated calibration budget.** An earlier calibration release ran each of the seven queries at ε = 1.8, so it spent 12.6 while reporting 1.8. Each query's noise had the correct size, so only the composition check, written by another member, could see it. The module now splits a total budget over the queries, the check passes on the corrected release, and the dataset was captured again from it.
* **Fixed public seed.** A later review found that the calibration noise had been drawn from a pseudo-random generator seeded with 42 from a public configuration file, which would let an attacker regenerate and remove the noise. The reconstruction check assumes the attacker does not know the noise, so every check passed. The fault was corrected before the dataset was captured again. Treat the `random_seed` in `Privacy-Integration/config.yaml` as public: it is for repeatable tests, not for a real release.
* **Label problem, not a DP problem.** The first private and non-private classifiers predicted the majority class. In 62.3% of positive frames the largest pedestrian was under about 8 × 8 pixels after resizing, and frames were stretched to a square. Keeping the aspect ratio and excluding frames with only small targets raised the baseline test AUC from about 0.50 to 0.84.
* **Keys in source.** A code review found the AES and HMAC keys written in source files. Both were replaced and moved to environment files, and a regression test fails if an old key reappears.

### Group Objectives

| No. | Objective | Evidence | Status |
|---|---|---|---|
| GO1 | Cumulative training budget within ε = 10 | Final ε of 0.991, 3.000 and 7.995; logs equal the reports in the independent audit | Fully achieved |
| GO2 | Membership inference success at or below 55% on private models | 50.3% to 51.6% with DP, 56.7% without; stress test confirms the effect | Fully achieved |
| GO3 | Perception accuracy within 15% of a real-data model | 10.46% of the real-data mAP50 on real images | **Not achieved** |
| GO4 | Calibration privatized before the simulator, within ε = 2 | Corrected release composes to 1.8; dataset captured from it; fixed seed found and corrected | Fully achieved |
| GO5 | Every generated file protected | SHA-256, AES-256 and RBAC; all audit tests passed; 40,000 of 40,000 files matched | Fully achieved |
| GO6 | Every label file checked for poisoning | Planted defects caught; full dataset screened (FILTERED) | Fully achieved |
| GO7 | Independent verification mapped to the GDPR | All checks passed; 8 of 8 provisions verified; signed certificate | Fully achieved |

At the model level the cost of privacy is moderate and well understood. At the data level the gap between synthetic and real images is large, and it cannot be attributed to differential privacy alone: the dataset was produced at one calibration budget, with no non-private dataset to compare against, and rendering, object size and image shape all add to the gap.

---

## Threat Model

The simulation host and the pipeline code are trusted. Storage and the people who can reach it are not fully trusted. The adversary can read or change files in storage, or try to act through a role that lacks a permission.

| Threat | Example | Control |
|---|---|---|
| Tampering | Images or labels changed before training | SHA-256 manifest per town; GCM tag; poisoning gate for labels |
| Disclosure | Dataset files copied from storage or a stolen disk | AES-256 encryption; plaintext removed after encryption |
| Privilege misuse | An ML user reads raw simulation output | RBAC gateway with deny-by-default permissions |
| Duplicate detection | Identical images give identical ciphertext | Fresh random IV or nonce for every encryption |
| Membership leakage | A model reveals whether a frame was in its training set | DP-SGD with a hard budget limit, tested with attacks |
| Recovery of source statistics | Published calibration values reveal the real values | Laplace noise on every query; audit log never released |
| Altered evidence | A signed report is edited after training | HMAC-SHA256 signature checked by Risk and Compliance |

| RBAC role | Permitted operations | Purpose |
|---|---|---|
| `Simulation_Engineer` | create, verify, encrypt | Produces and protects raw simulation output |
| `Privacy_Engineer` | verify | Checks integrity before privacy processing |
| `ML_Researcher` | read_processed | Uses processed datasets only |

**Out of scope:** attacks on the running simulation host, memory scraping, theft of an encryption key and side-channel attacks.

---

## Limitations

* **Utility gap.** A detector trained on the synthetic data keeps 10.46% of the real-data mAP50 on KITTI. The synthetic data cannot yet replace real data for the final training of a detector.
* **One calibration budget.** The dataset was captured at one calibration budget only, so the share of the utility gap caused by calibration noise is unknown.
* **Integrity of the delivered dataset.** It uses AES-256-CBC, which has no built-in tamper detection; integrity rests on the SHA-256 manifests. The GCM engine was tested on a separate Town01 capture.
* **Statistical power.** Three seeds separated private from non-private models but not the budgets; only the five-seed stress test separated them. The attacks are realistic but not the strongest known.
* **Image shape and labels.** CARLA images are 800 × 600 and KITTI images about 1242 × 375. The detectors were trained before the batch layer of the poisoning gate was added.
* **Scope of the checks.** A passing statistical test shows only that no weakness was found in that property; it does not replace cryptanalysis. The checks cover the listed evidence files, not all the software of the other components; the fixed calibration seed is an example of a fault every check passed. Some GDPR mappings rest on indirect evidence (Article 17 is linked to k-anonymity). k-anonymity holds exactly at the threshold of 5.
* **Keys and access.** Keys are kept in local files without a key management service. The HMAC key is shared by producer and auditor, so either could create a valid signature. The RBAC gateway has no user authentication and no persistent access log. Plaintext is removed with a normal file delete, not crypto-shredding.
* **Opacus secure mode** is off, because the library it needs does not support current PyTorch versions. Both certificates declare it.
* **Generalizability.** One simulator, five towns, one privacy classifier and one detector family; one calibration source (the Waymo Open Dataset). No formal user study with external users.
* **Licenses.** The Waymo Open Dataset and KITTI are licensed for non-commercial use only.

---

## Future Work

In order of priority:

| Priority | Direction | Components | Strengthens |
|---|---|---|---|
| 1 | **Seed check:** a check that fails on a fixed or public seed, so a fault like the one found here is caught automatically | PI, RC | GO4, GO7 |
| 2 | **Close the utility gap:** capture at several calibration budgets for a full privacy and utility curve; retrain on the filtered labels; use a real baseline with a closer image shape; fine-tune on a small real subset | DG, UE | GO3 |
| 3 | **Stronger attacks and more seeds:** evaluate the reference-calibrated attack in full and repeat the membership inference study with separate code in Risk and Compliance | PI, RC | GO2, GO7 |
| 4 | **Strengthen file protection:** move the delivered dataset to AES-256-GCM, encrypt frames in memory so plaintext is never written, and keep per-batch keys in a key management service | DG | GO5 |
| 5 | **Harden access and evidence:** user authentication and a tamper-evident access log for the RBAC gateway; public-key signatures in place of HMAC, so only the signer can sign and anyone can verify | DG, PI, RC | GO5, GO7 |

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
| Dataset Generation | CARLA client | `python scripts/capture_town.py --town town01` (add `--cipher gcm` for authenticated encryption); `python scripts/security_audit.py --mode gcm` |

Every command takes `--help`. Each component's README has the full sequence.

### Tests and Checks

```bash
cd shared && pytest                  # 50 tests
cd pipeline && pytest                # 37 tests
cd Privacy-Integration && pytest     # 135 tests
cd Utility-Evaluation && pytest      # 117 tests
cd Risk-Compliance && pytest         # 79 tests
cd Dataset-Generation && pytest      # 26 tests, CARLA stubbed
R26_E2E=1 pytest pipeline/tests/test_end_to_end.py   # the whole pipeline on generated data
ruff check .
```

The suites cover decryption and manifest checks, key loading, every handoff contract, the stage graph and gates, Laplace noise and its composition, DP-SGD budgets and the halt, the attack features, the stress test and calibrated attack, certificate signing, the PDF certificates and the tamper tests, the security engine and audit, the poisoning detector against injected defects, size-stratified AP, FID sampling, budget audits, the reconstruction attack, leakage patterns and the regulatory mapping. Each Risk and Compliance check is tested with a planted fault. The end-to-end test runs every real stage on an encrypted fixture and expects an overall PASS.

| Integration test | Expected | Actual |
|---|---|---|
| Manifest gate | Every decrypted file matches its digest | 40,000 of 40,000 matched |
| Poisoning gate, full dataset | Structural faults rejected; outlier batches excluded | 0 structural faults; 24 batches excluded; FILTERED |
| Poisoning gate, planted defects | Out-of-range box, unknown class and relabeled batch caught | All caught |
| Calibration reaches the simulator | Town01 profile equals the privatized file | Matched field by field |
| Budget limit | Training halts at the limit | Halted in a low-limit test |
| Per-step budget logging | One row per step | 3,090 rows per private run |
| Repeatability | Same seed gives the same result | Same loss and test AUC |
| Signed report tampering | Changed report is rejected | Rejected by PI and by RC |
| End-to-end pipeline | Every stage runs on the fixture | Overall PASS |

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
| Calibration | Laplace | ε = 1.8 in total, 0.2571 per query over 7 queries | The real driving statistics | Audit log, reconciliation, reconstruction attack, composition check |
| Training | DP-SGD (Opacus, RDP) | ε 1, 3 or 8, δ = 1e-5, halt at 10 | The synthetic training frames | Signed audit report and budget log per run; independent budget audit |
| Validation | Membership inference | | | Shadow-model and loss-threshold attacks per configuration and seed; a five-seed stress test against an overfitted control; a reference-calibrated attack |

The two budgets protect different data, so they are reported separately and never added.

---

## Under the Hood (Architecture)

* `shared/r26_common`: decryption and manifest checks, YOLO label parsing, key loading and hashing, used in the same form by every component.
* `shared/r26_contracts`: the 16 handoff schemas and the validator.
* `pipeline/r26_pipeline`: the runner, run state and lineage, the integrated report, and one stage module per component.
* `Privacy-Integration/`: calibration, the DP-SGD trainer, the attacks, the stress tests and the certificate.
* `Utility-Evaluation/`: decryption, the poisoning detector, dataset building, YOLO training and evaluation.
* `Risk-Compliance/`: the independent checks, the regulatory mapping and the compliance certificate.
* `Dataset-Generation/`: the CARLA capture script, its security engine, the RBAC gateway and the security audit. Runs on its own; its output feeds the pipeline.
* `Website/`: the project website, published to GitHub Pages, with the project documents and slides in `Website/docs/`.

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
  scripts/              capture_town, security_engine, rbac_gateway, security_audit
  notebooks/            security_audit
Website/
  docs/                 documents/, slides/, images/
.github/workflows/      tests.yml, pages.yml
```

### Handoffs

| File | From | To |
|---|---|---|
| `calibration_stats.json` | Privacy Integration | Dataset Generation, Risk and Compliance |
| `calibration_audit_log.json` | Privacy Integration | Risk and Compliance only |
| Encrypted towns and `*_manifest.json` | Dataset Generation | Utility Evaluation, Privacy Integration, Risk and Compliance |
| Frame metadata (CSV) | Dataset Generation | Risk and Compliance |
| `clean_labels/`, `poisoning_audit.json` | Utility Evaluation | Privacy Integration, Risk and Compliance |
| `audit_report.json`, `audit_certificate.json`, `budget_log.csv` | Privacy Integration | Risk and Compliance |
| `summary.json` | Privacy Integration | Utility Evaluation |
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

Values already set in the environment win. Otherwise keys come from the file named by `R26_ENV_FILE`, then a `.env` in the current folder or any folder above it, then the one at the repository root. A missing or wrong-length key raises an error; no key is ever generated silently, because that would make the dataset unrecoverable.

---

## Documents and Slides

All documents are also on the [project website](https://shndnth.github.io/R26-CS-010/).

| Document | Link |
|---|---|
| Research paper | [research-paper.pdf](Website/docs/documents/research-paper.pdf) |
| Final report (common integrated solution) | [final-report-group.pdf](Website/docs/documents/final-report-group.pdf) |
| Final reports (individual) | [IT22309556](Website/docs/documents/final-report-IT22309556.pdf) · [IT22541284](Website/docs/documents/final-report-IT22541284.pdf) · [IT22110220](Website/docs/documents/final-report-IT22110220.pdf) · [IT22066916](Website/docs/documents/final-report-IT22066916.pdf) |
| Project proposals | [IT22309556](Website/docs/documents/proposal-IT22309556.pdf) · [IT22541284](Website/docs/documents/proposal-IT22541284.pdf) · [IT22110220](Website/docs/documents/proposal-IT22110220.pdf) · [IT22066916](Website/docs/documents/proposal-IT22066916.pdf) |
| Research logbooks | [IT22309556](Website/docs/documents/logbook-IT22309556.pdf) · [IT22541284](Website/docs/documents/logbook-IT22541284.pdf) · [IT22110220](Website/docs/documents/logbook-IT22110220.pdf) · [IT22066916](Website/docs/documents/logbook-IT22066916.pdf) |
| Topic assessment form | [taf.pdf](Website/docs/documents/taf.pdf) |
| Checklists | [Checklist 1](Website/docs/documents/checklist-1.pdf) · [Checklist 2](Website/docs/documents/checklist-2.pdf) |

| Presentation | Slides |
|---|---|
| Proposal | [proposal.pdf](Website/docs/slides/proposal.pdf) |
| Progress review 1 | [progress-1.pdf](Website/docs/slides/progress-1.pdf) |
| Progress review 2 | [progress-2.pdf](Website/docs/slides/progress-2.pdf) |
| Final | [final.pdf](Website/docs/slides/final.pdf) |

### Technical Documentation

| Document | For |
|---|---|
| [shared/INTERFACES.md](shared/INTERFACES.md) | Every handoff between components: format, producer and consumer |
| [Privacy-Integration/README.md](Privacy-Integration/README.md) | Calibration, DP-SGD, the attacks, the stress tests, the certificate and the design decisions |
| [Utility-Evaluation/README.md](Utility-Evaluation/README.md) | The working folder, the steps in order, poisoning detection and interpreting results |
| [Risk-Compliance/README.md](Risk-Compliance/README.md) | Each check, its method, the regulatory mapping, declared weaknesses and the certificate |
| [Dataset-Generation/README.md](Dataset-Generation/README.md) | Running a capture, per-town settings, the cipher modes, the security audit and the output layout |

---

## Team

| Member | Student ID | Component | Folder |
|---|---|---|---|
| Priyadarshi S.A.S.D. | IT22309556 | Privacy Integration, framework integration | `Privacy-Integration/`, `pipeline/` |
| Yubitha S. | IT22541284 | Dataset Generation | `Dataset-Generation/` |
| Apeksha M.K.S.R. | IT22110220 | Utility Evaluation | `Utility-Evaluation/` |
| Wijesinghe R.D.P.T. | IT22066916 | Risk and Compliance | `Risk-Compliance/` |

**Supervisor:** Prof. Anuradha Jayakody · **Co-supervisor:** Mr. Dinith Primal

Each member contributed 25% of the project; the shared repository, library, contracts, pipeline runner, research paper, website and final report belong to the whole group.

BSc (Hons) in Information Technology, specializing in Cyber Security. Faculty of Computing, Sri Lanka Institute of Information Technology. Project R26-CS-010, 2026.

---

## Citation

If you use this work, please cite the research paper:

```bibtex
@misc{r26cs010_2026,
  title        = {Privacy-Enhanced Synthetic Data Generation for Autonomous Vehicle Training},
  author       = {Priyadarshi, S. A. S. D. and Yubitha, S. and Apeksha, M. K. S. R. and
                  Wijesinghe, R. D. P. T. and Jayakody, Anuradha and Primal, Dinith},
  year         = {2026},
  institution  = {Sri Lanka Institute of Information Technology},
  note         = {Project R26-CS-010},
  howpublished = {\url{https://github.com/shndnth/R26-CS-010}}
}
```

---

## Acknowledgements

We thank our supervisor, Prof. Anuradha Jayakody, and our co-supervisor, Mr. Dinith Primal, for their guidance throughout this research, and the academic staff of the Faculty of Computing whose comments at the proposal and progress reviews changed how we tested and reported our results. We also thank the open-source communities behind CARLA, PyTorch, Opacus, Ultralytics, PyCryptodome and ReportLab. The calibration statistics were computed from the [Waymo Open Dataset](https://waymo.com/open/) and the real-data baseline used [KITTI](https://www.cvlibs.net/datasets/kitti/), both under their non-commercial licenses.
