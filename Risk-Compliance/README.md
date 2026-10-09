# Risk and Compliance

Component of R26-CS-010: Privacy-Enhanced Synthetic Data Generation for
Autonomous Vehicle Training.

**Owner:** IT22066916, Wijesinghe R.D.P.T.

Independent verification of the privacy controls built by the other components,
and the mapping of those controls to the GDPR. The component
does not implement any privacy mechanism itself. It measures whether the
mechanisms behave as claimed and issues the compliance verdict.

## Scope

| Check | Question answered | Output |
|---|---|---|
| Certificate verification | Has any signed audit report been altered since signing? | `certificate_verification.json` |
| Budget audit | Does each reported epsilon match the per-step log, and stay under the ceiling? | `budget_audit.json` |
| Reconstruction attack | How well can the real calibration statistics be recovered from the published ones? | `reconstruction_attack.json` |
| Calibration composition | Do the calibration queries compose to the epsilon the calibration stage claims? | `calibration_composition.json` |
| Manifest verification | Does every decrypted file match the generator's SHA-256 manifest? | `manifest_verification.json` |
| Leakage scan | Do released artifacts expose paths, credentials or identifiers? | `leakage_scan.json` |
| k-anonymity | Can a frame be singled out from its metadata? | `anonymity.json` |
| Compliance report | Which provisions are verified by passing evidence? | `compliance_report.json` |

The verifiers are implemented independently of the code they check. Certificate
signatures, for example, are verified here rather than with Privacy
Integration's signer, so a defect in the signer cannot pass unnoticed.

## Setup

```bash
conda create -n compliance python=3.11
conda activate compliance
pip install -e ../shared -e ".[dev]"
```

`R26_AUDIT_HMAC_KEY` comes from the project `.env` at the repository root, shared
out of band.

## Usage

```bash
rc-verify-certs    --runs-dir <runs>
rc-audit-budgets   --runs-dir <runs>
rc-reconstruct     --true-values <true values json> --published <calibration_stats.json> \
                   --audit-log <calibration_audit_log.json>
rc-calibration-composition --audit-log <calibration_audit_log.json> --published <calibration_stats.json>
rc-verify-manifest --encrypted-dir <delivered dataset> --decrypted-dir <decrypted dataset>
rc-scan-leakage    --root <release folder>
rc-anonymity       --data-dir <folder with *_metadata.csv>
rc-report          --utility-dir <Utility Evaluation outputs>
```

Each command writes JSON to `outputs/` unless `--output` is given. `rc-report`
combines whatever results exist with the regulatory mapping. In the integrated
pipeline every check runs automatically, and the pipeline stops if manifest
verification fails.

## Method

**Certificate verification.** Every audit report carries an HMAC-SHA256
signature. All signatures are verified, then a copy of one report is altered
(its recorded epsilon is changed) to confirm the alteration is rejected.

**Budget audit.** The final epsilon in each audit report is compared with the
cumulative epsilon in its budget log. Epsilon must never decrease between steps,
since RDP composition is cumulative, and must never exceed the ceiling. The
non-private baseline has no budget log by design; its report is an explicit
declaration that no privacy mechanism was applied, and is treated as such.

**Reconstruction attack.** An adversary holding `calibration_stats.json` can do
no better than take each published value as its estimate, because Laplace noise
has zero mean. For sensitivity `s` and budget `ε` the expected absolute error is
`s/ε`; the check reports observed error against that expectation, where a ratio
near 1.0 indicates noise of the correct magnitude. The audit log is reconciled
separately: `true + logged noise = published` must hold for every field.

**Calibration composition.** Each calibration query's spend is recovered from
the audit log as `sensitivity / laplace_scale`. The seven queries read the same
records, so their costs add under sequential composition, and the total must not
exceed the `calibration_epsilon_spent` published with the statistics. When it
does, the result states the per-query budget that would meet the claim.

**Manifest verification.** Every decrypted file is re-hashed and compared with
the manifest shipped with the encrypted delivery. No key is needed, because the
manifest records digests of the plaintext.

**Leakage scan.** Encryption protects file contents, not file names, metadata,
logs or configuration. The scan looks for absolute paths, e-mail and IP
addresses, coordinates and credential-shaped strings. A finding is a prompt for
review rather than an automatic failure.

**k-anonymity.** `k` is the size of the smallest group of frames sharing the same
quasi-identifiers (weather, time of day, event type, town). `k = 1` means at
least one frame is uniquely identifiable from metadata alone.

**Compliance report.** A provision is marked verified only when the check backing
it passed. The verdict is `PASS` when every check ran and passed, `FAIL` when one
failed, and `INCOMPLETE` when one has not run, which is not a failure.

## Regulatory mapping

`risk_compliance/mapping.py` links each provision to the control that addresses
it and the check that evidences it: GDPR Articles 5(1)(b), 5(1)(c), 5(2), 17, 25
(training and calibration) and 32. The summaries are working notes and should be
confirmed against the regulation text before submission.

## Declared weaknesses

Recorded in `mapping.py` and carried into every report:

- **Opacus secure_mode disabled** (medium). DP noise comes from a standard
  pseudo-random generator, because torchcsprng is unmaintained and only
  supports PyTorch 1.8.1. Remediation: use a secure noise source once one
  supports current PyTorch.
- **Calibration total epsilon** (medium, corrected). Seven queries at 1.8 compose
  to 12.6 under sequential composition, but earlier outputs reported 1.8 as the
  total. `rc-calibration-composition` found it; the total is now reported
  correctly and the published statistics are unchanged. Running each query at
  0.257 would restore a 1.8 total but needs new statistics and a new capture.

## Planned

An independent replication of the membership inference study. The `attack`
extra (`pip install -e ".[attack]"`) reserves its dependencies; no check uses
it yet.

## Layout

```
risk_compliance/
├── settings.py         configuration and credential loading
├── verification.py     certificate verification and tamper test
├── budget_audit.py     epsilon claims against per-step evidence
├── reconstruction.py   Laplace reconstruction attack
├── composition.py      calibration budget composition
├── manifest.py         independent manifest verification
├── leakage.py          leakage scan
├── anonymity.py        k-anonymity
└── mapping.py          GDPR mapping, declared weaknesses

scripts/                command line entry points
tests/
```

File formats are defined in `shared/INTERFACES.md`. Never commit `.env`, the
true-values calibration file, or anything under `outputs/`.
