# R26-CS-010: Privacy Integration

Epsilon-differential privacy enforcement and empirical membership inference
validation for synthetic autonomous-vehicle training data.

**IT22309556, Priyadarshi S.A.S.D.**

The component enforces ε-DP at two points in the pipeline and validates the
guarantee empirically rather than assuming it:

1. **Calibration:** the Laplace mechanism privatizes aggregate statistics
   before they configure the CARLA simulator. A total budget of ε = 1.8 is
   split evenly over the seven queries (ε ≈ 0.257 each), so they compose to
   1.8 under sequential composition, reported as `calibration_epsilon_spent`.
   Earlier releases ran each query at 1.8 (12.6 in total); `pi-calibrate
   --epsilon` still reproduces that per-query mode.
2. **Training:** Opacus DP-SGD applies per-sample gradient clipping and
   Gaussian noise, with RDP accounting and an automatic budget halt.
3. **Validation:** a shadow-model membership inference attack with a
   two-feature signal (confidence and per-sample loss) measures whether the
   formal ε bound translates into real attack resistance.

## Setup

```bash
conda create -n research python=3.11
conda activate research
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ../shared -e ".[dev]"
```

Keys come from the project `.env` at the repository root (copy `.env.example`
there). No credentials appear in source.

## Pipeline

```bash
pi-decrypt        --input-dir  <encrypted> --output-dir <decrypted>
pi-analyse-labels --data-dir   <decrypted>
pi-train          --data-dir   <decrypted> --seed 42
pi-attack         --data-dir   <decrypted> --seed 42
python -m scripts.run_threshold_attack --data-dir <decrypted>
pi-report
```

Each stage is also runnable as `python -m scripts.<name>`.

### DP stress test (PP2 panel response)

```bash
pi-stress --data-dir <decrypted> --seeds 42 43 44
pi-stress-rescore          # recompute the verdict from saved results, no retraining
pi-stress-rescore --from-runs   # combine seeds run at different times (e.g. 42-44, then 45-46)
```

Every target trains on 500 members for 60 epochs, so the model without DP
overfits and leaks (the positive control). DP-SGD models at ε 1, 3 and 8 train
in the same regime and face the same loss-threshold attack, so any drop in
attack AUC comes from DP. Results go to `outputs/stress_test/`.

- **Positive control:** every no-DP run's attack AUC 95% bootstrap CI lies
  wholly above chance (0.5).
- **DP reduction:** for every ε, a paired one-sided t-test over seeds gives
  p < 0.05.
- **Budget ordering (reported, not part of the verdict):** paired one-sided
  t-tests between neighbouring budgets check that a larger ε leaves a
  stronger attack.

Each run writes one file per seed and configuration
(`seed_<n>_<config>.json`), and the summary covers only the seeds of that run.
`pi-stress-rescore --from-runs` rebuilds the summary from every per-run file.

### Stronger attack: reference-calibrated (offline LiRA-style)

```bash
pi-stress-calibrated --data-dir <decrypted> --seeds 42 43 44
```

Retrains the same stress-test targets (same seed, members and regime) and
attacks them with a reference-calibrated attack after Carlini et al. (2022)
and Watson et al. (2022). Four non-private reference models per seed, trained
on frames that are never attacked, estimate how easy each frame is; the attack
scores how much better the target does on a frame than the references expect,
using the logit of the true-label probability and one shared variance. This
can catch individual members at a low false-positive rate, where the plain
loss threshold cannot.

Results go to `outputs/stress_test/calibrated/` (`calibrated_summary.json`),
with the same verdict rules plus a paired comparison of TPR at 1% FPR. Each
record also reports the plain loss attack on the retrained target and whether
it reproduces the earlier `pi-stress` run. The command is safe to stop and
rerun: finished runs are skipped and reference results are cached per seed.
About 2.5 hours per seed on an RTX 3050 Ti.

Limits: references are non-private and there are only four per seed, so this
is weaker than full LiRA with many shadow models trained like each target.

The first version used stricter, arbitrary cut-offs (no-DP CI lower bound at
least 0.60, and fully separate CIs on every seed). They were replaced by the
standard tests above after the first run. The summary still reports the
first-version verdict under `first_version_rules`, and `pi-stress-rescore`
keeps the earlier verdict under `previous_verdicts`.

### In the integrated pipeline

`pipeline/` runs these stages for every configured seed and uses three hooks,
all off by default so standalone results are unchanged:

- `R26_OUTPUTS_DIR` moves `outputs/` into the run workspace.
- `--labels-dir <clean_labels>` trains on the labels that passed Utility
  Evaluation's poisoning gate. Rejected frames are skipped and counted as
  `gated`, never treated as empty frames.
- `build_certificate.py --utility-results <utility_results.json>` fills the
  utility section of the joint certificate.

## Artifact layout

Runs are namespaced by seed, so repeating a configuration under a new seed is
additive rather than destructive:

```
outputs/runs/seed_42/config_a/{model.pt, metrics.json, budget_log.csv,
                               audit_report.json, audit_certificate.json,
                               mia_results.json}
outputs/runs/seed_42/calibration_curve.json
outputs/runs/summary.json
outputs/figures/*.png
```

## Design decisions

| Decision | Reason |
| --- | --- |
| SGD with momentum for DP training | Adam's second-moment estimate accumulates injected DP noise, destabilizing training across every learning rate from 1e-5 to 1e-4 |
| Index duplication, not `WeightedRandomSampler` | Opacus replaces the DataLoader sampler inside `make_private` because RDP accounting requires Poisson subsampling |
| GroupNorm, not BatchNorm | Opacus rejects BatchNorm; using GroupNorm avoids depending on `ModuleValidator.fix` |
| 448×336 input, 8× downsampling, max pooling | At 256×256 with 16× downsampling and average pooling, 62% of positive targets fell below the resolvable size |
| Frames with only sub-threshold targets excluded | Labeling them negative asserts the absence of a present-but-unresolvable object; follows the KITTI minimum-size convention |
| AUC as the headline metric | The test set is ~10% positive while training is rebalanced 50/50, so accuracy reflects the prior shift; AUC is threshold-independent |
| Size-matched MIA evaluation sets | Unbalanced sets make attack accuracy track the class prior rather than attack skill |

## Reproducibility

`seed_everything` seeds Python, NumPy and Torch and disables cuDNN autotuning.
Verified end to end: the same seed reproduces identical per-epoch loss and an
identical final AUC across separate runs.

## Tests

```bash
pytest                       # full suite
pytest -m "not slow"         # skip torch-dependent tests
```
