# R26-CS-010: Privacy Integration

Epsilon-differential privacy enforcement and empirical membership inference
validation for synthetic autonomous-vehicle training data.

**IT22309556, Priyadarshi S.A.S.D.**

The component enforces ε-DP at two points in the pipeline and validates the
guarantee empirically rather than assuming it:

1. **Calibration:** the Laplace mechanism privatizes aggregate statistics
   before they configure the CARLA simulator. Seven queries at ε = 1.8 each
   compose to a total of 12.6, reported as `calibration_epsilon_spent`.
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
