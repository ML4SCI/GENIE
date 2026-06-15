# pinnde_eval

A small, shared, **quantitative** evaluation module for the PINNDE project, so the
score-based and flow-matching tracks can be compared with the same numbers. It
works on toy distributions today and is designed to take calorimeter high-level
features later without an API change.

Everything is plain functions plus the seeded toy generators. Inputs are torch
tensors **or** numpy arrays of shape `(N, d)`; conversion is internal.

## Install

```bash
pip install torch numpy scipy scikit-learn jetnet matplotlib
```

Runs on Google Colab (CPU or single GPU). `jetnet` provides the peer-reviewed
FPD/KPD reference implementations; if it is missing, every other metric still
runs and FPD/KPD come back as `None`.

The package is not pip-installed; put the folder that contains `pinnde_eval/` on
the path, e.g. in Colab after cloning the repo:

```python
import sys
sys.path.append(".../GENIE/PINN_Flow_Tina")   # the folder that contains pinnde_eval/
import pinnde_eval
```

## The three tiers

| Tier | Metric | What it tells you | Cost |
|------|--------|-------------------|------|
| 1 | `classifier_two_sample_test` | Test AUC (mean ± std over k retrainings) of an MLP separating real vs generated. **0.5 = indistinguishable.** | medium |
| 1 | `histogram_chi2` | Reduced χ² per feature between real/generated 1D histograms. **≈1 = agree within stats.** | cheap |
| 2 | `fpd` | Fréchet physics distance (2-Wasserstein between Gaussian fits) with infinite-sample extrapolation → `(value, error)`. | medium |
| 2 | `kpd` | Kernel physics distance (polynomial-kernel MMD) with batched uncertainty → `(median, error)`. | medium |
| 2 | `wasserstein_per_feature` | 1D Wasserstein-1 distance per coordinate (vector + mean). | cheap |
| 3 | `mmd` | RBF-kernel squared MMD, median-heuristic bandwidth. Pure torch. | cheap |
| 3 | `swd` | Sliced Wasserstein distance (random 1D projections). Pure torch. | cheap |

**Which tier when?**

- **Tier 3** — inside the training loop, every few hundred steps on ~1–5k
  samples. Fast, pure-torch, no extra dependencies. Use it to watch convergence.
- **Tier 2** — the headline numbers you report and compare between tracks. FPD
  and KPD come with error bars, so two models are comparable.
- **Tier 1** — the CaloChallenge-standard sanity checks. AUC near 0.5 and
  reduced χ² near 1 are the bar a good generator must clear.

## Usage

```python
from pinnde_eval import evaluate, report, mmd, swd

# Full suite (Tiers 1+2+3) -> flat dict
results = evaluate(real, gen, tier="full")
report(results)
#            mmd : 0.0021
#            swd : 0.013
#            auc : 0.52 +/- 0.01
#       chi2_mean : 1.08
#        w1_mean : 0.046
#            fpd : 0.00031 +/- 4e-05
#            kpd : ...

# Tier 3 only -- cheap monitors for the training loop
results = evaluate(real, gen, tier="monitor")   # {"mmd": ..., "swd": ...}
loss_monitor = mmd(real_batch, gen_batch)        # or call directly
```

### Conditional evaluation

Use `evaluate_by_condition` when a global score may hide failures in one class,
energy range, or detector region. Pass one label/bin per sample:

```python
from pinnde_eval import evaluate_by_condition

by_energy = evaluate_by_condition(real, gen, real_energy_bin, gen_energy_bin,
                                  tier="monitor")
print(by_energy["20-50 GeV"]["swd"])
```

### Calorimeter hook (deferred, no API change needed)

`evaluate(..., features_fn=...)` maps raw samples → high-level features before any
metric runs. For toys the features are the raw coordinates; later, layer
energies / shower widths (or a wrapper around the official CaloChallenge
`evaluate.py`) plug in here:

```python
results = evaluate(real, gen, tier="full", features_fn=shower_observables)
```

## Reproducibility

Everything stochastic (classifier splits/inits, projections, bandwidth
subsampling, FPD/KPD batching) takes a `seed`. The default `seed=0` makes runs
deterministic.

## Validation

`validate_toys.py` demonstrates the module is calibrated on Gaussian-mixture
toys (run `python -m pinnde_eval.validate_toys`):

1. **Null test** — two independent draws from the same 3D GMM give AUC ≈ 0.5,
   reduced χ² ≈ 1, and Tier-3 distances ≈ 0.
2. **Sensitivity test** — perturbing the GMM (mean shift, variance scale, drop a
   component) makes every metric grow with the perturbation size.
3. **Speed check** — Tier-3 monitors run in well under a second on 5k samples.

## Tests

```bash
pytest pinnde_eval/tests -q
```

## Files

- `tier1.py`, `tier2.py`, `tier3.py` — the metrics, grouped by tier.
- `evaluate.py` — the `evaluate()` entry point, `report()`, `plot_histograms()`.
- `data.py` — seeded GMM generators (mirror `flow_de/gendata.py`) for validation.
- `validate_toys.py` — null / sensitivity / speed checks.
- `tests/` — pytest edge-case tests.
