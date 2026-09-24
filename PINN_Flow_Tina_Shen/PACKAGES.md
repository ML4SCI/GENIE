# Tina — flow-matching track (GSoC 2026, ML4SCI / GENIE PINNDE)

Two self-contained Python packages for the PINNDE project:

| Package | What it is |
|---|---|
| **`pinnde_eval/`** | A shared, quantitative **evaluation module** — one function compares any two point clouds (real vs. generated) with three tiers of metrics. Meant to be the common yardstick for *both* tracks (score-based and flow-matching). |
| **`flow_matching/`** | A **conditional flow-matching generator** — learns a velocity field and generates by integrating an ODE from noise to data. Simulation-free: training never integrates the ODE, so there is no O(N·M) Monte-Carlo score estimate inside the loss. |

Everything is plain functions + a couple of small `nn.Module`s, fully seeded, and
tested. Inputs are torch tensors **or** numpy arrays of shape `(N, d)`.

---

## Install

```bash
pip install torch numpy scipy scikit-learn matplotlib pytest
pip install jetnet          # OPTIONAL — only needed for Tier-2 FPD/KPD (see note below)
```

or:

```bash
pip install -r requirements.txt
```

Runs on CPU (a GPU helps only the flow-matching training, which is tiny here).

> **Threading gotcha:** if numpy links a threaded OpenBLAS (e.g. numpy 1.26
> wheels on macOS), run the scripts with `OPENBLAS_NUM_THREADS=1`. The Tier-1
> classifier issues thousands of tiny matrix multiplies and OpenBLAS's
> thread busy-wait can turn a 9-second `evaluate()` into >30 minutes of
> spinning that looks like a hang. Colab's default numpy is unaffected.

Nothing is pip-installed as a package — put this `Tina/` folder on the path:

```python
import sys; sys.path.append(".../GSOC_2026_PINNDE/Tina")
import pinnde_eval, flow_matching
```

(The scripts and tests already insert the path automatically when run from this folder.)

---

## Part 1 — `pinnde_eval`: how good is a generator?

One entry point returns a flat dict of metrics, grouped in three tiers:

| Tier | Question it answers | Metrics |
|------|---------------------|---------|
| **Live monitors** | *Is it converging?* (cheap, run during training) | `mmd`, `swd` |
| **Distribution distances** | *How far apart, with error bars?* | `fpd`, `kpd`, `wasserstein_per_feature` |
| **Standard tests** | *Can you even tell real from generated?* | `classifier_two_sample_test` (AUC), `histogram_chi2`, `separation_power` |

```python
import numpy as np
from pinnde_eval import evaluate, report, mmd, swd

real = np.random.default_rng(1).normal(size=(5000, 3))
gen  = np.random.default_rng(2).normal(size=(5000, 3))

# cheap monitors only (use inside a training loop)
evaluate(real, gen, tier="monitor")     # {"mmd": ..., "swd": ...}
swd(real, gen)                           # or call a metric directly

# full suite (everything) -> dict, then a printed table
res = evaluate(real, gen, tier="full")
report(res)
```

`report` prints:

```
            mmd : 0.0001
            swd : 0.049
            auc : 0.502 +/- 0.004     # ~0.5  -> indistinguishable (good)
chi2_per_feature: [1.5, 0.6, 1.1]
      chi2_mean : 1.09                # ~1    -> histograms agree (good)
        w1_mean : 0.046               # ->0   -> distributions close (good)
            fpd : 0.0003 +/- 4e-05    # (None if jetnet not installed)
            kpd : ...
```

**Interpretation cheat-sheet:** AUC → 0.5, χ² → 1, MMD/SWD/W1/FPD → 0 all mean *good*.

### Per-condition evaluation
A global score can hide a bad slice (e.g. one energy bin). Pass one label per sample:

```python
from pinnde_eval import evaluate_by_condition
by_bin = evaluate_by_condition(real, gen, energy_bin_labels, tier="monitor")
print(by_bin["20-50 GeV"]["swd"])
```

### Real calorimeter showers
`evaluate(..., features_fn=...)` maps raw samples → high-level features *before* any
metric runs. `pinnde_eval.observables` supplies that function for CaloChallenge
voxelised showers:

```python
from pinnde_eval import load_calochallenge, shower_features_fn, evaluate

showers, e_inc = load_calochallenge("dataset_2_1.hdf5", n=8000)
fn = shower_features_fn(e_inc, geometry="ds2")
res = evaluate(real_showers, gen_showers, features_fn=fn, standardize=True)
```

Get ds2 (electrons) from [Zenodo](https://zenodo.org/records/6366271) and drop
both files in `Tina/`; they are gitignored (~1.4 GB each, over GitHub's limit).

Two traps, both documented in `pinnde_eval/DEVLOG.md` §11:

- **Voxel order is `(layer, alpha, r)`.** ds2's 6480 = 45 × 16 angular × 9
  radial. The natural-looking `reshape(45, 9, 16)` is wrong and silently
  corrupts `sigma_r` while leaving `<z>` plausible. The tests assert the
  convention.
- **Pass `standardize=True` for observables in mixed units.** SWD and W1 are
  scale-dependent; on the Geant4 null `swd` reads 1128 unstandardized against
  0.022 standardized, because `E_tot` in MeV swamps `sparsity` in [0,1].

```bash
python -m pinnde_eval.validate_calo    # null test + separation-power floor law
```

Two independent Geant4 draws give the floor a *perfect* generator hits
(N=8000): `auc 0.4971 ± 0.0101 · chi2 1.096 · swd 0.0222 · sep 0.0030`.

### Stability vs. sample size — *how many samples do I need to trust a number?*
Every metric has a finite-N **null floor** (the value a *perfect* generator shows)
and a finite-N **spread**. The stability study measures both, plus the smallest N
at which a fixed perturbation separates from the floor by 2 combined error bars:

```python
from pinnde_eval import stability_study, separation_z, min_resolvable_n

study = stability_study(n_grid=(250, 1000, 4000), kind="mean", eps=0.2)
min_resolvable_n(study)   # e.g. {"swd": 250, "auc": 4000, ...}
```

```bash
python -m pinnde_eval.stability     # full table + figures/stability.png (takes a few min)
```

Rule of thumb from the study: **never compare a metric value against zero —
compare it against the null floor at your N** (the floors are in `DEVLOG.md` §8).

### Local discrepancy maps — *where does the generator fail?*
A global score hides localized failures (a dropped or shifted mode barely moves
SWD). Three maps in `pinnde_eval.local` point at the failing *region*:

| Map | Question | Reading |
|---|---|---|
| `mmd_witness(real, gen, points)` | where is probability mass wrong? | > 0 real over-dense (missed), < 0 gen over-dense (hallucinated) |
| `classifier_discrepancy(real, gen)` | which samples give it away? | out-of-fold P(real\|x) per sample; sort gen ascending → worst fakes |
| `binned_residual_map(real, gen, features=(0,1))` | which histogram bins disagree? | per-bin z ~ N(0,1) under null; \|z\| > 3 = genuine local failure |

```bash
python figure_scripts/make_local_figure.py        # demo: all three maps light up on the broken modes
```

All three run in feature space, so for showers they apply to any observable pair
(e.g. layer energy vs. width) through the same `features_fn` hook.

### Validate the metrics on toys
```bash
python -m pinnde_eval.validate_toys
```
Three checks (all asserted): **null** (same distribution → AUC≈0.5, χ²≈1, distances≈0),
**sensitivity** (perturb → every metric grows monotonically), **speed** (Tier-3 < 1 s on 5k).
Calibration details and the chosen thresholds are in `pinnde_eval/DEVLOG.md`.

---

## Part 2 — `flow_matching`: a toy generator

Learn a velocity field `vθ(x, t)` by regressing it onto the straight-line target,
then generate by integrating the ODE from noise (`t=0`) to data (`t=1`):

```
x_t   = (1 - t)*x0 + t*x1          # x0 ~ N(0,I) noise, x1 data
target = x1 - x0                    # constant velocity along the path
loss   = || vθ(x_t, t) - target ||^2
```

```python
from flow_matching import train_flow_matching, sample

# data: a (N, d) torch tensor
model, history = train_flow_matching(data, dim=2, n_steps=4000, seed=0)
gen = sample(model, n=5000, dim=2, steps=50)     # integrate noise -> data
```

Run the end-to-end toy demo (trains on a 2-D Gaussian mixture, scores with `pinnde_eval`):

```bash
python -m flow_matching.demo
```

**Time convention:** `t=0` = noise, `t=1` = data, sample `0 → 1` (the flow-matching /
Lipman convention). The diffusion / score track uses the opposite labelling
(`t=0` data → `t=1` noise); the two are related by `t → 1−t`.

### Conditional generation — *the actual calorimeter target is p(shower | E_inc)*
Pass a per-sample condition (e.g. normalized log incident energy) and the field
becomes `vθ(x, t, c)`; the condition enters as its raw value plus a
low-frequency Fourier embedding:

```python
model, history = train_flow_matching(x, dim=3, cond=c)     # c: (N, 1)
gen = sample(model, n=5000, dim=3, cond=0.9)               # everyone at c=0.9
gen = sample(model, n=5000, dim=3, cond=c_eval)            # matched per-row conds
```

Run the end-to-end conditional demo (calo-flavored toy: sampling fraction /
depth / width vs. energy, correlated + skewed):

```bash
python -m flow_matching.demo_conditional
```

It scores the model three ways, strictest last: pooled over all energies,
**per energy bin** (`evaluate_by_condition` — a bad bin cannot hide), and at
**fixed unseen conditions** c* ∈ {0.1, 0.5, 0.9} against fresh truth draws at
exactly those c* (the interpolation test a marginal model would fail).

### On real CaloChallenge showers

```bash
python -m flow_matching.demo_calo    # needs both ds2 files in Tina/
```

Trains `p(shower observables | E_inc)` on `dataset_2_1` and scores it against
the held-out `dataset_2_2` plus the measured Geant4 null floor (~20 min on CPU).
The model now sits **at the floor in every energy bin** — AUC 0.487–0.510 per
quartile, 0.4998 pooled against a 0.4971 floor.

Getting there is the clearest argument in the project for per-condition
evaluation. The first configuration was pooled-indistinguishable from Geant4
while its **lowest energy quartile sat at AUC 0.787** — a quarter of the data
badly modelled, entirely invisible in the pooled score. Separation power being
*fine* in that bin is what proved the marginals were innocent and the joint was
wrong; two plausible fixes (dequantization, more ODE steps) moved it by under
0.01, and capacity was the real bottleneck. Full chain in
`pinnde_eval/DEVLOG.md` §13–14.

---

## Tests

```bash
python -m pytest pinnde_eval/tests flow_matching/tests -q     # 171 tests
```

The CaloChallenge observable tests run on synthetic voxel grids with
analytically known answers, so the suite needs no downloaded data; the one
real-data integration test skips itself when the ds2 file is absent.

## Reproduce the figures (in `figures/`)
```bash
python figure_scripts/make_figures.py        # sensitivity + flow-matching result figures
python figure_scripts/make_flow_figure.py    # velocity field + noise->data trajectories
python figure_scripts/make_local_figure.py   # local discrepancy maps on a broken generator
python -m pinnde_eval.stability   # metric stability vs sample size
```

| Figure | Shows |
|---|---|
| `sensitivity.png` | every metric grows as the distribution is perturbed (module is calibrated) |
| `stability.png` | null floor, spread, and resolvability of every metric vs. sample size N |
| `local_discrepancy.png` | the three local maps lighting up exactly on a dropped + shifted mode |
| `fm_convergence.png` | flow-matching MMD/SWD falling to the null floor during training |
| `fm_scatter.png`, `fm_histograms.png` | generated vs. true (2-D GMM) |
| `fm_flow.png` | the learned velocity field and noise→data trajectories |
| `fm_conditional.png` | conditional FM tracking p(observables \| energy): means vs. c + per-bin histograms |
| `fm_calo.png` | conditional FM on **real** ds2 showers: observable means vs. incident energy + marginals |
| `fig_calibration.png` | KS p-values under an exact null vs. across the two Geant4 files; false-positive rate per observable |
| `fig_power.png` | detection rate of KS / CvM vs. separation power on a controlled shift |
| `fig_voxel_order.png`, `fig_scale_bug.png`, `fig_sep_floor.png`, `fig_energy_bins.png`, `fig_zero_inflation.png` | real-data findings, see `pinnde_eval/DEVLOG.md` §11–16 |

### Toy result (2-D GMM, 6 modes)
Generated-vs-truth lands near the statistical floor:
**SWD 0.066** (floor ≈0.049) · **MMD ~1e-4** · **AUC 0.54 ± 0.003** (0.5 = perfect).

---

## Note: FPD / KPD need `jetnet`
FPD and KPD wrap `jetnet.evaluation`. If `jetnet` is not installed they return
`None` and everything else still runs (the module degrades gracefully — this path
is tested). A plain `pip install jetnet` works on Google Colab but fails on
macOS/Python 3.14 because its `wasserstein` dependency tries to compile with
OpenMP. The working local recipe (Python 3.12, no compiler needed — `wasserstein`
is skipped entirely, FPD/KPD don't use it):

```bash
pip install --no-deps jetnet
pip install "numpy<2" "scipy<1.14" numba energyflow tables h5py pandas awkward coffea pyyaml requests tqdm
python -m pinnde_eval.validate_toys     # FPD/KPD now reported instead of "n/a"
```

---

## Layout
```
Tina/
├── pinnde_eval/        evaluation module
│   ├── evaluate.py     evaluate(), evaluate_by_condition(), report()
│   ├── tier1.py        classifier AUC, histogram chi^2
│   ├── tier2.py        FPD, KPD, per-feature Wasserstein
│   ├── tier3.py        MMD, sliced Wasserstein, Sinkhorn divergence (pure torch)
│   ├── stability.py    metric stability vs sample size (null floor, resolvability)
│   ├── local.py        local maps: MMD witness, classifier P(real|x), binned residuals
│   ├── observables.py  CaloChallenge shower observables + HDF5 loader (features_fn)
│   ├── data.py         seeded GMM toys
│   ├── validate_toys.py null / sensitivity / speed checks
│   ├── validate_calo.py real-data null test + separation-power floor law
│   ├── classical.py    KS / Cramér–von Mises / Anderson–Darling tests, p-value combination
│   ├── validate_classical.py calibration, power and independence checks for classical.py
│   ├── DEVLOG.md       calibration record + chosen thresholds
│   └── tests/
├── flow_matching/      conditional flow-matching generator
│   ├── model.py        VelocityField (Fourier time + condition embeddings, GELU)
│   ├── core.py         fm_loss, sample (Euler/Heun ODE, optional condition)
│   ├── train.py        train_flow_matching (Adam + cosine LR, optional monitoring)
│   ├── demo.py         toy GMM demo, scored with pinnde_eval
│   ├── demo_conditional.py  p(observables | energy) toy demo, scored per energy bin
│   ├── demo_calo.py    p(observables | E_inc) on real ds2, vs the Geant4 floor
│   ├── demo_voxels.py  the same, generating the 6480 raw voxels instead
│   └── tests/
├── figures/            generated result figures
└── figure_scripts/     the scripts that produce them, one per figure set
```
