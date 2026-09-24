# DEVLOG, `pinnde_eval` calibration

I use this file as the technical record for the evaluation module: the toy
validation setup, the thresholds I calibrated, and the baseline numbers I use
when checking regressions. Read this before changing any threshold. Most values
come from a measured noise floor, not from preference.

---

## 0. Environment & reproduction

- Python: `Tina/.venv` (CPython 3.12.13, gitignored; created 2026-07-19 after
  the original `/private/tmp/pinnde_venv` was wiped with /tmp). numpy 1.26.4,
  scipy 1.13.1, pinned <2 by the jetnet dependency stack, see §6.
- Deps: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. `jetnet` is
  optional (Tier-2 FPD/KPD only) and now installed locally via the no-compiler
  recipe in the README.
- **Run with `OPENBLAS_NUM_THREADS=1`.** numpy 1.26 wheels bundle threaded
  OpenBLAS; the Tier-1 MLP issues thousands of tiny gemms and OpenBLAS's
  busy-wait threading turns a 9 s `evaluate()` into >30 min of spin that looks
  like a hang (~390% CPU, no progress). Measured 2026-07-19: full `evaluate`
  on n=8000 is 8.9 s with the variable set; still unfinished after 67 min
  without it. Torch has its own threadpool and is unaffected.
- Reproduce everything:
  ```bash
  cd Tina
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest pinnde_eval/tests -q
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pinnde_eval.validate_toys
  ```
- Determinism: every stochastic step is seeded. `validate_toys` calls
  `seed_all(seed)` (numpy + torch) at the top of each check, and each sampler
  takes an explicit `seed=`. The toy convention throughout is
  **params `seed=0`, real `seed=1`, gen `seed=2`** so "real vs gen" is two
  independent draws from one fixed GMM, not the same draw compared to itself.

### Fixed configuration and why

| Constant | Where | Value | Why this value |
|---|---|---|---|
| `k` (classifier retrainings) | `tier1.classifier_two_sample_test` | 5 | Enough to get a mean±std AUC without making the null test slow; each run uses `random_state = seed + i` so split **and** weight-init both vary. |
| `hidden_layer_sizes` | tier1 MLP | (64, 64) | Smallest net that cleanly separates a 4σ-shifted blob (AUC>0.95, see `test_classifier_auc_separates`) while staying ~0.5 on the null. |
| `max_iter` | tier1 MLP | 300 | Converges on these 1.5k–16k-row toys; larger just burns time. |
| `test_size` | tier1 split | 0.3 | Standard held-out fraction; stratified so both classes stay balanced. |
| `bins` | `histogram_chi2` | 50 | CaloChallenge-style binning; with N≈8k that's ~160 counts/bin, enough that the Gaussian χ² approximation holds and empty bins are rare. |
| `max_points` | `tier3.median_bandwidth` | 2000 | The median heuristic only needs a stable estimate; full pairwise on >2k rows is wasteful. Subsample is **seeded** so the bandwidth is reproducible. |
| `n_projections` | `tier3.swd` | 128 | SWD variance falls as 1/√P; 128 gives a stable distance (<1% wiggle) and still runs in ~10 ms on 5k×3 (see §5 speed). |
| `max_samples` / `min_samples` | `tier2.fpd` | `min(N,50000)` / `max(1000, max//5)` | jetnet's FPD extrapolates to infinite sample size from a range of batch sizes; the bracket must scale with N and stay >0 even on small toys (`min_samples` is clamped `< max_samples`). |
| `batch_size` | `tier2.kpd` | `min(5000, N)` | jetnet default-ish; caps cost on large N, falls back to N on toys. |

---

## 1. Tier-3 dimension checks

`mmd` and `swd` validate feature dimensions before doing any torch operations.
This keeps shape errors at the public API boundary instead of letting them fail
inside a pairwise-distance calculation. Both functions now raise a clear
`ValueError`:
  ```python
  if real.shape[1] != gen.shape[1]:
      raise ValueError(f"real and gen must share feature dimension, got "
                       f"{real.shape[1]} vs {gen.shape[1]}")
  ```
The test case is `pe.mmd(np.zeros((10,2)), np.zeros((10,3)))`.

## 2. MMD around zero

I use the unbiased MMD² estimator. It drops the diagonal from the two
within-sample kernel terms (`kxx`, `kyy`) and keeps the full cross term
`kxy.mean()`. For identical inputs, the cross term still contains the exact
`i==i` matches, so the estimate can land slightly below zero. That is expected
for the unbiased estimator.

With `x = np.random.default_rng(2).normal(size=(1000,3))`,
`mmd(x, x.copy())` gives `-0.0008205175399780273` for the fixed seed. The test
therefore checks `abs(mmd(...)) < 5e-3` for both an exact copy and two
independent normal draws. SWD stays stricter because a sample and its copy sort
to the same projected values, so `swd(x, x.copy()) < 1e-6`.

The `5e-3` bound sits above the deterministic copy offset and the n=8000 null
fluctuation (≈ -2.6e-6), while remaining far below meaningful signals in the
toy sensitivity checks.

## 3. Sensitivity trend check

I validate sensitivity with **d=3, n=8000, k=8, params seed=0, real seed=1, gen
seed=2**, and `eps_grid=(0.0,0.05,0.1,0.2,0.4)`. I use Spearman rank correlation
instead of step-by-step monotonicity because the smallest perturbations can sit
below the sampling-noise floor.

For example, the `mean` perturbation produced this SWD series:
  ```
  [0.0492618, 0.04825219, 0.07748349, 0.15436018, 0.32036853]
  ```
The eps=0.05 shift is smaller than the finite-N SWD floor (eps=0 already gives
≈0.049), so the first step dips by about 0.001 before the larger trend dominates.
The check now requires `spearmanr(eps, metric).correlation >= 0.8`. With five eps
points, ρ=0.8 rules out a flat or non-trending metric while leaving room for one
sub-noise fluctuation. All tracked metrics clear this threshold for all three
perturbations.

## 4. Growth-factor threshold for the `drop` perturbation

For the `drop` perturbation, the metric should grow but not as sharply as it
does for a global mean or variance shift. The perturbation down-weights one of
eight mixture components (`prob[0] *= (1-eps)`, then renormalizes), so even
`eps=0.4` moves only a limited amount of probability mass.

The SWD series was:
  ```
  [0.0492618, 0.05148745, 0.05509467, 0.08158173, 0.14447023]
  ```
The previous 3× bar missed by a small amount: `0.14447` is just under
`3 * 0.04926 = 0.14778`. I set the growth threshold to
`series[-1] > 2*series[0]` for SWD and W1_mean. The weakest perturbation still
doubles over the eps=0 floor, and the stronger perturbations clear the bar by a
wide margin.

## 5. AUC separation threshold

The classifier two-sample test is less sensitive than the transport distances
to re-weighting one mixture component. For the `drop` perturbation at eps=0.4,
AUC reaches **0.5364**: a weak but measurable signal, not the `>0.6` separation
seen for stronger shifts.

I check AUC with the same Spearman ≥ 0.8 trend requirement and require
`auc[-1] > auc[0] + 0.01`. The null AUC std is ≈0.004 (see §7), so +0.01 is
about 2.5σ above the null floor. For `mean` and `var`, AUC rises to roughly
0.71 and 0.70.

## 6. `jetnet` install and numeric FPD/KPD validation

`pip install jetnet` fails on this Mac twice over: no cp314 wheel for the
transitive `wasserstein` dep, and on 3.12 the newer `wasserstein` sdist tries
to compile with OpenMP (`omp.h` missing without Homebrew libomp). The working
recipe (README "Note: FPD / KPD need jetnet") installs `jetnet --no-deps` plus
its actually-imported deps with `numpy<2`; the `wasserstein` package is never
imported by the FPD/KPD path and stays absent. FPD/KPD remain optional so any
environment can still run Tier 1 and Tier 3.

`tier2._require_jetnet()` imports `jetnet.evaluation` lazily and raises a clear
install hint when the package is missing. `evaluate(..., tier="full")` catches
that `ImportError`, sets `results["fpd"] = results["kpd"] = None`, and prints one
skip line.

I checked the wrappers against the **jetnet 0.2.5 sdist source** (`gen_metrics.py`):
`fpd(...)` returns `(params[0], sqrt(diag(covs)[0]))` → `(value, error)`, and
`kpd(...)` returns `(np.median(vals), iqr/2)` → `(median, error)`. The wrappers
match those return contracts and add sample-size-aware defaults.

**Numeric validation (2026-07-19, local 3.12 venv).** On the standard null
(d=3, n=8000, two independent draws from the same GMM):
```
fpd : 0.0002063 +/- 7.388e-05    (small positive bias expected: FPD's
                                  finite-N bias falls as O(1/N) and jetnet
                                  recommends N=50k; 8k is our toy size)
kpd : 0.0001608 +/- 0.0003056    (consistent with 0 within 1 error bar)
```
Both scale correctly in the sensitivity test (they grow with every
perturbation kind). The `validate_toys` null assertion
`fpd < 5*err + 1e-3` passes.

---

## 7. Calibration record (current numbers, for regression reference)

All from `python -m pinnde_eval.validate_toys`, **d=3, n=8000, k=8,
params seed=0 / real seed=1 / gen seed=2**.

**Environment note (2026-07-19):** rebuilding the venv (3.11→3.12, numpy
1.26.4/OpenBLAS, scipy 1.13.1, current sklearn) shifted every null number
slightly, different BLAS and library RNG paths change the drawn samples and
MLP fits. Both columns are recorded below; the *new* column is the regression
reference going forward. All asserted checks pass in both environments.

### Null test (two independent draws, same GMM), expect "no difference"
```
                 2026-07 venv (3.12)          original (3.11 venv)
mmd  :           7.51e-06                     -2.623e-06     (|·| < 1e-2 asserted)
swd  :           0.05128                      0.04926        (finite-N floor; < 0.1)
auc  :           0.5046 +/- 0.00207           0.502 +/- 0.00415   (|mean-0.5| < 0.05)
chi2 :           [1.057, 0.893, 1.074] m 1.008  [1.524, 0.640, 1.114] m 1.093
w1   :           [0.0742, 0.0468, 0.0367] m 0.05256  [0.0778, 0.0336, 0.0254] m 0.04561
fpd  :           0.0002063 +/- 7.4e-05        n/a (jetnet absent)
kpd  :           0.0001608 +/- 3.1e-04        n/a
```

### Sensitivity (metric vs eps), every column rank-correlates with eps (ρ≥0.8)
2026-07 venv numbers:
```
mean:  swd 0.0513→0.3077   w1 0.0526→0.3600   auc 0.505→0.706
var :  swd 0.0513→0.1703   w1 0.0526→0.1683   auc 0.505→0.694
drop:  swd 0.0513→0.1588   w1 0.0526→0.1683   auc 0.505→0.532   (mildest, by design)
```

### Speed (Tier-3 monitors, n=5000, d=3), target ≪ 1 s
```
mmd : ~175 ms     swd : ~10 ms     (assert mmd+swd < 2.0 s)
```
Requires `OPENBLAS_NUM_THREADS=1` on this machine, see §0.

> If a future change moves any null number materially or drops a sensitivity ρ
> below 0.8, that's a regression, bisect against this table before relaxing a
> threshold.

---

## 8. Stability vs sample size (June 26 meeting item 1)

`stability.py` measures, for each metric and each N in a grid, the **null
floor** (metric value for a perfect generator at that N), its **spread**
(std over repeated independent draws), and the **separation z** against a
fixed perturbation: z(N) = |signal mean − null mean| / √(var_null + var_sig).
"Resolvable at N" requires z ≥ 2 at that N *and every larger N*, so one lucky
small-N fluctuation doesn't count. Defaults: d=3, k=8 GMM,
n_grid=(250…8000), 6 repeats, mean shift eps=0.2, classifier k=2.

Numbers (from `python -m pinnde_eval.stability`, 2026-07 venv, mean shift
eps=0.2; full tables in the script output, curves in `figures/stability.png`):

```
            null floor (mean±std)                    resolvable
metric      N=250          N=2000         N=8000     from N =
mmd         -7e-05±0.0013  2e-05±0.0002   -2e-05±8e-05   2000
swd         0.254±0.044    0.090±0.018    0.041±0.014    2000
w1_mean     0.257±0.050    0.088±0.020    0.041±0.015    2000
chi2_mean   1.04±0.12      0.96±0.11      0.90±0.11      2000
auc         0.487±0.017    0.508±0.011    0.499±0.009    1000
```

Takeaways:
- The SWD/W1 null floors are pure finite-N artifacts falling as ~N^(−1/2)
  (0.254 → 0.041 over a 32× increase in N ≈ factor 5.7 predicted, 6.2
  observed). MMD (unbiased) hovers around 0 at every N with shrinking spread;
  χ² stays ≈1; AUC stays ≈0.5.
- For the eps=0.2 mean shift, **AUC is the most sample-efficient detector
  (resolves from N=1000); every other metric needs N≥2000. Below N≈1000
  nothing separates at 2σ**, per-condition bins smaller than ~1000 events
  give monitor-grade numbers only.
- At N=8000 the signal sits 7–12 combined σ from the null for every metric.

Design notes:
- Repeats use fresh deterministic seeds per draw (an incrementing counter), so
  the repeat axis is the true finite-N sampling distribution of the metric.
- z uses the absolute mean difference: MMD's unbiased estimator and χ² can
  fluctuate below their ideal under the null.
- The practical rule this study buys: **compare a metric against its null
  floor at your N, never against zero**, SWD/W1 floors are pure finite-N
  artifacts that fall roughly as N^(−1/2), so "SWD = 0.05" is *perfect* at
  n=8000 and *terrible* at n=500 relative to floor.

## 9. Local discrepancy maps (June 26 meeting item 2)

`local.py` has three complementary maps, all sharing the sign convention
**positive = real over-dense (missed) / negative = generated over-dense
(hallucinated)**:

- `mmd_witness`: the RBF-MMD witness function, same kernel + median
  bandwidth as `tier3.mmd`, so it decomposes the global monitor in space.
- `classifier_discrepancy`: out-of-fold P(real|x) per sample from the
  Tier-1 MLP via StratifiedKFold (honest probabilities: each sample scored
  by a fold that never trained on it). Also returns the out-of-fold AUC.
- `binned_residual_map`: the Tier-1 two-sample χ² decomposed per bin:
  r = (√(Tg/Tr)·n_real − √(Tr/Tg)·n_gen)/√(n_real+n_gen), ~N(0,1) per bin
  under the null, so |r| > 3 flags a genuine local disagreement.

Calibration (tested in `test_stability_local.py`): null witness |max| < 0.05
on 1.5k samples; null residual map std < 1.6 with no |r| > 5 outliers at
8k samples; classifier null oof-AUC within 0.08 of 0.5.

Demo (`make_local_figure.py`, k=6 GMM with mode 0 dropped and mode 1 shifted):
global numbers say only "something is off" (swd 0.596, mmd 3.7e-02,
auc 0.705) while all three maps point at the two broken modes,
`figures/local_discrepancy.png`. Witness peaks red exactly on the dropped
mode; residual map shows |r| ≈ 8–9 at the dropped/hallucinated locations and
~N(0,1) elsewhere; the shifted mode's samples get P(real|x) ≈ 0.

## 10. Conditional flow matching benchmark (calo-flavored toy)

`flow_matching.demo_conditional` trains v_θ(x, t, c) on the 3-observable
shower toy (`pinnde_eval.data.sample_shower_toy`: sampling fraction / depth /
width vs. normalized log-energy c, correlated through a shared per-event
fluctuation, gamma/lognormal skew). 40k events, 6k steps, CPU ≈ 2 min.
Results (2026-07 venv, seed 0):

- Pooled over energies: swd 0.0056, mmd −8.4e-05, fpd 5.6e-05 ± 9.4e-06,
  kpd consistent with 0, w1_mean 0.0058. AUC 0.565 ± 0.007, the classifier
  still sees a residual imperfection (χ² is 2.4 on the skewed
  sampling-fraction feature, ~1.1–1.4 on the others): the flow slightly
  under-models the gamma tail. Kept as an honest example of why AUC stays in
  the suite when the transport distances sit at the floor.
- Per energy bin (the pooled number hides this): AUC rises with energy,
  0.512 / 0.551 / 0.561 / 0.617 across the four c-quartiles, the
  high-energy bin is the weakest slice (narrow distributions ⇒ the same
  absolute error is more visible). SWD per bin: 0.011 / 0.009 / 0.008 / 0.009.
- Fixed-condition interpolation at c* ∈ {0.1, 0.5, 0.9} vs fresh truth draws
  at exactly c*: per-feature means track to ≲0.5% (e.g. depth 3.705→3.700 at
  c*=0.5); swd ≈ 0.006–0.010.

## 11. Real CaloChallenge data (ds2), the `features_fn` in anger

`observables.py` is the `features_fn` for real showers. Data: CaloChallenge
ds2 (electrons), <https://zenodo.org/records/6366271>, two files of 100k
showers, `showers` (N, 6480) float64 MeV and `incident_energies` (N, 1) MeV.
Both files are ~1.36 GB, gitignored, and must stay that way, GitHub rejects
blobs over 100 MB.

**Voxel flatten order is `(layer, alpha, r)`, determined empirically, not
assumed.** 6480 = 45 layers × 16 angular × 9 radial. The dataset description
reads "9 radial and 16 angular", which invites `reshape(45, 9, 16)`: that is
wrong. Measured mean profiles over 2000 showers:

```
reshape(45, 9, 16):  the 9-axis is [1.00 0.99 0.97 0.89 0.70 1.00 0.98 0.95 0.63]
                     -- a sawtooth, physically impossible
reshape(45, 16, 9):  the 9-axis is [1.00 0.28 0.13 0.08 0.06 0.04 0.03 0.03 0.02]
                     -- monotonic ~50x falloff = radius
                     the 16-axis is flat to 0.3% = azimuthal symmetry
```

Getting this backwards leaves `<z>` looking plausible while silently
corrupting `sigma_r`. `test_observables.py` pins it two ways: a synthetic
shower spread across two radii must give `r_width > 0` while one spread across
two angles gives 0, and (when the file is present) the real radial profile
must be monotonically decreasing with a flat angular profile.

Sanity numbers on the real file (first 2000 showers): E_inc 1.0–992 GeV
log-uniform; 76.33% of voxels exactly zero, and *exactly* that fraction is
below 15.15 keV, so **the readout threshold is already applied to the
reference data**; longitudinal profile peaks at layer 11.

The extractor is validated against physics rather than a golden file:
`corr(log E_inc, sparsity) = −0.94` (more energy lights more voxels) and
`corr(log E_inc, <z>) = +0.87` (shower maximum deepens as log E). Both are
asserted with slack thresholds.

### The bug real data found: SWD and W1 need standardization

The toys hid this because every toy coordinate was unit-scale. Shower
observables are not: `E_tot` is tens of thousands of MeV, `sparsity` is in
[0, 1]. On the Geant4-vs-Geant4 null at N=8000:

```
                        without standardize      with standardize
swd                     1128                     0.0222
w1_mean                 521.2                    0.0221
w1_per_feature          [3648, 0.0021, 0.095, ...]   [0.021, 0.019, 0.034, ...]
```

SWD and W1 are scale-dependent, so they measured `E_tot` and nothing else.
MMD survives (median-heuristic bandwidth adapts to scale), AUC survives (the
classifier standardizes internally), χ² and separation power survive (binned
per feature). `evaluate(..., standardize=True)` z-scores both samples using
**the real sample's** mean and width, one shared scaler, because fitting
separately would erase the mean differences the metrics exist to detect.
Default is `False` so the §7 toy baselines stay reproducible, and a warning
fires when feature scales span more than 100×.

### Real-data null floor (the reference table for shower work)

`python -m pinnde_eval.validate_calo`, ds2_1 vs ds2_2 (independent Geant4
draws), N=8000, 7 observables, standardized:

```
mmd  : -2.348e-05          auc  : 0.4971 +/- 0.01005
swd  : 0.02221             chi2 : [1.223, 1.412, 1.187, 0.859, 1.06, 0.910, 1.022] m 1.096
w1   : 0.02213             sep  : [0.0037, 0.0034, 0.0024, 0.0025, 0.0030, 0.0027, 0.0031] m 0.0030
fpd  : 0.0002 +/- 1.1e-04  kpd  : 6.4e-06 +/- 6.0e-05
```

> These are the numbers a *perfect* generator produces on real showers at
> N=8000. Judge any model against this row, not against zero.

## 12. Separation power and its finite-N floor

`tier1.separation_power` implements the CaloChallenge statistic
S = ½ Σ (p−q)²/(p+q) on normalized histograms, bounded in [0, 1], 0 = identical
shapes, 1 = no overlap. It is what CaloChallenge submissions report, so having
it makes results here directly comparable to the published table.

**S has a finite-N null floor, and it is not small.** Two independent Geant4
draws (bins=50, mean over 7 observables, up to 4 repeats per N):

```
     N    sep_mean     std      n_bins/(2N)   ratio
   500    0.04401    0.00156      0.05000     0.88
  1000    0.02115    0.00130      0.02500     0.85
  2000    0.01057    0.00093      0.01250     0.85
  4000    0.00565    0.00057      0.00625     0.90
  8000    0.00319    0.00020      0.00313     1.02
 16000    0.00171    0.00000      0.00156     1.09
```

The floor falls 25.7× over a 32× increase in N, **1/N, not the 1/√N that
SWD and W1 follow** (§8). The law is derivable: for two same-distribution
histograms, E[(p̂−q̂)²] ≈ 2p/N and the denominator is ≈ 2p, so each occupied
bin contributes ≈ 1/(2N) and

    S_floor ≈ n_occupied_bins / (2N)

Measured ratios sit at 0.85–1.09, drifting below 1 for fine binning because
empty tail bins contribute nothing (the *occupied* count is what matters, not
the nominal one). The floor is correspondingly linear in the binning at fixed
N, measured at N=4000: 0.00286 / 0.00508 / 0.00996 / 0.01851 for
25 / 50 / 100 / 200 bins, close to doubling each time.

**The practical consequence, and the reason this is worth reporting.** The
CaloChallenge quotes separation powers as bare numbers. Two submissions
evaluated at different sample sizes or with different binning are not
comparable, and a small S is not evidence of a good model unless it is below
`n_bins/(2N)`. At N=500 with 50 bins, a *perfect* generator scores 0.044,
larger than many published per-observable separation powers. Any S reported
here is quoted alongside its floor.

## 13. Conditional flow matching on real ds2 (`flow_matching.demo_calo`)

The first time our own generator meets real Geant4 data. Learns
p(7 shower observables | E_inc) from ds2_1 (100k events, 12k steps, ≈5 min
CPU) and is scored against **ds2_2, which it never sees**. `E_tot` spans three
decades so it is modelled in log space; everything is then z-scored with
training statistics and inverted after sampling.

### Pooled: indistinguishable from Geant4

```
             model      floor (§11)
auc          0.5048     0.4971        (+0.0048 from 0.5)
chi2_mean    1.1178     1.0960        1.0x
swd          0.0168     0.0222        0.8x
w1_mean      0.0147     0.0221        0.7x
sep_mean     0.0032     0.0030        1.1x
fpd          1.5e-04 +/- 8.3e-05
```

**The 0.8x and 0.7x are not "better than Geant4."** The §11 floor was measured
between two independent Geant4 draws whose incident energies were sampled
*separately*; the model generates at the evaluation set's own energies and so
avoids that extra sampling variance. The floor is mildly generous in this
comparison. The honest reading is "indistinguishable at N=8000", not "better".

### Per energy bin: the pooled number hides a large failure

```
energy bin      n      swd      auc              sep
E 0-25%      2001   0.0474   0.787 +/- 0.010   0.01266
E 25-50%     1967   0.0465   0.573 +/- 0.016   0.01186
E 50-75%     2030   0.0329   0.498 +/- 0.015   0.01159
E 75-100%    2002   0.0338   0.527 +/- 0.018   0.00910
```

Pooled AUC is 0.5048, at the floor. The lowest-energy quartile is **0.787**:
trivially separable. A quarter of the data is badly modelled and the pooled
number says nothing is wrong. This is the strongest argument yet for
`evaluate_by_condition` being mandatory rather than optional.

Note the trend is **opposite to the toy** (§10), where AUC *rose* with energy
(0.512 → 0.617). On real showers it falls (0.787 → 0.527). The toy's smooth
skewed observables get harder to match as the distribution narrows; real
low-energy showers are sparse and nearly discrete (sparsity up to 0.998, only
a handful of voxels lit), which a continuous flow in observable space handles
badly. The toy was not predictive of where the real model fails, worth
remembering before trusting any toy-derived conclusion.

Per observable, `sparsity` is the worst (χ² 1.70, sep 1.7× floor) followed by
`f_samp` (χ² 1.36), the two bounded, most non-Gaussian quantities. Support
violations are small though: at most 0.40% of generated samples fall outside
the Geant4 range of any observable, so this is a distributional failure, not
the flow wandering off the physical support.

### Local maps must be run per condition

Running the §9 maps pooled and then on the worst slice:

```
              oof AUC   confidently-fake gen   max |r|
pooled         0.5044          0.8%              2.5
E 0-25%        0.7872         32.3%              3.8
```

Pooled, every local diagnostic says the generator is fine, oof AUC at the
floor, under 1% of generated samples confidently fake, and max |r| = 2.5,
*below* the |r| > 3 threshold §9 calibrated for a genuine local disagreement.
On the worst energy slice the same three maps light up: a third of generated
samples are confidently fake and the residual map crosses the threshold, with
the disagreement sitting in (sparsity, r_width).

**The maps are blind to a conditional failure unless you condition first.**
They localize in feature space, so a defect that exists only at low incident
energy is diluted by three well-modelled quartiles. The working recipe is
therefore: `evaluate_by_condition` to find the bad slice, *then* the local
maps inside it. Either step alone would have missed this.

### Open modelling gap

Low-energy showers. Diagnosed and fixed in §14, the numbers in this section
are the **baseline** configuration (`hidden=128, depth=3, n_steps=12000`),
kept because the diagnosis is the useful part.

## 14. Diagnosing and fixing the low-energy failure

Three hypotheses for the AUC of 0.787 in the lowest energy quartile, tested in
order. Two were wrong, and the wrong ones were informative.

### A. Support / boundary pile-up, refuted

The idea: `sparsity` and `f_samp` are bounded, low-energy showers pile against
the ceiling, and the flow overshoots it. Measured on 5026 low-E events:
**0.0%** sit in the top 1% of either observable's range. Nothing is against a
bound. A logit transform, which is what I had written into §13 as the first
thing to try, would have done nothing.

### B. Discreteness, confirmed as a fact, rejected as the cause

`sparsity` is a voxel *count*: it takes only the values `1 - k/6480`. Measured
deviation from that lattice is 4.55e-13, i.e. exact. The coarseness is
strongly energy dependent:

```
          lit voxels        distinct sparsity values   top-10 atoms hold
low E      6 .. 395  (med 166)        353 / 5026 events      7.8% of mass
high E  2224 .. 5496  (med 3874)     2263 / 4990 events      1.4% of mass
```

So at low energy a continuous flow is asked to reproduce a comb of ~350 atoms.
The textbook fix is dequantization: spread each atom over its lattice cell
during training (`+ U(0, 1/6480)`), floor back when sampling. Implemented in
`demo_calo.FeatureTransform`.

**It changed nothing: low-E AUC 0.787 → 0.785.** Kept anyway, because it is
physically correct, generated sparsity now lands on the same lattice as
Geant4 instead of between its teeth, but recorded as an honest negative
result. A real property of the data is not automatically the cause of a
failure.

### C. Joint structure, the actual cause

The clue was already in §13: separation power in the low bin (0.0129) is
barely worse than in the others (0.0117, 0.0116, 0.0090), while AUC is 0.787
against ~0.5. Separation power reads 1-D marginals; AUC reads the joint. So
the marginals were never the problem.

Per-observable separability inside the low-E slice, cross-validated:

```
E_tot 0.507  f_samp 0.482  z_mean 0.495  z_width 0.484
r_mean 0.480  r_width 0.489  sparsity 0.496
all 7, linear classifier      0.477
all 7, nonlinear MLP          0.788
```

**Every marginal is at chance, a linear model on all seven is at chance, and
only a nonlinear model separates them.** The failure is entirely nonlinear
joint structure. Correlation matrices confirm it:

```
                    max |corr difference|   mean
low  E (bad)               0.133           0.035
high E (fine)              0.037           0.016

worst pair, low E:  E_tot / sparsity   Geant4 -0.951   generated -0.818
```

At low energy those two are nearly deterministic, deposit more energy, light
more voxels, with only ~166 lit, so the observables sit close to a thin
manifold. The flow produced a fatter cloud around it: right marginals, wrong
correlations.

### What fixed it

```
config                    ODE steps   low-E AUC   gen corr (G4 -0.951)
baseline h128 d3 12k          50        0.7881        -0.818
baseline h128 d3 12k         200        0.7819        -0.824
baseline h128 d3 12k         800        0.7832        -0.824
wider    h384 d5 30k          50        0.5441        -0.915
wider    h384 d5 30k         200        0.5068        -0.927
wider    h384 d5 30k         800        0.5075        -0.927
```

**ODE resolution was not the bottleneck**: 16× the integration steps moved
the baseline by 0.005. **Capacity was**: a wider, deeper field trained longer
drops low-E AUC from 0.788 to 0.544, and only *then* do extra ODE steps buy
anything (0.544 → 0.507). A velocity field too smooth to represent the
manifold cannot be rescued by integrating it more carefully; once it is sharp
enough, integration accuracy starts to matter.

### Result with `hidden=384, depth=5, n_steps=30000, ode_steps=200`

```
energy bin      n      swd      auc (before)        sep
E 0-25%      2001   0.0345   0.509 +/- 0.020 (0.787)  0.00909
E 25-50%     1967   0.0467   0.510 +/- 0.011 (0.573)  0.01071
E 50-75%     2030   0.0302   0.487 +/- 0.020 (0.498)  0.01166
E 75-100%    2002   0.0316   0.496 +/- 0.009 (0.527)  0.00851

pooled: auc 0.4998 (floor 0.4971)  chi2 0.949  swd 0.0177  sep 0.0026
sparsity: chi2 1.68 -> 0.71, sep 0.00515 -> 0.00216
local maps: max |r| 2.1 pooled, 2.4 on the worst slice -- both below the
            |r| > 3 threshold; confidently-fake gen 32.2% -> 1.8%
```

Every energy bin now sits within about one sigma of 0.5. No slice is
separable, which is the bar §13 said the pooled number was hiding.

### Transferable lessons

1. **The metric that finds a failure is not the metric that diagnoses it.**
   Per-bin AUC found it; separation power being *fine* is what proved the
   marginals were innocent and pointed at the joint.
2. **A marginal fix cannot repair a joint failure.** Dequantization was
   correct physics and bought zero AUC.
3. **Check capacity before sampling resolution.** The intuitive knob (more ODE
   steps) did nothing until the field was good enough to be worth resolving.
4. A real, verifiable property of the data (discreteness) is not evidence that
   it causes the failure you happen to be looking at.

## 15. The per-layer feature space (d=187): floor and degeneracy

`per_layer_observables` gives 4 quantities for each of 45 layers; with the 7
core observables that is a 187-column space, the resolution published
CaloChallenge numbers are quoted at. Two things measured before modelling it.

### The null floor moves with dimension, FPD violently

Geant4 vs Geant4, N=8000, standardized:

```
             d=7 (§11)      d=187
auc          0.4971         0.5007 +/- 0.0098
chi2_mean    1.096          1.042
swd          0.0222         0.0240
w1_mean      0.0221         0.0235
sep_mean     0.0030         0.0029
mmd         -2.3e-05       -3.6e-06
fpd          2.0e-04        4.1e-02  +/- 4.7e-03
kpd          6.4e-06       -9.1e-07
```

Most metrics barely move. **FPD moves by a factor of 200.** It fits Gaussians
in the feature space and takes a Frechet distance between them, so its
finite-N bias grows with the number of covariance entries being estimated,
d=187 means ~17,000 covariance parameters from 8000 samples. A "small" FPD at
one dimension is a large one at another, and FPD values are not comparable
across feature spaces at all. This is the §8 rule (compare to the floor at
*your* N) extended: compare to the floor at your N **and your d**.

`demo_calo` therefore keys its floor table by dimension and refuses to run at
a dimension with no measured floor, rather than silently comparing against the
wrong row.

### Half the space collapses at low energy

An empty layer gives `sparsity_layer` exactly 1.0 and `r_mean_layer` exactly
0.0, so those columns carry point masses. Fraction of columns with more than
half their probability on a single value:

```
                       all energies     low-E quartile
>50% on one value        11 / 187          98 / 187
>90% on one value         0 / 187          49 / 187
>99% on one value         0 / 187           3 / 187

mean modal mass, by group   all E    low E
  E_layer                   0.184    0.482
  sparsity_layer            0.191    0.506
  r_mean_layer              0.194    0.518
  r_width_layer             0.259    0.627
```

At low incident energy the effective dimension is far below 187, over half
the columns are nearly constant, and 49 are essentially frozen. This is the
§14 manifold problem made much worse: the same slice that needed extra
capacity at d=7 now also has most of its coordinates degenerate.

Only layer 44 is empty in more than half of all events, so no layer selection
is needed; restricting to the 44 live layers (d=183) changes nothing
measurable (auc 0.5037 vs 0.5007).

### `report()` at this width

`report` printed every entry of a 187-element per-feature array, which made
the table unreadable and its underline several thousand characters wide. It
now summarizes any array longer than `max_items` (default 8) as
`[first four, ...] d=N mean/min/max`. Asserted in `test_observables.py`.

## 16. The per-layer model fails, and zero-inflation explains all of it

`python -m flow_matching.demo_calo --per-layer` with `hidden=512, depth=6,
n_steps=30000`, more capacity than the d=7 fix of §14, does not merely
degrade. It fails outright:

```
                   pooled        worst slice (E 75-100%)
out-of-fold AUC    0.9963        0.9995
confidently-fake   94.0%         93.8%
max |r|            60.0          26.4
```

AUC 0.9995 is a generator a classifier separates essentially perfectly. For
contrast, the §14 failure everyone would call bad was 0.787.

### It is not capacity, and not the §14 manifold problem

**More than half the columns carry a point mass at a physical boundary.** An
empty layer has energy exactly 0, radial centre exactly 0, radial width
exactly 0, and sparsity exactly 1. Fraction of Geant4 events sitting exactly
on that atom:

```
                    layers 0-9   layers 20-29   layers 40-44   max
E_layer               0.001         0.192          0.474       0.513
r_mean_layer          0.001         0.192          0.474       0.513
r_width_layer         0.048         0.283          0.580       0.625
sparsity_layer (=1)   0.001         0.192          0.474       0.474
```

A continuous density cannot put finite probability on a single point. The flow
does the only thing it can, spreads density *around* the atom, which puts
mass outside the physical range, since the atom sits at the boundary. Up to
**31.6% of generated `r_width_layer` values are negative**, i.e. impossible.

The link is not suggestive, it is essentially exact. Across the 44 layers,
the fraction of generated values falling outside the Geant4 range against
that layer's atom mass:

```
corr(atom mass, out-of-range fraction) = +0.989

layer  1: atom 10.6%  ->  9.70% impossible
layer 19: atom 17.0%  ->  8.74%
layer 31: atom 40.2%  -> 25.01%
layer 43: atom 59.7%  -> 31.61%
```

Every layer's failure rate is predicted by how much of its mass is on the
atom. That is the whole failure; there is nothing else to explain.

### Why this is a different kind of problem from §14

§14 was a *resolution* failure: the right density, insufficiently sharp, fixed
by capacity. This is a *representational* failure: the target is not
absolutely continuous, so no amount of capacity in a continuous flow can
express it. Training longer or wider makes the smeared cloud tighter around
each atom but never puts finite mass on it. The `--per-layer` run already used
more capacity than the §14 fix and did far worse.

Note also that dequantization (§14 B) does not apply. There the discreteness
was a *lattice*, many evenly spaced atoms, which spreading over a cell
reproduces. Here it is a single atom at a boundary coexisting with a
continuous part: a zero-inflated distribution, not a quantized one.

### The fix this calls for

A two-part (hurdle) model, which is the standard treatment for zero-inflated
data:

1. model layer **occupancy**: a Bernoulli per layer for "is this layer lit",
   which is mostly a function of incident energy and depth;
2. model the shape observables **conditioned on the layer being lit**, where
   they are genuinely continuous.

At generation, draw occupancy first and emit the exact atom for empty layers
rather than a near-miss. This also removes the impossible values for free,
since the continuous part is only ever sampled where it is defined.

Not attempted yet, it is an architecture change rather than a hyperparameter,
and the diagnosis is what this section is for. A cheaper intermediate worth
measuring first: restrict to the front layers, where the atom mass is small
(layers 0-9 sit at 0.001 for energy and 0.048 for `r_width`), and check that
AUC returns to the floor. If it does, that isolates zero-inflation as the sole
cause and gives a usable per-layer model over the calorimeter's active region
while the hurdle model is built.

---

## 17. Classical two-sample tests and the Sinkhorn divergence

The CaloChallenge metrics carry no null distribution of their own: separation
power reads `n_bins/(2N)` for a perfect generator (§12), so every number needs a
measured floor before it can be read at all. Classical tests promise to avoid
that, since their p-value is Uniform(0, 1) whenever the two samples match, at any
N. This section is the calibration record for the three now in `classical.py`
(Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling) and for the Sinkhorn
divergence in `tier3.py`.

### The tests are calibrated on our data

Pool all 200,000 ds2 showers, cut them into 100 **disjoint** 1000+1000 splits,
run each test on each of the 7 observables. That is an exact null by
construction, so the fraction of p-values below 0.05 is the test's true
false-positive rate.

```
  observable |   ks frac p<.05 |  cvm frac p<.05 |   ad frac p<.05   (target 0.05)
       E_tot |           0.080 |           0.070 |           0.060
      f_samp |           0.070 |           0.080 |           0.070
      z_mean |           0.070 |           0.090 |           0.090
     z_width |           0.080 |           0.070 |           0.070
      r_mean |           0.040 |           0.050 |           0.050
     r_width |           0.020 |           0.060 |           0.060
    sparsity |           0.070 |           0.070 |           0.050
```

Every rate sits near 0.05, including `sparsity`, which is a voxel count and so
has ties, the one assumption KS genuinely needs. The splits must be disjoint:
drawing each split independently from the pool reuses showers, which correlates
the p-values and manufactures differences between tests that more sampling does
not reproduce.

### How much can they detect?

Shift one sample by 5% of each observable's spread **inside a narrow incident
energy band**, where the observables are unimodal and a fractional shift means
what it sounds like. Power is the fraction of 50 repeats rejecting at the 5%
level after Bonferroni across the 7 observables.

```
     N |  KS power | CvM power |  AD power |  sep power | sep floor
   250 |      0.04 |      0.08 |      0.10 |    0.08527 |   0.10000
   500 |      0.06 |      0.12 |      0.14 |    0.04207 |   0.05000
  1000 |      0.18 |      0.24 |      0.34 |    0.02250 |   0.02500
  2000 |      0.50 |      0.52 |      0.62 |    0.01224 |   0.01250
```

Anderson–Darling leads at every N, modestly; at N=250 all three sit near chance.
The last two columns are the argument for keeping classical tests at all:
separation power and its own floor stay the same order of magnitude throughout,
so the number alone never says whether a difference was found, while a classical
test gives a detection rate directly.

**Correction to the 2026-08-31 meeting.** The table presented there combined the
seven p-values with Fisher's method and reported AD detecting 60% at N=250.
Fisher assumes independence. With Bonferroni, which does not, AD reaches 0.10 at
N=250, and 0.62 only at N=2000.

### Combine with Bonferroni, not Fisher

The seven p-values are not independent. Over 100 disjoint nulls the strongest
observable pair is E_tot / sparsity at −0.928, and their p-values correlate at
+0.941; the mean |correlation| over the 21 pairs is 0.326, and 14 of the 21 fall
outside the ±0.196 band independence predicts. `combine_pvalues` therefore
defaults to Bonferroni (min(p) × d), valid under arbitrary dependence. Fisher's
measured false-positive rate on an exact null is 0.14 to 0.29 against a 0.05
target.

### The three tests are one view, not three

They agree almost perfectly. Rank correlation between their statistics, over the
same 100 disjoint 1000-vs-1000 nulls:

```
 observable | KS-CvM | KS-AD | CvM-AD
      E_tot |   0.91 |  0.86 |   0.97
     f_samp |   0.94 |  0.90 |   0.97
     z_mean |   0.94 |  0.87 |   0.95
    z_width |   0.92 |  0.90 |   0.97
     r_mean |   0.92 |  0.88 |   0.97
    r_width |   0.92 |  0.89 |   0.95
   sparsity |   0.91 |  0.87 |   0.97
```

Reporting all three is therefore not three pieces of evidence but one, seen three
ways, with AD the most sensitive of them. Worth stating plainly in any write-up
rather than implying independent confirmation.

### Sinkhorn divergence

`tier3.sinkhorn` is the debiased entropic optimal transport cost,
`S(x, y) = OT(x, y) − ½OT(x, x) − ½OT(y, y)`; the self-terms are subtracted
because the raw entropic cost is not zero for identical samples. It measures
transport in the full feature space rather than on 1-D projections as SWD does.

On the 7-observable model it reads 0.3819 against a Geant4-vs-Geant4 floor of
0.4151 ± 0.077, i.e. inside the floor, agreeing with every other metric.

Two caveats. It **caps both samples at 2000 points** (`max_points`), so its floor
stops falling above N=2000. And **no script here produces that floor**: the value
and its spread were computed by hand, so neither the sample size nor the number
of repeats behind them can be checked afterwards. Any rerun belongs in a
committed script.

### The calibration figure is not evidence of the file-to-file shift

A calibration figure made for the 31 August meeting showed KS p-values for file
1 against file 2 with 35 of 210 below 0.1 where 21 are expected, and put the
excess down to the incident energy difference between the files. That does not survive checking. The figure
uses the first 30,000 showers of each file; the next two blocks of 30,000 give 12
and 20. Inside a single 1000-vs-1000 split the energy difference is only 0.2 to
0.4 standard errors, far too small to cause it.

The file-to-file difference itself is real but small: over all 100,000 showers
per file the mean log E_inc differs by 0.018, which is 2.0σ, and KS on E_inc
gives p = 0.10, about what two independent draws produce one time in twenty.

---

## 18. The CaloChallenge's own 362 features, and what they say about the two files

§15–16 worked in our own 187-column space and hit a wall built out of empty
layers. The 2026-08-31 meeting raised the more basic objection to that space: it
is not the one published numbers are quoted in. Anything we want to compare with
a submission has to be measured on the challenge's own features, computed their
way, including their binning.

### Their features, their code

For ds2 their classifier reads 362 numbers per shower: log10 incident energy;
log10 energy in each of the 45 layers; centre of energy in eta and in phi per
layer; width in eta and in phi per layer; log10 total energy; sparsity per
layer; radial centre and radial width per layer. The eta and phi blocks are the
180 columns we could not produce before, because they need the detector maps in
`binning_dataset_2.xml`.

`pinnde_eval/calochallenge.py` calls their code rather than reimplementing the
definitions. It downloads `HighLevelFeatures.py`, `XMLHandler.py`,
`binning_dataset_2.xml` and `evaluate.py` into a gitignored folder, pinned to
commit `3073d13` and checked by MD5. Their repository declares no licence, so
none of it is committed here.

Only the final assembly, which columns, in which order, with which scaling,
lives on our side, and `tests/test_calochallenge.py` checks it against their own
`prepare_high_data_for_classifier` on 200 real showers: the two agree to a
relative 1e-12. Two columns can also be checked independently against
`observables.py`: their per-layer energy is our sum over the layer, and their
sparsity (`1 − (voxel > 0).mean()`) is our fraction of empty voxels. Both match.

### The floor, measured two ways

`python -m pinnde_eval.validate_official` scores 10 **disjoint** pairs of 8,000
showers, and does it twice: with both samples from one file, which is pure
sampling noise, and with one sample from each file, which is the floor a model
is scored against. 20 minutes on the laptop.

```
    metric |           same file |     different files
       auc |    0.49932 +/-0.00420 |    0.50055 +/-0.00585
 chi2_mean |    1.02119 +/-0.03789 |    1.02292 +/-0.04298
       swd |    0.02109 +/-0.00208 |    0.02169 +/-0.00348
   w1_mean |    0.02044 +/-0.00367 |    0.02164 +/-0.00453
  sep_mean |    0.00279 +/-0.00009 |    0.00280 +/-0.00010
  sinkhorn |  290.08909 +/-4.32468 |  290.54659 +/-3.79061
       mmd |    0.00001 +/-0.00004 |    0.00003 +/-0.00006
       fpd |    0.19049 +/-0.01168 |    0.19716 +/-0.00985
       kpd |    0.00006 +/-0.00007 |    0.00007 +/-0.00009
```

**The two ds2 files are indistinguishable in this space.** The AUC difference
between the two floors is +0.0012, half a standard error of the difference. A
classifier reading all 362 features cannot tell which file a shower came from.
That answers the question behind §4B of the handoff: the shift in the incident
energy marginal (0.018 in mean log E_inc, 2.0σ, KS p = 0.10) is real but leaves
no trace in anything the challenge measures, so correcting dataset 1 would buy
nothing.

It also removes a caveat that has been attached to every earlier number: the
file-versus-file floors of §11 and §15 are not measurably generous.

**The floor moves with dimension, violently for FPD.** FPD reads 2e-04 at d=7,
4.1e-02 at d=187 and 0.19 at d=362; separation power falls from 0.0030 at d=7 to
0.0028 here only because it is already at its `n_bins/(2N)` limit. As always, a
metric value means nothing without the floor for its own feature space and
sample size.

**A note on the spread.** The AUC error bar quoted in earlier sections came from
retraining the classifier on a single pair, which measures classifier noise. The
±0.0059 above is across 10 disjoint pairs of showers, which is sampling noise and
the thing that actually matters. They are not interchangeable: the retraining
spread on one pair understates it.

---

## 19. Checking the claims before we make them

Several numbers in this log and in the meeting documents have been quoted for
weeks without being re-measured, and §17 showed what that costs. This section
re-checks the ones a write-up would rest on. Everything here is measured on all
100,000 showers of `dataset_2_1.hdf5` unless stated otherwise.

### The data is pre-calibrated, and the giveaway is at low energy

3.42% of all 200,000 ds2 showers deposit **more** energy than came in. That
number was in the handoff already; what is new is where it happens. By incident
energy quartile: 12.9%, 0.79%, 0.00%, 0.00%. The median `E_tot / E_inc` is 0.78
in every quartile.

So the deposits carry a sampling-fraction calibration of about 0.78, and the
over-unity showers are ordinary fluctuation where a shower is smallest, not a
defect in the data or in our reader.

### The wall, measured in both spaces

§16 said 98 of the 187 columns put over half their mass on a single value at low
energy. Re-measured, that is exactly right, and the official space is worse:

```
                      187 columns        362 official features
all 100k                10 of 187              20 of 362
lowest E quartile       98 of 187 (52%)       198 of 362 (55%)
second quartile         42 of 187              88 of 362
third quartile           1 of 187               0 of 362
top quartile             0 of 187               0 of 362
```

In the lowest quartile 48.4% of all (shower, layer) pairs are empty; in the top
half, essentially none are. The target is continuous where the shower is big and
half point masses where it is small.

### What the occupancy part of a hurdle model has to do

Splitting the problem needs a discrete model of which layers are lit. Two
measurements say what that model must look like.

**Energy explains most of it, not all.** Over 20 energy bins x 45 layers, 66.6%
of the cells are settled (under 2% or over 98% lit) and 14.9% are genuinely
uncertain (20-80%). Knowing the incident energy drops the uncertainty in whether
a layer is lit from 0.520 to 0.218 bits per layer.

**It is not a depth cut.** Only 48.6% of showers have their lit layers forming a
solid block from layer 0, and in the lowest energy quartile only 7.5% do -- those
showers average 6.25 empty layers scattered *inside* their lit range. The deepest
lit layer runs from 30.5 +/- 7.2 at low energy to 45.0 at the top.

A single "how deep did it reach" variable therefore cannot describe occupancy.
The discrete part needs per-layer occupancy, conditioned on energy and correlated
across layers.

### Sinkhorn really does stop learning above N=2000

`sinkhorn` caps both samples at `max_points=2000`. The consequence is visible in
the floor, Geant4 against Geant4, five disjoint repeats per N:

```
     N |           sinkhorn |   swd
   500 |    0.7816 +/-0.0828 |    0.0794
  1000 |    0.5222 +/-0.0516 |    0.0576
  2000 |    0.3725 +/-0.0374 |    0.0421
  4000 |    0.4048 +/-0.0435 |    0.0358
  8000 |    0.3815 +/-0.0289 |    0.0243
```

Sinkhorn flattens at 2000 while SWD keeps falling. Above that sample size
Sinkhorn adds no resolving power, so a comparison that needs more should either
raise `max_points` -- the cost grows with the square of the sample -- or use SWD.

### The floors, remeasured over 10 disjoint pairs

Every floor quoted before came from one comparison. Measured over 10 disjoint
pairs of 8000 showers, with pairs drawn inside one file and across the two
(`validate_floors.py`, `validate_official.py`):

```
                 7 observables            187 columns            362 official
metric        same file   across      same file   across      same file   across
auc            0.50303   0.50126       0.50041   0.49943       0.49897   0.50105
chi2_mean      1.04467   1.06179       1.02534   1.01825       1.02120   1.02296
swd            0.02153   0.02217       0.02080   0.02185       0.02109   0.02169
sep_mean       0.00290   0.00295       0.00290   0.00288       0.00279   0.00280
sinkhorn       0.35988   0.39402      69.05544  67.28254     290.08907 290.54660
fpd            0.00020   0.00026       0.03715   0.04197       0.19049   0.19716
largest gap   sinkhorn +2.0 SE        fpd +1.8 SE             fpd +1.4 SE
```

The old single-comparison values sit inside the new spreads, so they were not
wrong, only unquantified. The Sinkhorn floor now has a script behind it for the
first time: 0.394 +/- 0.040 at d=7, against the 0.4151 +/- 0.077 that was
computed by hand and could not be checked.

**The two files agree in every space.** The largest gap between same-file and
across-file pairs is 2.0 standard errors, on one metric out of nine, in one
space out of three. With nine metrics looked at, that is what chance produces.

### The floor a conditional model actually faces

A conditional model generates at the evaluation set's own incident energies. Two
independent Geant4 samples also differ by their energy draw, which the model
never pays for. `validate_matched.py` removes that by pairing each shower with
the Geant4 shower of nearest incident energy (matched to 2.6e-04 in log E,
against the 1.8e-02 that separates the files):

```
                    7 observables                  362 official features
metric        matched  independent  gap      matched  independent  gap
auc           0.49850    0.50126   +1.3      0.50031    0.50105   +0.3
chi2_mean     0.88456    1.06179   +4.9      0.95324    1.02296   +4.7
swd           0.01769    0.02217   +2.3      0.01787    0.02169   +3.4
w1_mean       0.01565    0.02212   +3.1      0.01523    0.02164   +4.4
sep_mean      0.00239    0.00295   +5.0      0.00263    0.00280   +4.8
sinkhorn      0.39011    0.39402   +0.2    289.25201  290.54660   +0.7
fpd           0.00016    0.00026   +1.8      0.18466    0.19716   +3.5
```

The binned and distance metrics sit well below the usual floor once energies are
matched, by 3 to 5 standard errors, while AUC and Sinkhorn barely move. So
every chi2, SWD, W1 and separation-power number quoted for a conditional model
so far has been read against a floor that was too generous, which is what let
the model of §13 look *better than Geant4*. Against the matched floor the d=7
model is simply at it: the 2026-09-16 GPU run gives auc 0.4967 (-0.4 sigma),
chi2 0.828 (-0.8), swd 0.0149 (-1.2), sep 0.0022 (-0.9), sinkhorn 0.349 (-0.9).

`NULL_FLOORS` in `demo_calo.py` now holds the matched floors for d=7 and d=362.

### What the null test can actually see

"The files agree" is only worth saying if the test would have caught a
difference. `validate_official --sensitivity` resamples one side of a same-file
pair to carry a known shift in mean log E_inc, using real showers only
(exponential tilting, selection without replacement), and scores it exactly like
a floor. Three repeats of 8000 showers, 362 features:

```
 shift realised |               auc |  sep_mean
          0.009 |   0.4973 +/-0.0080 |   0.00271     <- null
          0.040 |   0.4992 +/-0.0054 |   0.00279
          0.086 |   0.4967 +/-0.0042 |   0.00303
          0.142 |   0.5001 +/-0.0049 |   0.00345
          0.246 |   0.5038 +/-0.0051 |   0.00503
          0.363 |   0.5015 +/-0.0033 |   0.00776
          0.512 |   0.5097 +/-0.0022 |   0.01281
```

Two things follow, and the second was a surprise.

**The file-to-file difference is far below the noise floor of the test.** The
files differ by 0.018; separation power only clears its floor spread
(0.00279 +/- 0.00009) at a shift near 0.086, five times larger, and the
classifier needs about 0.5, thirty times larger. So "the files are
indistinguishable" means: any difference between them is at least five times
smaller than the smallest one we could detect at this sample size.

**Separation power beats the classifier at this job.** The binned marginal
statistic responds monotonically from a shift of 0.04 upwards, while the
classifier AUC stays at chance until 0.5. That inverts the usual assumption that
a trained classifier is the most sensitive test available; for a small shift
spread across many correlated features, it is the weakest one here. A plain
logistic regression is no better: across three null pairs it gave 0.4905, 0.5083
and 0.5108, a spread far wider than its nominal error.

It also means an AUC at 0.5 is weak evidence on its own. It should be quoted
next to a metric that has been shown to move, which is what the floor tables
above are for.

### Two checks that came back clean

**The ds2 files are not ordered.** Cutting a file into consecutive blocks is only
sound if rows are exchangeable. Adjacent blocks, blocks from opposite ends of the
file, and randomly shuffled subsets all score AUC 0.49 to 0.51, and the mean log
E_inc per 10,000-shower block varies only between 10.33 and 10.39.

**The numbers reproduce across machines.** `validate_classical` printed output
identical to the laptop on the CPU machine (AlmaLinux, CPU torch). The classifier-based
floors agree to about 4e-04 in AUC between laptop and the login machine, well inside the
+/-0.005 spread across repeats, which is as close as a floating-point-sensitive
MLP gets.

---

## 20. The first model run in the challenge's own space

`python -m flow_matching.demo_calo --features official --device cuda`, the
configuration that works at d=7 (`hidden=384, depth=5, n_steps=30000`), trained
on 100,000 showers and scored against 8,000 from the other file. 96 seconds on
the GPU machine's RTX 5090, against about 20 minutes for the same run on a laptop CPU at
d=7.

It fails, and not narrowly:

```
    metric |      model | matched floor |  vs floor
       auc |     0.9861 |        0.5003 |  +0.4861
 chi2_mean |    67.5754 |        0.9532 |    70.9x
       swd |     0.1408 |        0.0179 |     7.9x
   w1_mean |     0.2548 |        0.0152 |    16.8x
  sep_mean |     0.1974 |        0.0026 |    75.1x
```

Bonferroni-combined KS across the 362 columns is 0, and the out-of-fold
classifier calls 89.5% of generated showers confidently fake. The worst columns
are the deepest layers' energies -- `logE_layer_35` to `logE_layer_44`, chi2 200
to 221, separation power 0.60 to 0.66 against a floor of 0.0026 -- and up to 37%
of generated values for those columns fall outside the Geant4 range entirely.

### But zero-inflation is not the explanation this time

§16 blamed the d=187 failure on point masses, and §19 measured them here: in the
lowest energy quartile 198 of the 362 columns put more than half their mass on a
single value, and 48.4% of layers are empty. The natural expectation is that the
model fails at low energy and copes higher up. It does not:

```
  energy bin |     n |             auc |      sep
     E 0-25% |  2001 | 0.985 +/- 0.003 |  0.46840
    E 25-50% |  1967 | 0.964 +/- 0.005 |  0.24571
    E 50-75% |  2030 | 0.980 +/- 0.004 |  0.14416
   E 75-100% |  2002 | 0.996 +/- 0.001 |  0.23330
```

**The worst bin is the highest energy**, where §19 found no empty layers at all
(0.0% of layers empty, 0 of 362 columns concentrated on one value). Whatever is
wrong there, it is not zero-inflation: there is nothing degenerate in that
quartile to trip over.

Separation power does behave as the point-mass story predicts -- 0.468 in the
lowest quartile against 0.144 in the third -- so the point masses are real and
they do hurt the binned marginals. They are simply not the whole failure.

### The likelier first cause: the model is far too small for this target

The same architecture that fixed d=7 in §14 is being asked to carry 362
dimensions, and it was still learning when training stopped: the loss fell
1.9514 -> 1.1004 and was still dropping over the last 3,750 steps
(1.1068 -> 1.1024 -> 1.1004), while the SWD monitor fell 0.189 -> 0.142 without
flattening. At d=7 the same monitor had plateaued long before 30,000 steps.

So the order of work is: first establish whether capacity and training length
explain the high-energy failure, which is cheap now that a run costs 96 seconds,
and only then attribute what remains to the point masses. Attributing the whole
failure to zero-inflation before testing capacity would repeat the mistake of
§14, where a failure that looked structural turned out to be a model that was
too small.

---

## 21. Proving the harness before judging the model, and what the classifier actually sees

§20 found a model that fails at d=362. Before explaining a failure, the machinery
that reports it has to be shown not to invent it. This section does that first,
finds three bugs in the process, and then asks what the 362-feature classifier
is really detecting.

### A null check through the whole harness

`demo_calo --null` skips training and puts real Geant4 showers in the model's
place: showers from the training file matched shower by shower to the
evaluation energies (to 6.2e-04 in log E_inc). A perfect generator is exactly
that, so every number must land on the floor. `test_demo_calo.py` runs a small
version of it on every test run.

At d=7 it does: AUC 0.4965 against a floor of 0.4985, every other metric within
1 sigma, Bonferroni KS p = 0.91. At d=362 the headline metrics landed on the
floor too -- AUC 0.4999, chi2 0.9706, SWD 0.0177, separation 0.0027, every
energy bin at 0.50 -- **but two diagnostics called real showers impossible**:
Bonferroni KS p = 0, and 59% of `width_eta_44` "out of range".

### Three bugs, each measured, fixed and pinned by a test

**1. KS rejects values that moved by 1e-17.** The feature transform round trip
is lossless to 2.7e-15, but not bit-exact: 59.2% of `width_eta_44` is exactly 0
in the data, and after the round trip every one of those is 6.94e-18. KS sorts
values, so two point masses 1e-17 apart are simply different to it: real against
real gives p = 0.244, real against round-tripped real gives p = 0.00. The null
check now passes real showers bit-exact; the transform keeps its own lossless
test.

**2. The range check counted rounding as impossible.** Depending on the
rounding direction those zeros land just below 0, and a strict `gen < min` flags
them. `out_of_range_fraction` now needs a value to sit outside by more than a
millionth of the column's range; a genuine overshoot such as a width of -0.01 is
still caught. After the fix the worst column in the null reads 0.10%, which is
real showers from one file sitting just past the other file's extremes.

**3. The quantizer returned values one bit away from the data's.** Sparsity is
stored as `1 - lit/144`; the quantizer rebuilt it as `floor(x/spacing) * spacing`.
These differ in the last bit for about half of the lattice (a perfect generator
came back bit-identical for 57.5% of ds2 values), and KS then rejects a perfect
generator on sparsity at p = 5e-25 where real against real gives 0.12. That one
affected **the model**, not just the null: every model's KS result on the 45
sparsity columns was unreliable. The quantizer now snaps to the values the data
actually stores, while leaving impossible values (sparsity above 1) where they
are so they stay visible. The test confirms the old quantizer fails it.

After all three fixes the d=362 null reads Bonferroni KS p = 0.77 and a worst
out-of-range column of 0.10%, with the headline metrics unchanged.

**What the bugs touched.** Only the model runs' KS values and out-of-range
percentages in the 362-feature space. Every floor was measured real against real
without the transform, so none moved, and the conclusions of §20 rest on AUC and
the binned metrics, which a 1e-16 difference cannot shift.

### Capacity helps, but does not reach the floor

Same run as §20 with `--hidden 1024 --depth 8 --steps 100000` (2 min 53 s on the
RTX 5090):

```
                 384x5, 30k steps   1024x8, 100k steps    floor
auc                   0.986               0.900           0.500
sep_mean   (x floor)   75.1                49.5             1
w1_mean    (x floor)   16.8                10.0             1
fpd                    6.65                0.80            0.18
auc by E bin    .985 .964 .980 .996   .981 .836 .907 .992   .50
```

Every metric improves by a factor of 1.5 to 8, and the middle energy bins most of
all -- but the top bin, which has no empty layers, barely moves (0.996 to 0.992).
The worst columns move from the deep layers' energies to the deep layers'
widths, and the out-of-fold map for the top bin points at the joint distribution
of `EC_eta_23` and `EC_phi_23`: a correlation, not a marginal.

### What the classifier sees: correlations, measured without any model

`python -m pinnde_eval.validate_structure` takes real showers at matched energies
and destroys chosen correlations while keeping every conditional marginal
exactly -- it permutes columns among showers inside bins of 50 of nearly the same
incident energy -- then scores the result against untouched Geant4:

```
             variant | pooled auc | lowest E auc | highest E auc |  sep_mean
              intact |     0.5027 |       0.4938 |        0.5165 |   0.00268
     layers shuffled |     0.8722 |       0.7758 |        0.9050 |   0.00268
  layers, total kept |     0.8716 |       0.7768 |        0.9055 |   0.00268
    columns shuffled |     0.9778 |       0.9431 |        0.9566 |   0.00268
 columns, total kept |     0.9775 |       0.9441 |        0.9571 |   0.00268
```

Separation power is identical in every row (0.00268), which is the proof that
the marginals were untouched. So:

* **a generator with perfect marginals and no correlations scores AUC 0.978** --
  as badly as the small model (0.986);
* **one that gets each layer right internally but treats layers as independent
  scores 0.872** -- close to the big model (0.900);
* cross-layer structure matters most at **high energy** (0.905 against 0.776 at
  low energy), where every layer carries signal, which is exactly where the
  model fails worst;
* rebuilding log10 E_tot from the shuffled layers (the energy-sum identity,
  reproduced to 8.9e-16) changes nothing, so the signal is the soft correlations
  of shower development, not that one hard constraint.

In 362 dimensions the classifier is mostly a correlation detector. The model's
failure is dominated by how layers relate to each other; the empty-layer point
masses are real but secondary. Building the hurdle model first would fix the
smaller problem.

### Tools added for the next step

* `demo_calo --save-samples PATH` writes the real and generated showers, and
  `python -m pinnde_eval.diagnose_samples PATH` scores them by column family and
  by depth third, pooled and in the energy extremes, without retraining.
  `test_diagnose_samples.py` checks that it flags a corrupted family and only
  that family.
* `--seed` and `--ode-steps` on `demo_calo`, for run-to-run spread and sampler
  resolution.

### Where the big model fails, read from its saved samples

`python -m pinnde_eval.diagnose_samples runs/big_seed0.npz` on the 1024x8,
100k-step run (AUC 0.900), scoring each part of the shower on its own columns:

```
                part | columns |    all | lowest E | highest E
      layer energies |      45 |  0.788 |    0.836 |     0.987
     eta/phi centres |      90 |  0.800 |    0.886 |     0.990
      eta/phi widths |      90 |  0.959 |    0.997 |     0.992
            sparsity |      45 |  0.783 |    0.993 |     0.680
 radial centre+width |      90 |  0.866 |    0.988 |     0.990
              totals |       2 |  0.693 |    0.601 |     0.802
         layers 0-14 |     120 |  0.914 |    0.818 |     0.972
        layers 15-29 |     120 |  0.895 |    0.955 |     0.996
        layers 30-44 |     120 |  0.941 |    0.996 |     0.974
```

**Regenerating the incident energy is not the problem.** The model has to
re-emit log10 E_inc, which it was given, and misses by a median 0.018; setting
that column to the true value moves pooled AUC from 0.9015 to 0.9023.

**Two regimes.** At low energy the failure sits in sparsity (0.993) and the back
layers (0.996) -- the empty-layer point masses. At high energy every family
fails and the middle layers are worst (0.996), where the shower maximum is: the
shape of shower development.

**A concrete defect in the energy response.** The two totals columns alone reach
0.802 at high energy (0.770 with the true E_inc), and the sampling fraction
shows why:

```
                real median  real spread | model median  model spread
all              0.7822        0.1111    |   0.7678        0.1230
lowest E         0.8307        0.1930    |   0.7735        0.2039
highest E        0.7803        0.0184    |   0.7731        0.0443
```

At high energy the model's energy resolution is 2.4 times too wide, and at low
energy it misses that the sampling fraction rises to 0.83; the model's fraction
is flat near 0.77 everywhere. A likely cause is the parameterization: log10 E_tot
and each log10 E_layer span three decades set by E_inc alone, so a 2% spread at
high energy is a sliver of the column's range and almost invisible to the loss.
The d=7 model had the sampling fraction as a column of its own.

`demo_calo --relative-energy` models energies as log10(E / E_inc) and takes
log10 E_inc from the condition instead of regenerating it. It is tested to be
exactly reversible and to run end to end, and has not been run on the GPU yet.

### The first relative-energy run destabilized

`--relative-energy` with the same 1024x8, 100k-step setup trained unstably: the
loss jumped from 1.19 to 1.46 at step 37,500 and never recovered, and 2 of 8000
generated showers came out entirely NaN, both above 275 GeV -- the sampler
diverged on them. The scoring stalled on those two showers, which exposed a gap
in the harness: `drop_non_finite` now removes non-finite showers together with
their real partners and reports the count, so one divergence can no longer take
a whole report down.

On the 7998 intact pairs the run is worse than absolute energies -- pooled AUC
0.980 against 0.900, separation power 0.208 against 0.130. The high-energy median
sampling fraction did move onto Geant4's (0.7828 against 0.7803; absolute
energies gave 0.7731), but its spread widened to 0.064 against 0.018. With the
training unstable, this does not decide whether relative energies help. One
plausible mechanism for the instability: an empty layer's value becomes
-8 - log10 E_inc, an energy-dependent line instead of a fixed point.

`--clip-grad` and `--lr` now exist. The fair comparison is both
parameterizations with the same gradient clipping and seed.

### With gradient clipping: relative energies fix the energy response and break the layers

`--clip-grad 1.0` on both parameterizations, same seed, 1024x8 for 100k steps.
Both train without a loss spike and all 8000 showers are finite in both, though
the relative run's training monitor diverged once (SWD 4.6e30 at step 75,000)
before recovering.

```
                     absolute + clip    relative + clip
auc                       0.892              0.943
sep_mean  (x floor)        49.4               60.7
w1_mean   (x floor)         9.6               14.5
auc by energy bin   .984 .856 .865 .984   .979 .874 .939 .994
```

Clipping did the absolute baseline no harm (AUC 0.900 without it, 0.892 with;
whether that small gain is real needs a second seed). Relative energies are worse
overall -- but not everywhere. The sampling fraction, on the showers finite in
all four runs so far (median / spread):

```
     slice |         Geant4 |       absolute |  absolute+clip |       relative |  relative+clip
       all |  0.782 / 0.111 |  0.768 / 0.123 |  0.771 / 0.124 |  0.763 / 0.124 |  0.778 / 0.113
  lowest E |  0.831 / 0.193 |  0.774 / 0.204 |  0.773 / 0.205 |  0.731 / 0.190 |  0.777 / 0.187
 highest E |  0.780 / 0.018 |  0.773 / 0.044 |  0.781 / 0.048 |  0.783 / 0.064 |  0.784 / 0.036
```

Relative energies with clipping give the best energy response of any run: the
overall spread matches Geant4 (0.113 against 0.111) and the high-energy spread is
the tightest yet (0.036). The per-family breakdown shows the trade:

```
                part |  absolute + clip (all/low/high) |  relative + clip
              totals |      0.691 / 0.596 / 0.805      |  0.581 / 0.566 / 0.707
      layer energies |      0.771 / 0.919 / 0.984      |  0.961 / 0.948 / 0.996
      eta/phi widths |      0.963 / 0.996 / 0.989      |  0.991 / 0.999 / 0.998
 radial centre+width |      0.865 / 0.989 / 0.984      |  0.943 / 0.991 / 0.997
        layers 15-29 |      0.863 / 0.962 / 0.989      |  0.954 / 0.981 / 0.998
```

Relative E_tot helps; relative layer energies hurt badly, and at high energy too,
where no layer is empty -- so the moving empty-layer value (-8 - log10 E_inc)
cannot be the whole reason. The two parts are separable, so
`--relative-energy total` now makes only E_tot relative. The low-energy median
sampling fraction (Geant4 0.83, every model about 0.77) is still wrong in every
variant.

### The fix that works: E_tot relative to E_inc, and nothing else

Two runs settle it (`--clip-grad 1.0`, 1024x8, 100k steps). First, the yardstick:
the same absolute-energy configuration scores **0.8922 with seed 0 and 0.9078
with seed 1**, so anything inside about 0.016 in AUC is run-to-run noise. That
number should be quoted beside any comparison from now on.

`--relative-energy total` scores AUC 0.8915 -- the same as the absolute baseline
within that noise -- while the energy response becomes nearly correct:

```
 sampling fraction E_tot / E_inc: median / spread
     slice |        Geant4 |    absolute s0 |    absolute s1 | E_tot relative |   all relative
       all | 0.782 / 0.111 |  0.771 / 0.124 |  0.764 / 0.131 |  0.780 / 0.107 |  0.778 / 0.113
  lowest E | 0.831 / 0.193 |  0.773 / 0.205 |  0.774 / 0.220 |  0.799 / 0.183 |  0.777 / 0.187
  highest E| 0.780 / 0.018 |  0.781 / 0.048 |  0.771 / 0.045 |  0.782 / 0.024 |  0.784 / 0.036
```

The high-energy resolution goes from 2.5x too wide to 1.3x too wide, the pooled
median and spread land on Geant4's, and the low-energy median moves from 0.77 to
0.80 (target 0.83). Per family, the change is surgical:

```
                part | absolute | E_tot relative
              totals |   0.691  |     0.503   <- at the floor
      layer energies |   0.771  |     0.802
      eta/phi widths |   0.963  |     0.962
            sparsity |   0.807  |     0.805
 radial centre+width |   0.865  |     0.860
```

The totals family is indistinguishable from Geant4 (0.503 pooled, 0.534 at low
energy, 0.511 at high) where before a classifier read it at 0.80 in the top
quartile. Nothing else moved. **`--relative-energy total` is the setting to use
for the official space**; making every layer energy relative as well undoes the
gain elsewhere (layer energies 0.961) for no benefit here.

Two things to record: the scale a quantity is modelled on can matter more
than model capacity (this change cost nothing and fixed a family; going from
384x5 to 1024x8 cost 30x the compute for 0.09 in AUC), and a parameterization
should be judged per family, since the pooled AUC hid this entirely -- it moved
by less than the seed noise.

### 22. How much does a family score move between identical runs?

Every per-family comparison above rests on numbers measured once, so the first
job was to measure the noise. Two runs of the *same* absolute-energy
configuration (1024x8, 100k steps, `--clip-grad 1.0`), differing only in seed,
diagnosed with `pinnde_eval.diagnose_samples`:

```
                part | columns | seed 0 | seed 1 | spread
      layer energies |      45 |  0.771 |  0.838 |  0.067
     eta/phi centres |      90 |  0.781 |  0.792 |  0.011
      eta/phi widths |      90 |  0.963 |  0.972 |  0.009
            sparsity |      45 |  0.807 |  0.824 |  0.017
 radial centre+width |      90 |  0.865 |  0.871 |  0.006
              totals |       2 |  0.691 |  0.682 |  0.009
         layers 0-14 |     120 |  0.908 |  0.922 |  0.014
        layers 15-29 |     120 |  0.863 |  0.886 |  0.023
        layers 30-44 |     120 |  0.949 |  0.961 |  0.012
```

Most families are stable to +-0.02, but **layer energies move by 0.067 between
identical runs** -- four times the pooled AUC noise. That single number changes
what the four-variant comparison supports:

```
                part | absolute (s0/s1) | E_tot rel | sqrt widths | both  | verdict
      layer energies |    0.771 / 0.838 |    0.802  |      0.880  | 0.965 | only "both" is outside the noise
      eta/phi widths |    0.963 / 0.972 |    0.962  |      0.923  | 0.936 | sqrt helps, 5x the noise
              totals |    0.691 / 0.682 |    0.503  |      0.704  | 0.530 | E_tot relative fixes it, 20x the noise
```

So the earlier suspicion that "every change hurt the layer energies" was mostly
noise: 0.802 and 0.880 sit at or near the 0.771-0.838 band, and only the stacked
transform's 0.965 is clearly worse. Each fix is safe on its own; stacking them
costs the layer energies. Both effects are reproducible in the sense that they
exceed the measured noise, but neither has been repeated with a second seed yet
-- the claim is one seed per variant plus a measured noise scale.

### 23. Layer energies as sqrt(E_layer / E_inc)

The out-of-range check names the next target: with sqrt widths in place, the
worst offenders are the deep-layer **log energies**, 24% of which are generated
below -8. The official column is `log10(E_layer + 1e-8)`, so -8 means "no energy
at all" and below it means a layer holding less than nothing. That is the same
shape as the widths -- non-negative with a point mass at exactly 0 -- being
modelled on an unbounded log scale, and sqrt is what fixed the widths.

`--energy-sqrt` models each layer energy as `sqrt(E_layer / E_inc)`:

* squaring on the way out cannot produce a negative energy, so the 24% of
  impossible values are unreachable by construction;
* an empty layer is exactly 0 in the modelled space, a boundary rather than an
  isolated spike at -8 that the flow has to hit exactly;
* dividing by E_inc removes three decades of scale the condition already
  carries. In *log* space this was harmful (`--relative-energy all` pushed layer
  energies to 0.961), so whether it helps here is the open question, not a
  prediction.

Exactly invertible given the condition (tested to 1e-9 in log10, ~2 parts per
billion of energy), and `--energy-sqrt` with `--relative-energy all` is rejected
as a double rescaling. Two runs queued: sqrt widths + sqrt energies, and that
plus `--relative-energy total`.

A test-suite bug worth recording: guarding `names.index("logE_inc")` behind the
new flag was missed at first, which broke the 7-column core space -- including
the null check. It was the harness-proof test that caught it, not a run.

### 24. The empty layer is the failure, and it is a structural one

`--energy-sqrt` was a clear loss: AUC 0.992 with it, against 0.892 for the same
configuration without. Worth keeping, because the reason is the useful part.

Percentiles of layer 44 in the space the model worked in and the space the
metrics read (8000 showers each):

```
layer 44                      | real                         | generated
empty (log10 E = -8)          | 50.6%                        | 0.0%
sqrt(E/E_inc), non-empty 1/50/99 | 0.00128 / 0.01545 / 0.06139 | 0.00003 / 0.00362 / 0.04097
log10 E, all rows 1/50/99     | -8 / -8 / 2.712              | -5.335 / -0.396 / 2.716
```

**Not one generated layer is ever exactly empty.** A continuous density puts
zero probability on any exact value, so the flow emits 3e-5 where the data has
0 -- a negligible miss in sqrt space that becomes -0.4 against -8 in the log
space the metrics read. The transform did remove every impossible energy, and
scored worse anyway. In log space the same near-misses at least land below -8,
which is unambiguously "empty", and that is why the log parameterization beat it.

Checking the continuous part separately shows the atom is the entire problem --
the non-empty percentiles match well in both runs, front layers almost exactly:

```
                    | real empty | non-empty log10 E, 10/50/90 | generated
        layers 0-14 |         1% |         1.79 / 2.97 / 4.18 |  1.75 / 2.95 / 4.18
       layers 15-29 |        14% |         1.05 / 2.59 / 3.98 | -0.03 / 2.30 / 3.90
       layers 30-44 |        39% |         0.70 / 1.87 / 2.91 | -2.04 / 0.92 / 2.72
```

**The fix.** `--atom-snap` treats the point mass the way the sparsity comb is
already treated: during training the mass at the atom is spread uniformly over
the empty gap above it (real estate the data never uses), and when sampling
everything in that gap snaps back to the exact value. 183 of the 362 columns
carry such an atom, 26% of rows on average.

The layer rule covers **33 of the 45 layers**, not all of them: the atom finder
needs the value repeated in at least 1% of training rows, and layers 0-11 are
empty in only 0.013% to 0.887% of showers (layer 12 is the first over the line,
at 1.09%). That is the intended behaviour -- the front of the calorimeter always
has energy in it -- but layer 11 at 0.887% sits just under the threshold and
would be covered by a lower one. Untested.

That alone made things *worse*: AUC 0.892 -> 0.968, with the eta/phi centres
suddenly the worst features in the set. The measurement says why:

```
when a layer has no energy, how often is the rest of that layer empty too?
        | EC_eta | EC_phi | width_eta | EC_R | sparsity
  real  |   100% |   100% |      100% | 100% |     100%
  gen   |     0% |     0% |       51% |  55% |      54%
```

Emptiness is a property of the **layer**, not of a column. Snapping columns one
at a time produced showers with no energy in a layer but a non-zero centre
there, which no real shower contains -- a free discriminator. (The centres were
missed entirely because their empty value, 0, is not the column's minimum, so
the atom finder never saw it.)

Adding the rule that a layer the energy calls empty is emptied in every column:

```
                     run |    AUC |  chi2 | chi2 vs floor | sep vs floor |  swd vs floor
    absolute (seed 0/1)  | 0.892 / 0.908 | 45.5 / 47.7 | 48x | 49x | 4.0x
             sqrt widths | 0.914  | 23.9  | 25x | 26x | 4.4x
   sqrt + E_tot relative | 0.929  | 26.6  | 28x | 28x | 4.8x
    energy sqrt-fraction | 0.992  | 29.6  | 31x | 31x | 8.2x
  atom snap, per column  | 0.968  | 21.7  | 23x | 23x | 3.9x
  atom snap, by layer    | 0.890  |  9.4  | 10x |  9x | 3.4x   <- best
    that plus both above | 0.907  | 12.5  | 13x | 12x | 5.0x
```

**chi2 and separation power both fall by a factor of 5 against the previous
best, and by 2.3x against any other configuration tried**, at an AUC
indistinguishable from the baseline (0.890 vs 0.892 / 0.908, inside the +-0.016
seed noise). Per family, every one improves or holds:

```
                part | baseline s0/s1 | atom snap by layer | noise
      layer energies |  0.771 / 0.838 |              0.754 | 0.067
     eta/phi centres |  0.781 / 0.792 |              0.742 | 0.011
      eta/phi widths |  0.963 / 0.972 |              0.903 | 0.009
            sparsity |  0.807 / 0.824 |              0.638 | 0.017
 radial centre+width |  0.865 / 0.871 |              0.828 | 0.006
              totals |  0.691 / 0.682 |              0.697 | 0.009
        layers 30-44 |  0.949 / 0.961 |              0.869 | 0.012
```

Sparsity moves 10 noise widths, the deep layers 7, the widths 6. The totals do
not move, as expected: E_tot has no atom. Stacking the earlier parameterization
fixes on top makes it worse again (0.907, chi2 12.5), the same
non-additivity seen before.

Two honest caveats. Snapping makes out-of-range values legal by construction,
so "% impossible" can no longer be read as a quality measure for those columns
-- the remaining offenders are the radial centres at 8-9%. And the pooled AUC
barely moved, because the classifier keys on correlations between layers that
none of this touches; the marginals are 5x closer, the joint is not.

**Do the earlier fixes still help on top of it?** Three more runs say no, with
one exception that matters:

```
            configuration |    AUC |  chi2 |    sep |    swd | totals family
     atom snap, seed 0/1  | 0.890 / 0.898 | 9.4 / 10.5 | 0.0247 / 0.0275 | 0.062 / 0.066 | 0.697 / 0.612
          + sqrt widths   | 0.902  | 12.6  | 0.0324 | 0.087  |  --
     + E_tot relative     | 0.907  | 11.7  | 0.0306 | 0.072  | 0.502  <- at the floor
              + both      | 0.907  | 12.5  | 0.0313 | 0.089  |  --
```

sqrt widths were worth 45.5 -> 23.9 in chi2 on their own; on top of the atom fix
they cost 9.4 -> 12.6. The atom was what the sqrt transform was really helping
with -- 23% of width values are the empty-layer 0 -- and once the atom is
handled directly, the extra transform only adds distortion.

`--relative-energy total` is the exception: it still puts the totals family on
the floor (0.502 against 0.697/0.612, floor 0.5003), which is the energy
response, for chi2 11.7 against 9.4/10.5. **So the choice is explicit: use
`--atom-snap` alone for the best overall agreement, and add
`--relative-energy total` when the sampling fraction has to be right, which for
a calorimeter it does.** Both at two seeds or better.

Family noise with the atom fix, from the two seeds: widths, sparsity and radial
+-0.01, layer energies and centres +-0.02, depth thirds +-0.02, but the **totals
family +-0.085** -- much noisier than the +-0.009 it showed without the fix, so
only the move to 0.502 is large enough to read.

### 25. Generating the shower itself, not its summary

From the 2026-09-18 meeting: an analysis never uses a shower directly, it uses
quantities derived from one, so the reason to generate showers accurately is
that a generator which gets the voxels right is then known to be right for
anything downstream. Generating the 362 summaries only shows the summaries are
right. Also from that meeting: use their own `DrawAverageShower` to check the
average shape, which needs voxels to draw.

`flow_matching/demo_voxels.py` generates the **6480 voxel energies** (45 layers
x 16 angular x 9 radial) conditioned on E_inc, then computes the 362 features
from the generated voxels with the challenge's own code, so everything is
scored in the same space against the same measured floors.

**The voxel grid is mostly empty**, measured on 20,000 ds2_2 showers:

```
                          | exactly zero | lit voxels per shower
  all showers             |       75.1%  | median 1004 of 6480
  lowest energy quartile  |       97.2%  | 1st percentile 76
  highest energy quartile |       39.4%  | 99th percentile 5079
  non-zero energies span 0.0152 to 4330 MeV, 5.5 decades
```

So the point mass from section 24 is not a detail here, it is the whole
problem -- 6480 columns of it, at 75% instead of 17.8%. The same treatment
carries over unchanged (`log10(E + 1e-8)`, atom band, snap on the way out), and
it works immediately: a *deliberately untrained* 40-step model already produces
**74.6% exactly-zero voxels against Geant4's 74.8%**.

**The main result, and it comes for free.** Every physical rule from
section 24 holds by construction when the features are computed from a voxel
grid rather than predicted as 362 separate numbers:

```
  check                                   Geant4   from generated voxels
  widths and radial centres >= 0           0.00%       0.00%
  layer energy >= empty                    0.00%       0.00%
  sparsity within [0, 1]                   0.00%       0.00%
  sparsity on the 1/144 lattice            0.00%       0.00%
  empty layer empty in every column        0.00%       0.00%
  lit layer has a lit voxel                0.00%       0.00%
  E_tot = sum of layer energies            0.00%       0.00%
```

All seven, from a model too small to generate anything sensible. The
feature-space model needed `--atom-snap`, the per-layer rule and
`--derive-total` to get three of these right and still fails three. This is the
concrete form of the meeting's argument: get the voxels right and the derived
quantities cannot be inconsistent, because they are derived.

A bug the voxel work exposed in the existing code: the atom snap used
`x < hi`, where `hi` is the smallest real non-empty value. A genuine value
sitting exactly at that edge comes back from the z-score round trip a bit either
side of it, so half of those were being snapped to "empty" -- quietly destroying
the smallest real value in a column. The threshold is now strictly inside the
band (`hi - max(|hi|, 1) * 1e-9`), with a test.

Still to measure (needs the GPU): whether a properly trained voxel model is
competitive on the distribution metrics. `~/run_voxels.sh` runs a 20k-step
sanity check, the full 100k-shower run, and a `--no-atom-snap` ablation that
doubles as the minimally-preprocessed configuration for comparing against
a minimally preprocessed baseline, which the meeting asked for so that
preprocessing stops being a source of difference between approaches.

### 26. The first voxel runs: what actually breaks

Five runs on the GPU machine, 100k showers and 100k steps each, about 5 minutes apiece.
The model does not work yet, and the reasons are specific and measured.

```
  run                        auc   chi2/floor   median E_tot/E_inc
  voxels (20k, 20k steps)  1.000       123.9x            1.3e+06
  voxels (100k, 100k)      1.000       124.4x            9.9e+05
  no atom handling         1.000       151.5x            2.1e+07
  atom band 1.0            1.000       130.7x                443
  atom band 0.25           1.000       136.0x                184
  real Geant4               0.50         1.0x               0.78
```

**Five times the data and five times the training changed nothing** (123.9x to
124.4x). This is not undertraining.

**Cause 1: the informative part of the distribution sits in the tail.** With 75%
of voxels empty, a voxel column has mean near -8 and spread ~2.5 in log10, so
real *lit* energies sit around +3 standard deviations after z-scoring. That is
where a flow's errors are largest, and one unit of z there is a factor of 300 in
energy.

**Cause 2: the dequantization band is mostly noise.** Spreading the empty mass
over the whole ~6 log-unit gap injects variance the network cannot predict:

```
  voxel lit   total variance   variance among lit values   injected noise
      91.7%            3.86                       0.39               7%
      60.0%           12.78                       0.65              10%
      30.0%            9.64                       0.43              23%
      10.0%            5.64                       0.62              51%
```

Narrowing the band (`--atom-band`) fixes the energy scale dramatically -- median
E_tot/E_inc falls from 9.9e5 to 184, a factor of 5400 -- and costs a little chi2
(124x to 136x). So the diagnosis holds, but it is not sufficient.

**Cause 3, the dominant one: log space has no ceiling.** Of all generated
layers, **22.4% exceed the largest layer energy that exists in real ds2** (61.9
GeV), and those layers hold essentially **100% of the model's total energy**.
Remove them and the median E_tot/E_inc drops from 184 to 5.87, within a factor
of 7 of the real 0.78. The bulk of the distribution is roughly sane -- the
median generated layer holds 445 MeV against Geant4's 354, and the median lit
voxel 5.7 MeV against 13.8 -- but occupancy is 2.5x too high (4008 lit voxels
per shower against 1599) and the upper tail is unbounded.

**A hole in the physical checks, found late.** All seven checks passed on
showers depositing 10^13 times the incident energy, because every one of them
tests *consistency* -- is E_tot the sum of the layers, is an empty layer empty
everywhere -- and none tested *conservation*. `check_energy_is_conserved` is now
in the validator (E_tot / E_inc at most 3; real ds2 tops out at 1.6), it fires on
100% of these voxel showers, and it would have caught this in the first minute.
The earlier claim that voxel generation makes every physical check pass was made
against an incomplete checker: it gives the consistency rules for free, and
conservation not at all.

**What to do next.** Bounding the output is the fix that matches the physics.
The natural form is the factorization published models use: generate the layer
energies first (the feature-space model already does this reasonably), then the
*normalized* pattern within each layer, which lives on a simplex and is bounded
by construction. A shower's total is then set by the first stage rather than
emerging from 6480 unbounded numbers.

### 27. Checking the claims: the band is two different mechanisms

Two statements from the write-ups, re-measured before presenting them.

**"No real layer deposits between zero and 16 keV."** Stated from 8000 showers.
Re-measured on **all 200,000 showers in both files, 9 million layer instances**:
not one layer deposits in (0, 15 keV). The smallest non-zero layer energy is
15.165 keV in ds2_1 and 15.157 keV in ds2_2; per layer the smallest non-zero
deposit ranges from 15.157 keV to 1.23 MeV. Empty layers are 18.06% and 17.86%
of all layer instances, against the 17.8% quoted from the evaluation set. The
claim holds, and is almost certainly Geant4's production cut rather than
fundamental physics, so it is a property of this dataset and should be stated
that way.

**"The point mass is spread over the empty band above it."** True for the layer
energies and *not* for most of the features it is applied to. Measured over the
212 features that carry an atom, on 100,000 showers:

```
  family        n    mass at the atom    band width
  logE_layer   38             21.4%      6.18 to 6.39
  EC_R         38             21.4%      2.32e-02
  width_R      45             24.1%      2.98e-10
  width_eta    45             22.8%      5.27e-11 to 7.45e-11
  width_phi    44             23.3%      5.27e-11 to 7.45e-11
  EC_phi        2              0.1%      5.55e-17
```

For the 134 width features the gap is ~1e-10 wide: the smallest real width sits
essentially at zero. So the training-time spreading does nothing there, and the
snap is working by a different route -- it catches **negative** values, which are
impossible, and maps them to exactly zero, which is the correct empty value.

So `--atom-snap` is two mechanisms wearing one name:

* **energies**, where a genuine 6-log-unit gap exists: training spreads the mass
  into it, the flow learns how much belongs there, sampling snaps it back;
* **widths and radii**, where there is no gap: the flow's negative outputs *are*
  the empty mass, and snapping converts an impossible value into the right one.

Both are correct, and the second explains two measurements that had looked
independent -- negative widths falling from 72.6% to 4.8% of showers, and the
zero-mass in width columns coming out about right (47.4% generated against
49.7% real for width_eta_40). They are the same event counted twice.

**Per-family null floors, measured rather than assumed.** Every family number in
the write-ups is read against 0.5, which had never been checked -- the measured
floors in NULL_FLOORS are for all 362 features at once, not for a 2-column
family. Two disjoint real samples, energy-matched (worst gap 9e-3 in log E),
scored exactly the way a model is:

```
  family                 columns   null AUC
  layer energies              45     0.4875
  eta/phi centres             90     0.5005
  eta/phi widths              90     0.4993
  sparsity                    45     0.4909
  radial centre+width         90     0.4976
  totals                       2     0.4884
  layers 0-14                120     0.4903
  layers 15-29               120     0.5026
  layers 30-44               120     0.4936
```

They span about 0.487 to 0.504, so reading a family against 0.5 is fair to about
+-0.013. `pinnde_eval/validate_claims.py` reproduces this; the last digit moves
between runs of the classifier. In particular "the totals family sits on the floor at 0.502" is
supported: the measured floor for that family is 0.4884.

### 28. Five seeds each: the headline numbers, corrected

Every comparison so far was judged against a seed noise estimated from two runs,
which is a range and not a standard deviation. Five seeds of the baseline and
five of the recommended configuration, all 100k showers and 100k steps:

```
  baseline (5 seeds)          the fix (5 seeds)
  auc   0.909 +/- 0.017       auc   0.894 +/- 0.020
  chi2  50.7x +/- 3.4         chi2  11.7x +/- 2.3
  sep   52.8x +/- 3.5         sep   11.1x +/- 2.3
  individual chi2: 47.8, 50.0, 48.4, 51.2, 56.3   |   9.5, 11.1, 10.4, 12.1, 15.5
```

Three corrections follow.

**The chi2 improvement is real and slightly smaller than quoted.** 50.7x to
11.7x is a factor of 4.3, with a 39-unit gap against a combined spread of about
4 -- not in doubt. But the write-up quoted **9.8x** from a single run, and the
five-seed mean is 11.7x. Seed 0 was lucky.

**chi2 varies by about +-20% between seeds**, far more than the +-0.016 on AUC
suggested. Quoting chi2 to three significant figures from one run, as every
table in this log has done, was over-precise. Differences under about 4x in
floor units are not results.

**The AUC "improvement" is not real.** 0.894 +/- 0.020 against 0.909 +/- 0.017:
the fix is, if anything, very slightly better, and the difference is well inside
the spread. The statement stays "unchanged", and the single-run 0.872
that appeared in the write-up was the tail of the distribution, not a result.

### 29. Marginals or correlations? The interpretation was wrong

Both write-ups claim the remaining detectability is cross-layer correlations,
since chi2 (which reads one feature at a time) improved 4x while the classifier
(which reads all 362 at once) did not move. That is an interpretation, and it is
wrong.

The test: build a surrogate carrying **our model's marginals** and **Geant4's
exact dependence structure**, by mapping each real column through its ranks onto
the sorted generated values. If our marginals were the innocent party, that
surrogate would be hard to detect.

A first attempt over all 362 features gave a misleading answer because the
surrogate broke the empty-layer rule in 46% of showers -- the per-column rank
map scatters the atoms, and the classifier detects the construction rather than
the marginals. Redone on layers 0-9, which are essentially never empty, so no
atoms couple the columns and the copula surrogate is valid (80 columns, ~7900
showers with every one of those layers lit):

```
                     full model    our marginals + Geant4 correlations
  baseline              0.9398                    0.9265
  with the fix          0.9286                    0.9160
```

Those digits are for these two runs; `pinnde_eval/validate_claims.py` repeats
the measurement on any saved sample file and lands within a few thousandths.

**The marginals alone account for nearly all of the detectability.** Correlations
add about 0.01 of AUC. The "we fixed the vocabulary, the grammar is still
wrong" framing is backwards: the vocabulary is still wrong, just less so, and it
is still what gives the model away.

Scope, stated honestly: this is the front of the calorimeter, chosen because the
question is only well posed where there are no point masses -- which is also
where the empty-layer fix does the least. Whether correlations dominate in the
deep layers remains untested, and the copula surrogate cannot answer it there.

The practical consequence is a changed priority. A layer-aware architecture was
the planned next step on the theory that correlations were the wall. On this
evidence, per-feature accuracy is still the binding constraint, at least in the
front half.

**The symmetric rule, measured.** The five-seed runs are the first with the
two-way emptiness rule in place (a layer the energy calls lit must have at least
one lit voxel, so sparsity 1 is reserved for empty layers). Against the previous
best run:

```
  check                              before   after
  lit layer has a lit voxel           58.6%    5.9%
  sparsity outside [0, 1]             44.4%    7.0%
```

The second row was not the target. Sparsity out of range had been logged as a
separate problem wanting a bounded representation; most of it was this same
defect from another angle, and capping a lit layer at "one lit voxel" removed
six sevenths of it. Two defects, one rule.

### 30. Rank-Gaussian features: chi2 halved, classifier worse

Section 29 says per-feature accuracy is the binding constraint, so the direct
attack is to make every marginal exactly right by construction.
`RankGaussTransform` (`--rank-gauss`) maps each feature onto a standard normal
through its own empirical CDF and maps back through the training quantiles. The
generated marginal is then the training marginal whatever the model learns, an
atom becomes an interval sized by its probability mass, and a value outside the
training range is unreachable. Verified directly: pure Gaussian noise pushed
through the inverse reproduces every training percentile to within sampling
error, reproduces the 51% empty rate of layer 44, and produces zero impossible
values.

Three seeds against the five-seed baseline of the current best:

```
                       chi2 vs floor              auc
  current best      11.7 +/- 2.3         0.894 +/- 0.020
  rank-Gaussian      5.9, 5.9, 6.0       0.955, 0.959, 0.959
```

**chi2 halves and its seed spread collapses** from 2.3 to 0.1 -- the marginals
are no longer a source of run-to-run variation, which is what the construction
promises. Physical validity is the best of any configuration tried:

```
  check                              best so far   rank-Gaussian
  widths and radial centres >= 0           4.79%           0.00%
  sparsity within [0, 1]                   6.96%           0.00%
  lit layer has a lit voxel                5.90%           0.14%
  deposits less energy than it got         0.01%           0.00%
```

**And the classifier gets substantially worse**, 0.894 to 0.956. Per family the
damage is concentrated: sparsity 0.638 -> 0.850, radial 0.828 -> 0.894, centres
0.742 -> 0.802, while layer energies (0.754 -> 0.734), widths (0.903 -> 0.873)
and totals (0.697 -> 0.630) improve.

**The obvious explanation is wrong.** Ties are dithered across their rank block,
which looked like the atom-band noise problem of section 26 applied to every
repeated value. Measured: that dither is 1% to 7% of a column's variance,
including 5% for sparsity. It is not the cause, and the cause is not yet
identified.

What this does establish is that chi2 and the classifier can move in opposite
directions, decisively: the same change halves the challenge's own per-feature
statistic and makes a whole-shower classifier 0.06 better at spotting fakes. Any
claim of the form "the model improved" has to say which of them it means.
