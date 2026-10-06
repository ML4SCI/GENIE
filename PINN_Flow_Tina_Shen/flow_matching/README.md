# flow_matching

Conditional flow-matching generator for the PINNDE flow track — a
simulation-free alternative to an O(N·M) Monte-Carlo score
estimate.

**Idea.** Learn a velocity field `v_theta(x, t)` by regressing it onto a
straight-line interpolant. For noise `x0 ~ N(0,I)` and data `x1`:

```
x_t   = (1 - t) * x0 + t * x1          # straight-line (OT) path
target = x1 - x0                        # constant velocity along the path
loss   = || v_theta(x_t, t) - target ||^2
```

The target is exact (no kernel sum, no ODE solve in training), so it is cheap
and low-variance. Generate by integrating `dx/dt = v_theta(x, t)` from `t=0`
(noise) to `t=1` (data).

**Time convention:** `t=0` = noise, `t=1` = data, sample `0 → 1` (the
flow-matching / Lipman convention). The diffusion / score track labels it the
other way (`t=0` data → `t=1` noise); the two are related by `t → 1−t`.

## Files
- `model.py` — `VelocityField`: MLP with Fourier-feature time **and condition**
  embeddings + GELU.
- `core.py` — `fm_loss` (the objective) and `sample` (Euler/Heun ODE solver),
  both condition-aware.
- `train.py` — `train_flow_matching`: Adam + cosine LR decay; optional Tier-3
  (`pinnde_eval`) monitoring during training.
- `demo.py` — trains on a 2-D GMM toy and reports `pinnde_eval` metrics.
- `demo_conditional.py` — trains `p(observables | energy)` on the calo-flavoured
  shower toy and scores it pooled, per energy bin, and at fixed conditions.
- `demo_calo.py` — the same on **real CaloChallenge ds2 showers**: trains on
  `dataset_2_1`, scores against the held-out `dataset_2_2` and the measured
  Geant4 null floor.
- `_utils.py` — seeding and array helpers.
- `tests/` — edge-case + "it actually learns" tests.

## Run

```bash
cd Tina
python -m flow_matching.demo              # train on a 2-D GMM, score vs truth
python -m flow_matching.demo_conditional  # train p(observables | energy), toy
python -m flow_matching.demo_calo         # the same on real ds2 showers
pytest flow_matching/tests -q
```

`demo_calo` works in one of three spaces, and on the GPU when asked:

```bash
python -m flow_matching.demo_calo --features core                # 7 observables
python -m flow_matching.demo_calo --features per-layer           # 187 columns
python -m flow_matching.demo_calo --features official --device cuda  # the challenge's 362
```

In the official space, how the features are *parameterised* matters more than
model size. The settings that were measured, best first:

```bash
# an empty layer is an exact value, not a small one: log10 E = -8, centres and
# widths 0, sparsity 1, in 17.8% of all (layer, shower) pairs. A continuous flow
# hits an exact value 0% of the time. --atom-snap gives that point mass a region
# to land in during training and snaps the whole layer back when sampling.
python -m flow_matching.demo_calo --features official --atom-snap   # chi2 45.5 -> 9.4

# add this when the energy response has to be right: it models E_tot as
# log10(E_tot / E_inc) and puts the totals on the floor, for a little chi2
python -m flow_matching.demo_calo --features official --atom-snap --relative-energy total
```

`--positive-sqrt` (widths as sqrt(x)) and `--energy-sqrt` (layer energies as
sqrt(E/E_inc)) are kept for the record and are **not** recommended: the first is
subsumed by `--atom-snap`, the second is much worse. DEVLOG §24 says why, with
numbers.

```python
from flow_matching import train_flow_matching, sample
model, history = train_flow_matching(data, dim=2)   # data: (N, d) tensor
gen = sample(model, n=5000, dim=2)                  # integrate noise -> data
```

## Conditional generation

The actual calorimeter target is the *conditional* density `p(shower | E_inc)`,
not a marginal. Pass a per-sample condition and the field becomes
`v_theta(x, t, c)`; the condition enters as its raw value plus a low-frequency
Fourier embedding.

```python
model, history = train_flow_matching(x, dim=3, cond=c)   # c: (N, 1)

gen = sample(model, n=5000, dim=3, cond=0.9)      # everyone at c = 0.9
gen = sample(model, n=5000, dim=3, cond=c_eval)   # matched per-row conditions
```

`demo_conditional.py` scores the result three ways, strictest last:

1. **pooled** over all energies — the number a marginal model could also fake;
2. **per energy bin** via `evaluate_by_condition` — a bad bin cannot hide inside
   a good average;
3. **at fixed unseen conditions** `c* ∈ {0.1, 0.5, 0.9}` against fresh truth
   draws at exactly those `c*` — the interpolation test a lookup table fails.

## Results

**2-D GMM, k=6, 4000 steps (unconditional).** Generated-vs-truth lands near the
statistical null floor: **SWD 0.066** (floor ≈0.049 from two true draws),
**MMD ~1e-4**, **AUC 0.54 ± 0.003** (0.5 = perfect). Cosine LR decay was needed —
without it the samples drifted off the data manifold late in training
(final SWD 0.166 → 0.066 with decay).

**Shower toy, 40k events, 6000 steps (conditional).** Pooled: **SWD 0.0056**,
**MMD −8.4e-05**, **W1 0.0058**, FPD/KPD consistent with zero. **AUC 0.565 ±
0.007** — the transport distances sit at the floor but the classifier still
finds a residual imperfection, and χ² localizes it to the skewed
sampling-fraction feature (2.4 vs ~1.1–1.4 on the others): the flow slightly
under-models the gamma tail. This is kept as an honest example of *why* AUC
stays in the suite even when the distances look perfect.

Per energy bin the pooled number hides a trend — AUC rises 0.512 / 0.551 /
0.561 / 0.617 across the four condition quartiles, so the high-energy slice is
the weakest (narrow distributions make the same absolute error more visible).
At fixed `c*`, per-feature means track truth to ≲0.5%.

**Real CaloChallenge ds2, 100k events, 12000 steps (conditional, ~5 min CPU).**
Trained on `dataset_2_1`, scored against the held-out `dataset_2_2`. Pooled, it
is **indistinguishable from Geant4**: AUC 0.5048 against a measured null floor
of 0.4971, χ² 1.118 vs 1.096, SWD 0.0168 vs a floor of 0.0222.

Per energy bin, the first configuration told a completely different story —
and fixing it is the most instructive result in the project:

| energy bin | auc, `h128 d3 12k` | auc, `h384 d5 30k` |
|---|---|---|
| E 0–25% | **0.787 ± 0.010** | 0.509 ± 0.020 |
| E 25–50% | 0.573 ± 0.016 | 0.510 ± 0.011 |
| E 50–75% | 0.498 ± 0.015 | 0.487 ± 0.020 |
| E 75–100% | 0.527 ± 0.018 | 0.496 ± 0.009 |

The baseline had a quarter of the data trivially separable while its pooled
AUC sat at the floor. Note the trend is the *opposite* of the toy, where AUC
rose with energy — the toy did not predict where the real model fails.

**The diagnosis mattered more than the fix.** Separation power in the bad bin
was barely worse than elsewhere, which proved the marginals were fine and the
*joint* was wrong: every one of the 7 observables was individually at chance
(AUC 0.48–0.51), a linear classifier on all 7 was at chance (0.477), and only
a nonlinear one separated them (0.788). At low energy `E_tot` and `sparsity`
are nearly deterministic (Geant4 correlation −0.951); the flow produced −0.818,
a fatter cloud around a thin manifold.

Two plausible fixes did nothing. `sparsity` really is discrete — exactly
`1 − k/6480` — so it gets dequantized during training and floored back when
sampling, which is correct physics and moved AUC by 0.002. More ODE integration
steps moved it by 0.005 over a 16× increase. **Capacity was the bottleneck**,
and extra ODE steps only started paying off once the field was sharp enough to
resolve (0.544 → 0.507 at 200 steps).

The local maps only see the failure once you condition: pooled they gave
out-of-fold AUC 0.504 and max |r| 2.5 (below the |r| > 3 threshold); on the bad
slice, 0.787 and max |r| 3.8 with 32% of generated samples confidently fake.
Find the bad slice with `evaluate_by_condition`, *then* localize inside it.

Full numbers and the full diagnostic chain: `../pinnde_eval/DEVLOG.md` §10
(toy), §13 (real data), §14 (diagnosis and fix).

## Per-layer observables: a representational wall

```bash
python -m flow_matching.demo_calo --per-layer     # 187 columns
```

Scaling to the 187-column per-layer space **fails outright**, with more
capacity than the d=7 fix used: out-of-fold AUC **0.9963** pooled and 0.9995 on
the worst slice, 94% of generated samples confidently fake. For scale, the
low-energy failure above that everyone would call bad was 0.787.

Zero-inflation explains all of it. An empty layer has energy exactly 0, radial
centre exactly 0, radial width exactly 0 and sparsity exactly 1, and deep
layers are empty in up to 60% of showers. A continuous flow cannot put finite
probability on a point, so it spreads density around the atom — and since the
atom sits at a physical boundary, that mass lands outside it. Up to **31.6% of
generated radial widths are negative**. Across the 44 layers,
`corr(atom mass, out-of-range fraction) = +0.989`: every layer's failure rate
is predicted by how much of its mass sits on the atom.

This is a different kind of failure from the low-energy one. That was
*resolution* — the right density, not sharp enough, fixed by capacity. This is
*representational*: the target is not absolutely continuous, so no continuous
flow expresses it at any capacity. Dequantization does not help either; that
treats an evenly spaced lattice, not a single atom coexisting with a
continuous part.

## Next
The per-layer target needs a **two-part (hurdle) model**: a Bernoulli per layer
for occupancy (mostly a function of incident energy and depth), and the shape
observables modelled only where the layer is lit, emitting the exact atom
otherwise. That removes the impossible values for free. Worth measuring first,
because it is much cheaper: restrict to the front layers where the atom mass is
small (0.001–0.048 for layers 0–9) and confirm AUC returns to the floor, which
would isolate zero-inflation as the sole cause.

Architecture ablations still open: adaptive collocation (non-uniform `t`
sampling), richer Fourier features, and a physics-informed
continuity-equation residual, each scored against the null floor.
