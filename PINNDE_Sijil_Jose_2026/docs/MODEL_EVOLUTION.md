# How the models evolved: autoencoder → direct PINN → two-stage

CaloChallenge Dataset 2. This document is the *argument*, not the reference:
why each model was built, what test exposed its limit, what the next model
changed in response, and what the measurements supported at each step.

Read [REPLICATION.md](REPLICATION.md) for the specifications and commands,
and [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the final numbers. This
document is the connective tissue between them.

**A note on epistemic status.** Everything labelled *measured* came from a run
in this repository. Where a conclusion was reached, stated confidently, and
then overturned, that is recorded too — §9 collects those, because the
reversals are more instructive than the successes. Three times a defect was
attributed to a cause that a later control disproved.

---

## Contents

1. [The problem, and the three properties that drive everything](#1-the-problem)
2. [Stage 0 — baselines, to know what "working" means](#2-stage-0--baselines)
3. [Stage 1 — the autoencoder](#3-stage-1--the-autoencoder)
4. [Stage 2 — latent diffusion, and how we concluded the reconstruction was not good enough](#4-stage-2--latent-diffusion)
5. [Stage 3 — the direct PINN sampler, and its five failure modes](#5-stage-3--the-direct-pinn-sampler)
6. [Stage 4 — the two-stage model, and what actually made it different](#6-stage-4--the-two-stage-model)
7. [Two-stage PINN vs CaloDREAM](#7-two-stage-pinn-vs-calodream)
8. [Hyperparameters, as they changed](#8-hyperparameters-as-they-changed)
9. [What was concluded wrongly, and what corrected it](#9-what-was-concluded-wrongly)
10. [What is still open](#10-what-is-still-open)

---

## 1. The problem

Learn `p(shower | E_inc)` over 6480 voxels, with E_inc spanning 1 GeV – 1 TeV,
such that a trained classifier cannot separate generated showers from Geant4.

Three properties of the data generate every design decision that follows, and
it is worth naming them up front because each one eventually got its own
dedicated mechanism:

| Property | Measured | Mechanism that eventually addressed it |
|----------|----------|----------------------------------------|
| Extreme, spatially correlated sparsity | ~75% of voxels exactly zero | occupancy head + gate (§5.4) |
| Huge dynamic range at the faint end | 15 keV – 6 GeV per voxel | layer-logit representation (§6.2) |
| A tight conservation law | E_dep/E_inc median 0.7826 | two-stage factorization (§6.1) |

The order in which these were *discovered* is roughly the order of this
document. None of them was obvious from the outset; each announced itself as a
metric that refused to improve.

---

## 2. Stage 0 — baselines

Before any generative model, three baselines were built, and this turned out
to be the highest-leverage thing done in the whole project:

| Baseline | Relative reconstruction error |
|----------|------------------------------:|
| all zeros | 1.000 |
| per-energy-bin mean shower | **0.638** |
| PCA-96 (same budget as the AE latent) | ~0.60 |
| PCA-256 | 0.461 |
| PCA-512 | 0.387 |

`ae_baselines.py` computes these and prints a verdict.

**Why this mattered.** The autoencoder v3 had a validation loss that plateaued
and *looked* converged. Against these baselines it was reconstructing at
rel 0.688 — **worse than the energy-binned mean shower**, and far worse than a
linear PCA with the identical 96-number budget. A model that cannot beat a
linear method with the same bottleneck is not capacity-limited; it is
*objective*-limited.

Without the baselines, the flat validation curve would have been read as
convergence, and the next step would have been a bigger network. That would
have been wasted. **The general lesson: a loss curve tells you whether
optimization is progressing, never whether the model is any good.**

---

## 3. Stage 1 — the autoencoder

### 3.1 Why an autoencoder first

The strategic plan was latent diffusion: compress showers to a small latent,
then run a generative model there. 6480 dimensions is a hostile space for
diffusion, and a 96-dimensional one is not. The autoencoder was the
compression half of that plan.

Architecture (`FactorizedCaloAE`), and the structural idea that makes 96
numbers sufficient: **treat the 45 longitudinal layers as 45 separate 16×9
images**, encode them with a *shared* 2D convolutional stack, then model the
longitudinal direction with 1D convolutions.

```
[B, 45, 16, 9]
  → per-layer 2D encoder (shared):  (1,16,9) → (16,16,9) → (32,8,5) → (64,4,3)
  → 768 features per layer, reshape to [B, 768, 45]
  → 1D encoder along z: 768 → 256 → 128
  → flatten [B, 5760] ⊕ energy_embed [B, 64] → Linear → [B, 96] → [B, 8, 12]
  → decoder: exact mirror, energy_embed injected AGAIN
```

Two choices here paid off later:

- **The latent is structured** (8 channels × 12 longitudinal positions), not a
  flat 96-vector, so a downstream diffusion model can use convolutions on it.
- **The decoder also receives the energy embedding.** This means a latent
  generator only has to model *shape*; the decoder handles energy scale.

3,110,882 parameters.

### 3.2 The v3 plateau, and its four-part post-mortem

v3's validation loss stopped improving around epoch 20 and stayed flat whether
trained 85 epochs or 460. `diagnose_training_plateau.py` was written to
decompose the objective rather than guess. Four separate findings, each
measured on the epoch-20 checkpoint over 2048 validation showers:

| Finding | Measurement | Why it broke the objective |
|---------|-------------|----------------------------|
| **The hit BCE shared the value output** | 63% of total loss, stuck at 0.496 — worse than a constant predictor (0.361) | The same number had to be a ±3 logit *and* equal ~4×10⁻⁴ |
| **The energy-sum L1 dominated the gradient** | 3× the gradient of the reconstruction term, cosine 0.015 with it | It moved every voxel together, carrying no shape information |
| **The physical-space energy term was inert** | 5.9×10⁻⁷ | Contributed nothing at all |
| **The representation was nearly linear** | `log1p(E/E_inc)` with values ~1e-4 | Log-space MSE/L1 barely saw shower shape |

Plus a units bug found in the same pass: only `energies` had been converted to
GeV, so the model was training on `log1p(shower_MeV / energy_GeV)` — a ratio
inflated ~1000×. It stretched the target range ~15× (max 6.3 vs 0.4), which
fed into a `1 + 10·x_log` per-voxel weighting (max weight 63 vs 5).

**The plateau was not one bug. It was four terms of the objective each failing
in a different way, and a units error amplifying one of them.**

### 3.3 v4: what changed, and the result

```python
true = expm1(x_log);  pred = expm1(v * x_scale)
rel2 = ((pred - true)².sum(1) / true².sum(1)).mean()
bce  = BCEWithLogits(h, (x_log > 0).float())
p_e  = pred.clamp(min=0) * sigmoid(h)          # SOFT-gated
e2   = ((p_e.sum(1) - e_true) / e_true)².mean()
total = rel2 + 0.1*bce + 1.0*e2
```

Each change answers one row of the table above:

- **Two output heads** (value, hit logit) — the BCE no longer fights the
  regression.
- **`rel2` is the square of the reported metric.** Objective and metric now
  agree. Optimizing something you do not report is precisely how v3 plateaued
  invisibly.
- **The energy term uses the soft-gated prediction** `pred · σ(h)`. The
  reported output zeros voxels the hit head calls empty, so an energy term on
  *ungated* values optimizes energy the gate then discards. Measured: with
  ungated `e2`, the gated energy error got *worse* in epoch 1 (0.29 → 0.37).
- **`x_scale` ≈ 8×10⁻⁴** divides the target so inputs and the value head are
  O(1).
- **Linear output activation, deliberately.** In log space an empty voxel's
  target is exactly 0, and Softplus has a floor near 0.69 that would forbid it.

Result: **rel 0.367 at 255 epochs** — beating PCA-512 (0.387) with 96 numbers
instead of 512. Energy error 0.021, lit fraction 0.256 against a real 0.250.
Training to 400 epochs gained nothing.

So as a *compression* result, v4 is genuinely good. Whether that is good
enough for latent diffusion is a different question, and §4 is how it was
answered — **with the serious caveat, established late, that §4 was answered
on the pre-v4 model.** The v4 checkpoint lives only on the workstation; no
file in this working copy has `two_head=True`, and v4 has never been through
the official evaluation. See §4.7.

---

## 4. Stage 2 — latent diffusion

### 4.1 The methodological move: measure the ceiling first

A latent generator can never produce anything its decoder cannot reconstruct.
So before training any latent sampler, `train_latent.py --ceiling` was added:
it **encodes and decodes real showers with no diffusion model in the loop**,
and exports the result through the identical pipeline.

Every latent-sampler number is then read against that file, not against real
showers. This single test is what made the eventual conclusion unambiguous —
without it, a mediocre latent sampler and a mediocre decoder are
indistinguishable.

### 4.2 The first answer was a false positive

On the **v3** autoencoder, the latent sampler matched its ceiling exactly, and
that was initially read as success — the sampler is doing its job, the
remaining error is the decoder's.

That test was far easier than it looked. v3's latents were **nearly
deterministic given the incident energy**: per-dimension std 0.036–1.6 with
several dead dimensions, and the latent sampler's training loss reached 1e-5.
Reproducing a near-delta distribution is trivial. The conclusion was correct
by accident, resting on no real task.

v4's latents use all 96 dimensions (std 3.1–4.6) and its sampler's loss
plateaus at 0.44, which is about the average posterior variance over
t ~ U(0,1) — what convergence actually looks like for this objective. The
re-test on v4 therefore meant something.

**An operational catch worth repeating:** `train_latent.py` prints generated
latents' per-dimension std against 1.0 on every run. At 600 steps it read
**0.019** — the latents had collapsed to their mean — while E_dep/E_inc still
looked healthy at 0.759, because the mean latent decodes to an average shower
with roughly the right energy. An aggregate metric hid a total collapse.

### 4.3 On v4, the latent path led on physics metrics

| Metric | latent sampler | ceiling (AE recon) | direct sampler (film 100k) | real |
|--------|---------------:|-------------------:|---------------------------:|-----:|
| E_dep/E_inc | **0.782 ± 0.118** | 0.774 ± 0.120 | 0.701 ± 0.126 | 0.782 ± 0.120 |
| 5/50/95% | 0.61/0.78/1.00 | 0.59/0.78/0.97 | — | 0.58/0.78/0.98 |
| longitudinal error | **0.050** | 0.055 | 0.114 | — |
| radial error | **0.108** | 0.114 | 0.118 | — |
| NN / real-to-real | 1.30 | 1.27 | 1.29 | 1.00 |

This looked like the winning path. The energy distribution matched real
showers in *both* mean and spread, per energy bin as well as overall — the
defect that resisted every fix in voxel space. And every generated number
tracked the ceiling to within noise.

### 4.4 How the official evaluation settled it

The physics metrics above are **aggregates over voxels**. A classifier looks at
individual voxels. The official evaluation:

| | cls-low AUC |
|---|---:|
| AE reconstruction (the ceiling) | **0.989** |
| latent diffusion | **0.991** |
| direct conv U-Net | 0.683 |

**The latent sampler reached its own ceiling to within noise — and the ceiling
is 0.989.** That is the whole conclusion, and it is why the `--ceiling` test
was worth building. The sampler is not the problem. A ceiling of 0.989 means
the decoder's reconstructions are *themselves* trivially separable from real
showers, so no generator built on it can be competitive, however good it is.

The warning had been in the data all along: samples sat 1.26–1.30× further
from real showers than real showers are from each other, and the quick metrics
had no view of the per-voxel energy distribution at all.

### 4.5 Why the reconstruction is not good enough — the mechanism

Two independent measurements say the same thing.

**Occupancy.** The AE reconstruction lights **41.98% of voxels against a real
24.95%** — and a 42% lit fraction is a no-hit-head signature, which is itself
evidence that the evaluated file is pre-v4 (§4.7). Total energy is near-perfect (median E_dep/E_inc 0.7826, matching the
reference to four digits, χ² = 0.0013 — the best of any model here). So the
energy is right and spread over too many voxels. FPD and KPD punish exactly
this, because they compare distributions in a feature space rather than
marginal histograms:

| | KPD ×10³ | FPD ×10³ |
|---|---:|---:|
| Geant4 vs itself | −0.028 | 10.78 |
| CaloDREAM | 0.023 | 24.65 |
| CaloVAE+INN (worst published) | 45.61 | 5443 |
| **AE pre-v4 (ours)** | **91.60** | **11328** |

Worse than the worst published submission by a factor of two.

**Azimuthal position.** The latent model's worst histogram χ² values are not
energies but **EC Phi** — the azimuthal energy centroid — at ≈0.445 for layers
26–29, far worse than any other model in the comparison. The 96-number
bottleneck does not preserve φ position faithfully enough, so errors the
diffusion model makes in latent space become *structured spatial* errors after
decoding. Stacking a generative error on a lossy reconstruction compounds
rather than cancels.

### 4.6 Independent corroboration

Two external data points, both found after the conclusion was already reached:

- **CaloLatent**, the one published Dataset-2 submission that also runs
  diffusion in a learned latent space, placed **15th of 17** at AUC 0.983.
- **CaloDREAM's own latent variant** overestimates the low-energy regime with
  sparsity discrepancies that the authors attribute **explicitly to the
  autoencoder reconstruction rather than the generative model**, and mitigate
  with a 1 MeV cut.

That the group with the best submission reached the same diagnosis by the same
route is the strongest available evidence that this is a property of the
approach on this dataset, not of our particular autoencoder.

**Verdict: the latent path needs a better autoencoder, not a better sampler.**
PCA-256 (0.461) and PCA-512 (0.387) show there is headroom in the bottleneck
size, and `--latent-channels/--latent-length` make it a one-flag sweep. But
that is a different project, and the voxel path had no such ceiling.

### 4.7 Correction: which autoencoder all of §4 was measured on

Everything in §4.4 and §4.5 — the 0.989 ceiling, the 41.98% occupancy, the
FPD/KPD figures — comes from `best_calo_ae_cond_improved.pt`, which is
**epoch 85, pre-v4, trained under the old units convention**. It has no
`two_head` key and no `shower_units_gev` marker. I originally presented these
results as v4's, quoting v4's reconstruction quality (rel 0.367) alongside
them; those are different checkpoints.

Two things follow, and they point in opposite directions:

- **The ceiling argument is weakened.** A pre-v4 decoder with no hit head has
  every reason to produce a poor ceiling. The v4 ceiling is unmeasured.
- **But the latent-dimension sweep supports the conclusion independently.**
  Five latent sizes were trained (8 → 128, rel 0.598 → 0.394); only the
  96-dim one was evaluated, at two different epochs. Reconstruction error
  improved and the classifier AUC did not move (0.9888 at epoch 85 → 0.9919
  at epoch 460, FPD 11328 → 12437). Reconstruction quality and classifier
  separability are close to decoupled, so a better autoencoder is not
  obviously a route to a competitive latent generator — which is the same
  conclusion by a different route.

The experiment that settles it is cheap and specified in
[EXPERIMENTS.md](EXPERIMENTS.md): copy v4 back, run `--ceiling`, evaluate.
Until then §4's conclusion should be read as *probably right, on stale
evidence*.

---

## 5. Stage 3 — the direct PINN sampler

### 5.0 The formalism, and the one decision that mattered

A stochastic interpolant from data to noise, and generation as a reverse-time
ODE:

```
x(t) = (1-t)·x₀ + σ(t)·ε           σ(t) = σ₀ + (1-σ₀)·t,  σ₀ = 1e-2
dx/dt = G(t, x, E) = [(1-σ₀)·x − q(t, x, E)] / σ(t)
q(t, x, E) = E[x₀ | x_t, E]
```

**Failure mode 0 — the global flow-map ansatz.** The original PINNDE
formulation has the network learn the *trajectory* `x(t; z)` as a function of
its initial noise — the whole flow map. At 1–3 dimensions this works, which is
where the formalism had been validated. At 6480 it fails, three ways, all
measured:

1. **It cannot generalize.** 512 → 4096 anchor trajectories (8× more data) cut
   held-out MSE only 1.84×, i.e. **MSE ~ N^−0.29**. Reaching usable error would
   need ~10⁶× more data.
2. **It cannot train from cold start.** The physics residual is evaluated at
   the model's own prediction, which early in training is off-manifold, where
   the q-kernel degenerates and `G ~ x/σ_t` becomes radially repulsive. The
   residual **rose** during training (6862 → 22002) — it could not minimize its
   own objective.
3. **Samples came out worse than noise**: E_dep/E_inc ≈ 3.4 against a true 0.78.

**The fix, which is the project's central decision: parameterize the local
field `q` and recover `G` algebraically.** This keeps the formalism — same
interpolant, same ODE, same σ(t) — and changes only what the network learns:

- `q` is a **local** function of the current state: smooth, low-complexity, so
  it generalizes instead of memorizing.
- Training data becomes **unlimited and always on-manifold**: any real x₀, any
  t, any ε gives an exact target. No trajectories, no anchor set, no
  collocation points.
- `q` is **bounded** (a convex combination of real showers, ‖q‖ ≈ 68) while `G`
  ranges 90–600 across t. Predicting the well-conditioned quantity and deriving
  the stiff one is the same reason standard diffusion models predict x₀ or ε
  rather than the raw score.

Measured: E_dep/E_inc 0.92 against a true 0.78, versus 3.4 for the flow map.

**The physics therefore lives in the parameterization, not in a residual
penalty.** The ODE structure, the σ(t) schedule and the analytic G *are* the
physics-informed content. An explicit residual term was tried and measurably
hurt, for reason (2) above.

**Failure mode 0b — the scale mismatch.** Before standardization, data occupied
a ball of radius ‖x₀‖ ≈ 0.0555 while unit-variance noise lives on a shell of
radius √6480 ≈ 80.5: **the noise is 1450× larger than the signal**. `G` was
then ~99.9% noise direction (‖G‖ = 79.61 at every t), so a physics residual
could be driven to ~1.5 and still carry error 30× the entire data signal. That
is why the residual and data terms were irreconcilable — they were describing
different things. The toy 1–3D cases had the opposite regime (signal radius
4.5 vs noise 1.0–1.7). Standardizing x₀ restores it.

### 5.1 Failure mode 1 — the Monte-Carlo target collapses samples

The formalism supplies `q` as a Monte-Carlo sum over a reference ensemble. The
obvious move is to distill that. It was tested head-to-head against simply
regressing the clean shower x₀ (8k steps, same net/batch/seed/noise):

| metric | clean x₀ | MC target (n_ref 128–512) | real |
|--------|---------:|--------------------------:|-----:|
| spread across showers | 0.276 | 0.149–0.176 | 0.408 |
| pairwise distance | 58 | 24–27 | 86 |
| distance to nearest training shower | 32 | 7–9 | 47 |

Clean won on every diversity metric and trained 2–4× faster. A larger ensemble
does not fix it (128 → 512 moved spread only 0.149 → 0.176 at 2.3× the cost).

**The underlying reason, and this is the interesting part: the exact
finite-reference q-field is a *memorizing* sampler.** Run as an ODE it lands
within 1.76 of a training shower, against 47 for a genuinely new one. It is a
poor thing to distill. And regressing clean x₀ needs no reference ensemble at
all, because the MSE minimizer of `E[(v − x₀)²]` *is* `E[x₀|x_t,E] = q`.

### 5.2 Failure mode 2 — a conditional mean cannot be sparse

With the `concat` MLP and no occupancy modelling, **77% of generated voxels
landed above the 15.15 keV threshold against 25% for real showers**, carrying
the excess energy with them.

This is structural, not a training deficiency. `q` is a conditional mean, so a
voxel uncertain between empty and lit comes out at an intermediate value — in
any continuous representation, a small spurious deposit. Thresholding the
output cannot fix it without also destroying real small deposits (measured).

**Fix: a second output head** predicting a per-voxel "is this lit in x₀" logit,
trained with BCE at weight 0.1. It does **not** enter the ODE — `G` is built
from `q` alone — it only gates the final read-out at t=0, writing gated voxels
to *exactly* `empty_value` so they invert to exactly 0 MeV.

Note this is the **same defect with the same fix as the autoencoder's**
(§3.3). Two independent architectures, one shared lesson: sparse data needs
explicit occupancy modelling, separate from value regression.

The gate rule was then swept at sample time only (no retraining — seconds
instead of hours):

| gate | hits/shower | radial err | NN distance | E_dep/E_inc |
|------|------------:|-----------:|------------:|------------:|
| `threshold` (σ(h) > 0.5) | 1359 | 0.321 | 138 | 0.718 |
| `sample` (Bernoulli) | 1502 | 0.283 | 161 | 0.711 |
| **`topk`** | 1543 | 0.300 | 151 | **0.780** |
| `gumbel` (top-k without replacement) | ~1540 | — | 1.35–1.48× | 0.53–0.65 |

Real: ~1520 hits, E_dep/E_inc 0.784, real-to-real NN ≈ 120. **No rule wins
everywhere.** `threshold` stays closest to the manifold but is biased sparse;
`sample` is right per-voxel but real occupancy is spatially *correlated*, so
independent draws add salt-and-pepper noise; `topk` matches energy best and is
the default. `gumbel` exists because `topk` ranks by probability, probability
tracks brightness, and so it systematically drops the **dim** voxels real
showers carry at the edges — 0.6% of hits below 1 MeV against a real 7.2%.

### 5.3 Failure mode 3 — two scalars cannot condition 6480 inputs

With `concat`, the *mean* E_dep/E_inc came out right (0.79 vs a real 0.78)
while the per-energy response did not: **1–10 GeV 15% high and over-dispersed,
10–100 GeV 8% low, 100+ GeV 3% low**. Per-shower energy spread was 2.3× too
wide.

Getting the average right but the dependence wrong is what weak conditioning
looks like. Two scalars competing with 6480 inputs at the first linear layer is
a weak way to condition.

**Fix: FiLM.** Sinusoidal features of t and standardized log E (16 frequencies)
→ a 128-dim MLP → per-layer scale and shift on every hidden activation. The
FiLM projections are **zero-initialized**, so the model starts at identity
modulation and learns the conditioning.

Measured: energy spread 2.3× → 1.2× too wide, and sampling divergences
**236 → 1** per 100k draws.

### 5.4 Failure mode 4 — the decisive test: capacity vs inductive bias

Even with FiLM at 100k steps, the per-voxel energy distribution stayed wrong.
The natural hypothesis was insufficient capacity. **A width sweep at fixed
steps tested it directly, and refuted it:**

| hidden | params | dim_frac | vox5 (MeV) | longitudinal | radial |
|-------:|-------:|---------:|-----------:|-------------:|-------:|
| 512 | 10.9M | 0.0050 | 3.55 | 0.133 | 0.214 |
| 1024 | 22.8M | 0.0065 | 3.36 | 0.108 | 0.190 |
| 2048 | 49.8M | 0.0064 | 3.42 | 0.112 | 0.184 |
| **real** | — | **0.072** | **0.63** | — | — |

**The dim-voxel fraction is flat at 0.005–0.0065 against a real 0.072, and the
5th-percentile lit voxel is flat at 3.4–3.6 MeV against a real 0.63, across
4.6× more parameters.** Profile errors improve from 512 to 1024, then stop.

This is a clean result: the flat MLP's limit is **its lack of spatial
structure, not its size**. It has no notion that voxel 100 and voxel 101 are
neighbours, which is exactly what the dim edge voxels require.

**Fix: a 3D conv U-Net over (z, φ, r)** with circular padding in φ (the
azimuth wraps; zero padding asserts a wall that is not there), FiLM
conditioning in every block, position channels for normalized z and r index,
and a zero-initialized output conv.

| Metric | MLP 16k (any width) | MLP film 100k | **conv 8k** | real |
|--------|--------------------:|--------------:|------------:|-----:|
| dim_frac | 0.0050–0.0065 | ~0.006 | **0.0326** | 0.072 |
| vox5 (MeV) | 3.36–3.55 | — | **1.47** | 0.63 |
| longitudinal error | 0.108–0.133 | 0.114 | **0.036** | — |
| radial error | 0.184–0.214 | 0.123 | **0.037** | — |
| E_dep/E_inc | 0.68–0.70 | 0.709 | **0.760** | 0.783 |
| parameters | 10.9–49.8M | 49.8M | **2.47M** | — |

Profile errors fell 3–5×, the energy deficit went 10% → 3%, and the dim-voxel
fraction — *flat across the entire width sweep* — moved 5×. **At 2.47M
parameters, 20× fewer than the MLP, and after only 8k steps.** That last point
is what makes the improvement attributable to spatial structure rather than
capacity.

### 5.5 The two defects that would not yield

After the hit head, FiLM and the conv trunk, two things remained, and both
resisted every further sampling-side and architectural change:

1. **A persistent ~10% energy deficit.** It survived every read-out, every
   gate, every representation. `--value-correct` — dividing kept voxel values
   by their hit probability, which is *algebraically* the right undilution —
   made it worse: E_dep/E_inc 0.780 → 0.963 against a real 0.784, longitudinal
   error 0.139 → 0.257. The diagnosis: the aggregate had been right only
   through *cancellation* (the field recovered 0.79 of the energy in truly-lit
   voxels while inventing 0.26 in truly-empty ones), so undiluting the kept
   voxels also inflates the spurious energy that survived the gate.
2. **Dim voxels 12× too rare.** Improved by the conv trunk (0.006 → 0.033) but
   still far from 0.072.

**Both turned out to be representation and factorization problems, not model
problems — which is why stage 4 fixed them and stage 3 could not.**

### 5.6 The tests that diagnosed the direct model, in one table

Every row is a measurement, not an opinion. The "verdict" column is what the
result licensed — and in four cases that was the opposite of the hypothesis
being tested.

| # | Test | What it varied | Result | Verdict |
|--:|------|----------------|--------|---------|
| 1 | Anchor-data scaling | 512 → 4096 anchor trajectories (8×) | held-out MSE fell only 1.84×, i.e. **MSE ~ N^−0.29** | The global flow-map ansatz cannot generalize at 6480-D. ~10⁶× more data needed. **Killed the original PINNDE parameterization.** |
| 2 | Physics residual from cold start | residual loss on vs off | MSE_f **rose** 6862 → 22002 during training | The residual is evaluated off-manifold where the q-kernel degenerates and G ~ x/σ_t is radially repulsive. It cannot minimize its own objective. **Physics belongs in the parameterization, not a penalty.** |
| 3 | Signal-vs-noise norm audit | — | ‖x₀‖ ≈ 0.0555 against ‖ε‖ ≈ 80.5; ‖G‖ = 79.61 at every t | Noise is 1450× the signal; G is ~99.9% noise direction. Residual and data terms were describing different things. **Forced global standardization of x₀.** |
| 4 | MC target vs clean x₀ | regression target, 8k steps, matched seed/noise | spread 0.149–0.176 vs **0.276** (real 0.408); NN-to-train 7–9 vs **32** (real 47); 2–4× slower | The exact finite-reference q-field is a *memorizing* sampler (lands within 1.76 of a training shower). **A bad thing to distill.** |
| 5 | n_ref scaling | 128 → 512 reference showers | spread moved 0.149 → 0.176 at 2.3× cost | Not a sample-size problem. **Closed off "just use more references".** |
| 6 | Occupancy audit | — | **77%** of generated voxels above threshold vs a real **25%** | A conditional mean cannot be sparse — uncertain voxels land at intermediate values. **Motivated the hit head.** |
| 7 | Output thresholding | post-hoc cut instead of a head | destroyed real small deposits | Sparsity must be modelled, not clipped. |
| 8 | Gate-rule sweep (sample-time only) | threshold / sample / topk / gumbel | hits 1359 / 1502 / 1543 / ~1540 (real ~1520); E_dep/E_inc 0.718 / 0.711 / **0.780** / 0.53–0.65 | **No rule wins everywhere.** topk matches energy but drops dim voxels (0.6% of hits <1 MeV vs real 7.2%); sample restores them but scatters them. The gate chooses *which* voxels are lit — it cannot fix their *values*. |
| 9 | Per-energy-bin response | concat conditioning | 1–10 GeV **+15%** and over-dispersed, 10–100 GeV −8%, 100+ GeV −3%, while the *mean* was right (0.79 vs 0.78) | Average right, dependence wrong = weak conditioning. Two scalars cannot condition 6480 inputs. **Motivated FiLM.** |
| 10 | FiLM A/B at matched depth/width | conditioning only | energy spread 2.3× → **1.2×** too wide; divergences **236 → 1** | Conditioning, confirmed. |
| 11 | Divergence control | 400 integration steps **from scratch** | same divergence rate as the 100-step run | **Refuted** the "integrator stiffness" diagnosis — the earlier fix had come from a fresh noise draw, not more steps. |
| 12 | Width sweep at fixed steps | hidden 512 → 1024 → 2048 (10.9M → 49.8M params) | dim_frac **flat** 0.0050/0.0065/0.0064 vs real 0.072; vox5 **flat** 3.55/3.36/3.42 MeV vs real 0.63 | **The decisive test.** The defect is insensitive to 4.6× capacity. The flat MLP's limit is its lack of spatial structure, not its size. **Motivated the conv U-Net.** |
| 13 | Conv trunk at 1/20 the parameters | trunk only, 8k steps, 2.47M params | dim_frac 0.006 → **0.0326**; longitudinal 0.114 → **0.036**; radial 0.123 → **0.037**; E_dep/E_inc 0.709 → **0.760** | Attributable to spatial structure, since it improved while parameters *fell* 20×. |
| 14 | `--value-correct` | divide kept values by p(lit) | E_dep/E_inc 0.780 → **0.963** (real 0.784); longitudinal 0.139 → 0.257 | **Refuted** an algebraically correct fix. The aggregate had been right only by cancellation (0.79 of lit-voxel energy recovered, 0.26 invented in empty ones), so undiluting inflates the spurious part and never recovers the gated-away part. |
| 15 | Denoiser decomposition (`--check-denoiser`) | lit vs empty voxels separately | 0.79 of truly-lit energy recovered; 0.26 invented in truly-empty | Exposed the cancellation in row 14. |
| 16 | Representation gap measurement | — | empty-to-threshold gap **5.5×10⁻⁴** standardized units | Below any achievable regression error. **The model was structurally blind at the faint end. Motivated the layer-logit representation.** |
| 17 | Voxel energy distribution | — | hits <1 MeV: **0.006** vs real **0.072** (12× too rare); 5th-pct voxel 3.88 vs **0.632** MeV | The defect every aggregate metric missed. **Now reported by `quick_metrics`.** |
| 18 | `compare_to_reference.py` | generated vs the evaluation's own reference | E_inc distributions agree; E_dep/E_inc within 1%; empty voxels exact zeros in both; no voxel in the forbidden (0, 15.15 keV) band; smallest non-zero on the threshold | **Ruled out** units/threshold artifacts as the cause of AUC ≈ 1. A valuable negative — it made the per-voxel defect the only remaining explanation. |
| 19 | float32 resolution check | — | a threshold voxel sits 1054 float32 ticks above empty even at E_inc = 1 TeV | **Ruled out** numerical precision. The blindness in row 16 is the representation's, not the format's. |
| 20 | `lit_head` (E[value\|lit]) | third output head | loss normalized by **lit** voxels instead of all → ~25× too large, damaged q | **Untrustworthy result**, not a negative one. Needs re-running with the weighting fixed. |
| 21 | Official evaluation | film MLP, 100k samples | cls-low **0.999**, cls-high 1.000, cls-resnet 1.000 | Superseded every physics metric. Correct means, correct hit counts and 5–11% profile errors are all aggregates; a classifier looks at individual voxels. |

**Reading the table as a whole.** Rows 1–3 killed the original formulation and
fixed the parameterization. Rows 4–5 settled the training target. Rows 6–8
addressed sparsity as far as a read-out can. Rows 9–13 are the architecture
story, and row 12 is the one that mattered — it is the only test here designed
specifically to separate two competing hypotheses, and it did.

Rows 14, 11 and 20 are the cautionary ones: an algebraically sound fix that
measured worse, a confident diagnosis refuted by a from-scratch control, and an
experiment whose own bug made it uninterpretable.

And rows 16–17 are why stage 4 exists. After everything above, two defects
remained — the ~10% energy deficit and the 12× dim-voxel shortfall — and both
were **properties of the coordinates, not of the network.** No architectural or
sampling change could reach them, which is what the two-stage factorization was
for.

---

## 6. Stage 4 — the two-stage model

### 6.1 The reframing

The two residual defects share a root cause. In the single-stage setup, one
network must simultaneously produce:

- the **total** energy and its longitudinal distribution (a tight conservation
  law, ~45 effective degrees of freedom), and
- the **per-voxel** shape, including voxels at 15 keV.

These have wildly different scales, and in a single standardized
representation the second is what sets the resolution. A relative error that is
negligible for the total energy is catastrophic for a faint voxel, and vice
versa. The network has to split its precision budget between them and does
badly at both ends.

**Factorize the joint instead:**

```
p(shower | E_inc) = p(u | E_inc) · p(w | E_inc, u)

u  (45 numbers)   total energy + its distribution along z, as bounded ratios
w  (6480)         each voxel's share of its OWN layer; every layer sums to 1
```

The two blocks are modelled by separate networks and recombined
**analytically** — the inverse is algebra, not learning. Measured round-trip
error on total energy: **3.4×10⁻⁸** relative.

### 6.2 Three separable mechanisms, each measurable

**1. Energy conservation stops being learned.** The total is one of 45 numbers
a small network models directly, and the shape model *cannot affect it at all*
because `w` is normalized per layer. E_tot/E_inc χ²: **0.127 → 0.0020**, a
factor of 63. The 10% deficit of §5.5 cannot exist here by construction.

**2. The logit representation becomes usable.** Once layer totals live in `u`,
`w` can go through a logit with α = 1e-6, which sends w → 0 to about −13.8 and
gives the network room at the faint end:

| | `fraction` rep. | `layer` rep. |
|---|---:|---:|
| empty voxel, standardized | 0.0 | −0.5488 |
| median lit voxel | ~5.5e-4 above empty | **1.729** |
| **empty-to-lit gap** | **5.5e-4** | **2.28** |
| voxel-distribution χ² | 0.024 | **0.0025** |

A factor of 4000 in resolution where the model was blind, and a 10×
improvement in the voxel distribution **from a change of variables alone, with
no architectural change**. This is the dim-voxel defect of §5.5 resolved.

It also removes the Jensen bias that made the log-of-absolute `threshold`
representation undershoot energy by 2.4×, because per-voxel means can no
longer move layer totals.

**3. Each network solves an easier problem.** Stage A is a 45-dimensional
density. Stage B is a *normalized* field with no scale freedom at all.

### 6.3 Configuration

| | Stage A (energy) | Stage B (shape) |
|---|---|---|
| variable | `u`, 45 | `w`, 6480 |
| trunk | FiLM MLP | conv 3D U-Net |
| hidden | 512 | 1024 |
| hit head | no (data is dense) | **yes** |
| conditioning | E_inc | E_inc **and all 45 `u`** |
| batch | 256 | 64 |
| steps | 20,000 | 100,000 |
| params | 993,325 | 2,474,242 |

Everything measured to work in stage 3 carries over unchanged: the hit head,
FiLM, cosine LR with warmup, EMA 0.999, the Heun sampler, the divergence
handling. **The factorization is orthogonal to all of it** — which is why it
was cheap to build once the pieces existed.

### 6.4 Result

| | cls-low | cls-high | cls-resnet | mean χ² |
|---|---:|---:|---:|---:|
| PINN film MLP | 0.999 | 1.000 | 1.000 | 0.182 |
| PINN conv U-Net | 0.683 | 0.925 | 0.894 | 0.0082 |
| **Two-stage** | **0.578** | **0.695** | **0.804** | **0.0017** |

Best or tied-best in six of seven histogram observable classes. Low-level AUC
statistically indistinguishable from CaloDiffusion's 0.577, which placed 2nd of
17 published submissions.

**And its remaining weakness, stated plainly:** cls-high 0.695 and cls-resnet
0.804 lag cls-low 0.578 by 0.10–0.12, which is ~8× the run-to-run scatter. We
match the second-best published submission on raw voxels and fall to ~4th on
derived quantities. The diagnosis follows from what each classifier sees:
per-voxel marginals are close to right, **voxel-to-voxel correlations are
not**. The deep-layer χ² values (E layer 40–44) localize it to the sparse
shower tail.

---

## 7. Two-stage PINN vs CaloDREAM

CaloDREAM (arXiv:2405.09629, SciPost Phys. 18, 088) is the best published
Dataset-2 submission at cls-low AUC 0.531. The factorization here is taken
directly from it, so the comparison is worth being precise about: **what is
shared is the problem decomposition; what differs is every network and the
generative framework itself.**

### 7.1 What is genuinely the same

| | Both |
|---|---|
| Factorization | `p(u\|E_inc) · p(x\|E_inc, u)`, two networks trained independently, linked at sampling |
| Energy ratios | **Identical definition.** Their eq. (2): `u₀ = ΣEᵢ/(f·E_inc)`, `uᵢ = Eᵢ/Σ_{j≥i}Eⱼ` for i=1..44 |
| Scale factor `f` | Both rescale u₀ into [0,1]. **They fix f = 2.85; we compute 1.05 × max = 2.9756 from the data.** |
| Logit preprocessing | **Identical.** Their eq. (3), α = 1e-6, then per-feature standardization |
| Per-layer normalization | Voxels normalized by their layer's energy, losing no information |
| Post-processing | Both renormalize generated voxels to restore exact per-layer sums |
| Conditioning on E_inc | Both log-transform and rescale |
| Analytic invertibility | Both rely on it to impose energy conservation exactly |

The agreement on the representation is exact, not approximate. That is
deliberate — it was adopted from the paper after our own measurements showed
the representation was the binding constraint.

### 7.2 Where the approaches genuinely differ

| Aspect | CaloDREAM | Two-stage PINN (ours) |
|--------|-----------|------------------------|
| **Generative framework** | Conditional Flow Matching. Linear trajectories; the network predicts the **velocity** `v(x,t)` directly, trained on `‖v((1−t)ε+tx, t) − (x−ε)‖²` | Stochastic interpolant with a **σ(t) schedule** (σ₀=1e-2). The network predicts the **denoiser** `q = E[x₀\|x_t]`; the velocity `G = [(1−σ₀)x − q]/σ(t)` is recovered **analytically** |
| **Time convention** | t=0 noise → t=1 data | t=1 noise → t=0 data |
| **Energy network** | **Autoregressive Transfusion**: transformer encoder/decoder with masked self-attention and cross-attention, generating `u` **layer by layer** to exploit the causality of energy deposition | A single **FiLM-conditioned MLP** modelling all 45 `u` **jointly**, non-autoregressively |
| **Shape network** | **3D Vision Transformer**: non-overlapping voxel patches (3,16,1 for DS2), embedding dim 480, 6 heads, 6 blocks, learnable position encoding | **3D conv U-Net**: 3 resolution levels (45,16,9)→(23,8,5)→(12,4,3), circular φ padding, FiLM per block, position channels |
| **Sparsity handling** | No occupancy head. Sparsity is addressed **post hoc** with a 1 MeV threshold cut | **A dedicated hit head** (per-voxel occupancy logit, BCE weight 0.1) gating the read-out, plus four selectable gate rules |
| **ODE solver** | Runge–Kutta 4, 50 steps (energy) / 20 steps (shape), uniform | **Heun (2nd order)** on a **quadratic grid** `t_k=(1−k/n)²`, 50–100 steps, concentrated near t=0 where `G ~ x/σ_t` is stiffest |
| **Fast sampling** | **Bespoke Non-Stationary (BNS) solvers** — learned solvers initialized to Euler, giving large speedups at fixed quality | None. Plain Heun |
| **Dimensionality reduction** | A separate **laViT** variant using an autoencoder, for DS3's 40,500 voxels | Tried and **abandoned** (§4) — our decoder ceiling was too low |
| **Training budget** | 500 epochs (energy, batch 4096), 800 epochs (shape, batch 64), max LR 1e-3 cosine | 20k steps (energy, batch 256), 100k steps (shape, batch 64), peak LR 2e-4 cosine + 500-step warmup, EMA 0.999 |
| **Datasets** | DS2 and DS3, separate energy networks per dataset | DS2 only |
| **Result (cls-low)** | **0.531** | 0.578 |

### 7.3 Which differences plausibly explain the remaining gap

The gap is 0.047 in cls-low — small — but 0.17 in cls-high and 0.12 in
cls-resnet, which is where the real difference lies. Our deficit is in
*derived* quantities and spatial correlations (§6.4), and three of the
differences above speak directly to that. Ranked by how much I would bet on
each:

1. **The shape network.** A ViT's self-attention is global: every patch
   attends to every other patch, so long-range correlations across the
   calorimeter are modelled directly. Our U-Net's receptive field grows only
   with depth, and at 3 levels over a 45×16×9 grid it is local-to-medium
   range. Since our measured defect is precisely *correlations* rather than
   marginals, this is the most likely single cause.
2. **The autoregressive energy network.** Energy deposition along z is
   genuinely causal — layer i's energy depends on what was already deposited.
   Transfusion models `p(uᵢ | u_{<i}, E_inc)` exactly; our joint MLP models the
   45 numbers in one shot and must learn that structure implicitly. This should
   show up in layer-energy observables, which is exactly what cls-high reads.
3. **Training budget.** 800 epochs over 100k showers at batch 64 is ~1.25M
   steps against our 100k. We are plausibly undertrained by an order of
   magnitude, and stage B was still improving when the run was cut.

Two further differences I would *not* bet on as causes:

- **CFM vs denoiser-parameterized interpolant.** These are closely related
  reparameterizations of the same object, and both are well-conditioned. Our
  σ(t) schedule plus analytic G is arguably better conditioned near t=0. I
  would be surprised if this mattered.
- **The hit head vs their 1 MeV cut.** Ours is the more principled mechanism
  and our sparsity metrics are good. Their needing a post-hoc cut suggests they
  have the *harder* time here, not us.

**An honest caveat on all of the above:** these are hypotheses ranked by
plausibility, not measurements. The cheap experiments that would test them —
averaging our classifiers over 10 runs to make the comparison precise, running
FPD/KPD, and simply training stage B longer — have not been done. Nothing in
§7.3 should be quoted as a finding.

### 7.4 What CaloDREAM independently confirms about our own work

Three of this project's conclusions, reached by our own measurements, are
corroborated by the paper:

- **FiLM-style affine conditioning** — they use it inside the ViT. We arrived
  at it from the per-energy-bin response defect (§5.3).
- **Circular padding in φ** — their "cylindrical convolution". We arrived at it
  from the geometry.
- **The latent ceiling** — their laViT variant's sparsity failure is attributed
  by the authors to the autoencoder reconstruction, not the generator (§4.6).
  Same diagnosis, same route.

---

## 8. Hyperparameters, as they changed

### 8.1 The PINN sampler, across the sequence

| | concat MLP | film MLP | conv U-Net | two-stage B |
|---|---|---|---|---|
| trunk | 3×Linear(hidden) | 3×Linear + FiLM | 3-level 3D U-Net | 3-level 3D U-Net |
| hidden / width knob | 1024 | 1024–2048 | 1024 → c=(32,64,128) | 1024 |
| params | 22.8M | 22.8–49.8M | 2.47M | 2.47M |
| representation | `fraction` | `fraction` | `fraction` | **`layer` logit** |
| steps | 8k–16k | 100k | 8k–16k | 100k |
| batch | 32–64 | 64 | 64 | 64 |
| peak LR | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| schedule | cosine→0 + 500 warmup | same | same | same |
| EMA | 0.999 | 0.999 | 0.999 | 0.999 |
| grad clip | 1.0 | 1.0 | 1.0 | 1.0 |
| hit weight | 0.1 | 0.1 | 0.1 | 0.1 |
| σ₀ | 1e-2 | 1e-2 | 1e-2 | 1e-2 |
| sampling | 100 Heun | 100 Heun | 100 Heun | 50 Heun |
| gate | topk | topk | topk | topk |
| read-out | denoise | denoise | denoise | **raw** |

Notably **the optimizer settings never changed.** Peak LR 2e-4, cosine to zero,
500-step warmup, EMA 0.999 and grad clip 1.0 were set early and were never
implicated in any defect. Every real improvement came from the
parameterization, the representation, the conditioning or the trunk.

One hyperparameter that is a trap: **`--steps` sets the cosine schedule, not
just a stopping point.** And with the plain unpreconditioned net, samples got
measurably *worse* past ~8k steps at batch 32, and blew up to inf by 16k at
constant LR. The decay to zero is what makes long runs stable.

### 8.2 The autoencoder

| | v3 | v4 |
|---|---|---|
| decoder output | 1 channel (shared) | **2 channels** (value, hit logit) |
| loss | log-space MSE + L1 + hit BCE + energy L1 | **rel² + 0.1·BCE + 1.0·soft-gated e²** |
| per-voxel weighting | `1 + 5·x` (then `1 + 10·x_log`) | **none** |
| target scaling | raw (~1e-4) | **`x_scale` ≈ 8e-4** |
| units | `shower_MeV / energy_GeV` (bug) | `shower_GeV / energy_GeV` |
| φ padding | zero | **circular** |
| latent | 8×12 = 96 | 8×12 = 96 (now sweepable) |
| optimizer | AdamW 3e-4, wd 1e-5 | same |
| batch | 4096 | 4096 |
| schedule | cosine, 5 warmup epochs | same |
| **rel error** | **0.688** (worse than mean shower) | **0.367** (beats PCA-512) |

Note: the v4 column describes the workstation checkpoint. The AE results in
[MODEL_COMPARISON.md](MODEL_COMPARISON.md) are the pre-v4 model's (§4.7).

### 8.3 Tests built along the way, and what each caught

| Test | What it exists to catch | Did it catch something? |
|------|-------------------------|-------------------------|
| `ae_baselines.py` | a model worse than PCA / the mean shower | **Yes** — the entire v3 plateau |
| `train_latent.py --ceiling` | crediting the sampler for the decoder's limits | **Yes** — settled §4 |
| `fixed_eval` | a loss curve too noisy to read (3× swings from t alone) | **Yes** — made plateaus visible |
| `sweep.py` width sweep | capacity vs inductive bias | **Yes** — refuted the capacity hypothesis |
| `sweep.py` gate sweep | sample-time choices costing a retrain | Yes — four rules compared in seconds |
| `compare_to_reference.py` | a units/threshold artifact faking a bad AUC | Ruled one out (valuable negative) |
| `selftest.py` (21 tests) | regressions in the bugs that cost real runs | **Yes** — the Gumbel precedence bug, immediately |
| `--check-denoiser` | lit/empty decomposition of the energy error | **Yes** — showed §5.5's cancellation |
| `quick_metrics` voxel distribution | per-voxel defects invisible to aggregates | **Yes** — the 12× dim-voxel shortfall |

---

## 9. What was concluded wrongly

Four corrections, each stated confidently before being tested.

**1. "The radial profile error is an architecture limit."** Attributed to the
flat MLP's lack of geometry and declared insensitive to training. It then
nearly **halved** under more training and width (0.184 → 0.123). The
architectural claim was right in the end — the conv trunk did fix it — but the
*insensitivity to training* claim was wrong, and was asserted without a
control.

**2. "The divergent showers are integrator stiffness."** Based on a retry that
fixed them at 400 steps. A from-scratch run at 400 steps showed the **same
divergence rate** — the fresh noise draw had been doing the work, not the step
count. FiLM later cut divergences 236 → 1, confirming it was a conditioning
symptom.

**3. "The latent sampler matches its ceiling, so the formalism works at 96
dimensions."** True, but the test was trivial: v3's latents were nearly
deterministic given E_inc (§4.2). The conclusion survived re-testing on v4, but
it had rested on nothing.

**4. "`--value-correct` should undo the occupancy dilution."** Algebraically
correct and measurably worse (E_dep/E_inc 0.780 → 0.963). The reasoning was
incomplete: the aggregate had been right only through cancellation (§5.5).

**5. "The reconstruction errors I measured are the models' reconstruction
errors."** They were measured with a decoder at the wrong intermediate
resolution. The encoder downsamples the transverse plane (16, 9) → (8, 5) →
(4, 3), so a mirrored decoder must go (4, 3) → **(8, 5)** → (16, 9). The
training script had been corrected to that at some point, but **every
checkpoint in this project was trained with (8, 4)**, and two standalone
scripts still carried the (8, 4) version.

Conv weights do not encode spatial size, so an (8, 4)-trained checkpoint loads
into an (8, 5) model with **zero missing or unexpected keys** and silently
reconstructs about 10% worse. Nothing errors; nothing warns.

Corrected values, measured at the resolution each model was actually trained
with:

| checkpoint | measured at (8,5) | true, at (8,4) |
|---|---:|---:|
| `best_calo_ae_cond_improved.pt` | 0.4307 | **0.3947** |
| latent sweep, 8 | 0.5980 | **0.5886** |
| latent sweep, 24 | 0.5559 | **0.5431** |
| latent sweep, 48 | 0.4878 | **0.4463** |
| latent sweep, 96 | 0.4314 | **0.3983** |
| latent sweep, 128 | 0.3943 | **0.3507** |

Every model is better than reported, by 0.01–0.044.

The decoder resolution is now an explicit constructor argument, recorded in
every new checkpoint, and `from_checkpoint()` defaults a file without that
field to the legacy (8, 4) with a printed note. The v4 figure of 0.367 was
measured on the workstation by the training script itself and is
self-consistent with whatever that script used; it is **not** independently
verified here.

This one was found only because a request to deduplicate two "verbatim" copies
turned out to involve copies that were not verbatim — and the one that
disagreed with the main script was the one that matched the data.

**The pattern.** Every one of these was a *mechanism* asserted from a plausible
story, where the measurement available at the time was consistent with several
mechanisms. The fix in each case was a control: the same experiment with one
variable changed. The width sweep (§5.4) is the model of how this should go —
it was designed specifically to separate two hypotheses, and it did.

---

## 10. What is still open

All five are written up as runnable experiments, each with a control and a
falsification criterion, in [EXPERIMENTS.md](EXPERIMENTS.md).

In the order the measurements support:

1. **FPD/KPD for the two-stage model.** One command, no retraining. These are
   the metrics the summary paper leads with, and the only measurement we have
   is the autoencoder's, which is worse than the worst published submission.
2. **Voxel-to-voxel correlations** — the binding defect (§6.4). cls-high and
   cls-resnet lag cls-low by 0.10–0.12. §7.3 ranks three candidate causes;
   a global-attention shape network is the leading one.
3. **Train stage B longer.** It was still improving, and CaloDREAM's shape
   network saw roughly 10× more steps.
4. **Average classifiers over 10 runs**, as the published tables do. Our
   ±0.015 single-run scatter makes comparisons at the published ±0.004
   precision meaningless.
5. **The autoencoder, if the latent path is to be revisited at all.** PCA-256
   (0.461) and PCA-512 (0.387) show headroom in bottleneck size, and the
   decoder's φ-position fidelity (§4.5) is the specific thing to fix. §4.6's
   two independent corroborations argue for leaving this alone.

---

## References

- CaloChallenge 2022 summary — arXiv:2410.21611 (Tables C15, C16)
- CaloDREAM — arXiv:2405.09629, SciPost Phys. 18, 088
- CaloDiffusion — arXiv:2308.03876
- Conditional Flow Matching — Lipman et al., arXiv:2210.02747
- EDM preconditioning — Karras et al., arXiv:2206.00364
- χ² metric definition — eq. 15 of arXiv:2009.03796
