# End-to-end replication guide

CaloChallenge Dataset 2 generative models. Everything needed to reproduce
each model from raw data: preprocessing, architecture, loss, training
schedule, sampling, export and evaluation.

Companion documents:
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md) — results, ranking, comparison
  with the 17 published submissions.
- [EXPERIMENTS.md](EXPERIMENTS.md), and
[NOVA_TRANSFER.md](NOVA_TRANSFER.md) for the sparse-autoencoder techniques
adapted from the NOvA CAE report — the open experiments, with controls and
  predictions.
- [MODEL_EVOLUTION.md](MODEL_EVOLUTION.md) — why each model was built, what
  test exposed its limit, and how the two-stage design differs from CaloDREAM.
- `docs/PINNDE_LAB_NOTEBOOK.md` — the running lab notebook: every experiment tried,
  including the ones that failed.

Every number quoted as "measured" in this document came from a run in this
repository. Where a design choice was tested and rejected, that is stated
with the numbers, because knowing what *not* to try is most of the value.

---

## Table of contents

1. [Environment and data](#1-environment-and-data)
2. [The problem](#2-the-problem)
3. [Preprocessing — three representations](#3-preprocessing--three-representations)
4. [The generative formalism](#4-the-generative-formalism)
5. [Network architecture](#5-network-architecture)
6. [Loss functions](#6-loss-functions)
7. [Training](#7-training)
8. [Sampling](#8-sampling)
9. [Export and evaluation](#9-export-and-evaluation)
10. [Model A — two-stage PINN (recommended)](#10-model-a--two-stage-pinn-recommended)
11. [Model B — single-stage conv U-Net](#11-model-b--single-stage-conv-u-net)
12. [Model C — autoencoder](#12-model-c--autoencoder)
13. [Model D — latent diffusion](#13-model-d--latent-diffusion)
14. [Verification](#14-verification)
15. [Pitfalls](#15-pitfalls)

---

## 1. Environment and data

### 1.1 Dependencies

```
python >= 3.10
torch >= 2.0        # CUDA build
h5py
numpy
matplotlib          # plots only
tqdm                # autoencoder progress bars only
```

No other dependencies. The PINN code (`src/calods2/`) imports only `torch`,
`numpy` and `h5py`.

### 1.2 Hardware

| Model | VRAM | Wall time |
|-------|-----:|----------:|
| Two-stage (A 20k + B 100k steps) | ~12 GB | ~14 h on an RTX 5090 |
| Single-stage conv (16k steps) | ~10 GB | ~3 h |
| Autoencoder (250 epochs, batch 4096) | ~20 GB | ~20 h |
| Sampling 100k showers, 50 Heun steps | ~8 GB | ~40 min |

All of it fits in 12 GB except the autoencoder, which needs batch 4096;
lower it with `CALO_BATCH_SIZE=1024`.

### 1.3 Getting the data

Download Dataset 2 from the CaloChallenge Zenodo record
(https://calochallenge.github.io/homepage/). You need two files:

- `dataset_2_1.hdf5` — 100k showers, used for **training**
- `dataset_2_2.hdf5` — 100k showers, used as the **evaluation reference**

Never train on `dataset_2_2.hdf5`. It is the file `evaluate.py` compares
against, and training on it makes every metric meaningless.

Convert the training file to the tensor format the code expects:

```python
import h5py, torch
with h5py.File('dataset_2_1.hdf5') as f:
    showers = torch.tensor(f['showers'][:], dtype=torch.float32)
    energies = torch.tensor(f['incident_energies'][:], dtype=torch.float32)
torch.save({'showers': showers.reshape(-1, 45, 16, 9),
            'energies': energies.reshape(-1, 1)}, 'data/Dataset2.pt')
```

**Units: both arrays are in MeV.** The HDF5 stores MeV and this conversion
keeps MeV. Getting this wrong is the single most expensive mistake available
here — see §15.1.

Verify:

```
showers  (100000, 45, 16, 9)  float32  min 0.0      max 6103.57   [MeV]
energies (100000, 1)          float32  min 1000.55  max 999795.8  [MeV]
```

### 1.4 Geometry

| Axis | Size | Meaning | Periodic? |
|------|-----:|---------|-----------|
| z | 45 | longitudinal layer (shower depth) | no |
| φ | 16 | azimuthal angle | **yes** |
| r | 9 | radial bin | no |

6480 voxels total. The φ periodicity matters: it is wired into every
convolution in this codebase as circular padding, and asserting a wall at
φ = 0 (zero padding, the naive default) is wrong.

Readout threshold: **15.15 keV**. Voxels below it are zero in the data, and
generated showers must have the same cut applied before export. About 75% of
voxels are exactly zero.

---

## 2. The problem

Learn `p(shower | E_incident)` over 6480 voxels, conditioned on incident
energy spanning three decades (1 GeV – 1 TeV, log-uniform), such that a
trained classifier cannot distinguish generated showers from Geant4 ones.

Three properties of the data drive every design decision:

1. **Extreme sparsity.** ~75% of voxels are exactly zero, and occupancy is
   spatially correlated — a shower is a connected object, not random dust.
2. **Enormous dynamic range.** Voxel energies span ~15 keV to ~6 GeV, and
   total deposited energy tracks incident energy across three decades.
3. **A hard conservation law.** Deposited energy is a tight function of
   incident energy (median E_dep/E_inc = 0.7826). A model that gets this
   wrong by 10% is visibly wrong, and no amount of shape accuracy hides it.

Sparsity is addressed by a dedicated occupancy head (§5.4), dynamic range by
the logit representation (§3.3), and conservation by the two-stage
factorization (§10) — which makes it exact rather than learned.

---

## 3. Preprocessing — three representations

The codebase implements three. **The choice of representation mattered more
than any architectural change**, so this section is worth reading before any
other.

### 3.1 `fraction` — the baseline (`gendata.py`)

```
x = log1p(E_voxel_GeV / E_inc_GeV)
x0 = (x - mean(x)) / std(x)          # single GLOBAL scalar mean/std
```

**Why standardize at all.** The interpolant adds unit-variance Gaussian
noise. In raw `log1p` space the data occupies a ball of radius ‖x₀‖ ≈ 0.0555
while the noise lives on a shell of radius √6480 ≈ 80.5 — the noise is 1450×
larger than the signal. The vector field is then ~99.9% noise direction
(measured ‖G‖ = 79.61 at every t), so the physics residual can be driven to
~1.5 and still carry error 30× larger than the entire data signal. This is
why the PINN residual and data terms were irreconcilable before
standardization: they were describing different things.

**Why a single global scalar, not per-dimension.** ~75% of voxels are exactly
zero and many are near-always-zero. Per-dimension division would blow up dead
voxels into pure amplified noise. Global scaling preserves relative structure
across voxels.

**Its fatal flaw.** Voxel fractions are ≤ 0.28, so `log1p` is nearly the
identity, and an empty voxel and a 15.15 keV hit end up a median
**5.5 × 10⁻⁴ standardized units apart** — far below any achievable
regression error. The model literally cannot resolve "empty" from "faint".
Measured consequence: dim voxels (<1 MeV) came out 12× too rare, and the
5th-percentile lit voxel was 3.4 MeV against a real 0.63 MeV.

### 3.2 `threshold` — an intermediate fix

```
x = log1p(E_voxel_MeV / 0.01515)
```

Empty = 0, threshold = log 2, 1 GeV = 11.1. The empty-to-hit gap becomes 0.22
standardized units — 400× larger. But being log-of-absolute-energy, the
per-voxel conditional means bias the total energy downward through a Jensen
effect, measured at a 2.4× undershoot. Useful as a diagnostic, not for
production.

### 3.3 `layer` — the two-stage representation (`layerdata.py`)

This is the one to use. It splits a shower into two pieces modelled
independently, with an **exact analytic inverse**.

**Energy block `u` — 45 numbers:**

```
u[0] = E_tot / (f · E_inc)                              ∈ [0, 1]
u[i] = E_{i-1} / Σ_{j≥i-1} E_j        for i = 1..44      ∈ [0, 1]
```

Each `u[i]` for i ≥ 1 is the fraction of *remaining* energy deposited in layer
i−1. The 45th layer is implied by the other 44, so 45 numbers describe 45
layers exactly. `f` (`f_scale`) is a constant chosen as
`1.05 × max(E_tot/E_inc)` so `u[0]` stays inside (0, 1); **measured
f_scale = 2.9756** over the full 100k training set. Note this is
data-dependent: computing it on a subset gives a smaller value and a shower
with a larger E_tot/E_inc would then encode to u[0] > 1.

**Shape block `w` — 6480 numbers:**

```
w = E_voxel / E_layer       → every layer sums to exactly 1
```

Layer totals carry *no* information in `w`. All scale lives in `u`.

**Both blocks then go through a logit transform**, which is the piece the
earlier representations were missing:

```
logit(p) with α = 1e-6:   p ← α + (1-2α)·clamp(p, 0, 1)
                          y = log(p) - log1p(-p)
```

`logit(w)` sends w → 0 to about −13.8, giving the network room to represent
faint voxels. Standardization then follows: **per-dimension** for `u` (45
active dimensions, no dead coordinates) and a **single global scalar** for
`w` (same reasoning as §3.1).

Measured constants on Dataset 2:

```
f_scale   = 2.9756
w_mean    = -11.6914     w_std = 3.8700
empty voxel sits at       -0.5489   (standardized)
median lit voxel at        1.7286
```

**The empty-to-lit gap is now 2.28 standardized units**, against 5.5 × 10⁻⁴
in the `fraction` representation — a factor of 4000. This single change
dropped voxel-distribution χ² from 0.024 to 0.0025.

**Round-trip accuracy.** `decode(encode(x)) == x` to a measured
**3.4 × 10⁻⁸ relative error on total energy.** The inverse is analytic, not
learned, so energy conservation is exact by construction.

Why the logit also fixes the energy bias: because layer totals live in `u`,
the per-voxel means in `w` can no longer bias total energy. That removes the
Jensen effect of §3.2 entirely.

### 3.4 Energy conditioning (all models)

```
log_e = log10(E_inc_GeV)
e_cond = (log_e - mean) / std
```

Measured on Dataset 2: **mean = 1.4973, std = 0.8660** (E in GeV). A single
standardized scalar, saved in every checkpoint. Never re-derive it at sampling time from a
different sample — use the saved value.

---

## 4. The generative formalism

### 4.1 Stochastic interpolant

A noising path from data (t=0) to noise (t=1):

```
x(t) = α(t)·x₀ + σ(t)·ε           ε ~ N(0, I)
α(t) = 1 - t
σ(t) = σ₀ + (1 - σ₀)·t            σ₀ = 1e-2
```

At t=1: pure noise. At t=0: the clean shower plus a residual σ₀ = 0.01 of
noise in every voxel — which the read-out removes (§8.2).

### 4.2 Reverse-time ODE

Generation integrates backwards from t=1 to t=0:

```
dx/dt = G(t, x, E)
G(t, x, E) = [(1 - σ₀)·x - q(t, x, E)] / σ(t)
q(t, x, E) = E[x₀ | x_t, E]
```

`q` is the posterior mean of the clean shower given the noisy state — the
denoiser. `G` follows analytically.

### 4.3 What the network parameterizes, and why

**The network predicts `q`, and `G` is recovered algebraically.** This is the
single most important decision in the project, and it was reached by
measuring the alternative fail.

The original PINNDE formulation has the network learn the **flow map**
`x(t; z)` — the whole trajectory as a function of its initial noise. At 1–3
dimensions this works. At 6480 it does not, quantitatively:

- **It cannot generalize.** 512 → 4096 anchor trajectories (8× more data) cut
  held-out MSE only 1.84×, i.e. MSE ~ N^−0.29. Reaching usable error would
  need ~10⁶× more data.
- **It cannot train from cold start.** The physics residual is evaluated at
  the model's own prediction, which early in training is off-manifold, where
  the q-kernel degenerates and G ~ x/σ_t becomes radially repulsive. The
  residual *rose* during training (measured 6862 → 22002) — it could not
  minimize its own objective.
- **Result:** samples came out worse than noise (E_dep/E_inc ≈ 3.4 against a
  true 0.78).

Predicting the local field `q` fixes all three at once:

- `q` is a **local** function of the current state — smooth, low-complexity,
  so it generalizes instead of memorizing.
- Training data is **unlimited and always on-manifold**: take any real x₀,
  any t, any ε, form x_t, and you have an exact target. No trajectory
  generation, no anchor set, no collocation points.
- `q` is **bounded** (a convex combination of real showers, ‖q‖ ≈ 68) while
  `G` ranges 90–600 across t. Predicting the well-conditioned quantity and
  recovering the stiff one analytically is the same reason standard diffusion
  models predict x₀ or ε rather than the raw score.

Measured head-to-head: E_dep/E_inc 0.92 against a true 0.78, versus 3.4 for
the flow-map ansatz.

**The physics-informed content therefore lives in the parameterization, not
in a residual penalty term.** The ODE structure, the σ(t) schedule and the
analytic G are the physics; the loss is a plain regression (§6.1). An
explicit residual loss was tried and measurably hurt — see §15.4.

---

## 5. Network architecture

`VelocityNet` in `src/calods2/models/velocity_net.py`. One class, three
interchangeable trunks selected by `cond=`.

### 5.1 Conditioning features (shared)

```python
e, extra = energy[:, :1], energy[:, 1:]
a = t * freqs * π          # freqs = 2^[0..n_freq-1], n_freq = 16
b = e * freqs * π
features = [t, e, sin(a), cos(a), sin(b), cos(b), extra]     # 4·n_freq + 2 + cond_extra
```

Sinusoidal features of both diffusion time and standardized log-energy. The
`extra` channel is how the two-stage shape model receives the 45 energy
numbers (`cond_extra=45`) without changing any call signature.

These go through a 2-layer MLP to a 128-dim conditioning vector `c`, which
drives FiLM modulation everywhere.

### 5.2 Trunk options

**`cond="concat"` — flat MLP, two extra inputs.** Baseline only.

```
Linear(1 + 6480 + 1 + cond_extra → hidden) → SiLU
Linear(hidden → hidden) → SiLU            ×2
Linear(hidden → out_dim)
```

Why it is insufficient: the *mean* E_dep/E_inc comes out right (0.79 vs a
real 0.78) while the per-energy response does not — 1–10 GeV 15% high and
over-dispersed, 10–100 GeV 8% low, 100+ GeV 3% low. Getting the average right
but the dependence wrong is exactly what weak conditioning looks like, and
two scalars competing with 6480 inputs at the first layer is a weak way to
condition.

**`cond="film"` — flat MLP + FiLM.** Same depth and width as `concat`, so an
A/B isolates the conditioning. Each hidden activation is modulated:

```
h = layer(h)
scale, shift = film(c).chunk(2)
h = SiLU(h · (1 + scale) + shift)
```

The FiLM projections are **zero-initialized**, so the model starts at
scale 1 / shift 0 (identity modulation) and learns the conditioning. This cut
sampling divergences from 236 to 1 and fixed the energy over-dispersion.

Still insufficient: a flat MLP has no notion that voxel 100 and voxel 101 are
neighbours. **A width sweep at fixed budget proved this is not a capacity
problem** — 512 → 1024 → 2048 hidden (10.9M → 49.8M params) left the
per-voxel defect completely flat: dim-voxel fraction 0.0050 / 0.0065 / 0.0064
against a real 0.072, and 5th-percentile voxel energy 3.55 / 3.36 / 3.42 MeV
against a real 0.63. Profile errors improved from 512 to 1024 and then
stopped. The limit is the lack of spatial structure, not the size.

**`cond="conv"` — 3D U-Net over (z, φ, r). Use this.**

```
input: (B, 1, 45, 16, 9), or (B, 3, …) with position channels

enc0: ConvBlock3D(3 → c1)            (B, c1, 45, 16, 9)
enc1: ConvBlock3D(c1 → c2, stride 2) (B, c2, 23,  8, 5)
enc2: ConvBlock3D(c2 → c3, stride 2) (B, c3, 12,  4, 3)
mid:  ConvBlock3D(c3 → c3)
dec1: upsample(nearest) + skip h1 → ConvBlock3D(c3+c2 → c2)
dec0: upsample(nearest) + skip h0 → ConvBlock3D(c2+c1 → c1)
out:  Conv3d(c1 → n_blocks, k=3)     zero-initialized

c1 = max(8, hidden//32),  c2 = 2·c1,  c3 = 4·c1
```

With `hidden=1024`: c1, c2, c3 = 32, 64, 128 → **2.47 M parameters**, i.e.
9× *fewer* than the film MLP and 22× better mean χ².

`_ConvBlock3D` is Conv → GroupNorm → FiLM → SiLU, twice, with a residual
shortcut:

```python
h = norm1(conv1(pad_phi_circular(x)))
scale, shift = film(c).chunk(2)
h = silu(h * (1 + scale) + shift)
h = norm2(conv2(pad_phi_circular(h)))
return silu(h + skip(x))
```

**Circular padding in φ, zero padding in z and r.** φ wraps around the
calorimeter; z and r end there. `padding=(1, 0, 1)` on the conv plus an
explicit circular pad on the φ axis.

**Position channels.** The normalized layer index z and radial index r,
concatenated to every input. A convolution is translation-invariant by
construction, but a calorimeter is not — depth and radius both matter, and
the dim voxels we under-produce live at the edges. CaloDiffusion
(arXiv:2308.03876) feeds exactly these.

**The output conv is zero-initialized**, so the network starts predicting
q = 0 and learns from there.

### 5.3 Output blocks

`out_dim = shower_dim × n_blocks`, where n_blocks counts the enabled heads:

| Block | Always? | Purpose |
|-------|---------|---------|
| value | yes | `q` — the posterior-mean clean shower |
| hit | `hit_head=True` | per-voxel "is this lit in x₀" logit |
| lit value | `lit_head=True` | `E[value \| lit]`, read-out only |

### 5.4 The hit head — why it is essential

The q-regression is a **conditional mean**. A voxel that is uncertain between
empty and lit comes out at an intermediate value, which in any continuous
representation is a small spurious deposit. Measured: **77% of generated
voxels landed above the 15.15 keV threshold against 25% for real showers**,
carrying the excess energy with them.

Thresholding the output cannot fix this without also destroying real small
deposits (measured). A separate occupancy head can, because occupancy is a
classification problem with exact labels.

**The hit head does not enter the ODE.** `G` is built from `q` alone. The hit
head only gates the final read-out at t=0, where the sample is handed back as
a shower. Gated voxels are set to *exactly* `empty_value`, so they invert to
exactly 0 MeV.

### 5.5 EDM preconditioning — implemented, measured harmful

`coeffs(t)` implements Karras et al. (2022) preconditioning:
`q̂ = c_skip·x_t + c_out·F(c_in·x_t, t, E)`.

The motivation is real: without it the MLP must pass the identity through its
hidden layers (at t≈0 the ideal output is ≈ the input), and it cannot —
measured, the plain net recovers only 64–79% of the RMS structure of real
showers at t=0.

**But it measured worse** with a single global σ_d, and is off by default
(`precond=False`). Kept in the code for comparison. If retried, it needs a
per-dimension or per-t data-scale estimate.

---

## 6. Loss functions

### 6.1 The PINN loss

```python
def loss(t, xt, energy, y, hit_target, hit_weight=0.1):
    v, h, u = self._raw(t, xt, energy)
    mse = (v - y).pow(2).mean()                       # y = x0, the clean shower
    total = mse + hit_weight * BCEWithLogits(h, hit_target)
    if u is not None:                                  # lit_head
        n = hit_target.sum().clamp(min=1.0)
        total = total + ((u - y).pow(2) * hit_target).sum() / n
    return total
```

Three components, and the weighting of the first two is the whole objective:

**1. Value MSE against the clean shower x₀.** Not against a Monte-Carlo
estimate of `q`. The MSE minimizer of `E[(v − x₀)²]` over the training
distribution *is exactly* `E[x₀ | x_t, E]`, which is `q`. So regressing the
clean shower gives the right target for free, with no reference ensemble.

This was tested head-to-head against the Monte-Carlo target (`target="mc"`,
8k steps, same net/batch/seed/noise). **Clean won on every diversity
metric:**

| metric | clean | mc (n_ref 128–512) | real showers |
|--------|------:|-------------------:|-------------:|
| spread across showers | 0.276 | 0.149–0.176 | 0.408 |
| pairwise distance | 58 | 24–27 | 86 |
| distance to nearest training shower | 32 | 7–9 | 47 |

and trained 2–4× faster. The mc target collapses samples toward the mean, and
a larger reference ensemble does not fix it (128 → 512 moved spread only
0.149 → 0.176 at 2.3× the cost). The underlying reason: the exact
finite-reference q-field is a *memorizing* sampler — run as an ODE it lands
within 1.76 of a training shower, against 47 for a genuinely new one — so it
is a poor thing to distill.

**2. Hit-head BCE, weight 0.1.** Target is `x0 > empty_value + 1e-6`, i.e.
"this voxel is lit in the clean shower". The data is already cut at 15.15 keV,
so this reproduces the real sparsity directly.

**3. Lit-value head (optional, off by default).** Supervised only where the
voxel is lit. **The one experiment in this project whose result is
untrustworthy:** its loss was normalized by the number of *lit* voxels rather
than all voxels, making it ~25× too large and damaging `q`. If revisited, fix
the weighting first.

### 6.2 The autoencoder loss (`calorimeter_loss_v4`)

```python
true = expm1(x_log)
pred = expm1(v * x_scale)
rel2 = ((pred - true)².sum(1) / true².sum(1)).mean()        # per-shower
bce  = BCEWithLogits(h, (x_log > 0).float())
p_e  = pred.clamp(min=0) * sigmoid(h)                        # SOFT-gated
e2   = ((p_e.sum(1) - e_true) / e_true)².mean()
total = rel2 + 0.1 * bce + 1.0 * e2
```

Every term here replaced something that was measurably broken in v3. The v3
model **reconstructed worse than the energy-binned mean shower** (rel 0.688
vs 0.638; linear PCA with the same 96 numbers got 0.539). The flat validation
curve was the *objective*, not capacity. Diagnosis of each v3 term:

- **Hit BCE shared the value output.** It needed logits of ±3 from a number
  that must also equal ~4 × 10⁻⁴. It was 63% of the loss and stuck at 0.496 —
  worse than a constant predictor (0.361). Fixed by a separate head `h`.
- **The energy-sum L1 had the largest gradient** (3× the rel term) with
  cosine 0.015 against it — it moved every voxel together, carrying no shape
  information. The physical-space term was 5.9 × 10⁻⁷: inert.
- **log1p(E/E_inc) is nearly linear with values ~1e-4**, so log-space MSE/L1
  barely saw shower shape at all. Fixed by `rel2`, which is scale-free.

Two subtleties worth copying:

- **`rel2` is the square of the reported metric**, so objective and metric
  agree. Optimizing something that isn't what you report is how v3 plateaued
  invisibly.
- **The energy term uses the SOFT-gated prediction** `pred · sigmoid(h)`, not
  the raw values. The reported output zeros voxels the hit head calls empty,
  so an energy term on ungated values optimizes energy the gate then removes.
  Measured: with ungated `e2` the gated energy error got *worse* in epoch 1
  (0.29 → 0.37).

### 6.3 `x_scale`, and why the decoder output is linear

`x_scale` is the global std of `x_log` over a 20k-shower subset of the train
split, ≈ 8 × 10⁻⁴. The model works on `x_log / x_scale` so its inputs and
value head are O(1) instead of ~1e-4. It is saved as a buffer so a resumed
run keeps the value it trained with.

**The decoder's final activation is linear, deliberately.** In log space the
target for an empty voxel is exactly 0, and Softplus has a floor near 0.69
that would prevent the model from ever predicting it.

---

## 7. Training

### 7.1 The PINN training loop

```python
for step in range(1, steps + 1):
    lr = lr_peak * min(1, step/warmup) * 0.5 * (1 + cos(π·step/steps))
    idx = randint(0, N, (batch,))                  # fresh sample every step
    x0b, eb = x0_all[idx], e_all[idx]
    t = rand(batch, 1)                             # uniform in [0, 1]
    xt = (1-t)*x0b + (σ₀ + (1-σ₀)*t) * randn_like(x0b)
    hit_target = (x0b > empty_value + 1e-6).float()
    loss = model.loss(t, xt, eb, x0b, hit_target, hit_weight)
    loss.backward()
    clip_grad_norm_(params, 1.0)
    opt.step()
    ema.mul_(0.999).add_(params, alpha=0.001)       # EMA is what you sample
```

| Hyperparameter | Value | Note |
|----------------|-------|------|
| optimizer | Adam | no weight decay |
| peak LR | 2e-4 | |
| warmup | 500 steps, linear | |
| schedule | cosine to **0 at `steps`** | `steps` sets the schedule, not just a stop |
| grad clip | 1.0 | |
| EMA decay | 0.999 | **sample from the EMA, not the raw weights** |
| hit weight | 0.1 | |
| batch | 64 (conv), 256 (45-dim energy model) | |

**There is no epoch structure and no fixed training set pass.** Every step
draws fresh showers, a fresh t and fresh noise, so every training point lies
on the interpolant manifold and the effective dataset is unlimited.

**`--steps` is not a stopping point, it is the schedule.** The cosine decays
to zero at `steps`. Resuming with a different value puts the LR on a
different curve (see §15.5).

### 7.2 Deterministic validation — `fixed_eval`

The per-step training loss is **uninformative**, and this is not a minor
annoyance. The objective's value depends strongly on t — near the data
variance at t=1, near 0 at t=0 — so consecutive printed losses differ by 3×
(measured 0.07 to 0.27 within one run) for reasons that have nothing to do
with learning. The trend is completely hidden.

`fixed_eval` holds everything but the weights fixed: **fixed showers (the last
256), fixed noise (seed 1234), fixed t grid (8 points at (i+0.5)/8)**. Now a
flat curve really means "not improving". It returns the mean and the per-t
breakdown, which is the more useful read — a model usually stops improving at
high t long before low t.

**Caveat, measured:** validation loss and sample quality come apart. The
pre-hit-head MLP degraded in sample quality while its loss stayed flat. So the
`.best`-by-val checkpoint is a **weak** selector, and sampling uses the latest
EMA by default.

### 7.3 Checkpointing and resume

Every 1000 steps the checkpoint saves EMA weights, raw weights, optimizer
moments, step number, full architecture config, `stats`, and loss history.
`resume=True` restores all of it and continues on the same schedule.

A separate `<path>.best` copy is written when validation improves — read §7.2
before trusting it.

### 7.4 Autoencoder training

| Hyperparameter | Value |
|----------------|-------|
| split | 90/10 train/val, `manual_seed(42)` |
| batch | 4096 (`CALO_BATCH_SIZE` to override) |
| optimizer | AdamW, lr 3e-4, weight decay 1e-5 |
| schedule | cosine, 5 warmup epochs |
| grad clip | 1.0 |
| AMP | fp16 + GradScaler on CUDA |
| epochs | 250 (improvement stops ~250; ran to 400 with no gain) |

---

## 8. Sampling

### 8.1 Heun integration on a power grid

```python
x = randn(n, shower_dim)
grid = [(1 - k/n_steps)² for k in range(n_steps + 1)]      # t: 1 → 0
for k in range(n_steps):
    t0, t1 = grid[k], grid[k+1];  h = t0 - t1
    G1 = model.G_hat(t0, x, E)
    G2 = model.G_hat(t1, x - G1*h, E)
    x = x - (G1 + G2) * h / 2
```

Second-order Heun: two network calls per step, instead of the
O(n_ref × 6480) Monte-Carlo sum the formalism started with — which is the
speedup the whole project is after.

**The time grid is quadratic, `t_k = (1 - k/n)²`**, concentrating steps near
t=0 where `G ~ x/σ_t` is stiffest (|λ| ~ 1/σ₀ = 100).

Step count: 50 is a good default; 100 is the conservative choice. A
step-count sweep showed returns flattening well before 100.

### 8.2 Read-out

At t=0 the state still carries σ₀ = 0.01 of noise in every voxel. Three
options:

| `readout` | Values from | Note |
|-----------|-------------|------|
| `denoise` | `q̂(0, x)` | a conditional mean, **diluted by occupancy uncertainty** |
| `raw` | the ODE end point itself | already a sample; **use this for the two-stage model** |
| `lit` | the lit-value head | needs `lit_head=True`; see §6.1 |

Occupancy comes from the hit head either way.

Why `raw` for two-stage: `x(0)` is already a draw from the model's
distribution, while `q̂(0, x)` is its conditional mean — diluted by the
probability each voxel is empty. Measured on real showers, the mean carries
only 0.85 of the energy truly-lit voxels hold, which was most of the
sampler's energy deficit. The denoised read-out existed to remove the σ₀
noise, and the gate now does that for the empty voxels where it mattered.

### 8.3 Gate rules — measured comparison

How the hit head decides a voxel is lit. Measured on the 16k checkpoint
(real: ~1520 hits/shower; a real-to-real NN distance is ~120):

| gate | hits/shower | radial err | NN distance | E_dep/E_inc |
|------|------------:|-----------:|------------:|------------:|
| `threshold` (σ(h) > 0.5) | 1359 | 0.321 | 138 | 0.718 |
| `sample` (Bernoulli) | 1502 | 0.283 | 161 | 0.711 |
| **`topk`** (default) | 1543 | 0.300 | 151 | **0.780** |

Real E_dep/E_inc is 0.784, so `topk` wins there decisively, which is why it is
the default. **No option wins everywhere** — `threshold` stays closest to the
data manifold but is biased sparse, dropping every voxel the model is
genuinely unsure about; `sample` is right per-voxel but real occupancy is
spatially *correlated*, so independent draws add salt-and-pepper noise.

`topk` keeps the k most probable voxels with `k = round(Σ p)` — the head's own
expected hit count. It matches the hit count like `sample` while keeping the
spatial coherence of `threshold`.

A fourth rule, `gumbel`, samples without replacement proportional to p
(Gumbel top-k): exactly k voxels like `topk`, chosen stochastically like
`sample`. It exists because `topk` ranks by probability, probability tracks
brightness, and so it systematically drops the **dim** voxels real showers
carry at the edges — measured 0.6% of hits below 1 MeV against a real 7.2%.

### 8.4 `--value-correct`: principled, and measured worse

Since `q` is a conditional mean over both outcomes,
`q − empty = p(lit) · (E[value|lit] − empty)`, so a voxel about to be declared
lit arguably should carry `q/p`, not `q`.

**Measured worse and off by default:** E_dep/E_inc 0.780 → 0.963 against a
real 0.784; longitudinal error 0.139 → 0.257; radial 0.302 → 0.379.

The reasoning is sound but incomplete. That model's aggregate was right only
through *cancellation* — the field recovered 0.79 of the energy in truly-lit
voxels while inventing 0.26 in truly-empty ones. Undiluting the kept voxels
also inflates the spurious energy that survived the gate, and never recovers
the true energy the gate discarded. It is also too aggressive for voxels
`topk` keeps on marginal confidence (p = 0.3 scales a voxel by 3.3×).

Documented here because "obviously correct" reasoning that measures worse is
exactly what is worth recording.

---

## 9. Export and evaluation

### 9.1 Export (`export.py`)

```python
recon_mev, diag = export.physical_showers_mev(x0_std, eg_all, stats)
export.write_hdf5(path, recon_mev, eg_all * 1000.0, stats, threshold_kev=15.15)
```

Two things this does that matter:

**It clamps instead of asserting.** A 100k-step run produced a few voxels so
large that `expm1` overflowed float32, and the old inline export died on
`assert np.isfinite(flat).all()` — *after* training and sampling had both
finished. A blow-up is a result worth measuring, not an assertion failure.
`physical_showers_mev` clamps to the physically possible and reports exactly
how much it had to clamp.

The clamp is physics, not a fudge: a single voxel cannot hold more than the
incident energy, so `x_log = log1p(E/E_inc) > log 2` is unphysical by
construction. **A large `clipped` count means the sampler is diverging and
the downstream numbers are untrustworthy — fix the model, don't lean on the
clamp.** Measured at 100k steps: 5 showers in 2048 (0.24%), all with
E_inc in 1–10 GeV, the widest part of the distribution where the field is
sharpest and the reverse ODE stiffest.

**It applies the 15.15 keV threshold** before writing. Generated showers must
carry the same readout cut as the data.

Output layout: `incident_energies` (N, 1) and `showers` (N, 6480), both
float32 MeV, gzip-compressed.

### 9.2 Quick metrics (seconds)

```bash
python -m calods2.metrics.quick_metrics --gen generated/two_stage_ds2_100k.hdf5
```

E_dep/E_inc, hit fraction, dim-voxel fraction, vox5/vox50, longitudinal and
radial profile errors, NN distance ratio, voxel energy distribution. Run this
before the official evaluation — it catches gross errors in seconds rather
than hours.

Reference values for real showers: E_dep/E_inc median **0.7826**, lit
fraction **0.2495**.

### 9.3 Official evaluation

```bash
python run_full_evaluation.py --skip-generate \
    --gen-file generated/two_stage_ds2_100k.hdf5 \
    --out-dir evaluation_two_stage
```

**Generate exactly 100,000 showers.** `evaluate.py`'s splitter hard-codes
60000/20000/20000 for Dataset 2, so classifier modes require exactly 100k on
both sides. A 20k file fails on `assert len(data1) == len(data2)`.

Modes: `hist` (χ² of 227 histograms), `cls-low` (per-voxel DNN), `cls-high`
(layer energies + shape summaries), `cls-resnet` (CNN), `fpd-kpd`.

**The published numbers average 10 classifier trainings.** A single run has
~0.015 AUC scatter — measured, by evaluating one file twice. Average 10 runs
before comparing against the literature.

---

## 10. Model A — two-stage PINN (recommended)

Best model in this repository: low-level AUC **0.578**, statistically
indistinguishable from CaloDiffusion's 0.577, which placed 2nd of 17
published submissions. Mean χ² 0.0017, best or tied-best in six of seven
observable classes.

### 10.1 Why factorize

Three mechanisms, each measured independently:

1. **Energy conservation becomes exact by construction.** The single-stage
   models carried a persistent ~10% energy deficit that survived every
   read-out, gate and representation change tried. Here the total is one of
   45 numbers a small network models directly, and the shape model *cannot
   affect it at all*. Etot/Einc χ² = 0.0020 against 0.127 — a factor of 63.
2. **The logit representation becomes usable.** Once layer totals are stage
   A's job, the shape model works in a space where the empty-to-lit gap is
   2.28 standardized units instead of 5.5 × 10⁻⁴ (§3.3). Voxel-distribution
   χ² = 0.0025, 10× better than anything else here.
3. **Each network solves an easier problem.** Stage A is a 45-dimensional
   density. Stage B is a *normalized* field with no scale freedom. Neither has
   to learn "how much" and "where" simultaneously.

This is CaloDREAM's factorization (arXiv:2405.09629), confirmed by the
summary paper as the best Dataset-2 submission.

### 10.2 Configuration

| | Stage A (energy) | Stage B (shape) |
|---|---|---|
| variable | `u`, 45 numbers | `w`, 6480 numbers |
| trunk | `film` MLP | `conv` 3D U-Net |
| hidden | 512 | 1024 |
| hit head | **no** (dense data) | **yes** |
| `cond_extra` | 0 | **45** (receives `u`) |
| batch | 256 | 64 |
| steps | 20,000 | 100,000 |
| params | 993,325 | 2,474,242 |

### 10.3 Run it

```bash
python -u -m calods2.train_two_stage \
    --data data/Dataset2.pt \
    --checkpoint-energy checkpoints/ts_energy.pt \
    --checkpoint-shape  checkpoints/ts_shape.pt \
    --steps-energy 20000 --steps-shape 100000 \
    --cond conv --batch 64 \
    --export generated/two_stage_ds2_100k.hdf5 \
    --n-sample 100000 --sample-steps 50
```

Add `--resume` to continue an interrupted run (keep `--steps-*` identical).
Add `--skip-energy` to reuse a trained stage A. Both `--skip-*` together
skips training entirely and just samples.

Sampling order: draw `u` from stage A → draw `w` from stage B conditioned on
`[e_cond, u]` → recombine analytically with `layerdata.decode`. Both stages
use `readout="raw"`; stage B uses `gate="topk"`.

### 10.4 Expected output

```
E_dep/E_inc median ~0.78   (real 0.7826)
lit fraction       ~0.25   (real 0.2495)
```

Worst observables: deepest layers (E layer 40–44, χ² up to 0.019), the sparse
shower tail. That is everyone's worst region in this table.

### 10.5 Known weakness

cls-low 0.578 but **cls-high 0.695 and cls-resnet 0.804**. We match the
second-best published submission on raw voxels and fall 0.10–0.12 behind it on
derived quantities — far outside the 0.015 noise. The diagnosis: per-voxel
marginals are close to right, **voxel-to-voxel correlations are not**. This is
the highest-value remaining work.

---

## 11. Model B — single-stage conv U-Net

Simpler: one network, one sampling pass, no factorization. Low-level AUC
0.683, mean χ² 0.0082 — mid-table against the published submissions and ahead
of every flow-based one on low-level.

```bash
python -u -m calods2.train_velocity \
    --data data/Dataset2.pt --cond conv \
    --checkpoint checkpoints/velocity_conv.pt \
    --steps 16000 --batch 64 --hidden 1024 \
    --export generated/conv_recon_ds2.hdf5 \
    --n-sample 100000 --sample-steps 100
```

Operates on the `fraction` representation (§3.1), so it inherits that
representation's resolution limit — which is most of the gap to Model A.

**Do not simply raise `--steps`.** With the plain unpreconditioned net,
samples got measurably *worse* past ~8k steps at batch 32, and at constant LR
they blew up to inf by 16k. The cosine decay to zero is what makes 16k
stable.

Its telling failure is cls-high = 0.925: the high-level classifier, which
sees only layer energies and shape summaries, separates it almost perfectly.
That is a direct statement that its layer-energy distributions are wrong —
exactly the defect §10.1 removes.

---

## 12. Model C — autoencoder

Not a generative model. It compresses a 6480-voxel shower to **96 numbers**
and reconstructs it better than 512-component PCA (rel 0.367 vs 0.539). Its
value is (a) as a compression result and (b) as the **ceiling** for any
latent generator built on its decoder.

**Checkpoint warning.** The rel 0.367 figure is the v4 model, which exists
only on the workstation — no checkpoint in this working copy has
`two_head=True`. The AE results in [MODEL_COMPARISON.md](MODEL_COMPARISON.md)
were produced by `best_calo_ae_cond_improved.pt`: epoch 85, **pre-v4**, old
units convention, rel 0.395. Check `two_head` and `shower_units_gev` in any
checkpoint before quoting a number against it.

### 12.1 Architecture — factorized, not monolithic

The key structural idea: **treat the 45 z-layers as 45 separate 16×9 images**,
process them with a shared 2D encoder, then model the longitudinal direction
with 1D convolutions. This factorization is what makes 96 numbers enough.

```
input [B, 45, 16, 9]  (log-normalized)

2D encoder, applied per z-layer (shared weights):
  ResConvBlock2D(1 → 16)          (16, 16, 9)
  StridedDown2D(16 → 32)          (32,  8, 5)
  ResConvBlock2D(32 → 32)
  StridedDown2D(32 → 64)          (64,  4, 3)
  ResConvBlock2D(64 → 64)
  → 64·4·3 = 768 features per layer

reshape/permute → [B, 768, 45]

1D encoder along z:
  Conv1d(768 → 256) → GroupNorm → GELU
  Conv1d(256 → 128) → GroupNorm → GELU

bottleneck:
  flatten [B, 128·45 = 5760] ⊕ energy_embed [B, 64]
  Linear → [B, 96] → reshape [B, 8, 12]

decoder: exact mirror, plus energy_embed again
  final Conv2d → 2 channels (value, hit logit), LINEAR output
```

**3,110,882 parameters** at `width_mult=1.0`.

**The latent is structured — 8 channels × 12 longitudinal positions, not a
flat 96-vector.** This is deliberate: a diffusion model over a latent with
spatial structure can use convolutions.

**The decoder receives the energy embedding too**, not just the encoder. That
is what lets a latent generator model only the *shape* distribution while the
decoder handles energy scale.

`EnergyEmbed` lifts the standardized log-energy scalar through a 2-layer MLP
to 64 dims. A 1-dim conditioning signal forces every downstream linear layer
to learn its own scale for the energy term; a learned embedding is more
flexible.

### 12.2 Run it

```bash
python train_auto_encoder_improved.py \
    --epochs 250 --latent-channels 8 --latent-length 12 \
    --phi-circular --checkpoint-path best_calo_ae_cond_improved.pt
```

`--phi-circular` should always be on; it is a flag only so pre-existing
checkpoints still load.

### 12.3 Its measured defect

Total energy is excellent — median E_dep/E_inc 0.7826, matching the reference
to four digits, χ² = 0.0013, the best of any model here. But **41.98% of
voxels come out lit against a real 24.95%**: the energy is right and spread
over too many voxels.

That 42% is a no-hit-head signature — v4 measures 0.256 — and is why the
evaluated file is identifiable as pre-v4. It shows up brutally in FPD/KPD
(11328 / 91.6), which compare distributions
in a feature space rather than marginal histograms — worse than the worst
published submission by a factor of two.

---

## 13. Model D — latent diffusion

Train the PINN velocity field in the AE's 96-dim latent space, then decode
through the frozen decoder. **This did not work**, and the honest reading is
that it underperforms the plain conv U-Net it was meant to improve on: AUC
0.991 vs 0.683, mean χ² 0.151 vs 0.0082.

The appeal was real — the formalism was validated at 1–3 dimensions and
measurably fails at 6480, so 96 is far closer to the regime where it works,
latents are dense (no hit head needed) and the decoder already knows the
geometry. Measured lit-voxel fraction 0.256 against a real 0.250, so sparsity
really was handled.

**Why it failed.** Its worst observables are **EC Phi** — the azimuthal energy
centroid — at χ² ≈ 0.445 for layers 26–29, far worse than any other model.
The 96-number bottleneck does not preserve φ position faithfully enough, so
errors in latent space become *structured* spatial errors after decoding.

**The decisive diagnostic:** the AE reconstruction itself scores cls-low 0.989,
and latent diffusion scored 0.991. It essentially reached its own ceiling. The
sampler is not the problem — the decoder is. Any further work here must
improve the autoencoder, not the sampler.

Independent corroboration: **CaloLatent**, the one published submission that
also does latent-space diffusion, placed 15th of 17 (AUC 0.983).

Always measure the ceiling first:

```bash
# encode + decode REAL showers, no diffusion model involved
python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_cond_improved.pt \
    --ceiling --export generated/latent_ceiling.hdf5 --n-sample 20000
```

Note that latents use **per-dimension** standardization, unlike the voxel
path's single global scale — all 96 dimensions are active, so there are no
dead coordinates to amplify.

---

## 14. Verification

### 14.1 Regression tests

```bash
pytest
```

21 CPU tests, no GPU and no data file needed. Includes the
layer-representation round-trip, Gumbel gate hit counts, gate behaviour under
non-finite input, φ-periodicity of the conv trunk, and that export clamps
rather than raising. Run it after any change.

Two of these tests were written *because* the bug they catch had already
happened — the Gumbel NaN (§15.3) was caught by a test written minutes
earlier.

### 14.2 Smoke test

Before committing GPU-hours, verify the whole pipeline on a tiny run:

```bash
python -u -m calods2.train_two_stage --data data/Dataset2.pt \
    --steps-energy 200 --steps-shape 200 --cond conv --batch 16 \
    --checkpoint-energy /tmp/a.pt --checkpoint-shape /tmp/b.pt \
    --export /tmp/smoke.hdf5 --n-sample 512 --sample-steps 10
python -m calods2.metrics.quick_metrics --gen /tmp/smoke.hdf5
```

The numbers will be bad. What you are checking is that it runs end to end and
writes a valid file.

### 14.3 Denoiser check

```bash
python models/2_direct_pinn/sample.py --checkpoint ... --check-denoiser
```

Decomposes the energy error into truly-lit versus truly-empty voxels — the
diagnostic that showed the aggregate E_dep/E_inc was right only by
cancellation (§8.4).

---

## 15. Pitfalls

Every one of these cost real time in this project.

### 15.1 Units

`Dataset2.pt` stores **MeV** for both showers and energies. An earlier version
converted only `energies` to GeV, leaving the model training on
`log1p(shower_MeV / energy_GeV)` — a ratio inflated by ~1000×. It stretched
the log1p target range ~15× (max 6.3 vs 0.4), which fed into a
`1 + 10·x_log` weighting (max weight 63 vs 5) and very likely caused a
stubborn early plateau: the same architecture plateaued at the same val loss
whether trained 85 or 460 epochs.

This is a **breaking change for old checkpoints**, which is why the AE
checkpoint carries a `shower_units_gev` marker.

A related bug in `generate_caloeval_hdf5.py` always applied the MeV/GeV
convention, feeding units-fixed checkpoints 1000× inputs — with a stale
sanity check that expected 0.033 against an actual 0.78, so it looked fine.
**Check that your sanity thresholds were updated along with your units.**

### 15.2 Train/eval file confusion

Train on `dataset_2_1.hdf5`, evaluate against `dataset_2_2.hdf5`. Training on
the reference makes every number meaningless.

### 15.3 Operator precedence in the Gumbel gate

```python
g = -torch.log(u).clamp_min(eps)        # WRONG
g = -torch.log((-torch.log(u)).clamp_min(eps))   # right
```

The first parses as `-(log(u).clamp_min(eps))`. Since `log(u) < 0`, the clamp
flattens it to `eps`, and the outer log of a negative value is NaN — which
**silently gated every voxel off**, because NaN compares False. Zero lit
voxels, no error raised.

### 15.4 Physics-residual loss

Adding an explicit PDE-residual penalty measurably hurt. The residual is
evaluated at the model's own prediction, which early in training is
off-manifold, where the q-kernel degenerates and `G ~ x/σ_t` becomes radially
repulsive. The residual *rose* during training (6862 → 22002). The physics
belongs in the parameterization (§4.3), not in a penalty term.

### 15.5 Changing `--steps` on resume

The cosine schedule's length is set by `--steps`. Resuming a finished
2000-step run into a 4000-step schedule snaps the LR from ~0 back to half
peak with no warmup — a warm restart, which gave a measurably different final
val loss (0.659 vs 0.632 for an uninterrupted control). Not a bug, but keep
`--steps` fixed unless a warm restart is what you want.

### 15.6 Starting a run without `--resume`

At step 1000 a fresh run overwrites the checkpoint, and `best_val` resets to
infinity so the `.best` copy is clobbered too. Snapshot first, or use new
filenames.

### 15.7 Conv sampling batch and VRAM

A conv trunk holds ~16× the activations per sample that the MLP does: at batch
2048 one intermediate is 2048 × 32 × 6480 × 4 B = 1.7 GB, with several live at
once. `max_sample_batch` derives a safe batch from free VRAM (~809 on a 32 GB
card). An explicitly passed `--sample-batch` is always honoured and may OOM —
that is the caller's choice.

### 15.8 Buffered logs look like a hung job

Use `python -u` under `nohup`, or the log stays empty for minutes and looks
stalled.

---

## Appendix — file map

| File | Role |
|------|------|
| `src/calods2/models/velocity_net.py` | `VelocityNet`, `train`, `sample`, gates, read-outs |
| `src/calods2/data/layerdata.py` | two-stage representation, exact inverse |
| `src/calods2/data/dataset.py` | `fraction`/`threshold` representations, Dataset 2 loader |
| `models/3_two_stage/train.py` | Model A driver |
| `models/2_direct_pinn/train.py` | Model B driver |
| `models/1_latent_diffusion/train_latent.py` | Model D driver, `--ceiling` |
| `src/calods2/data/export.py` | clamping conversion + HDF5 writer |
| `src/calods2/metrics/quick_metrics.py` | fast metrics |
| `models/2_direct_pinn/sample.py` | sample from a checkpoint, `--check-denoiser` |
| `tests/test_selftest.py` | 21 CPU regression tests |
| `src/calods2/metrics/sweep.py` | hyperparameter sweeps |
| `src/calods2/models/flow_de.py` | Monte-Carlo q-field (reference/comparison only) |
| `train_auto_encoder_improved.py` | Model C: `FactorizedCaloAE`, `calorimeter_loss_v4` |
| `run_full_evaluation.py` | official evaluation driver |
| `docs/MODEL_COMPARISON.md` | results and published comparison |
| `docs/PINNDE_LAB_NOTEBOOK.md` | full experiment log, including failures |

## References

- CaloChallenge 2022 summary — arXiv:2410.21611
- CaloDREAM (two-stage factorization) — arXiv:2405.09629
- CaloDiffusion (logit preprocessing, position channels) — arXiv:2308.03876
- EDM preconditioning — Karras et al. 2022, arXiv:2206.00364
- χ² metric definition — eq. 15 of arXiv:2009.03796
