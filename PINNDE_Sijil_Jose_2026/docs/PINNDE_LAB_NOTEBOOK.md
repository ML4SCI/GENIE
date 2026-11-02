# calods2

A PINN-based reverse-time-diffusion sampler for Fast Calorimeter Challenge
Dataset 2, built by porting the formalism from Sijil Jose's GSoC 2025
project [PINNDE](https://github.com/sijil-jose/GENIE/tree/main/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose)
(ML4SCI/GENIE) from 1-3 dimensional toy Gaussian-mixture targets to real
6480-dimensional calorimeter showers, conditioned on incident energy.

This samples showers **directly** -- no autoencoder, no latent space: noise
`z` is drawn in the full 6480-dimensional voxel space and carried to a shower
by the reverse-time ODE.

**On one-pass sampling.** The reference formalism learns the TRAJECTORY
`x(t, z, E)` -- the whole flow map -- so a sample is one forward pass with no
integration at all. That is the headline appeal, and it is what an earlier
version of this paragraph claimed. It does not survive at 6480 dimensions:
held-out error on unseen `z` falls only as `N^-0.29` (8x the anchor
trajectories bought 1.84x), and the residual cannot train from a cold start
because it is evaluated at the model's own off-manifold predictions, where
`G ~ x/sigma_t` is radially repulsive and `MSE_f` *rises* during training.
Samples came out worse than noise.

What works instead is learning the LOCAL field `q(t, x, E) = E[x0 | x_t]`, from
which `G` follows by exact algebra -- but a local field has to be integrated,
so sampling costs ~200 network calls (100 Heun steps). One-pass sampling is
recoverable by DISTILLING this trained field into a few-step or one-step
student, which is the standard route and works precisely where direct flow-map
learning failed: the teacher supervises everywhere on the manifold, so the
student never depends on a residual evaluated off it. CaloDREAM reaches 8
network calls this way with its classifier AUC still below 0.6
(arXiv:2405.09629). That distillation step is NOT implemented here.

## IMPORTANT: which parameterization to use

There are two implementations in here. **Use the velocity one
(`velocity_net.py` / `train_velocity.py`).** The direct port of the reference
repo's trajectory ansatz is kept for reference and for the 1-3D regime, but it
does not work at 6480 dimensions, and the reason is measured, not suspected:

| | `‖x0‖` (real 68.0) | `E_dep/E_inc` (real 0.78) |
|---|---|---|
| trajectory ansatz `x(t; z)` (reference port) | 105 – 132 | 2.9 – 3.5 |
| **velocity ansatz `q(t, x, E)`** | 45 – 48 | **0.80 – 0.94** |

One consequence worth being explicit about: in the velocity parameterization
the reverse-time ODE is satisfied *by construction* -- the network learns `q`
and `G = [(1-sigma0) x - q]/sigma(t)` is then exact algebra -- so there is no
residual left to penalize. The PINN residual loss (`MSE_f`) belongs to the
trajectory path in `networks_calo.py`, and that is the path measured not to
scale here. The physics in the velocity path is in the interpolant and the
exact `G`, not in a soft penalty.

Three measurements drove the change:

1. **The flow map does not generalize in high dimension.** Going from 512 to
   4096 anchor trajectories (8x) cut held-out MSE only 1.84x -- `MSE ~ N^-0.29`.
   Extrapolating to a usable error needs ~10^6x more trajectories. The network
   memorizes its anchors rather than learning the map. (An earlier apparent
   "it generalizes" result came from a buggy train/held-out split that did not
   actually separate trajectories; with a correct split, held-out MSE ends up
   *worse* than at initialization.)
2. **The physics residual cannot train from cold start.** It is evaluated at
   the model's own prediction, which early on is off-manifold, where the
   q-kernel degenerates and `G ~ x/sigma_t` becomes radially repulsive. `MSE_f`
   *rises* during training (6862 -> 22002) -- it cannot minimize even its own
   objective. A warm start from a data-fitted model does not rescue it.
3. **The two terms are not inconsistent.** The residual evaluated on true
   anchor trajectories is ~1e-12, so both terms share a minimizer. The problem
   is purely the optimization landscape away from that minimizer, which is why
   no reweighting (tested across 10 -> 0.001, and data-weighted 1000:1 the
   other way) fixes it.

The velocity parameterization keeps the same interpolant, the same reverse-time
ODE and the same Monte-Carlo q-function, and changes only what the network
represents: the local field `q(t, x, E) = E[x0 | x_t]` instead of the global
flow map. See the module docstring in `velocity_net.py` for the full argument.

Units were verified independently and are clean: a real-data round trip
(standardize -> unstandardize -> expm1 -> physical) reproduces the raw file's
`E_dep/E_inc = 0.7815` to float32 precision.

### Usage (velocity path)

```bash
# train + sample + export in one go; no anchor generation step at all
python models/2_direct_pinn/train.py \
    --data data/Dataset2.pt --checkpoint checkpoints/velocity.pt \
    --steps 16000 --batch 32 --hidden 1024 --device cuda \
    --export generated/pinnde_recon_ds2.hdf5 --n-sample 100000

# fast physics check against real showers (iterate on this)
python -m calods2.metrics.quick_metrics --gen generated/pinnde_recon_ds2.hdf5

# official CaloChallenge evaluation
python run_full_evaluation.py --skip-generate \
    --gen-file generated/pinnde_recon_ds2.hdf5
```

Defaults worth knowing, each set from a measurement rather than a guess:

| Default | Why |
|---|---|
| `--target clean` | Regress the clean shower `x0`; its MSE minimizer *is* `E[x0\|x_t,E]`. No reference ensemble needed at all. Beat the Monte-Carlo target on every diversity metric and trained 2-4x faster. |
| `--n-ref` only used by `--target mc` | **Not** a quality knob. 128 -> 512 moved sample spread 0.149 -> 0.176 (real 0.408) at 2.3x the cost. An earlier version of this README claimed the opposite; that claim was wrong. |
| `--representation fraction` | `log1p(E/E_inc)`, which for these fractions (<= 0.28) is nearly linear. The alternative `threshold` (`log1p(E/15.15 keV)`) is genuinely log-scaled, and a denoiser predicts a conditional *mean* in whatever space it works in: exponentiating a mean-of-logs undershoots (Jensen), measured as a median `E_dep/E_inc` of 0.32 against a real 0.78, with a 5-95% band of 0.02-2.58. Under `fraction` the same run gives median 0.73, band 0.35-1.15. `threshold` existed only to make an empty voxel and a threshold hit resolvable (0.22 standardized units apart vs a median 5.5e-4 under `fraction`); the hit head does that job instead, so the log scaling is no longer needed. |
| hit head on (`--no-hit-head` to disable) | A conditional mean puts an uncertain voxel at an intermediate value, i.e. a small spurious deposit. Hit fraction against a real 0.23: **0.78** (`fraction`, no gate) -> 0.43 (`threshold`, no gate) -> **0.215** (`fraction` + gate). The second output block predicts "lit in `x0`" and gates the final read-out only -- `G` is still built from `q` alone. |
| cosine LR + EMA(0.999) | At constant LR the samples blew up to `inf` by 16k steps. |
| `--steps 16000` | Sets the cosine schedule length, not just a stopping point. Raising it is not free: in the pre-hit-head configuration samples got measurably *worse* after ~8k steps. |
| `--precond` off | EDM preconditioning with a single global `sigma_d` made it far worse (`E_dep/E_inc` 2.68 vs a real 0.78 -- essentially unit noise left in every voxel). Per-voxel data variance spans ~0 to 27, so one global scale is the wrong assumption; in near-dead voxels the skip injects noise the MLP must then cancel by reproducing it. |

Sampling integrates the reverse-time ODE with Heun's method on a power time
grid `t_k = (1-k/T)^2` (steps concentrated near `t=0`, where
`|lambda| ~ 1/sigma0 = 100` is stiffest), then returns the denoised, hit-gated
read-out rather than the raw ODE end point. With the exact Monte-Carlo field
that denoising step alone moved the hit fraction from 0.58 to 0.25 (real: 0.25).

### `--cond film` fixes the energy over-dispersion

With `--cond concat` (two extra inputs beside the 6480 voxels) the overall
mean `E_dep/E_inc` was right while the per-energy response was not, and every
energy bin was ~2x too broad. `--cond film` -- sinusoidal features of
`(t, log E)` through a small MLP, producing a per-layer scale and shift on
every hidden activation -- fixes that. Same depth, width, seed and budget
(16k steps, batch 32, hidden 1024; 22.8M vs 22.0M parameters):

| `E_inc` | `concat` 100k | `film` 16k | `film` 100k | real |
|---|---|---|---|---|
| 1-10 GeV | 0.920 +- 0.396 | 0.736 +- 0.193 | 0.681 +- **0.204** | 0.815 +- 0.174 |
| 10-100 GeV | 0.710 +- 0.125 | 0.684 +- 0.087 | 0.682 +- **0.083** | 0.777 +- 0.070 |
| 100+ GeV | 0.754 +- 0.038 | 0.769 +- 0.021 | 0.765 +- **0.025** | 0.780 +- 0.021 |
| overall sd | 0.256 | 0.130 | **0.132** | 0.110 |
| longitudinal err | 0.150 | 0.105 | **0.114** | - |
| radial err | 0.198 | 0.228 | **0.123** | - |
| hits/shower | 1513 | 1541 | **1550** | 1612 |
| diverged per 100k | **236** | - | **1** | - |

The per-bin spreads track the real ones (1.17-1.19x) instead of being ~2x
wide. The conditioning was the bottleneck: the model had been absorbing an
energy dependence it could not represent as extra shower-to-shower scatter.

The divergence count settles the open question in the next section: 236 runaway
showers per 100k with `concat`, **1** with `film`, same budget and sampler. The
runaway noise draws were a symptom of weak conditioning, not of the integrator
and not intrinsic to the learned field.

What is left is a systematic ~10% energy DEFICIT (0.709 vs 0.790), worst at low
energy (-16%, -12%, -2% across the three bins) -- a calibration bias rather
than a variance problem. `concat` had the overall scale right (+1.3%) and
everything else wrong, so this is a trade, not a strict win.

### How the hit head's gate is applied matters, and is a sampling-side choice

`--gate` costs nothing to change -- it applies at the final read-out, so any
saved checkpoint can be re-sampled with `sample_velocity.py` in minutes. On
the 16k checkpoint, 2048 showers (real reference in brackets):

| `--gate` | hits/shower (1592) | `E_dep/E_inc` mean (0.784) | long. err | radial err | NN / real-to-real |
|---|---|---|---|---|---|
| `threshold` (p > 0.5) | 1359 | 0.718 | 0.159 | 0.321 | **1.11** |
| `sample` (Bernoulli) | 1502 | 0.711 | 0.143 | **0.283** | 1.37 |
| **`topk`** (default) | **1543** | **0.780** | **0.138** | 0.300 | 1.23 |

`threshold` drops every voxel the model is unsure about, losing ~15% of the
hits and ~8% of the energy. `sample` fixes the marginal hit count but real
occupancy is spatially *correlated*, so independent per-voxel draws add
salt-and-pepper noise -- visible as the NN distance jumping to 1.37x
real-to-real. `topk` keeps the `k` most probable voxels with
`k = round(sum of the head's probabilities)`, matching the expected count
while staying spatially coherent. Nothing wins on every metric; `threshold`
remains the closest to the data manifold.

### A small fraction of NOISE DRAWS diverges, and more steps do not help

A few showers run away completely: a single voxel ends up above the incident
energy, which overflows float32 once exponentiated, and one such shower is
enough to make a mean `E_dep/E_inc` print `inf`. Measured on the 100k-step
model, all at `E_inc` = 1-10 GeV:

| run | diverged |
|---|---|
| 100000 showers, 100 steps | 236 (0.236%) |
| those 236 re-sampled at 400 steps (fresh noise) | 3 |
| those 3 re-sampled at 1600 steps (fresh noise) | 0 |
| **20000 showers, 400 steps from scratch** | **45 (0.225%)** |

The last row is the control that matters, and it kills the obvious reading of
the first three. The divergence rate is the SAME at 400 steps as at 100, so
the time grid is not the cause: what rescued the retried showers was the
fresh noise draw, not the finer grid. (The retry changes both at once, which
is why the first three rows looked like a stiffness problem and were
originally written up here as one.) Roughly 0.23% of standard-normal draws
land where the learned field runs away -- a property of the field, so fixing
it properly is a training-side problem.

Meanwhile `sample_velocity.py` detects these per shower
(`export.diverged_mask`) and re-samples just those (`--max-retries`), which
works because the noise is redrawn. The counts are always printed: rejecting
divergent draws is a sampling bias, tiny at 0.23% but real, and a large
retried fraction would mean the field is badly behaved rather than rarely so.
`export.physical_showers_mev` clamps anything left to the physically possible
maximum rather than letting an assertion discard a finished run, and every
gate tolerates non-finite logits (a NaN once produced an out-of-bounds index
and killed a 100k-shower sampling pass with a CUDA device-side assert).

## Latent PINNDE: the same sampler in the autoencoder's 96-d latent space

`train_latent.py` runs the identical field network and Heun sampler on the
autoencoder's latent instead of the 6480 voxels, then decodes:

```
showers --encode--> latents (96-d) --PINNDE--> latents --decode--> showers
```

Two of the voxel path's hard problems vanish by construction: the hit head is
switched off (latents are dense; the decoder reproduces sparsity itself) and
there is no geometry to learn (the decoder knows it). The formalism was
validated at 1-3 dimensions and measurably fails at 6480; 96 is much closer to
where it works, and it shows -- training loss reaches 1e-5.

In exchange there is a hard ceiling: whatever the autoencoder cannot
reconstruct, no latent sampler recovers. `--ceiling` measures it directly by
encoding and decoding REAL showers with no diffusion model involved. **Always
read a latent-sampler result against that file, not against real showers
alone.**

**This is the best configuration measured in this repository.** 20k steps on
the v4 autoencoder (rel 0.367), 2048 generated vs 2048 real:

| | latent sampler | ceiling (AE reconstruction) | real |
|---|---|---|---|
| `E_dep/E_inc` | **0.7819 +- 0.1181** | 0.7742 +- 0.1197 | 0.7822 +- 0.1200 |
| 5 / 50 / 95% | 0.606 / 0.776 / 0.999 | 0.587 / 0.777 / 0.973 | 0.576 / 0.782 / 0.979 |
| hits/shower | 1642 | 1633 | 1558 |
| longitudinal err | **0.0495** | 0.0546 | - |
| radial err | 0.1075 | 0.1135 | - |
| NN / real-to-real | 1.30 | 1.27 | 1.00 |

The energy distribution matches real showers in BOTH mean and spread, per
energy bin as well as overall -- the defect that resisted every fix in voxel
space (2.3x too wide with `concat`, 1.2x with `film`). Against the direct
sampler's best (`film`, 100k steps, raw read-out): energy 0.701 +- 0.126 vs
**0.782 +- 0.118**, longitudinal 0.114 vs **0.050**, radial 0.118 vs
**0.108**, NN ratio 1.29 vs 1.30.

Every generated number tracks the ceiling to within noise, so the latent
diffusion is doing its job and what remains is the autoencoder's. Note in
particular that the ceiling's own NN distance is 1.27x real-to-real: the
off-manifold gap neither approach has closed is a RECONSTRUCTION limit here,
not a generator one. The lever for this path is therefore the autoencoder --
PCA-256 reaches rel 0.461 and PCA-512 0.387 against the current 96-d model's
0.367, so a larger latent has headroom.

An earlier version of this section reported the same "matches the ceiling
exactly" conclusion from a run on the broken v3 autoencoder. That test was
much easier than it looked: v3's latents were nearly deterministic given the
incident energy (per-dim std 0.036-1.6, several dead dimensions, training loss
1e-5). v4 uses all 96 dimensions (std 3.1-4.6) and the loss plateaus at 0.44,
which is roughly the average posterior variance over `t ~ U(0,1)` and is what
convergence looks like here -- so the conclusion now rests on a real task.

```bash
# the ceiling first -- it predicts the outcome before training anything
python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_v4.pt --ceiling \
    --export generated/latent_ceiling.hdf5 --n-sample 20000

python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_v4.pt \
    --checkpoint checkpoints/latent_velocity.pt --steps 20000 --hidden 512 \
    --batch 256 --export generated/latent_recon_ds2.hdf5 --n-sample 100000
```

`train_latent.py` prints the generated latents' per-dimension std against 1.0
every run. At 600 steps it read 0.019 -- collapsed to the mean -- while
`E_dep/E_inc` still looked healthy at 0.759, which is exactly the failure an
aggregate metric hides.

## The formalism (unchanged from the reference repo)

Both the reference repo and this port use the same stochastic-interpolant
path between data and noise:

```
x(t) = alpha(t)*x0 + sigma(t)*eps,   alpha(t) = 1-t,   sigma(t) = sigma0 + (1-sigma0)*t
```

so `x(0) = x0` (data) and `x(1) ~ eps` (standard Gaussian, up to a small
floor `sigma0`). The reverse-time ODE this induces is

```
dx/dt = G(t,x) = [(1-sigma0)*x - q(t,x)] / sigma(t)
```

where `q(t,x) = E[x0 | x(t)=x]` is estimated by Monte Carlo importance
weighting over a fixed reference ensemble of real showers (`qVectorField`
in `flow_de.py`) -- this is the same Gaussian-kernel posterior-mean
estimator as the reference repo, generalized with an extra energy-
proximity term in the kernel so the estimate is conditional on incident
energy (see the module docstring in `flow_de.py` for the exact math).

Training follows the reference repo's PINN recipe exactly:

1. **Anchor data** (`generate_anchor_trajectories.py`): numerically solve
   the reverse ODE exactly (2nd-order Runge-Kutta, `FlowDE`) for a modest
   number of starting `z`'s, giving ground-truth `(t, z, E, x(t))` points.
   Expensive (that's why it's only done for a handful of trajectories,
   once, up front) -- this is the data the PINN's supervised loss term
   anchors to.
2. **PhysicsNN** (`networks_calo.py`): a network whose raw output is
   combined with `z` via `x(t,z,E) = z + net(t,z,E) - net(1,z,E)`, which
   hard-constrains the `t=1` boundary condition to hold exactly regardless
   of the network's weights -- the same ansatz trick as the reference
   repo's `x_function`.
3. **Loss**: `MSE_u + 10 * MSE_f`, where `MSE_u` fits the network to the
   anchor trajectories and `MSE_f` is the ODE residual `t*dx/dt - t*G(t,x)`
   (autograd through the ansatz), enforced on a much broader grid of
   collocation points than the anchor data covers -- identical structure,
   including the `t`-weighting of the residual, to the reference repo's
   `f`/`loss_function`.
4. **Ramp-step curriculum**: train on a growing fraction of the anchor
   data at a cooling learning rate, matching the reference repo's
   `train`/`train_adam`.

## What had to change for 6480 dimensions + energy conditioning

- **Energy conditioning** (new): `qVectorField`, `PhysicsNN`, and every
  training/sampling function take an extra `energy` argument throughout.
  The toy GMM targets don't have a conditioning variable at all.
- **Network capacity**: the reference repo uses a 3-hidden-layer,
  32-unit-wide MLP for its 1-3 dimensional targets. That has nowhere near
  the capacity to represent a calorimeter shower; `PhysicsNN` here defaults
  to 512-unit hidden layers (`--hidden`, tune freely). Still a plain MLP --
  a PINN's collocation-based training doesn't need the convolutional
  inductive bias the autoencoder uses.
- **Reference ensemble size M**: the toy case uses M=12000 reference points
  for a 1-D target. At 6480 dimensions, `qVectorField`'s `(N_batch, M, d)`
  intermediate tensor gets large fast, so M defaults to a few hundred
  (`--n-ref`) rather than tens of thousands -- a real, flagged
  precision/memory tradeoff, not a silent shortcut.
- **LBFGS vs. Adam**: the reference repo's primary optimizer is LBFGS with
  a full-batch closure (`max_iter=10000`). At this dimensionality that's
  far more expensive per step; `train_adam` (already present as a
  secondary option in the reference repo) is the practical default here,
  with `train_lbfgs` kept for fine-tuning, at a much lower `max_iter`.
- **Representation**: showers are preprocessed exactly as in
  `train_auto_encoder_improved.py` -- `log1p(shower_GeV / energy_GeV)`,
  flattened to 6480 dims, with the same `log10(E)` standardization -- so a
  PINNDE-sampled shower and an AE-reconstructed one are on identical
  footing, and `sample_and_export.py` writes the exact HDF5 layout
  (`incident_energies`, `showers`, MeV, 15.15 keV threshold) that
  `evaluate.py` / `run_full_evaluation.py` already expect.

## Files

| File | Reference-repo analog | Role |
|---|---|---|
| `flow_de.py` | `flow_de/flow_de.py` | `qVectorField` (energy-conditioned MC q-field), `FlowDE` (exact RK2 reverse-ODE solver) |
| `gendata.py` | `flow_de/gendata.py` | Loads Dataset 2 instead of a synthetic GMM; same log1p/energy preprocessing as the AE pipeline |
| `networks_calo.py` | `flow_de/networks_1d.py` (etc.) | `PhysicsNN`, the ansatz, the ODE residual, `train_adam`/`train_lbfgs` |
| `generate_anchor_trajectories.py` | data-prep block in `train_1d_GMM.py` | Precomputes exact `(t,z,E,x)` anchor points |
| `train_calo_pinnde.py` | `train_1d_GMM.py` | Main training entrypoint |
| `sample_and_export.py` | *(new -- toy cases only plot, don't need an exporter)* | One-forward-pass sampling -> CaloChallenge HDF5 (trajectory path) |
| `velocity_net.py` | *(new)* | **The path to use.** `VelocityNet` (learned `q` + hit head), training loop, Heun sampler |
| `train_velocity.py` | *(new)* | Entrypoint for the velocity path: train -> sample -> export, no anchor step |
| `quick_metrics.py` | *(new)* | Fast physics comparison of an exported HDF5 against real showers |
| `sample_velocity.py` | *(new)* | Sample/re-gate a trained checkpoint; divergence and denoiser diagnostics |
| `export.py` | *(new)* | Physical conversion with clamping, divergence mask, HDF5 writer |
| `train_latent.py` | *(new)* | Latent PINNDE: encode -> sample -> decode, and the `--ceiling` bound |
| `layerdata.py` | *(new)* | Two-stage representation: per-layer normalization + logit, and its exact inverse |
| `train_two_stage.py` | *(new)* | Energy model + shape model, sampled and combined analytically |
| `selftest.py` | *(new)* | 21 CPU regression tests, one per bug that cost a real run |
| `sweep.py` | *(new)* | Train-time and sample-time grid sweeps -> one metrics table |

## Usage

```bash
# 1. Precompute anchor trajectories (CPU is fine, one-time cost)
python -m calods2.generate_anchor_trajectories \
    --data data/Dataset2.pt --out src/calods2/anchors.pt \
    --n-traj 64 --n-ref 256 --n-steps 250 --time-stride 5

# 2. Train the PINN
python -m calods2.train_calo_pinnde \
    --anchors src/calods2/anchors.pt --checkpoint src/calods2/calods2.pt \
    --epochs 200 --ramp-steps 4 --device cuda

# 3. Sample showers and export in CaloChallenge format
python -m calods2.sample_and_export \
    --checkpoint src/calods2/calods2.pt --data data/Dataset2.pt \
    --n 100000 --out generated/pinnde_recon_ds2.hdf5

# 4. Evaluate exactly like the AE's output
python run_full_evaluation.py --skip-generate --gen-file generated/pinnde_recon_ds2.hdf5
```

## Status

The **velocity path** runs end to end and produces physically reasonable
showers. Best measured configuration: `--representation fraction`, hit head
on with `--gate topk`, **`--cond film`, 100k steps, batch 64, hidden 2048**,
cosine LR + EMA, 100 Heun steps. 2048 generated vs 2048 real held-out showers
(`python -m calods2.metrics.quick_metrics`):

| | generated | real |
|---|---|---|
| `E_dep/E_inc` mean (sd) | 0.709 (0.132) | 0.790 (0.110) |
| `E_dep/E_inc` median (5-95%) | 0.736 (0.48 - 0.88) | 0.784 (0.61 - 0.98) |
| hit fraction / hits per shower | 0.239 / 1550 | 0.249 / 1612 |
| longitudinal profile rel. L1 | 0.114 | - |
| radial profile rel. L1 | 0.123 | - |
| NN distance to real showers | 153.2 | 121.7 (real-to-real) |
| diverged noise draws | 1 per 100,000 | - |

Two things mattered, neither of them predicted here:

* **Scaling** (16k/32/1024 -> 100k/64/2048) halved the radial profile error,
  which had been flat across every other change and was wrongly written off
  in an earlier version of this file as an architecture limit.
* **FiLM conditioning** halved the energy spread AND cut runaway noise draws
  from 236 per 100k to 1, which had been wrongly attributed to integrator
  stiffness.

### Two-stage: energy first, then shape (`train_two_stage.py`)

The factorization used by CaloDREAM (arXiv:2405.09629), the best Dataset-2
submission, adapted here:

```
stage A   p(u | E_inc)        45 bounded ratios: total energy, and how it is
                              distributed along z
stage B   p(w | E_inc, u)     6480 numbers: each voxel's share of its OWN
                              layer, so every layer sums to 1
```

Both stages are the same `VelocityNet` and the same reverse-time ODE; only the
variables differ. `layerdata.py` holds the transform and its exact inverse.

Two measured problems this addresses:

* **Energy conservation stops being learned.** The single-stage model carried a
  ~10% energy deficit that survived every read-out, gate and representation
  change. Here the total is one of 45 numbers a small network models directly
  and the shape model cannot affect it: at 300 steps per stage -- essentially
  untrained -- `E_dep/E_inc` already came out **0.783** against a real 0.78.
* **Faint voxels become representable.** Because layer totals live in `u`, the
  shape variables can be logit-transformed (alpha = 1e-6) without biasing the
  energy. The gap between an empty voxel and a faint one goes from a median
  **5.5e-4** standardized units under `fraction` to **1.35**, about 2500x more
  resolution exactly where the model was blind (its 5th-percentile lit voxel
  was 3.4 MeV against a real 0.63).

The conv trunk also now takes **position channels** -- normalized layer and
radial index as extra inputs, as in CaloDiffusion (arXiv:2308.03876). A
convolution is translation invariant by construction but a calorimeter is not,
and the voxels being under-produced sit at the edges. phi-equivariance is
preserved (the channels vary only in z and r) and is covered by a test.

```bash
python models/3_two_stage/train.py --data data/Dataset2.pt \
    --checkpoint-energy src/calods2/two_stage_energy.pt \
    --checkpoint-shape src/calods2/two_stage_shape.pt \
    --steps-energy 20000 --steps-shape 100000 --cond conv \
    --export generated/two_stage_ds2.hdf5 --n-sample 20000
```

### The geometry-aware trunk (`--cond conv`) is the biggest single win

A 3D U-Net over `(z, phi, r)` -- circular padding in phi, zero in z and r,
FiLM conditioning per channel, same hit head -- replaces the flat MLP. At
**8k steps and 2.47M parameters**, against MLPs at 16k-100k steps and
10.9M-49.8M parameters:

| | MLP 16k (any width) | MLP film 100k | **conv 8k** | real |
|---|---|---|---|---|
| dim_frac | 0.0050-0.0065 | ~0.006 | **0.0326** | 0.072 |
| vox5 (MeV) | 3.36-3.55 | - | **1.47** | 0.63 |
| longitudinal err | 0.108-0.133 | 0.114 | **0.036** | - |
| radial err | 0.184-0.214 | 0.123 | **0.037** | - |
| `E_dep/E_inc` | 0.68-0.70 | 0.709 | **0.760** | 0.783 |
| spread | 0.101-0.132 | 0.126 | **0.107** | 0.115 |
| nn / real-to-real | 1.22-1.25 | 1.29 | 1.324 | 1.00 |

Profile errors fell 3-5x, the energy deficit went from 10% to 3% with a spread
that finally matches real, and the dim-voxel fraction -- flat at 0.005-0.0065
across the entire MLP width sweep -- moved to 0.0326. That last point is what
makes this attributable to spatial structure rather than capacity. The one
regression is the nearest-neighbour ratio (1.324 vs 1.241).

Costs: roughly 20x more time per step than the MLP on a small GPU, and conv
activations are ~16x larger per sample, so sampling needs a smaller chunk
(`sweep.py` drops to 128 automatically).

### Width is saturated; this is why the trunk had to change

The official evaluation puts both samplers at classifier AUC 0.99-1.00, i.e. the
bottom of the published table, despite the physics metrics above. The cause is
not units (ruled out directly against the evaluation's own reference file: see
`compare_to_reference.py`) but the **voxel energy distribution**: real showers
carry 7.2% of their hits below 1 MeV, mostly at the shower edges, and this
sampler produces 0.6%. Every metric above is an aggregate over voxels and
cannot see it; `quick_metrics.py` now reports it.

A width sweep (`python -m calods2.metrics.sweep --train "hidden=512,1024,2048"`,
16k steps, film, topk gate) says this is the architecture, not the capacity:

| hidden | params | ratio | hits | dim_frac | vox5 | long | radial |
|---|---|---|---|---|---|---|---|
| 512 | 10.9M | 0.681 | 1566 | 0.0050 | 3.55 | 0.133 | 0.214 |
| 1024 | 22.8M | 0.699 | 1643 | 0.0065 | 3.36 | 0.108 | 0.190 |
| 2048 | 49.8M | 0.702 | 1506 | 0.0064 | 3.42 | 0.112 | 0.184 |
| **real** | - | **0.783** | 1561 | **0.072** | **0.63** | - | - |

4.6x more parameters leaves `dim_frac` and `vox5` flat. Profile errors improve
from 512 to 1024 and then stop. The next step is therefore a geometry-aware
3D convolutional U-Net trunk over `(z, phi, r)` -- a flat MLP has no notion of
neighbourhood, which is what dim edge voxels need. Everything measured to work
here (hit head, FiLM, fraction representation, cosine LR + EMA, Heun sampler,
divergence handling) is orthogonal to the trunk and carries over.

Known gaps, in the order worth attacking:

1. **Systematic ~10% energy deficit** (0.709 vs 0.790), worst at low energy
   (-16%, -12%, -2% across the three bins). This is the one remaining
   first-order defect. Cheap things to test first, all sampling-side and
   needing no retraining: the hit gate discards the value in voxels it zeroes
   (`--gate threshold|sample|topk` differ by ~8% in total energy), and the
   denoised read-out replaces a sample with a conditional mean.
2. **Samples sit 1.26x further from real showers** than real showers are from
   each other. Not memorization (the exact Monte-Carlo solver scores 0.4 here,
   i.e. it copies training showers), but not on the manifold either.
3. The official CaloChallenge evaluation (classifier AUC, high-level
   histograms) has **not** been run on this sampler yet -- only the quick
   metrics above.

The **trajectory path** (`train_calo_pinnde.py`, the direct port of the
reference repo's PINN residual ansatz) is kept for reference and for the 1-3D
regime. It is measured not to scale to 6480 dimensions -- see the table at the
top of this file.
