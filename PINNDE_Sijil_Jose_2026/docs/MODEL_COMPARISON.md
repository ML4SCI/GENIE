# CaloChallenge Dataset 2 — model comparison and ranking

Compiled 2026-10-18 from the six evaluation directories copied off the
workstation. Every number below is read directly out of the official
`evaluate.py` output files; nothing is re-derived or estimated.

Dataset 2: 100,000 electron showers, 45 longitudinal layers × 16 φ × 9 r =
6480 voxels, incident energies 1 GeV – 1 TeV (log-uniform). Reference file
`dataset_2_2.hdf5`.

For how each model is built and trained, see
[REPLICATION.md](REPLICATION.md) — the end-to-end guide covering
preprocessing, architecture, loss functions, training schedules and sampling.
For why each one was built and what test exposed its limit, see
[MODEL_EVOLUTION.md](MODEL_EVOLUTION.md), which also compares the two-stage
design with CaloDREAM point by point.

---

## 1. Headline ranking

Ranked by low-level classifier AUC, which is the single hardest and most
discriminating metric in the challenge (0.5 = generated and real are
indistinguishable, 1.0 = trivially separable).

| # | Model | cls-low AUC | cls-high AUC | cls-resnet AUC | mean χ² | params |
|---|-------|------------:|-------------:|---------------:|--------:|-------:|
| **1** | **Two-stage PINN** | **0.578** | **0.695** | **0.804** | **0.0017** | 3.5 M |
| 2 | PINN conv U-Net | 0.683 | 0.925 | 0.894 | 0.0082 | 2.5 M |
| 3 | AE pre-v4 (reconstruction — see §5) | 0.989 | — | — | 0.139 | 3.1 M |
| 4 | Latent diffusion (AE + PINN) | 0.991 | 1.000 | 1.000 | 0.151 | 1.0 M + 3.1 M AE |
| 5 | PINN film MLP | 0.999 | 1.000 | 1.000 | 0.182 | 22.8 M |

`mean χ²` is the average over all 227 histogram χ² values in
`histogram_chi2_2.txt` (eq. 15 of arXiv:2009.03796; lower is better).

The ordering is the same whether you rank by classifier AUC or by histogram
χ², which is reassuring — the two metrics are measuring different things and
they agree.

### Run-to-run spread

Two of these directories (`evaluate_full_conv` and `evaluation_conv`) are
**the same generated file evaluated twice**, with the classifier retrained
from scratch each time. That accidental duplicate is the only error bar we
have, and it is worth quoting:

| mode | run A | run B | spread |
|------|------:|------:|-------:|
| cls-low | 0.6700 | 0.6827 | 0.013 |
| cls-high | 0.9224 | 0.9250 | 0.003 |
| cls-resnet | 0.8814 | 0.8935 | 0.012 |

The two-stage model was also evaluated twice in `cls-low` (0.5800, 0.5781)
and `cls-high` (0.6953, 0.7015), consistent with the same ~0.01 scatter.

**So treat AUC differences below ~0.015 as noise.** The two-stage model's
advantage over the conv U-Net (0.578 vs 0.683, a gap of 0.105) is roughly
eight times that scatter and is therefore real. The gap between the AE,
latent diffusion and film MLP at the bottom (0.989 / 0.991 / 0.999) is partly
within noise — they are all simply "trivially separable" and their exact
ordering is not meaningful.

---

## 2. The models

### 2.1 Two-stage PINN — rank 1

**Files:** `models/3_two_stage/train.py`, `layerdata.py`, `velocity_net.py`
**Checkpoints:** `ts_energy.pt` (stage A), `ts_shape.pt` (stage B)
**Evaluated file:** `two_stage_ds2_100k.hdf5` (100k showers, 50 Heun steps)

Factorizes the shower into two independent generative problems:

- **Stage A — energy.** 45 numbers: the total deposited energy and how it is
  distributed along z, encoded as bounded ratios `u`. A small FiLM-conditioned
  MLP (0.99 M params), trained 20k steps at batch 256.
- **Stage B — shape.** 6480 numbers: each voxel's *share of its own layer*,
  with every layer normalized to sum to 1. A 3D conv U-Net over (z, φ, r)
  (2.47 M params) conditioned on both the incident energy and all 45 stage-A
  numbers (`cond_extra=45`), trained 100k steps at batch 64.

Sampling draws `u`, then `w` conditioned on it, then recombines them
analytically through `layerdata.decode` (measured round-trip error on total
energy: 2.6 × 10⁻⁷).

**Why it wins — three distinct mechanisms, each of which we measured
separately:**

1. **Energy conservation becomes exact by construction.** The single-stage
   models carried a persistent ~10% energy deficit that survived every
   read-out, gate and representation change we tried. Here the total is one
   of 45 numbers a small network models directly, and the shape model
   *cannot affect it at all*. Result: Etot/Einc χ² = 0.0020, against 0.127
   for the film MLP — a factor of 63.
2. **The per-voxel logit representation becomes usable.** Once layer totals
   are stage A's job, the shape model can work in a logit space where an
   empty voxel sits at −0.539 and the faintest lit voxel at ~0.81 — a gap of
   1.35, against 5.5 × 10⁻⁴ in the older `fraction` representation. That gap
   is what the classifier was keying on: dim voxels were 12× too rare and the
   5th-percentile lit voxel was 3.4 MeV against a real 0.63 MeV. Voxel
   distribution χ² drops to 0.0025, the best of any model by a factor of 10.
3. **Each network solves an easier problem.** Stage A is a 45-dimensional
   density; stage B is a *normalized* field with no scale freedom. Neither
   has to simultaneously learn "how much" and "where".

This is the factorization used by CaloDREAM (arXiv:2405.09629), which is
confirmed by the summary paper as the strongest Dataset-2 submission
(low-level AUC 0.531, the best of seventeen — see §6). So the result is a
confirmation of a known good design rather than a novel one.

**Remaining weaknesses.** Its worst observables are the deepest layers —
E layer 44 (χ² = 0.019), 43, 41, 40. Those layers are almost empty in real
showers, so they are the hardest to get right, and they are the worst
observable for *every* model in this table. cls-resnet at 0.804 is also
notably worse than cls-low at 0.578, meaning a convolutional classifier still
finds spatial structure that the flat classifier misses.

### 2.2 PINN conv U-Net — rank 2

**File:** `models/2_direct_pinn/train.py --cond conv`
**Evaluated file:** `conv_recon_ds2.hdf5`

Single-stage: one 3D conv U-Net over the full 6480-voxel shower, FiLM-
conditioned on diffusion time and log incident energy, with circular padding
in φ so the network sees the azimuth as periodic, plus position channels for
normalized z and r index. 2.47 M params.

This model is where most of the architectural progress happened, and it
remains a strong, *simpler* baseline — one network, one sampling pass, no
factorization. Replacing the flat MLP with this conv trunk is what dropped
mean χ² from 0.182 to 0.0082, a factor of 22.

**Why it falls short of two-stage:** it still has to learn energy
conservation. Etot/Einc χ² = 0.0027–0.0035 is decent but its E-layer profile
χ² (0.0139 mean) is 3.5× worse than two-stage's 0.0039, and its voxel
distribution (0.024) is 10× worse. The cls-high AUC of 0.925 is the telling
number: the *high-level* classifier, which sees only layer energies and
shower-shape summaries, separates it almost perfectly. That is a direct
statement that its layer-energy distributions are wrong — exactly the defect
the two-stage factorization removes.

### 2.3 AE pre-v4 — rank 3, but not comparable (see §5)

**File:** `train_auto_encoder_improved.py`
**Checkpoint:** `best_calo_ae_cond_improved.pt`
**Evaluated file:** `ae_recon_ds2_full.hdf5`

Conditional autoencoder, 96-dimensional latent (8 channels × 12 length),
two-head decoder (value head + occupancy head), trained with
`calorimeter_loss_v4` = relative² + 0.1·BCE + 1.0·soft-gated energy term, with
circular φ padding. 3.11 M params, ~400 epochs (improvement stopped near 250).

**Genuine merit:** it compresses a 6480-voxel shower into **96 numbers** and
reconstructs it at relative error 0.395 (measured on 1024 held-out showers
under the units convention it was trained with). As a compression result that
is respectable, and it is what makes the latent-diffusion path conceivable at
all.

**Which checkpoint this is, because it is not the one originally claimed
here.** `best_calo_ae_cond_improved.pt` is **epoch 85, pre-v4** (no `two_head`
key, no `shower_units_gev` marker), trained under the old
`log1p(E_MeV / E_GeV)` units convention. The v4 model — two-head decoder,
`calorimeter_loss_v4`, rel 0.367 at 255 epochs, lit fraction 0.256 — exists
only on the workstation and **has never been officially evaluated.** No
checkpoint in this working copy has `two_head=True`. Every number in this
section therefore belongs to the pre-v4 model.

**Its measured defect** is occupancy: 41.98% of voxels come out lit against a
real 24.95%. Total energy is excellent (median Etot/Einc 0.7826, matching the
reference exactly to four digits, χ² = 0.0013 — the best of any model) but the
energy is spread over too many voxels. Its worst observables are the deep
layers, up to χ² = 0.81 for E layer 44.

This is also the only run with FPD/KPD computed: **FPD 11328.4 ± 9.8**,
**KPD 91.60 ± 1.23** (×10³). One other FPD/KPD exists, from the latent sweep's
96-dimensional run at epoch 460 (`sweeps/latent_sweep_runs/
lat_c8_l12_d96/eval_results/`): **FPD 12437.2 ± 25.9, KPD 97.7 ± 1.0**, with
AUC 0.9919. Same latent size, 5x the epochs, marginally *worse* on all three
— reconstruction error and classifier separability are close to decoupled
here. See §5 for the full latent sweep.

### 2.4 Latent diffusion (AE + PINN) — rank 4

**File:** `models/1_latent_diffusion/train_latent.py`
**Evaluated file:** `latent_recon_ds2.hdf5`

Trains the PINN velocity field in the AE's 96-dimensional latent space, then
decodes samples through the frozen AE decoder. The appeal is obvious: a
96-dimensional diffusion problem is vastly easier than a 6480-dimensional one.

**This did not work, and the honest reading is that it underperforms the
plain conv U-Net it was supposed to improve on** — mean χ² 0.151 against
0.0082, cls-low 0.991 against 0.683. It is the one clearly negative result in
this table.

The failure mode is visible in the χ² breakdown: its worst observables are
not energies but **EC Phi** — the azimuthal energy centroid — at χ² ≈ 0.445
for layers 26–29, far worse than any other model. The AE decoder's 96-number
bottleneck evidently does not preserve φ position faithfully enough, so
errors the diffusion model makes in latent space become *structured* spatial
errors after decoding. Stacking a generative error on top of a lossy
reconstruction error compounds rather than cancels.

Worth noting: the AE reconstruction itself (rank 3) sets a ceiling of
cls-low 0.989 for anything built on this decoder. Latent diffusion scored
0.991 — i.e. it very nearly reached its own ceiling. **The latent approach is
not failing because the diffusion model is bad; it is failing because the
ceiling is too low.** Any further work here has to improve the autoencoder,
not the sampler.

### 2.5 PINN film MLP — rank 5

**File:** `models/2_direct_pinn/train.py --cond film`
**Evaluated file:** `pinnde_recon_film.hdf5`

A flat MLP over all 6480 voxels with FiLM conditioning (sinusoidal features of
diffusion time and log E → per-layer scale and shift). 22.8 M params — by far
the largest model here, and the worst.

**Merit:** this is the step that fixed energy over-dispersion and cut sampling
divergences from 236 to 1, and it is the model that established the whole
reverse-time-ODE approach works at all. It is a necessary rung on the ladder,
not a dead end.

**But it is comprehensively beaten.** AUC ≈ 1.000 on all three classifiers
means a classifier separates its samples from real ones essentially perfectly.
Its mean χ² of 0.182 is the worst in the table, driven by the spatial
observables (EC Eta/Phi ≈ 0.254, Width ≈ 0.235) — a flat MLP has no notion
that voxel 100 and voxel 101 are neighbours, so it cannot get shower
morphology right at any parameter count. **9× more parameters than the conv
U-Net and 22× worse χ².** This is the clearest capacity-vs-inductive-bias
result in the project.

---

## 3. Histogram χ² by observable class

Mean over all layers in each class. Lower is better; best in each column
in bold.

| Model | Etot/Einc | E layer | EC Eta | EC Phi | Width Eta | Width Phi | Voxel dist |
|-------|----------:|--------:|-------:|-------:|----------:|----------:|-----------:|
| Two-stage | 0.0020 | **0.0039** | **0.0010** | **0.0010** | **0.0010** | **0.0009** | **0.0025** |
| PINN conv (A) | 0.0027 | 0.0139 | 0.0015 | 0.0015 | 0.0060 | 0.0065 | 0.0243 |
| PINN conv (B) | 0.0035 | 0.0138 | 0.0015 | 0.0015 | 0.0062 | 0.0067 | 0.0241 |
| AE pre-v4 | **0.0013** | 0.1866 | 0.1363 | 0.1382 | 0.1785 | 0.1852 | 0.1443 |
| Latent diffusion | 0.0146 | 0.1551 | 0.2119 | 0.2121 | 0.2004 | 0.1999 | 0.0629 |
| PINN film MLP | 0.1270 | 0.1208 | 0.2540 | 0.2541 | 0.2342 | 0.2361 | 0.0466 |

Three things stand out:

- **The two-stage model is best or tied-best in six of seven columns.** Its
  only loss is Etot/Einc, where the AE wins — unsurprising, since the AE is
  reconstructing a real shower's energy rather than generating one.
- **The conv trunk is what buys spatial accuracy.** EC Eta/Phi improves by
  two orders of magnitude from the film MLP (0.254) to either conv model
  (0.0015). Width follows the same pattern. No amount of MLP capacity
  substitutes for a convolution.
- **Voxel distribution is where the two-stage layer-logit representation
  pays.** 0.0025 against 0.024 for the conv U-Net — a 10× improvement from a
  change of variables alone, no architectural change.

---

## 4. What each model contributed

The chronology matters more than the ranking, because each model fixed a
specific measured defect in its predecessor:

| Step | Defect it fixed | Evidence |
|------|-----------------|----------|
| hit head | 77% of voxels came out lit (real: 25%) | occupancy gating |
| FiLM conditioning | energy over-dispersion; 236 sampling divergences | divergences → 1 |
| conv trunk | spatial morphology (centroids, widths) | EC χ² 0.254 → 0.0015 |
| layer-logit representation | dim voxels 12× too rare | voxel χ² 0.024 → 0.0025 |
| two-stage factorization | ~10% persistent energy deficit | Etot χ² 0.127 → 0.0020 |

Each row is a hypothesis that was tested and held. Several other
plausible-sounding changes were tested and **rejected on measurement**, and
they are worth recording so they are not retried:

- **EDM preconditioning** — measured harmful with a global σ_d.
- **`--value-correct`** (dividing kept voxel values by hit probability to undo
  occupancy dilution) — principled, but measured *worse*: Etot/Einc went from
  0.780 to 0.963. Left off by default.
- **Global flow-map ansatz** — works in low dimension, fails at 6480-D,
  scaling as N^−0.29.
- **`lit_head`** — the one experiment whose result is untrustworthy: its loss
  was normalized by lit voxels rather than all voxels, making it ~25× too
  large and damaging `q`. If this idea is revisited it needs re-running with
  the weighting fixed.

---

## 4b. The autoencoder latent-dimension sweep

`sweeps/latent_sweep_runs/`. Five latent sizes were trained.
The three whose `train.log` is empty did train — the blank logs come from a
later resume that died on a missing numpy; the checkpoints hold the real
state.

| latent dim | C × L | epochs | rel error | official evaluation |
|---:|---|---:|---:|---|
| 8 | 2 × 4 | 145 | 0.5886 | — |
| 24 | 4 × 6 | 134 | 0.5431 | — |
| 48 | 6 × 8 | 413 | 0.4463 | — |
| 96 | 8 × 12 | 460 | 0.3983 | AUC 0.9919 · FPD 12437 · KPD 97.7 |
| 128 | 8 × 16 | 611 | **0.3507** | — |
| 96 | 8 × 12 | 85 | 0.3947 | AUC 0.9888 · FPD 11328 · KPD 91.6 |

The last row is `best_calo_ae_cond_improved.pt`, the checkpoint that generated
`ae_recon_ds2_full.hdf5` and therefore the AE row of §1.

`rel` was measured here on one fixed protocol — the same 1024 validation
showers at the same split seed for every model — because each checkpoint's
stored `val_loss` is on the old composite objective and the values are not
comparable across runs.

**Three reasons not to read this as a latent-dimension effect:**

- **Epochs are confounded with dimension** (145, 134, 413, 460, 611). The
  128-dim run got four times the 8-dim run's training. The apparent monotone
  improvement 0.598 → 0.394 is partly just more compute, and nothing in this
  sweep separates the two.
- **All five predate the units fix and the v4 loss.** Verified directly:
  evaluated as `log1p(E_MeV/E_GeV)` they give the errors above; evaluated as
  `log1p(E_GeV/E_GeV)` they give 56–174. None of these numbers are comparable
  to v4's 0.367.
- **Only the 96-dim run was ever officially evaluated**, and that is the
  informative part: AUC 0.9919 at epoch 460 against 0.9888 at epoch 85, same
  latent size, with FPD marginally *worse* (12437 vs 11328) despite
  near-identical reconstruction error. **Reconstruction error and classifier
  separability are close to decoupled.** A better latent did not move the
  classifier at all.

That last point is the one with consequences, and the **mechanism is now
known** (NOVA_TRANSFER.md §5.2). A single feature — the number of sub-MeV
hits — nearly reproduces the whole classifier:

| autoencoder | sub-MeV hits | excess | sub-MeV AUC | full cls AUC |
|---|---:|---:|---:|---:|
| pre-v4, epoch 85 | 941 | 8.3× | 0.9660 | 0.9888 |
| sweep d96, epoch 460 | 1806 | **15.9×** | 0.9878 | 0.9919 |
| real | 114 | 1.0× | — | — |

Training longer reduced squared error by smearing energy ever more finely into
empty voxels: **better MSE, worse classifier.** That is why reconstruction
quality and separability came apart, and it means the ceiling is not "the
autoencoder is bad" — its energy is right to 0.5% and its reconstruction error
is a respectable 0.43 — but one specific, nameable and plausibly fixable
defect.

---

## 5. Caveats — read before quoting any of this

**The AE row is a reconstruction, not a generated sample.**
`ae_recon_ds2_full.hdf5` contains the autoencoder's reconstructions of *real*
showers — it was handed the true shower, compressed it to 96 numbers, and
decoded it back. It is not a generative model and cannot be ranked against
the others as one. Its AUC of 0.989 should be read as an **upper bound on the
fidelity any AE-based generator can reach with this decoder**, which is
precisely why the latent-diffusion result in §2.4 is so constrained. I have
left it in the table because that ceiling is the single most useful thing it
tells us, but the "rank 3" label is not a like-for-like comparison.

**The AE was evaluated with `--mode all`, the others with individual modes.**
So the AE has one AUC (0.9888) where others have three. The `all` mode
classifier sees both low- and high-level features; it is closest to cls-low
but not identical.

**FPD/KPD exists for only one model.** Those metrics are the ones most often
quoted in the literature, and we have them for the AE alone. Running them for
the two-stage model is the cheapest way to make this work comparable to
published results.

**Classifier AUC has ~0.015 run-to-run scatter** (§1). Several of the
differences in the bottom half of the ranking are within that.

**The autoencoder row is the pre-v4 model, not v4** (§2.3). I originally
labelled it "AE v4" and quoted v4's reconstruction quality (rel 0.367 at 255
epochs) next to this model's AUC and FPD. Those came from different
checkpoints. Everything in the AE row — AUC 0.9888, FPD 11328, 41.98%
occupancy, the χ² column in §3 — belongs to `best_calo_ae_cond_improved.pt`
at epoch 85, pre-v4, old units convention. The consequence for the
latent-diffusion argument is spelled out in §7, item 5.

**The two-stage result is from a 100k-sample file; earlier two-stage numbers
in `classifier_cls-low_2.txt` came from a 20k file that crashed the
evaluation.** The `evaluate.py` splitter hard-codes 60000/20000/20000 for
Dataset 2, so classifier modes require exactly 100k showers on both sides.
Only the 100k numbers are reported above; the failed 20k attempt left a
header in the file with no result after it.

**The published numbers in §6 were extracted from the paper's PDF, whose
table columns the text extractor interleaved.** I reconstructed Tables C15
and C16 by grouping the values into per-submission triplets and cross-checking
against the paper's own prose — which states that CaloDREAM and CaloDiffusion
have the lowest AUCs, that flow-based models follow, and that VAE- and
GAN-based submissions are worst. All three statements hold in my
reconstruction, and the Geant4-vs-itself floor came out at the expected 0.500.
I am therefore confident in it, but **spot-check CaloDREAM, CaloDiffusion and
Calo-VQ against Table C15 in the PDF before any of it goes into a writeup** —
a single misaligned row would change our apparent ranking.

**Our AUCs are single-run; the published ones are 10-run averages** with
isotonic calibration. Same `evaluate.py`, same modes, so the procedure
matches, but our ±0.015 scatter (§1) is wider than their ±0.004 quoted
uncertainties. Any comparison at the 0.01 level is not meaningful; the
0.10-level gaps in §6.2 are.

---

## 6. Comparison with the published CaloChallenge submissions

Source: **arXiv:2410.21611**, *CaloChallenge 2022: A Community Challenge for
Fast Calorimeter Simulation*, Tables C15 (AUCs) and C16 (KPD/FPD) for
Dataset 2. All 17 official submissions, averaged over 10 independent
classifier trainings with isotonic calibration. Our numbers come from the
same `evaluate.py` with the same three classifier modes, so the comparison is
apples-to-apples in procedure — but single-run, not 10-run averaged (§7).

### 6.1 Low-level AUC — the headline metric

| Rank | Model | low-level | high-level | ResNet |
|-----:|-------|----------:|-----------:|-------:|
| 1 | CaloDREAM | 0.531 ± 0.003 | 0.521 ± 0.002 | 0.681 ± 0.015 |
| 2 | CaloDiffusion | 0.577 ± 0.004 | 0.591 ± 0.009 | 0.680 ± 0.006 |
| — | **Two-stage PINN (ours)** | **0.578** | **0.695** | **0.804** |
| 3 | CaloScore | 0.595 ± 0.003 | 0.666 ± 0.002 | 0.795 ± 0.011 |
| 4 | SuperCalo | 0.694 ± 0.006 | 0.757 ± 0.004 | 0.986 ± 0.001 |
| — | **PINN conv U-Net (ours)** | **0.683** | **0.925** | **0.894** |
| 5 | conv. L2LFlows | 0.708 ± 0.004 | 0.737 ± 0.002 | 0.941 ± 0.003 |
| 6 | CaloScore distilled | 0.710 ± 0.002 | 0.891 ± 0.003 | 0.965 ± 0.002 |
| 7 | CaloINN | 0.743 ± 0.002 | 0.865 ± 0.003 | 0.994 ± 0.000 |
| 8 | CaloScore single-shot | 0.747 ± 0.003 | 0.902 ± 0.002 | 0.973 ± 0.002 |
| 9 | iCaloFlow teacher | 0.763 ± 0.004 | 0.837 ± 0.005 | 0.970 ± 0.002 |
| 10 | iCaloFlow student | 0.819 ± 0.004 | 0.886 ± 0.003 | 0.975 ± 0.002 |
| 11 | CaloPointFlow | 0.863 ± 0.005 | 0.908 ± 0.004 | 0.999 ± 0.000 |
| 12 | CaloVAE+INN | 0.907 ± 0.004 | 1.000 ± 0.000 | 0.993 ± 0.001 |
| 13 | MDMA | 0.942 ± 0.005 | 0.987 ± 0.001 | 1.000 ± 0.000 |
| 14 | DeepTree | 0.963 ± 0.002 | 0.927 ± 0.002 | 0.999 ± 0.000 |
| 15 | CaloLatent | 0.983 ± 0.001 | 0.995 ± 0.001 | 1.000 ± 0.000 |
| 16 | CaloDiT | 0.984 ± 0.001 | 0.912 ± 0.002 | 0.988 ± 0.001 |
| 17 | Calo-VQ | 0.986 ± 0.001 | 0.994 ± 0.000 | 0.999 ± 0.000 |
| — | **Latent diffusion (ours)** | **0.991** | **1.000** | **1.000** |
| — | **PINN film MLP (ours)** | **0.999** | **1.000** | **1.000** |
| — | Geant4 vs itself (floor) | 0.500 ± 0.002 | 0.499 ± 0.002 | 0.500 ± 0.004 |

**The two-stage model's low-level AUC of 0.578 is statistically
indistinguishable from CaloDiffusion's 0.577**, which placed second of
seventeen. Inserted into the published table it ranks **2nd–3rd of 18** on
this metric, behind only CaloDREAM.

That is a genuinely strong result and the honest headline. But it comes with
an immediate qualification, below.

### 6.2 Where we are weaker than the ranking suggests

Our model's advantage does **not** hold up across the other two classifiers:

| metric | ours | CaloDiffusion | CaloDREAM | our gap |
|--------|-----:|--------------:|----------:|--------:|
| low-level | 0.578 | 0.577 | 0.531 | tied 2nd |
| high-level | 0.695 | 0.591 | 0.521 | **+0.104** |
| ResNet | 0.804 | 0.680 | 0.681 | **+0.124** |

On low-level observables we match the second-best published submission. On
high-level observables we fall behind it by 0.104 — about seven times our
run-to-run scatter — and drop to roughly 4th place, behind CaloScore (0.666).
On the CNN-ResNet classifier we are 0.124 behind and again ~4th.

**This pattern is diagnostic, not noise.** The low-level classifier sees raw
voxels; the high-level one sees layer energies and shower-shape summaries;
the ResNet sees spatial structure convolutionally. We do well on raw voxels
and worse on *derived* quantities, which says our per-voxel marginals are
close to right while the **correlations between voxels** — the things that
determine layer energies, centroids and widths — are not. The deepest-layer
χ² values in §3 (E layer 40–44) point at the same place: the sparse tail of
the shower, where voxel-to-voxel correlation matters most and where there is
least training signal.

So the fair summary is: **competitive with the best published submissions on
one of three classifiers, mid-table on the other two.** Not "2nd place in the
CaloChallenge".

### 6.3 Our other models, placed

- **PINN conv U-Net** (0.683 low-level) sits between SuperCalo (0.694) and
  conv. L2LFlows (0.708) — genuinely mid-table, around **5th of 18**, and
  ahead of every flow-based submission on low-level. Its high-level AUC of
  0.925 is poor, though, worse than everything down to CaloPointFlow.
- **Latent diffusion** (0.991) and **film MLP** (0.999) land at the very
  bottom, below Calo-VQ, the worst published submission. These are not
  competitive and should be presented as development stages, not results.

### 6.4 CaloLatent — direct corroboration of our §2.4 finding

Worth singling out: **CaloLatent**, the one published submission that also
does diffusion in a learned latent space, scored **0.983 low-level — 15th of
17**. Our latent-diffusion attempt scored 0.991 and failed for the reason
diagnosed in §2.4 (pinned against its decoder's reconstruction ceiling).

That the only comparable published approach also lands near the bottom is
strong independent evidence that the problem is the latent-space formulation
itself on this dataset, not our particular autoencoder. It substantially
raises my confidence in the recommendation to drop that path (§7, item 5).

### 6.5 FPD / KPD — where we are clearly not competitive

Dataset 2, Table C16, selected rows (lower is better):

| Model | KPD ×10³ | FPD ×10³ |
|-------|---------:|---------:|
| Geant4 vs itself (floor) | −0.028 ± 0.022 | 10.78 ± 0.79 |
| CaloDREAM | 0.023 ± 0.036 | 24.65 ± 1.04 |
| CaloScore | 0.149 ± 0.057 | 112.48 ± 0.91 |
| CaloDiffusion | 0.174 ± 0.042 | 146.93 ± 0.87 |
| conv. L2LFlows | 0.271 ± 0.090 | 157.40 ± 0.97 |
| … | | |
| CaloVAE+INN (worst published) | 45.61 ± 0.83 | 5443.43 ± 27.23 |
| **AE pre-v4 (ours, reconstruction)** | **91.60 ± 1.23** | **11328.44 ± 9.84** |

Our only FPD/KPD measurement — the autoencoder's — is **worse than the worst
published submission, by a factor of two in both metrics.** This is a real
result, not a normalization artifact: it is consistent with the AE's measured
occupancy defect (41.98% of voxels lit against a real 24.95%), which FPD and
KPD are specifically sensitive to because they compare distributions in a
feature space rather than marginal histograms.

**We have no FPD/KPD for the two-stage model**, which is the gap that matters
most for publication, since these are the metrics the paper leads with for
distributional fidelity. Given the two-stage model's voxel-distribution χ² of
0.0025 (10× better than the conv U-Net) I would expect a far better number —
but that is an expectation, not a measurement, and it should not be stated as
anything else until it is run.

---

## 7. Recommended next steps, in order of value

Specified as runnable experiments with controls, predictions and falsification
criteria in [EXPERIMENTS.md](EXPERIMENTS.md), and
[NOVA_TRANSFER.md](NOVA_TRANSFER.md) for the sparse-autoencoder techniques
adapted from the NOvA CAE report.

1. **Run FPD/KPD on the two-stage samples.** One command, no retraining, and
   §6.5 shows this is now the single biggest hole: it is the metric the
   summary paper leads with, we have it for the autoencoder only, and that one
   measurement is worse than the worst published submission.
2. **Attack the high-level and ResNet classifiers.** §6.2 shows this is where
   we lose to CaloDiffusion — by 0.104 and 0.124, far outside noise — while
   matching it on low-level. The defect is voxel-to-voxel correlation, and the
   deep-layer χ² values localize it to the sparse shower tail. This is the
   highest-value modelling work remaining, and the clearest path from
   "competitive on one metric" to "competitive overall".
3. **Average the classifiers over 10 runs**, as the paper does. Cheap, and it
   removes the ±0.015 scatter that currently makes our numbers
   non-comparable at the precision the published table is quoted to.
4. **Train stage B longer.** It is the only model here still improving, and
   100k steps at batch 64 is modest. Its worst observables are the deep
   layers, which is exactly what item 2 is about.
5. **Re-measure the latent ceiling on the v4 autoencoder before deciding
   anything about that path.** §2.4's "pinned against its ceiling" argument
   used the *pre-v4* reconstruction's AUC of 0.989 (§2.3). The v4 model has a
   hit head, measures lit fraction 0.256 against pre-v4's 0.420, and has never
   been evaluated. Copy it back from the workstation, run
   `train_latent.py --ceiling`, and evaluate that file. The conclusion may
   well stand — CaloLatent at 15th of 17 (§6.4) is independent support — but
   the evidence currently quoted for it is stale.

---

## Appendix — source files

| Model | Evaluation directory | Generated file |
|-------|---------------------|----------------|
| Two-stage PINN | `evaluation_two_stage/` | `two_stage_ds2_100k.hdf5` |
| PINN conv (A) | `evaluate_full_conv/` | `conv_recon_ds2.hdf5` |
| PINN conv (B) | `evaluation_conv/` | `conv_recon_ds2.hdf5` |
| AE pre-v4 | `evaluation_results_ds2/` | `ae_recon_ds2_full.hdf5` |
| Latent diffusion | `evaluate_latent_recon/` | `latent_recon_ds2.hdf5` |
| PINN film MLP | `evaluate_full_pinnde/` | `pinnde_recon_film.hdf5` |

All classifier numbers from `classifier_<mode>_2.txt`; all χ² from
`histogram_chi2_2.txt`; FPD/KPD from `fpd_kpd_2.txt`.

Published numbers: arXiv:2410.21611, Tables C15 (AUC), C16 (KPD/FPD),
Figure 67. CaloDREAM: arXiv:2405.09629. CaloDiffusion: arXiv:2308.03876.
χ² definition: eq. 15 of arXiv:2009.03796.
