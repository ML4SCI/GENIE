# Transferring the NOvA sparse-CAE approach to Dataset 2

Source: *An AI Agent Trains a Sparse Autoencoder on NOvA Pixel Maps at NERSC*,
H. B. Prosper, Genesis Mission hackathon, October 2026
(`../nova_cae_report.pdf`).

That report solves the same problem we have — compressing sparse detector
images with an autoencoder — on a dataset that is **more** sparse than ours,
and it solves one part of it better than we do. This document maps its
approach onto Dataset 2, records what we already had independently, and
measures the gap.

**The headline, measured here and not in the report:** our autoencoder's
reconstruction has **precision 0.54 against recall 0.92** — it finds 92% of
real hits and then paints 2.3× too many. That is the exact over-painting
failure the report diagnoses, and no metric in this repository could see it,
because every one of them is a distribution comparison.

---

## 1. The two problems side by side

| | NOvA CVN maps | CaloChallenge Dataset 2 |
|---|---|---|
| object | 2 views × 100 planes × 80 cells = 16,000 pixels | 45 layers × 16 φ × 9 r = 6,480 voxels |
| value per cell | 8-bit charge, 1–165 ADC | energy, 15.15 keV – ~6 GeV |
| **sparsity** | **99.4% empty** (0.58% occupancy) | **75.5% empty** (24.5% occupancy) |
| conditioning | none | incident energy, 3 decades |
| structure | two views, different geometry | φ periodic, z and r are not |
| figure of merit | **hit F1** | **classifier AUC** (and χ², FPD/KPD) |
| latent | 8 → 256 numbers | 96 numbers (8 × 12) |
| events | 217,372 (balanced data/MC) | 100,000 |

Two differences matter for what transfers:

- **Our sparsity is far milder** — 3:1 empty-to-lit against their 171:1. Every
  argument in the report that scales with the imbalance is correspondingly
  weaker here, so nothing can be adopted on the strength of their numbers
  alone.
- **Our figure of merit is different, and it points the opposite way on one
  key decision.** See §5. This is the most important thing in this document.

---

## 2. What the report does

**A hurdle model.** The decoder emits two numbers per pixel: a logit `s_i`
giving the hit probability, and a log-charge `μ_i` giving the predicted
`log(1+y)` *if* the pixel is hit. The report derives this from a likelihood
(Mullahy's hurdle model) rather than asserting it:

```
L_i = (1 − p_i)^(1−h_i) · [ p_i · f(z_i | μ_i, σ) ]^(h_i)
```

Taking −log splits it exactly into a binary cross-entropy term and a charge
term over hits only. The training loss is that likelihood with four stated
modifications (focal hit term, dropped truncation, fixed σ absorbed into λ,
normalization by the hit count).

**Focal loss for the hit term**, and the reasoning is the part worth copying.
The textbook fix for class imbalance is to up-weight the rare class, and the
report works out what that does: for hit weight `w`, the loss-minimizing
prediction crosses 0.5 at true probability

```
q > 1/(w + 1)
```

With `w = 171` (the full rarity) the threshold is `q > 0.0058`, exactly the
mean occupancy — so **every pixel with even an average chance of being hit is
declared a hit.** Measured consequence: tracks painted ~3 pixels wide, 2.68×
too many hits.

Focal loss avoids this by down-weighting what is *already easy* rather than
up-weighting what is rare:

```
FL(p_t) = −α_t · (1 − p_t)^γ · log p_t        γ = 2
```

where `p_t` is the probability assigned to the **correct** answer. A pixel
already right at `p_t = 0.9` has its loss cut 100×, so the mass of obviously
empty pixels falls silent while uncertain pixels near a track keep full
weight. Critically it does **not** shift an uncertain pixel's prediction
toward "hit". Measured: thin tracks, 0.95 of the true hit count.

**A tuned read-out threshold.** `ŷ_i = 1[p_i > τ] · (e^{μ_i} − 1)`, with τ
chosen *on validation* to maximize F1 — and it came out at **0.35–0.40, not
0.5**. The report explains why: focal loss is not a proper loss, so it pulls
predictions toward 0.5, and α then shifts them toward "empty". `σ(s)` is
therefore **not a calibrated occupancy probability**, and τ must be fitted
separately from training.

**F1 as the figure of merit**, plus per-event hit-count and total-charge
closure and a charge RMS. Not pixel MSE, which an empty prediction scores
well on.

**An honest ablation.** CoordConv was argued for on good grounds and then
tested: it made no measurable difference (ΔF1 −0.003/0.000/+0.005), because
the architecture's fully connected bottleneck already supplies absolute
position. The report keeps the negative result and explains it.

Training: AdamW 1e-3 / wd 1e-4, one-cycle (5% warmup → cosine to zero), batch
512, bf16, grad clip 1.0, 50 epochs, keep best validation F1.

---

## 3. What we already had, independently

Worth stating first, so the transfer is not overclaimed. Our v4 autoencoder
and the PINN shape model both already have:

| Report ingredient | Our equivalent | Arrived at via |
|---|---|---|
| two-part decoder (hit logit + value) | v4's `two_head`; `VelocityNet.hit_head` | the v3 post-mortem: a shared output needed logits of ±3 from a number that must also equal 4e-4 |
| charge term that ignores empty cells | partially — `rel2` is scale-free | — |
| log-space value target | `log1p(E/E_inc)`, with `x_scale` | — |
| separate towers for different geometry | per-layer 2D encoder + 1D along z | the factorization that makes 96 numbers enough |
| coordinate channels | `pos_channels` in the PINN conv trunk | CaloDiffusion |
| cosine schedule with warmup, grad clip 1.0 | identical | — |
| latent-size scan | the d = 8…128 sweep | — |

So the architecture is already a hurdle model in both of our pipelines. **What
is missing is the hit term's form, the read-out threshold, and the metric.**

---

## 4. The measured gap

### 4.1 We were blind to placement

Every metric in `quick_metrics.py` was a *distribution* comparison. A model can
match the lit fraction exactly while lighting entirely the wrong voxels, and
nothing we had would notice.

Our reconstructions pair voxel-for-voxel with their inputs (verified: the
incident energies in `ae_recon_ds2_full.hdf5` match `data/Dataset2.pt` row for
row), so the report's metric is computable on files we already have. On 20,000
showers at the 15.15 keV readout threshold:

| | value |
|---|---:|
| true lit fraction | 0.2450 |
| reconstructed lit fraction | **0.4198** |
| **precision** | **0.5388** |
| **recall** | **0.9232** |
| **F1** | **0.6805** |
| median per-event F1 | 0.5059 |
| 10th-percentile per-event F1 | 0.3507 |
| reco/true hit count | **2.32 ± 0.75** |
| reco/true total charge | 1.005 ± 0.023 |
| charge RMS on true hits, log(1+E/MeV) | 1.24 |

**Recall 0.92 with precision 0.54 is over-painting**, and it is the same
signature the report measured for weighted cross-entropy — 2.68× too many
hits there against **2.32× here.** Two independent projects, two different
detectors, the same failure mode and nearly the same number.

Note what the aggregate metrics said about this same file: total charge
closure 1.005 ± 0.023, and E_dep/E_inc agreeing with real to four digits.
**The energy was right and distributed over twice as many voxels as it should
have been.** That is precisely the kind of defect a classifier sees and a
histogram does not.

(The checkpoint measured here is `best_calo_ae_cond_improved.pt` — pre-v4, see
MODEL_COMPARISON.md §2.3. The v4 model reports a lit fraction of 0.256 against
a real 0.250, so its occupancy *rate* is nearly right; whether its
*placement* is right is unmeasured, and §7 is how to find out.)

### 4.2 Our read-out threshold is hard-coded at 0.5

`FactorizedCaloAE.decode` gates on `h > 0`, i.e. τ = 0.5. Even for a
calibrated probability that is the accuracy-optimal cut, not the F1-optimal
one and not the one that conserves hit count. The report found 0.35–0.40.

Exercising the new `tune_threshold()` on a 1-epoch model gave **τ = 0.35**,
with F1 0.381 against 0.256 at τ = 0.5 — a large difference from a parameter
we never tuned.

---

## 5. Where the transfer breaks, and this is the important part

The report's figure of merit is **F1**. Ours is **classifier AUC**. On one
decision they disagree, and following the report blindly would make our
primary metric worse.

A post-hoc energy cut is the cheapest way to raise F1 — the same knob as τ,
and the same thing CaloDREAM does with its 1 MeV cut. Sweeping it on our
reconstruction:

| cut [MeV] | precision | recall | F1 | lit frac | hits r/t | charge r/t |
|---:|---:|---:|---:|---:|---:|---:|
| 0.015 (readout) | 0.539 | 0.923 | 0.681 | 0.420 | 2.32 | 1.005 |
| 0.5 | 0.686 | 0.831 | 0.752 | 0.297 | 1.39 | 0.998 |
| 1.0 | 0.717 | 0.804 | 0.758 | 0.275 | 1.24 | 0.992 |
| 2.0 | 0.754 | 0.768 | **0.761** | **0.249** | 1.08 | 0.977 |

F1 rises from 0.681 to 0.761 and the lit fraction lands on the true 0.245,
with total charge still within 2.3%. By the report's own figure of merit this
is a large free win, with no retraining.

**It would probably make our classifier AUC worse.** A 1–2 MeV cut works by
*deleting the dim population*, and real showers carry **7.2% of their hits
below 1 MeV**. Those dim voxels are the single defect we have spent the most
effort on: MODEL_EVOLUTION.md §5.5 records dim voxels being 12× too rare as
one of the two defects that motivated the entire two-stage model. A cut that
removes them raises F1 and destroys exactly the thing the classifier keys on.

CaloDREAM says as much about its own cut: it "completely adjusts the sparsity
up to a specific value by removing" low-energy voxels, and their classifier
still "misses the low-energetic noise affecting the sparsity and the voxel
energy distributions".

**Conclusion: adopt the diagnosis, not the remedy.** Focal loss fixes
over-painting at *training* time by not over-predicting in the first place,
which keeps the dim voxels. A threshold cut fixes it at *read-out* time by
deleting them. For an F1 metric these are equivalent; for a classifier metric
they are opposites.

This is the one place where the report's approach must not be copied
mechanically, and it is only visible because the two projects are scored
differently.

### 5.1 How much worse: the cut is non-monotonic

Measured directly, by computing single-feature AUCs against the real showers
(the evaluation's reference keeps its dim voxels; only our file is cut):

| cut [MeV] | sub-MeV-hit AUC | hit-count AUC | reco sub-MeV hits | real |
|---:|---:|---:|---:|---:|
| 0.015 (readout) | 0.9660 | 0.6897 | 941.0 | 113.6 |
| **0.5** | **0.6236** | 0.5732 | **143.1** | 113.6 |
| 1.0 | **0.9987** | 0.5472 | 0.0 | 113.6 |
| 2.0 | **0.9987** | 0.5142 | 0.0 | 113.6 |

**The F1-optimal cut is close to the classifier-pessimal one.** F1 rises
monotonically to 2 MeV; the classifier-visible feature has a sharp optimum
near 0.5 MeV and then collapses, because above ~1 MeV our file has *zero*
sub-MeV voxels while the reference has 114, which is trivially detectable in
the other direction.

So the two metrics do not merely disagree — following F1 drives this feature
from 0.62 to 0.9987. A 0.5 MeV cut, on the other hand, is close to free and
worth testing on the real classifier.

### 5.2 What actually carries the ceiling

Single-feature AUCs on the pre-v4 reconstruction, against the full cls-low DNN
score of 0.9888:

| feature | AUC | real mean | reco mean |
|---|---:|---:|---:|
| **hits below 1 MeV** | **0.9660** | 113.6 | **941.0** |
| mean lit-voxel energy | 0.7136 | 39.14 | 23.61 |
| hit count | 0.6897 | 1588 | 2720 |
| max voxel energy | 0.5077 | 703.3 | 688.4 |
| total energy / E_inc | 0.5064 | 0.783 | 0.787 |

**One feature nearly reproduces the whole classifier.** The autoencoder
invents 8.3× too many sub-MeV voxels; total energy and peak energy are
indistinguishable from real. The ceiling is not "the autoencoder reconstructs
poorly" — reconstruction error is a respectable 0.43 and the energy is right
to 0.5%. It is one specific, nameable defect: **dim-energy smearing into
empty voxels.**

And it tracks across both evaluated autoencoders:

| autoencoder | lit frac | sub-MeV hits | excess | sub-MeV AUC | full cls AUC |
|---|---:|---:|---:|---:|---:|
| pre-v4, epoch 85 | 0.4198 | 941 | 8.3× | 0.9660 | 0.9888 |
| sweep d96, epoch 460 | 0.5655 | 1806 | **15.9×** | 0.9878 | 0.9919 |
| **real** | 0.2450 | 114 | 1.0× | — | — |

**This explains the latent-sweep paradox** (MODEL_COMPARISON.md §4b): the
longer-trained model has a *worse* sub-MeV excess and a correspondingly worse
AUC. Training longer reduced squared error by spreading energy ever more
finely into empty voxels — better MSE, worse classifier. That is the mechanism
behind "reconstruction error and classifier separability are decoupled", and
it is exactly the failure mode a focal hit term exists to prevent.

A caveat on the inference: a single feature correlating with the DNN's score
does not prove the DNN is using it. But the correspondence holds across two
models with very different excesses (8.3× → 0.989, 15.9× → 0.992), and the
other four features are at or near 0.5, so there is not much else left for the
classifier to be using.

---

### 5.3 Consequence: the latent path deserves a retry

The latent-diffusion result (MODEL_EVOLUTION.md §4) was that the sampler
reached its ceiling: 0.991 against a reconstruction ceiling of 0.989. The
conclusion drawn was "this path needs a better autoencoder, not a better
sampler", and the recommendation was to drop it.

§5.2 changes the character of that conclusion. The ceiling is not diffuse
reconstruction error that would need a fundamentally better autoencoder. It is
**one defect, in one direction, of the kind focal loss is designed to
prevent** — and the v4 model already reports a lit fraction of 0.256 against a
real 0.250, so it may already have largely fixed it. The v4 ceiling has never
been measured.

So the ordering changes. Before investing in focal loss:

1. **Measure the v4 ceiling** (EXPERIMENTS.md experiment 3). If v4's sub-MeV
   excess is near 1×, the ceiling may already be far below 0.989 and the
   latent path has headroom *today*, with no new training at all.
2. **Test a 0.5 MeV model-side cut** on the existing files. §5.1 shows it
   takes the dominant feature from 0.966 to 0.624 for free. This is the
   cheapest experiment in the whole project — one evaluation pass.
3. **Only then** consider focal loss, which is the principled training-time
   version of the same fix.

**One caveat that limits all of this.** The latent sampler's own worst
observable was the azimuthal energy centroid, EC φ χ² ≈ 0.445 for layers
26–29 — far worse than any other model. That is a *positional* defect of the
96-number bottleneck, not an occupancy defect, and **focal loss will not touch
it.** So there may be a second, independent ceiling underneath the first. The
honest position is that one known ceiling contributor now looks fixable and
another does not, and experiment 3 is how to find out which dominates.

---

## 6. What is implemented

### `src/calods2/metrics/quick_metrics.py --paired`

Adds the occupancy-placement report: precision, recall, F1, lit fraction and
hit-count ratio at the readout threshold and at 1 and 2 MeV. Only meaningful
for reconstructions; it verifies the incident energies match row for row and
refuses otherwise.

```bash
python -m calods2.metrics.quick_metrics --gen ae_recon_ds2_full.hdf5 \
    --data data/Dataset2.pt --n 4096 --paired
```

### `calorimeter_loss_v5` in `train_auto_encoder_improved.py`

The focal hurdle objective:

```
L = focal(hit) + λ · charge(true hits only) + w_E · energy_closure
```

Three deliberate departures from the report, each with a reason:

- **The charge term is in the scaled log space** (`v` against `x_log/x_scale`),
  not on `x_log` directly. My first version used `(v·x_scale − x_log)²`, and
  the smoke test caught it: `x_log` is ~1e-4 here, so that term came out at
  ~1e-8 and was invisible next to the focal term — **the exact v3 defect this
  project already paid for once.** The report does not hit this because its
  charges are 1–165 ADC, so `log(1+y)` is naturally O(1). Switching to
  `log(1 + E/MeV)` to get the same effect would reintroduce the Jensen bias
  that made our absolute-log representation undershoot energy by 2.4×, so
  scaling is the right fix, not changing the variable.
- **The energy-closure term is kept** (the report has no equivalent, having no
  conservation law to respect) and still uses the *soft-gated* prediction, for
  the reason in v4's docstring.
- **λ defaults to 0.1, not 1.0.** At λ = 1 the charge term is **83%** of the
  loss and the hit term is starved — which would reintroduce the very
  over-painting this is meant to fix. Measured term balance:

| λ | focal | charge | energy |
|---:|---:|---:|---:|
| 0.05 | 0.43 | 0.20 | 0.37 |
| **0.1** | **0.36** | **0.33** | **0.31** |
| 0.25 | 0.24 | 0.55 | 0.21 |
| 1.0 | 0.09 | 0.83 | 0.08 |

Verified on CPU: all three terms carry real gradients (norms 6.8, 24.1, 15.1),
so none is inert — the failure that made v3's energy term useless at 5.9e-7.
`γ = 2` provably suppresses (focal 0.662 against 2.432 at γ = 0).

### `tune_threshold` in `train_auto_encoder_improved.py`

Picks τ on validation by maximizing F1 and returns the full
precision/recall/F1 table, so the choice is visible rather than implicit.

### CLI

```bash
python train_auto_encoder_improved.py --loss v5 --focal-lambda 0.1 --tau 0.40 \
    --phi-circular --epochs 250 --checkpoint-path best_calo_ae_v5.pt
```

`--loss v4` remains the default; nothing changes unless asked for. Both paths
were exercised end to end through the real training loop (train, validate,
checkpoint) on CPU.

---

## 7. The experiment

**This must be an A/B against v4, not a replacement.** Our imbalance is 3:1,
not 171:1, so the report's motivation is much weaker here and the result is
genuinely uncertain. Add to [EXPERIMENTS.md](EXPERIMENTS.md) as experiment 5.

```bash
# control: v4, current objective
python train_auto_encoder_improved.py --loss v4 --phi-circular --epochs 250 \
    --checkpoint-path ae_v4_ctl.pt

# treatment: focal hurdle
python train_auto_encoder_improved.py --loss v5 --focal-lambda 0.1 --tau 0.40 \
    --phi-circular --epochs 250 --checkpoint-path ae_v5.pt
```

Then, for each, generate reconstructions and run **both** metrics:

```bash
python -m calods2.metrics.quick_metrics --gen <recon>.hdf5 --data data/Dataset2.pt --paired
python scripts/cluster/run_full_evaluation.py --skip-generate --gen-file <recon>.hdf5 --out-dir <dir>
```

| quantity | v4 (pre-v4 measured) | v5 prediction |
|---|---|---|
| precision | 0.539 | **rises to 0.70–0.80** |
| recall | 0.923 | falls to ~0.80 |
| **F1** | 0.681 | **0.75–0.80** |
| lit fraction (true 0.245) | 0.420 | 0.25–0.30 |
| reco/true hits | 2.32 | 1.0–1.2 |
| rel error | 0.395 | unchanged or slightly worse |
| **cls-low AUC** | 0.989 | **the test that matters** |
| fraction of hits < 1 MeV (real 0.072) | — | **must not fall** |

**Falsified if** F1 improves but the AUC does not, *or* if the sub-MeV hit
fraction drops. Either would mean focal loss is buying placement accuracy the
same way a cut does — by suppressing dim voxels — in which case it is not the
fix we want and the honest record is a negative result in
MODEL_EVOLUTION.md §9.

That last row is the one to check first. It distinguishes "the model stopped
over-painting" from "the model got more conservative", which F1 cannot tell
apart.

---

## 8. One idea the report supplies for the PINN side

The report's framing retroactively justifies the stage-A occupancy head in
EXPERIMENTS.md experiment 4. Stage A's target has a point mass at zero —
17.4% of u entries, rising to 48.5% for u₄₄ — because a layer can be exactly
empty. That is a **hurdle structure in the energy block**, not just the voxel
block, and the report's likelihood derivation is the principled argument for
modelling it with a separate binary head rather than hoping a continuous
regression covers it.

The report also supplies the warning: once the hit term is focal, the
occupancy probabilities are not calibrated, so stage A would need its own
tuned threshold rather than 0.5 — and with 44 coordinates whose atom masses
span three orders of magnitude, plausibly a per-coordinate threshold.

---

## 9. What the report does better than us, procedurally

Two habits worth copying outright, independent of any model detail:

- **Phase gating.** Each phase was one cheap job that had to succeed before
  more compute was spent: access → profile → smoke test → scan → evaluate.
  Total cost 2.5 GPU node-hours for six trained models and an ablation. We
  have lost multi-hour runs to defects a 200-step smoke test would have caught.
- **Ablating a design choice that was argued for on good grounds.** CoordConv
  had a clean motivation, was tested anyway, and made no difference. That is
  the same discipline as our width sweep (MODEL_EVOLUTION.md §5.4), and the
  report's conclusion — "a plausible argument for a design choice is a
  hypothesis, and an ablation is how one tests it" — is the sentence this
  project has relearned four times.

---

## References

- Prosper, *An AI Agent Trains a Sparse Autoencoder on NOvA Pixel Maps at
  NERSC*, October 2026 — `../nova_cae_report.pdf`
- Lin et al., focal loss — arXiv:1708.02002
- Mullahy, hurdle models for count data — J. Econometrics 33 (1986) 341
- Liu et al., CoordConv — arXiv:1807.03247
- CaloDREAM (1 MeV cut on sparsity) — arXiv:2405.09629
