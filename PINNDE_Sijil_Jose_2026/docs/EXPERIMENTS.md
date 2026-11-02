# Experiment plan

Five experiments, ordered by information gained per GPU-hour. Each one states
its prediction *and* what result would falsify it, because three conclusions in
this project were asserted from a plausible mechanism and later overturned by a
control (see MODEL_EVOLUTION.md §9). Every experiment here has a control.

Companion docs: [MODEL_COMPARISON.md](MODEL_COMPARISON.md) (results),
[REPLICATION.md](REPLICATION.md) (how to build each model),
[MODEL_EVOLUTION.md](MODEL_EVOLUTION.md) (why each was built).

| # | Experiment | Training? | Cost | Answers |
|---|-----------|-----------|------|---------|
| 0 | FPD/KPD on the two-stage samples | none | ~20 min | the biggest reporting gap |
| 1 | Teacher-forced stage B | none | ~45 min | does the A→B handoff cost anything |
| 2 | 10-run classifier averaging | none | ~8 h unattended | makes our numbers comparable to published |
| 3 | v4 latent ceiling | none | ~1 h | is the latent path actually dead |
| 4 | Occupancy head on u₁…u₄₄ | stage A only | ~2 h | the atom-at-zero defect |
| 5 | Focal hurdle loss for the AE (v5) | AE, 2 runs | ~2×20 h | does over-painting cost us AUC |
| **0b** | **0.5 MeV cut on existing AE/latent files** | **none** | **~20 min** | **cheapest test in the project** |

Experiments 0–3 need no training at all. **Do them before 4.**

---

## Experiment 0 — FPD/KPD on the two-stage model

**Why first.** It is the metric the CaloChallenge summary paper leads with for
distributional fidelity, we have it for the autoencoder only, and that one
measurement is worse than the worst published submission. Right now the
two-stage model's headline result cannot be placed against Table C16 at all.

```bash
cd "$REPO"            # the checkout on the GPU workstation
python scripts/cluster/run_full_evaluation.py --skip-generate \
    --gen-file generated/two_stage_ds2_100k.hdf5 \
    --out-dir evaluation_two_stage --modes fpd kpd
```

If `--modes` is not wired through, call `evaluate.py` directly:

```bash
cd CaloChallenge/CaloChallenge/code
# NOTE the mode names are 'fpd' and 'kpd' separately -- there is no 'fpd-kpd'.
# Both write to fpd_kpd_<dataset>.txt. No network is trained: they call
# jetnet.evaluation.fpd/kpd on high-level features, so this is cheap and
# runs fine without a large GPU.
for mode in fpd kpd; do
  python evaluate.py --input_file ../../generated/two_stage_ds2_100k.hdf5 \
      --reference_file ../../dataset_2_2.hdf5 --dataset 2 --mode $mode \
      --output_dir ../../evaluation_two_stage
done
```

**Compare against** (arXiv:2410.21611 Table C16, ×10³):

| | KPD | FPD |
|---|---:|---:|
| Geant4 floor | −0.028 | 10.78 |
| CaloDREAM | 0.023 | 24.65 |
| CaloDiffusion | 0.174 | 146.93 |
| worst published (CaloVAE+INN) | 45.61 | 5443 |
| AE pre-v4 (ours) | 91.60 | 11328 |

**Prediction.** FPD in the 100–600 range, i.e. between CaloScore and
CaloScore-distilled. Reasoning: FPD/KPD are sensitive to the per-voxel energy
distribution, where our voxel-distribution χ² of 0.0025 is the best of any
model we have.

**Falsified if** FPD > 1000. That would mean the low-level AUC of 0.578 is
being carried by marginals while the joint distribution is badly wrong, and it
would redirect all of experiment 4's effort toward whatever FPD is seeing.

---

## Experiment 0b — a 0.5 MeV model-side cut, on files we already have

The cheapest experiment here, and it should be run first. NOVA_TRANSFER.md
§5.2 shows that a single feature — the count of sub-MeV hits — nearly
reproduces the autoencoder's whole classifier score (8.3× excess → AUC 0.966
against the full DNN's 0.9888). §5.1 shows a 0.5 MeV cut takes that feature
from **0.966 to 0.624**, while 1 MeV and above *collapse* it to 0.9987 because
our file then has zero sub-MeV voxels against the reference's 114.

So there is a sharp optimum near 0.5 MeV, found on one feature over 20k
showers, and it needs the real classifier to confirm.

```bash
python - <<'EOF'
import h5py, numpy as np
for src, dst in [("ae_recon_ds2_full.hdf5", "ae_recon_cut05.hdf5"),
                 ("generated/latent_recon_ds2.hdf5", "latent_recon_cut05.hdf5")]:
    with h5py.File(src) as f:
        sh = f["showers"][:]; e = f["incident_energies"][:]
    sh = np.where(sh > 0.5, sh, 0.0)
    with h5py.File(dst, "w") as g:
        g.create_dataset("incident_energies", data=e, compression="gzip")
        g.create_dataset("showers", data=sh, compression="gzip")
    print("wrote", dst)
EOF

python scripts/cluster/run_full_evaluation.py --skip-generate --gen-file ae_recon_cut05.hdf5 \
    --out-dir evaluation_ae_cut05
```

**Prediction.** cls-low drops from 0.989 to 0.75–0.90. Not to 0.5 — the
mean lit-voxel energy (single-feature AUC 0.714) and hit count (0.690) are
still wrong and the cut only partly addresses them.

**Falsified if** the AUC does not move, which would mean the DNN was never
using the sub-MeV count and §5.2's single-feature inference is wrong. That is
a real possibility worth testing cheaply before spending 40 GPU-hours on
experiment 5.

**Also sweep 0.25 and 0.75 MeV** if 0.5 helps, since the optimum was located
on a proxy rather than on the classifier itself.

---

## Experiment 1 — teacher-forced stage B

**The question.** Stage B is trained on the **true** u of each shower
(`train_two_stage.py` builds `cond_shape` from `layerdata.encode`, never from
stage A's output), but at sampling time it receives stage A's **generated** u.
It has never seen an imperfect u during training — textbook exposure bias.

Stage A's own metrics tell you how wrong u is. They do not tell you whether
stage B is *sensitive* to that wrongness, which is a different quantity. This
experiment measures the sensitivity by removing stage A from the pipeline:

```
end-to-end      u ~ stage A       → w ~ stage B(·|u) → decode(u, w)
teacher-forced  u = encode(data)  → w ~ stage B(·|u) → decode(u, w)
```

This is the same diagnostic move as `train_latent.py --ceiling`: feed the
pipeline real data at the join and see how much error belongs to the generator
versus the join.

`models/3_two_stage/teacher_forced.py` implements it (CPU smoke-tested).

```bash
python -u -m calods2.teacher_forced \
    --data data/Dataset2.pt \
    --checkpoint-shape checkpoints/ts_shape.pt \
    --export generated/ts_teacher_forced_100k.hdf5 \
    --n-sample 100000 --sample-steps 50

python scripts/cluster/run_full_evaluation.py --skip-generate \
    --gen-file generated/ts_teacher_forced_100k.hdf5 \
    --out-dir evaluation_teacher_forced
```

**Where to look, and where not to.** Not the energy observables. `decode`
reconstructs the total as `f·E_inc·u₀` whatever w is, and the per-layer
projection forces `Σ_j w = 1`, so **E_tot/E_inc is exact by construction in
both runs.** The smoke test confirmed this (median 0.788 from an untrained
random network). Any handoff cost lives in the shape observables:

| metric | end-to-end | teacher-forced | read |
|---|---|---|---|
| cls-low AUC | 0.578 | ? | overall |
| cls-high AUC | 0.695 | ? | **the one that matters** |
| cls-resnet AUC | 0.804 | ? | **the one that matters** |
| χ² EC η / EC φ | 0.0010 / 0.0010 | ? | centroid placement |
| χ² Width η / φ | 0.0010 / 0.0009 | ? | lateral spread |
| χ² E layer (mean) | 0.0039 | ? | should barely move |
| χ² E_tot/E_inc | 0.0020 | ? | will not move — ignore it |

**Prediction.** A small gap: teacher-forced cls-high better by 0.02–0.05. The
reasoning is that stage A's u errors are modest (its E_tot χ² is 0.0020) and
the conv U-Net is a smooth function of its conditioning, so it should degrade
gracefully.

**Falsified if** the gap exceeds ~0.08 in cls-high. That would mean the handoff
is a major error source and the fix is to train stage B on *perturbed* u —
feeding it stage A's samples, or real u plus calibrated noise — which is a
cheap change with a large payoff. **It would also mean the two trainings can no
longer be decoupled, so `--skip-energy` would stop being safe.**

**Honest limitation.** The real u come from `dataset_2_1.hdf5`, the file both
models trained on. This is a handoff-sensitivity test, not a generalization
test.

---

## Experiment 2 — average the classifiers over 10 runs

The published tables average 10 independent classifier trainings with isotonic
calibration and quote ±0.004. Our single runs carry **±0.015** scatter, which
we know because two directories turned out to hold the same generated file
evaluated twice (cls-low 0.6700 vs 0.6827).

So every comparison we make at the 0.01 level — including "indistinguishable
from CaloDiffusion's 0.577" — is currently not supportable at the precision the
published numbers are quoted to.

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python CaloChallenge/CaloChallenge/code/evaluate.py \
      --input_file generated/two_stage_ds2_100k.hdf5 \
      --reference_file dataset_2_2.hdf5 --dataset 2 \
      --mode cls-low --save_mem \
      --output_dir evaluation_ts_seed$seed
done
```

Check whether `evaluate.py` exposes a seed flag first; if it does not, the
re-initialization per run is what provides the variation, which is exactly what
the paper averages over. Collect with:

```bash
grep -h -A2 "Final result" evaluation_ts_seed*/classifier_cls-low_2.txt \
  | grep -oE "^0\.[0-9]+" | awk '{s+=$1; ss+=$1*$1; n++} \
    END {m=s/n; printf "mean %.4f  sd %.4f  n %d\n", m, sqrt(ss/n-m*m), n}'
```

**Do this for the two-stage model and the conv U-Net at minimum.** It costs
nothing but wall time and it is the difference between a quotable result and an
anecdote.

---

## Experiment 3 — the v4 latent ceiling

**Why this is now necessary.** The conclusion that the latent path is dead
rests on §4.4 of MODEL_EVOLUTION.md: the latent sampler scored 0.991 against a
reconstruction ceiling of 0.989. But that ceiling was measured on
`best_calo_ae_cond_improved.pt` — **epoch 85, pre-v4, old units convention,
no hit head.** The v4 model has a hit head, measures lit fraction 0.256 against
pre-v4's 0.420, and has never been evaluated.

A pre-v4 decoder with no occupancy modelling has every reason to produce a poor
ceiling. The argument may be right; the evidence quoted for it is stale.

```bash
# 1. bring v4 back from the workstation
rsync -av "$WORKSTATION:$REPO/best_calo_ae_v4.pt" ./   # set both to your own
python -c "
import torch; ck=torch.load('best_calo_ae_v4.pt',map_location='cpu',weights_only=False)
print('two_head', ck.get('two_head'), '| epoch', ck.get('epoch'),
      '| units_gev', ck.get('shower_units_gev'), '| x_scale', ck.get('x_scale'))"
# two_head MUST be True and shower_units_gev MUST be True, or it is not v4

# 2. the ceiling: encode+decode REAL showers, no diffusion model involved
python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_v4.pt --ceiling \
    --export generated/latent_ceiling_v4.hdf5 --n-sample 100000

# 3. evaluate it
python scripts/cluster/run_full_evaluation.py --skip-generate \
    --gen-file generated/latent_ceiling_v4.hdf5 \
    --out-dir evaluation_ceiling_v4
```

**Prediction.** The v4 ceiling will be better than 0.989 but still far from
competitive — somewhere in 0.90–0.97. Reasoning: v4's occupancy is nearly
correct (0.256 vs 0.250), which removes the pre-v4 model's dominant defect,
but a 96-number bottleneck still loses azimuthal position (the latent model's
EC φ χ² ≈ 0.445 is the worst of any model we have), and that is a bottleneck
property rather than a loss-function property.

**Falsified if** the v4 ceiling comes in below ~0.85. That would mean the
latent path was abandoned on the strength of a stale checkpoint and deserves a
proper retry with a latent diffusion model trained on v4.

**Independent evidence that the conclusion survives anyway** (§4b of
MODEL_COMPARISON): across the latent sweep, reconstruction error improved while
classifier AUC did not move — 0.9888 at epoch 85 vs 0.9919 at epoch 460 for the
same 96-dim latent, FPD 11328 vs 12437. If reconstruction quality and
separability really are decoupled, a better autoencoder will not rescue this
path regardless of what experiment 3 returns. **Experiment 3 tests the ceiling
argument; the sweep tests the premise underneath it. They can disagree, and
that would be the interesting outcome.**

---

## Experiment 4 — occupancy head on u₁…u₄₄

### The defect

A calorimeter layer can be *exactly* empty, which forces u_i = 0 exactly. So
stage A's target is not a continuous density but a mixture:

```
p(u_i | E_inc) = π_i · δ(u_i) + (1 − π_i) · p_cont(u_i | E_inc)
```

Measured atom masses, and the standardized value each atom occupies:

| u index | 1 | 10 | 20 | 30 | 40 | 44 |
|---|---:|---:|---:|---:|---:|---:|
| π_i | 0.0001 | 0.0032 | 0.0913 | 0.2666 | 0.4173 | **0.4846** |
| atom at | −6.60 | −13.04 | −2.94 | −1.55 | −1.10 | −0.95 |

**17.4% of all u entries (i ≥ 1) are atoms at exactly zero**, and u₀ is never
zero (minimum E_tot/E_inc = 0.0063), so only the ratio coordinates need this.

A continuous interpolant `x(t) = (1−t)x₀ + σ(t)ε` with Gaussian ε cannot
produce an atom. Stage B has exactly the right mechanism for its own 75.5% atom
at w = 0 — the hit head plus gate, which writes gated voxels to *exactly*
`empty_value`. **Stage A is built with `hit_head=False` and has no such
mechanism at all.**

### Required code change

`empty_value` is a **scalar** buffer, and that is sufficient for stage B
because w uses one global standardization, so every empty voxel lands at
−0.5489. It is **not** sufficient for stage A: u is standardized
per-dimension, so the atom sits at a different standardized value for every
coordinate — 44 distinct values spanning −14.35 to −0.95.

In `velocity_net.py`:

```python
# before
self.register_buffer("empty_value", torch.tensor(float(empty_value)))

# after — accept a scalar or a per-coordinate vector
ev = torch.as_tensor(empty_value, dtype=torch.float32).reshape(1, -1) \
     if not isinstance(empty_value, (int, float)) \
     else torch.tensor(float(empty_value))
self.register_buffer("empty_value", ev)
```

A `(1, D)` buffer broadcasts correctly through every existing use —
`torch.where(lit, q, self.empty_value.expand_as(q))`, `x0b > empty_value +
1e-6`, and the checkpoint field (store it as a list). Verified:
`(1,45)` against `(B,45)` expands to `(B,45)`.

Then add to `layerdata.py`:

```python
def empty_value_u(stats):
    """Standardized value each u_i takes when its layer is exactly empty.
    Unlike w, u is standardized per-dimension, so this is a VECTOR."""
    y = _logit(torch.zeros(45))
    return ((y - stats["u_mean"]) / stats["u_std"]).reshape(1, -1)
```

and in `train_two_stage.py`, build stage A with `hit_head=True` and
`empty_value=layerdata.empty_value_u(stats)`.

**Gate choice matters and is not obvious.** `topk` keeps `k = round(Σp)`
entries per sample — sensible for 6480 voxels with a shower-wide hit count, but
there is no analogous global constraint on 45 energy ratios. Start with
`threshold` (σ(h) > 0.5), and sweep `sample` and `threshold` before trusting
either. This needs a `--gate-energy` flag, since stage B's gate should stay
`topk`.

### Run it

Stage B does **not** need retraining — it is conditioned on real u, so it is
independent of stage A (this is the decoupling that makes `--skip-energy`
safe). Only stage A is retrained.

```bash
# treatment: stage A with per-coordinate occupancy head
python -u -m calods2.train_two_stage --data data/Dataset2.pt \
    --checkpoint-energy checkpoints/ts_energy_hit.pt \
    --checkpoint-shape  checkpoints/ts_shape.pt \
    --steps-energy 20000 --skip-shape --energy-hit-head \
    --export generated/ts_hit_100k.hdf5 --n-sample 100000 --sample-steps 50

# CONTROL: identical compute, no occupancy head
python -u -m calods2.train_two_stage --data data/Dataset2.pt \
    --checkpoint-energy checkpoints/ts_energy_ctl.pt \
    --checkpoint-shape  checkpoints/ts_shape.pt \
    --steps-energy 20000 --skip-shape \
    --export generated/ts_ctl_100k.hdf5 --n-sample 100000 --sample-steps 50
```

**The control is not optional.** Retraining stage A at all changes its seed and
its EMA trajectory, so any improvement in the treatment run is confounded with
"we retrained stage A" unless a matched-compute run without the head is
measured alongside it.

### The prediction, and the trap

The naive prediction is "deep-layer χ² improves". **That is not a valid test,**
and here is why. Atom mass correlates strongly with per-layer χ² — r = 0.924
for the two-stage model. But it correlates just as strongly for models that
never compute u at all:

| model | r(atom mass, χ²) | r(layer index, χ²) |
|---|---:|---:|
| two-stage | +0.924 | +0.854 |
| conv U-Net | +0.915 | +0.843 |
| film MLP | +0.789 | +0.714 |
| AE reconstruction | **+0.930** | +0.860 |

The autoencoder — no u, no stick-breaking, not even generative — shows the
strongest correlation. So π_i is acting as a proxy for "this layer is usually
empty", a property of the *data* that makes every model struggle. The
correlation is real and says nothing about the representation.

**The valid prediction is differential:** χ²(E layer 40–44) improves in the
treatment run relative to **the matched control**, while the conv U-Net and AE
baselines (not retrained, so fixed) stay put. Specifically:

| quantity | control | treatment | prediction |
|---|---|---|---|
| χ² E layer 40–44 (mean) | ~0.011 | ? | **falls ≥ 25%** |
| χ² E layer 0–20 (mean) | ~0.0008 | ? | unchanged |
| χ² E_tot/E_inc | 0.0020 | ? | unchanged (u₀ has no atom) |
| cls-high AUC | 0.695 | ? | improves 0.01–0.03 |
| fraction of generated u_i exactly 0 | ~0 | ? | **→ π_i per coordinate** |

That last row is the direct check, and it is the one to look at first: it
measures whether the mechanism does what it is supposed to do, independently of
whether that helps. **If generated u_i are exactly zero at the right rate and
deep-layer χ² does not improve, the atom was not the binding defect** — a clean
negative result, and worth recording in MODEL_EVOLUTION.md §9.

**Falsified if** treatment and control are within noise on deep-layer χ² while
the zero-rate check passes.

---

## Experiment 5 — focal hurdle loss for the autoencoder

Specified in full in [NOVA_TRANSFER.md](NOVA_TRANSFER.md) §7, adapted from the
NOvA sparse-CAE report. Summary of why it is here:

Our AE reconstruction has **precision 0.539 against recall 0.923** and lights
2.32× too many voxels — over-painting, measured with the new
`quick_metrics --paired`. No metric we had could see it, because they are all
distribution comparisons and the lit *fraction* and total charge both looked
fine.

`calorimeter_loss_v5` (implemented, CPU-tested) replaces v4's plain BCE with a
focal hit term, which fixes over-painting at training time rather than by a
post-hoc cut.

**The critical caveat, and the reason this needs a careful A/B.** A post-hoc
energy cut also fixes it — F1 0.681 → 0.761 at a 2 MeV cut, lit fraction
0.420 → 0.249 — but it does so by deleting voxels below 1 MeV, which is 7.2%
of real hits and precisely the defect the two-stage model exists to fix.
**Check the sub-MeV hit fraction before believing any F1 improvement.**

```bash
python train_auto_encoder_improved.py --loss v4 --phi-circular --epochs 250 \
    --checkpoint-path ae_v4_ctl.pt
python train_auto_encoder_improved.py --loss v5 --focal-lambda 0.1 --tau 0.40 \
    --phi-circular --epochs 250 --checkpoint-path ae_v5.pt
```

**Falsified if** F1 rises but cls-low AUC does not, or if the sub-MeV hit
fraction falls — either means focal loss is buying placement the same way a cut
does.

---

## Recording the results

Whatever comes back, write it into the docs the same day:

- Numbers → [MODEL_COMPARISON.md](MODEL_COMPARISON.md)
- A mechanism that held → the relevant section of
  [MODEL_EVOLUTION.md](MODEL_EVOLUTION.md)
- **A mechanism that did not hold → §9 of MODEL_EVOLUTION.md**, with the
  control that refuted it

The last one is the most valuable and the easiest to skip. Three of the five
experiments above exist only because an earlier confident claim was checked and
failed.
