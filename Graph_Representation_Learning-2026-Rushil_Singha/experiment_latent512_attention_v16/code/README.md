# v16 — Autoencoder-only receptive-field ablation

Built from `v_15.2` (STE-pT, ±2 η/φ bounds, multiplicity count loss).
Flow matching removed. Everything here is reconstruction only.

---

## 1. The two "k"s, in plain language

These are two completely different knobs and it is worth being precise about
which one does what, because they pull in **opposite** directions.

### kNN `k` — how the graph is *built*

Before the network sees anything, each jet is turned into a graph. Each
particle becomes a node, and we draw an edge to its `k` nearest neighbours on
the (η, φ) plane.

> **k = how many other particles each particle is directly wired to.**

- Small k (say 4) → a **sparse, stringy** graph. To get from a particle on one
  edge of the jet to a particle on the other edge you have to walk through
  many intermediate particles. The graph is *wide* — a large **diameter**.
- Large k (say 16) → a **dense, blobby** graph. Almost everything is 2–3 steps
  from everything else. Small diameter.

### ChebNet `K` — how far one layer *reaches*

`K` is the order of the Chebyshev polynomial. One `Euler_ChebConv` layer
computes

```
out = x + ε · ( W₀·T₀(L̃)x + W₁·T₁(L̃)x + … + W_{K-1}·T_{K-1}(L̃)x )
```

`T_k(L̃)x` mixes in information from particles exactly `k` steps away in the
graph. So:

| K | what the layer sees |
|---|---|
| K = 1 | only `T₀` = the node itself → **no message passing at all** (the floor) |
| K = 2 | adds `T₁` → **exactly 1 hop** — the true "1-hop baseline" |
| K = 5 | up to 4 hops in a single layer |
| K = 10 | up to 9 hops in a single layer |

With `L` stacked layers the total receptive field is `L × (K−1)` hops.

> **K = how many hops one layer can reach in a single shot.**

A plain 1-hop GNN (GraphSAGE, GCN) has to stack 9 layers to see 9 hops, and
every extra layer smears the signal (oversmoothing). ChebNet gets there in one
layer with 9 separate weight matrices — one per hop distance — so it can treat
"my neighbour" and "the particle 7 hops away" **differently** instead of
averaging them together. That is the entire thesis.

### Why they fight each other

```
receptive field of the model  =  L × (K−1)   hops
size of the thing to cross    =  graph diameter  hops   ← set by kNN k
```

- If you make the graph **denser** (bigger kNN k), the diameter shrinks, so a
  small K already reaches everything and **K stops mattering** → your K sweep
  comes out as a flat line and the experiment says nothing.
- If you make the graph **too sparse** (kNN k = 2–3), it fragments into
  disconnected pieces, reconstruction collapses, and you are measuring
  breakage rather than receptive field.

So you want a graph that is **connected but long**: big enough diameter that
K = 5 is genuinely starved and K = 10 is genuinely fed.

### The kNN k I chose: **k = 8** (constant across all six runs)

You asked me to pick one and hold it fixed. Reasoning:

- v15/v15.2 used k = 10; the context doc proposed dropping to k = 4.
- k = 4 on JetNet-150 leaves a noticeable fraction of jets with disconnected
  components, which hurts reconstruction for reasons unrelated to K.
- k = 8 keeps essentially every jet connected (good reconstruction — your
  "must not hamper reconstruction" requirement) while leaving a median
  diameter of roughly 4–6 hops on 150-particle jets.

With `L = 2` layers that gives:

| encoder | receptive field | vs a ~5-hop jet |
|---|---|---|
| 1-hop (SAGE) | 2 hops | **starved** |
| ChebNet K = 5 | 8 hops | just covers it |
| ChebNet K = 10 | 18 hops | comfortably saturated |

That is a real contrast between the three, which is what you need.

**Do this before burning GPU hours** — it takes ~10 minutes and the table goes
straight into the paper:

```bash
python check_graph_setup.py --ks 4 6 8 10 16 --num-particles 150
```

If the measured median diameter comes back at 3 or lower, drop to `KNN_K = 6`
in both run scripts. If it comes back at 8+, k = 8 is fine or you can go to 10.
Honest note: **increasing kNN k increases each particle's direct neighbourhood
but decreases how much K can add**, so 8 is already at the generous end for
this experiment.

---

## 2. What I removed and what replaced it

The problem with v15.2 for this study: **Set2Set and the decoder Transformer
both mix information globally, regardless of K.** They can reconstruct
long-range structure that the encoder never captured, which hides the effect
you are trying to measure.

### Set2Set → StatsPool

Set2Set runs an LSTM that re-reads the whole particle set several times with
softmax attention. That is content-based global routing — it *is* attention.

Replacement: **`StatsPool`** — concatenate four fixed statistics over the
nodes, then project.

```
[ sum(h) | mean(h) | max(h) | std(h) ]   →  1024 dims  →  MLP  →  512
```

Why these four:

| statistic | what it keeps |
|---|---|
| **sum** | total amount — scales with multiplicity, so the count signal survives |
| **mean** | average character of the particles, independent of how many |
| **max** | the leading / hardest particle per feature channel |
| **std** | the spread — how heterogeneous the jet is |

It is permutation-invariant (a jet is an unordered set), handles variable
multiplicity, and critically it is a **fixed symmetric function**: every node
enters the reduction independently, so it **cannot invent correlations between
particles**. Any pairwise structure in the latent had to be built by the
ChebNet layers. That is exactly the property the thesis needs.

It also empirically loses very little versus Set2Set — sum+mean+max+std is the
standard PNA-style aggregator set and is much richer than the plain mean+sum
the context doc suggested, so reconstruction should stay close.

### Decoder Transformer → per-slot FiLM residual MLP

Removed the 2-layer `TransformerEncoder` over the 150 slots. To avoid simply
making the decoder weaker (which would confound the comparison), the capacity
is put back as **3 residual blocks that are FiLM-conditioned on `z`**. The same
`z` is broadcast to every slot, so this adds depth and expressivity but carries
**zero slot-to-slot information**. No cross-particle mixing at decode time.

### Both bottlenecks are still exactly 512-d

Set2Set outputs `2 × 256 = 512`. StatsPool projects `4 × 256 = 1024` down to
512. The bottleneck is the measuring instrument — it must not move between
runs, and it doesn't.

---

## 3. The single-hop baseline

`EulerSAGEConv` — strictly 1-hop message passing:

```
out = x + ε · ( W_self·x + W_neigh · mean_{j∈N(i)} x_j )
```

Deliberately built with the **same** Euler residual and the **same**
anti-symmetric weight parametrisation as `Euler_ChebConv`, so the only thing
that differs between it and the ChebNet is the receptive field — not the
stability trick, not the residual structure. This is also the layer family
Hariri et al. (2020) used, so it doubles as the external baseline.

(Note for the paper: `EulerSAGEConv` has 2 weight matrices, ChebNet K=5 has 5,
K=10 has 10. Parameter counts are recorded in every `metrics.json` so a
reviewer can't claim you just added capacity — and that is exactly what the
`ptD` control metric is there to rule out.)

---

## 4. The six runs

| script | pooling | decoder attn | encoders |
|---|---|---|---|
| `run_ablation_local.py` | StatsPool | off | SAGE, Cheb K=5, Cheb K=10 |
| `run_ablation_global.py` | Set2Set | on | SAGE, Cheb K=5, Cheb K=10 |

Everything else is identical: kNN k = 8, 2 conv layers, hidden 256, latent 512,
JetNet-150, 200 epochs each, AdamW lr 1e-3 with cosine decay, batch 512.

### Data

| | source | jets |
|---|---|---|
| **train** | JetNet `split="train"` (70%) | **all of it** — no subsampling |
| **test** | JetNet `split="test"` (15%) | **all of it** — never seen during training |

`split_fraction = [0.7, 0.15, 0.15]` is passed explicitly so the split is
reproducible. JetNet-150 has roughly 170–180k jets per type, so expect
**≈ 360k training jets and ≈ 78k test jets** in total — the exact counts are
printed at the top of every `log.txt` and stored in each `metrics.json` under
`n_train_per_type` / `n_test_per_type`. The 15% validation slice is unused;
there is no early stopping, so nothing needs it.

Jets are dropped only if they have fewer than 2 surviving particles (the
`failed` counter in the log) — a handful at most.

### The question this answers

For each encoder define the **attention gap**:

```
gap = long-range error WITHOUT attention  −  long-range error WITH attention
```

- **gap large for 1-hop, small for K=10** → the wider spectral receptive field
  is doing the job attention was doing. **This is the thesis result.**
- **gap roughly constant** → attention contributes something K cannot buy.
- **gap ≈ 0 everywhere** → attention was never earning its overhead on this
  task. Also a publishable finding.

---

## 5. Metrics

Reconstruction is evaluated **paired, per jet, on the entire JetNet test
split** (v15/v15.2 evaluated on training jets and un-paired — this is
stricter).

For observable *X*:

```
paired relative error = mean_over_jets |X(recon_i) − X(real_i)| / σ(X_real)
```

Dividing by the real spread makes different observables comparable, which raw
W1 does not (that is the "multiplicity W1 looks red but is fine" issue from
your notes — multiplicity spans 0–150, pT spans 0–1).

**Long-range observables (should improve with receptive field):**

| observable | formula | what it is |
|---|---|---|
| `girth` | Σᵢ wᵢ·rᵢ | jet width |
| `ang_scale` | Σᵢ<ⱼ wᵢwⱼΔRᵢⱼ / Σᵢ<ⱼ wᵢwⱼ | pT-weighted mean pairwise angle |
| `eec2` | Σᵢ<ⱼ wᵢwⱼΔRᵢⱼ² | two-point energy correlator, β=2 (β=2 weights *large* separations hardest → most long-range-sensitive) |
| `mass` | relative jet mass | |

**Control observable (should NOT improve with receptive field):**

| `ptD` | √(Σᵢ wᵢ²) | pT dispersion — purely local, has no angular information at all |

**This is the falsifiability test.** If `girth`/`ang_scale`/`eec2` improve with
K but `ptD` stays flat → the claim holds. If everything improves together →
it is just "bigger model = better" and you should say so.

Also reported: per-jet EMD (same family as Hariri et al.), multiplicity MAE,
particle-level W1 and W1/σ for η/φ/pT, and a per-jet-type breakdown (top jets
carry the most long-range structure from the 3-prong W decay).

---

## 6. Outputs

Per run, in `results/<tag>/`:

| file | what it shows |
|---|---|
| `recon_pairs.png` | **real vs reconstructed, per model** — 6 test jets (2 gluon, 2 quark, 2 top), three panels each: real graph, encoder→decoder reconstruction, and the two overlaid. Marker size ∝ pT. **The same 6 jets in every one of the 6 runs**, so you can flip between runs and compare directly. |
| `multiplicity.png` | is *n* actually reconstructed, or just guessed? (see §7) |
| `recon_sample.npz` | the raw arrays behind `recon_pairs.png`, if you want to re-plot without re-running |
| `observables.png` | real vs reconstructed histograms for every substructure observable, each annotated with its paired relative error, and labelled LONG-RANGE / CONTROL |
| `dist_comparison.png` | 9-panel particle-level real vs recon (η, φ, pT, 1st/5th/20th pT, multiplicity, mass, Σ pT) with W1 and W1/σ |
| `ae_loss.png` | training curve |
| `metrics.json` | every number |
| `log.txt` | full run log incl. graph diameter and parameter counts |

Aggregate, in `results/_aggregate/`:

| file | what it shows |
|---|---|
| `fig1_receptive_field.png` | **the money figure** — long-range error vs receptive field, one line per arm |
| `fig2_attention_gap.png` | side-by-side bars + the gap per encoder |
| `fig3_longrange_vs_control.png` | per-observable; watch that `ptD` stays flat |
| `fig4_per_jet_type.png` | gluon / quark / top breakdown |
| `summary_table.md` / `.csv` | the full table |

`aggregate_results.py` also prints a DATA / TRAINING REPORT to stdout with the
exact jet counts, epochs, receptive field in hops, parameter counts, and final
training loss for all six runs.

---

## 7. Where does *n* come from? (there is no generation here)

You are right that nothing is being generated — this is pure reconstruction,
real test jet in, reconstruction out. But the particle count still has to be
*predicted*, and it is worth being precise about how.

**The decoder never sees *n*.** It sees exactly one thing: the 512-d latent
`z`. There is no path that leaks the true multiplicity to it.

The decoder always emits **150 slots**, no matter the jet. Each slot produces
4 numbers: η, φ, pT, and a **mask logit**. The mask logit becomes

```python
mask_soft = sigmoid(logit * 2.0)          # in [0, 1]
active    = mask_soft > 0.5               # this slot is a real particle
n_recon   = active.sum()                  # <-- this is n
```

So `n` is just "how many of the 150 slots turned themselves on", and that
decision is made from `z` alone. If the bottleneck loses the multiplicity, `n`
comes out wrong — which is exactly what you want it to be sensitive to.

Three terms in the loss push `n` to be right:

| term | what it does for *n* |
|---|---|
| `BCE(mask)` | per-slot on/off. Targets are pT-sorted and left-packed, so slot *i* should be on iff *i < n_true* — combined with the slot-ID embedding, each slot can learn "am I inside the first *n*?" |
| `0.5 × MSE(count)/150` | BCE treats the 150 slots independently and carries no signal about the **total**. This term penalises `Σ mask_soft − n_true` directly. This is the v15.2 multiplicity fix. |
| `Sinkhorn` | the transport weights are the masks, so a wrong count misallocates pT mass and is penalised again |

**Why StatsPool matters here:** the `sum` channel is the one that scales with
node count. Take it out (mean-only pooling) and multiplicity information is
substantially weakened on its way into the latent. Set2Set encodes it through
the LSTM readout instead. Both arms have a route; they are just different
routes.

**One consequence worth knowing:** the STE hard mask uses the same 0.5
threshold, and pT is normalised over active slots only. So every reconstructed
jet has Σ pT = 1 exactly, matching real jets by construction. The "jet pT sum
spreads below 1" problem from your notes was a *flow-sampled-latent* artefact —
it does not arise in reconstruction.

**How to check it actually works:** `multiplicity.png` per run.

- Left panel: `n_recon` vs `n_real` as a hexbin. Points should hug the red
  `y = x` line. If they instead form a **horizontal band** on the orange
  dotted line, the decoder is ignoring `z` and predicting the dataset average.
- Right panel: the error histogram, with MAE, bias, and exact-match rate.

In `metrics.json`: `mult_mae`, `mult_bias`, `mult_corr`, `mult_exact_frac`,
and `mult_mae_mean_baseline`. **That last one is the number to compare
against** — it is the MAE you would get by always predicting the average
multiplicity. If `mult_mae` is not clearly below it, `n` is not being
reconstructed, it is being guessed, and every other metric should be read with
suspicion.

---

## 8. How to run

```bash
cd v16_ablation
export JETNET_DATA_DIR=/pscratch/sd/r/rushil13/jetnet_data
export ABLATION_SAVE_ROOT=/pscratch/sd/r/rushil13/gsoc_rushil/v16_ablation/results
```

Step 0 — verify the graph geometry (do this first, ~10 min):

```bash
python check_graph_setup.py --ks 4 6 8 10 16 --num-particles 150
```

Step 1 — the primary arm (no global attention), 3 runs:

```bash
python run_ablation_local.py
```

Step 2 — the attention arm, 3 runs:

```bash
python run_ablation_global.py
```

Step 3 — figures and table:

```bash
python aggregate_results.py
```

Both run scripts checkpoint every 10 epochs and resume automatically, so a
killed job just picks up where it left off.

### Fast smoke test first

At the top of `run_ablation_local.py` set:

```python
EPOCHS            = 3
MAX_JETS_PER_TYPE = 20000    # train
MAX_TEST_PER_TYPE = 5000     # test
```

Run it, confirm the figures look sane, then revert to `200 / None / None`.
Strongly recommended before queuing the full thing — the graph caches built by
the smoke run are keyed separately, so they will not pollute the real run.

### Compute cost — read this before queuing

Six runs × 200 epochs × ~700 batches on ~360k jets. The Sinkhorn loss on
512×150×3 point clouds dominates each step. On a single A100 expect roughly
**8–14 hours per run, so 2–3.5 GPU-days for all six.** If that is too much,
in order of what costs you least scientifically:

1. `MAX_JETS_PER_TYPE = 60000` (≈180k train jets) — halves wall-clock, barely
   moves reconstruction quality at this bottleneck size
2. `EPOCHS = 120` — the loss curve is close to flat well before 200
3. Drop `chebK5` and keep only SAGE vs K=10 — but you lose the shape of the
   curve, so do this last

Every run checkpoints every 10 epochs and resumes, so splitting across
several jobs is fine.

### Single run

```bash
python ae_core.py --tag mytest --conv cheb --K 5 --knn 8 --layers 2 --epochs 200
```

---

## 9. What to check when the numbers land

1. **Is `ptD` flat across K?** If not, the effect is capacity, not receptive
   field — report it that way.
2. **Is the K=5 → K=10 line flat?** Likely, if the measured diameter is small.
   That is a saturation result, not a failure — say so, and cite the diameter
   table. The SAGE → K=5 step is the one carrying the real signal.
3. **Does the attention gap shrink with K?** That is the headline for the
   "how much attention overhead can ChebNet absorb" question.
4. **Do top jets show a bigger effect than gluon jets?** Predicted by the
   3-prong structure argument.
5. **Is `mult_mae` well below `mult_mae_mean_baseline`?** If not, the model is
   guessing the average particle count rather than reconstructing it — fix
   that before reading anything else (see §7).

---

## 10. Files

| file | purpose |
|---|---|
| `ae_core.py` | all models, data, training, metrics, plots. `run_one(cfg)` is the entry point |
| `run_ablation_local.py` | **Script 1** — no global attention, 3 encoders |
| `run_ablation_global.py` | **Script 2** — Set2Set + decoder attention, 3 encoders |
| `check_graph_setup.py` | graph diameter table (run first) |
| `aggregate_results.py` | combined figures + summary table |
