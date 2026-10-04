# v17_final — what a compressed jet keeps

Graph autoencoders on JetNet-150. Three experiments, one metric suite.

| # | experiment | question | runs | status |
|---|---|---|---|---|
| 1 | **Pooling** | Does learned attention pooling (Set2Set) beat a fixed summary (StatsPool)? | GraphSAGE, latent 512, round 1 | trained — re-score only |
| 2 | **Compression** | When does the bottleneck start to bind? | GraphSAGE at latent 512 / 128 / 64, scout | trained — re-score only |
| 3 | **Reach** | Does seeing further across the graph help, with the operator held fixed? | GraphSAGE, ChebNet K=2, ChebNet K=6 at latent 64, × 3 seeds | **9 runs to train** |

## Files

| file | what it does |
|---|---|
| `ae_core.py` | model, data, training, evaluation. `python ae_core.py --help` |
| `suite.py` | the metric suite — the only place any metric is defined |
| `reeval.py` | scores already-trained models with the suite (no training) |
| `aggregate.py` | builds `results.md` and the paper figures from every `suite.json` |
| `check_graph_setup.py` | the graph-geometry table for the setup section |
| `diagnostic.py` | why reach doesn't help: D₂ re-score, Test A (factorisation), Test B (latent probe), Experiment 1 count check |
| `slurm_reach.sh` | Experiment 3 — 9-job array |
| `slurm_reeval.sh` | Experiments 1 and 2 — re-score old runs, then aggregate |
| `slurm_final.sh` | the last 6 runs: ChebNet K=4 and GraphSAGE + Set2Set at latent 64, 3 seeds each |
| `slurm_diagnostic.sh` | the diagnostics, then aggregate again (~3-5 h, resubmit to resume) |

## Run it

Paste `v17_final/` next to `v16_ablation/` on NERSC. **Submit from the folder that contains `v17_final/`**, not from inside it.

```bash
# 1. Experiment 3 — the first job alone builds the shared graph cache (~30-40 min)
sbatch --array=0 v17_final/slurm_reach.sh
# wait for "cache saved" in logs/reach-*_0.out, then
sbatch --array=1-8 v17_final/slurm_reach.sh

# 2. Experiments 1 and 2 — any time, even while step 1 trains (~1-2 h)
sbatch v17_final/slurm_reeval.sh

# 3. once all 9 reach runs finish — the paper tables
python v17_final/aggregate.py \
    --pooling     $SCRATCH/gsoc_rushil/v16_ablation/results \
    --compression $SCRATCH/gsoc_rushil/v16_ablation/results_scout \
    --reach       $SCRATCH/gsoc_rushil/v17_final/results \
    --out         $SCRATCH/gsoc_rushil/v17_final/paper
```

Output: `paper/results.md` plus `fig_pooling`, `fig_compression`, `fig_reach` and `fig_reach_by_size`.

```bash
# 4. the last six runs (caches exist, so all six start together)
sbatch v17_final/slurm_final.sh
# 5. once they finish: diagnostics + results.md for everything
sbatch v17_final/slurm_diagnostic.sh
```

Output: `diagnostics/diagnostics.md` + `fig_probe.png`, and `paper/results.md` rebuilt with log D₂.

If a reach job hits its 24 h limit, resubmit only that index — `sbatch --array=5 v17_final/slurm_reach.sh`. It resumes from its last 10-epoch checkpoint.

## Experiment 3 design

| index | encoder | reach | seed | role |
|---|---|---|---|---|
| 0 1 2 | GraphSAGE | 2 hops | 0 1 2 | local baseline |
| 3 4 5 | ChebNet K=2 | 2 hops | 0 1 2 | **operator control** — same reach as GraphSAGE, same parameter count |
| 6 7 8 | ChebNet K=6 | 10 hops | 0 1 2 | **reach arm** |

- **GraphSAGE vs K=2** isolates the operator (spatial vs spectral) with reach held at 2 hops.
- **K=2 vs K=6** isolates reach with the operator held fixed. **This is the headline.**
- Reach = layers × (K−1). K=6 gives 10 hops, which is the 90th-percentile jet diameter at kNN k=6. More reach buys nothing: the widest jets are already covered.

Everything else is shared: StatsPool, no decoder attention, undirected kNN(6), 2 layers, 1/√K normalisation, 200 epochs on the full JetNet-150 training split. That is the same budget as the latent-512 pooling runs.

## The metric suite

Every metric is the **paired relative error**: compute the quantity on each real test jet and on its reconstruction, average `|recon − real|` over jets, and divide by the real spread. 0 is perfect.

| group | metric | why |
|---|---|---|
| **primary** | ECF(2, β=1) | energy correlation function; the √ inside ΔR doesn't factorise |
| | EEC at ΔR > 0.2 | a threshold on a *pair* can't be written as per-particle sums |
| diagnostic | log D₂, raw D₂ | two-prong vs one-prong (3-point correlator). Reported but **not in any group**: raw D₂ correlated −0.93 with training loss across the reach runs, and log D₂ still −0.77 — both reward worse models |
| **control** | ptD | no geometry at all. Nothing about reach can move it |
| **holistic** | EMD | energy-flow EMD (Komiske–Metodiev–Thaler) — the metric Hariri et al. 2020 used |
| **support** | girth, jet mass, ECF(2, β=2) | these **factorise exactly** into per-particle sums, so a zero-reach model computes them perfectly. They show reconstruction quality and are never evidence about reach |

**A reach effect** means the primary errors fall from K=2 to K=6 while the support errors and ptD stay level. If all three move together, the models differ in how well they trained, not in what they capture.

## What changed from v16, and why it matters

1. **Undirected graphs — a bug fix.** `kneighbors_graph` builds a *directed* graph, and on jet graphs about half the edges have no reverse edge. ChebNet's Chebyshev polynomials are only bounded when the Laplacian has a real spectrum, which needs a symmetric graph. On the directed graph the terms grow with order, which penalises higher K for reasons that have nothing to do with reach. GraphSAGE never builds these polynomials, so **the old GraphSAGE results are unaffected**. The old ChebNet results are not trustworthy, which is why Experiments 1 and 2 use GraphSAGE only.
2. **Resume actually works — a bug fix.** PyG's `Linear` saves an extra derived key next to each anti-symmetric weight, so every resume-after-timeout failed and silently restarted from epoch 0. No earlier run ever resumed, so no past result is affected. `load_state()` now fixes this.
3. **The correct EMD.** The old "EMD" weighted every particle equally and used pT as a coordinate. That is not the metric Hariri et al. used. `suite.emd_energy_flow` now implements the energy-flow EMD: pT-weighted, with distance ΔR/R.
4. **One suite, everywhere.** Training-time evaluation and `reeval.py` call the same `suite.score`, so every table comes from the same code.
5. **Kept:** 1/√K term normalisation, anti-symmetric Euler convolutions, StatsPool, the FiLM slot decoder, and JetNet's own train/test splits.

## Before you start

- Check that the round-1 and scout folders on NERSC still hold `encoder.pth` and `decoder.pth`. `reeval.py` skips any run without them and says so.
- If your old results live somewhere other than `$SCRATCH/gsoc_rushil/v16_ablation/results` and `…/results_scout`, edit the `ROUND1` / `SCOUT` lines in `slurm_reeval.sh`.
